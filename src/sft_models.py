import torch, os, sys
from pathlib import Path
sys.path.append(Path(__file__).parent)
from transformers import AutoTokenizer, AutoModelForCausalLM, QuantoConfig
from typing import List, Tuple, Union
from dataset import WikipediaDatasetHF
from torch.utils.data import DataLoader

class ModelForMLM(torch.nn.Module):
    def __init__(self, device: torch.device, model_name: str, vocab_size: int,  quant_config: Union[QuantoConfig, None]):
        super(ModelForMLM, self).__init__()
        self.device = device
        self.model_name = model_name
        self.model_name_srt = model_name.split("/")[-1]
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, quantization_config=quant_config, device_map="auto")
        self.loss_fn = torch.nn.CrossEntropyLoss()
    
    def forward(self, input_ids: torch.tensor, attention_mask: torch.tensor, labels: Union[None, torch.tensor]) -> torch.tensor:
        z = self.model(input_ids=input_ids, attention_mask=attention_mask)["logits"] # (b, T, V)
        if labels is not None:
            labels = labels.to(self.device)  
            loss = self.loss_fn(torch.flatten(z, start_dim=0, end_dim=1), torch.flatten(labels, start_dim=0, end_dim=1)) 
        else:
            loss = None
        return {"pred_labels": z.argmax(dim=-1), "loss": loss}

class CustomLinearLayer(torch.nn.Module):
    def __init__(self, linear: torch.nn.Linear, trainable_indices: List[int], frozen_indices: List[int]):
        assert len(trainable_indices) + len(frozen_indices) == linear.out_features, f"Indices should add to out_features"
        super(CustomLinearLayer, self).__init__()
        self._weight = linear.weight.clone().detach().T # (d, 4d)
        self.trainable_indices = trainable_indices
        self.frozen_indices = frozen_indices
        self.W_tr = torch.nn.Parameter(self._weight[:, self.trainable_indices].clone().detach())  # Trainable (d, d1)
        self.W_fr = self._weight[:, self.frozen_indices].clone().detach()  # Frozen, no gradient (d, d2) s.t. d1 + d2 = 4d
  
    def forward(self, x):
        """x.shape = (b, T, d)"""
        y_tr = torch.matmul(x, self.W_tr)  # Trainable part (b, T, d1)
        y_fr = torch.matmul(x, self.W_fr)  # Frozen part (b, T, d2)
        y = self.intelligent_concat(y_tr, y_fr, x.shape) # (b, T, 4d) s.t. d1 + d2 = 4d
        return y

    def intelligent_concat(self, y_tr, y_fr, input_shape):
        """Concatenates the outputs from trainable and frozen parts.
        y_tr.shape = (b, T, d1), y_fr.shape = (b, T, d2)
        input_shape = shape of the input (b, T, d)
        """
        b, T, _ = input_shape
        final_output = torch.zeros(b, T, self._weight.shape[1]).to(y_tr.device)  # (b, T, 4d)
        for i, idx in enumerate(self.frozen_indices):
            final_output[:, :, idx] = y_fr[:, :, i]
        for i, idx in enumerate(self.trainable_indices):
            final_output[:, :, idx] = y_tr[:, :, i]
        return final_output

class ModelForMLMWithSFT(torch.nn.Module):
    def __init__(self, device: torch.device, tokenizer: AutoTokenizer, model_name: str, neurons_frac: float, quant_config: Union[None, QuantoConfig]):
        super(ModelForMLMWithSFT, self).__init__()
        self.model_name = model_name
        self.device = device
        self.tokenizer = tokenizer
        
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        self.model = ModelForMLM(device=self.device, model_name=self.model_name, vocab_size=len(self.tokenizer), quant_config=quant_config)
        self.model.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.neurons_frac = neurons_frac
        self.k, lin_name = self._get_num_neurons()
        self.W_pretrained = {name: val.clone().detach().T for name, val in self.model.model.named_parameters() if f".mlp.{lin_name}" in name}
        
        for param in self.model.model.model.parameters():
            param.requires_grad = False
        for param in self.model.model.lm_head.parameters():
            param.requires_grad = True
    
    def _get_num_neurons(self) -> int:
        name, mlp = self.get_target_linear_module(layer_idx=0)
        linear = getattr(mlp, name)
        k = int(self.neurons_frac * len(self.get_layers()) * linear.weight.T.shape[1])
        return k, name
    
    def prepare_for_round1_sft(self) -> None:
        for param in self.model.model.model.parameters():
            param.requires_grad = False
        for i, layer in enumerate(self.get_layers()):
            name, mlp = self.get_target_linear_module(layer_idx=i)
            linear = getattr(mlp, name)
            linear.weight.requires_grad_(True)
        self.calc_num_sft_params()
    
    def _get_topk_neurons(self) -> List[Tuple[int, int]]:
        norm_diffs = []
        for i, layer in enumerate(self.get_layers()):
            name, mlp = self.get_target_linear_module(layer_idx=i)
            linear = getattr(mlp, name)
            
            for key in self.W_pretrained.keys():
                if key.endswith(f"{name}.weight"):
                    pretrain_weight = self.W_pretrained[key] # (d, 4d)
                    break
                
            ft1_weight = linear.weight.clone().detach().T # (d, 4d)
            pretrain_norms = torch.norm(pretrain_weight, dim=0)  # (4d,)
            ft1_norms = torch.norm(ft1_weight, dim=0)  # (4d,)
            
            norm_diff = torch.abs(pretrain_norms - ft1_norms)  # (4d,)
            for j in range(norm_diff.shape[0]):
                norm_diffs.append((i, j, norm_diff[j].item()))  # (layer_index, neuron_index, norm_diff)
        
        norm_diffs = sorted(norm_diffs, key=lambda x: x[2], reverse=True)
        top_k_neurons = [(layer, col) for layer, col, _ in norm_diffs[:self.k]]
        self.W_pretrained = None
        return top_k_neurons

    def prepare_for_round2_sft(self) -> None:
        self.top_k_neurons = self._get_topk_neurons()  # List[N x (i, j)]
        for param in self.model.model.parameters():
            param.requires_grad = False

        for i, layer in enumerate(self.get_layers()):
            name, mlp = self.get_target_linear_module(layer_idx=i)
            linear = getattr(mlp, name)
            weight = linear.weight.clone().detach().T  # (d, 4d)
            trainable_indices = [j for _, j in self.top_k_neurons if _ == i]  # (d1,)
            frozen_indices = [j for j in range(weight.shape[1]) if (i, j) not in self.top_k_neurons]  # (d2,) s.t. d1 + d2 = 4d
            if len(trainable_indices) > 0:
                custom_linear = CustomLinearLayer(linear=linear, trainable_indices=trainable_indices, frozen_indices=frozen_indices)
                setattr(mlp, name, custom_linear)
        self.calc_num_sft_params()

    def calc_num_sft_params(self) -> None:
        # Check if the requires_grad are set correctly
        train_params = 0
        total_params = 0
        for name, param in self.model.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                train_params += param.numel()
        print(f"Number of total parameters: {total_params}")
        print(f"Number of trainable parameters: {train_params}")
        print(f"SFT efficiency: {train_params * 100 / total_params:.3f}%")
    
    def get_layers(self) -> torch.nn.ModuleList:
        m = self.model_name.lower()
        if ("llama" in m) or ("mistral" in m) or ("sarvam" in m) or ("aya-23" in m):
            layers = self.model.model.model.layers
        elif "bloom" in m:
            layers = self.model.model.model.transformer.h
        else:
            raise NotImplementedError("Invalid model name!")
        return layers
    
    def get_target_act_module(self, layer_idx: int) -> torch.nn.Module:
        m = self.model_name.lower()
        mlp = self.get_layers()[layer_idx].mlp
        if ("llama" in m) or ("mistral" in m) or ("sarvam" in m) or ("aya-23" in m):
            target_module = mlp.act_fn
        elif "bloom" in m:
            target_module = mlp.gelu_impl
        else:
            raise NotImplementedError("Invalid model name!")
        return target_module

    def get_target_linear_module(self, layer_idx: int) -> Tuple[str, torch.nn.Linear]:
        m = self.model_name.lower()
        mlp = self.get_layers()[layer_idx].mlp        
        if ("llama" in m) or ("mistral" in m) or ("sarvam" in m) or ("aya-23" in m):
            target_module = mlp
            name = "gate_proj"
        elif "bloom" in m:
            target_module = mlp
            name = "dense_h_to_4h"
        else:
            raise NotImplementedError("Invalid model name!")
        return name, target_module
    
    def create_hook_function(self, intervene_config: dict):
        def hook_function(module: torch.nn.Module, inputs: torch.Tensor, outputs: torch.Tensor):
            indices = intervene_config["indices"] # List[n x Tuple[layer_index, neuron_index]] (n neurons to intervene)
            value = intervene_config["value"] # List[n] 
            for i, (layer_idx, neuron_idx) in enumerate(indices):
                if module == self.get_target_act_module(layer_idx=layer_idx):
                    outputs.index_fill_(dim=-1, index=torch.tensor([neuron_idx]).to(self.device), value=value[i])
        return hook_function

    def register_hook(self, intervene_config: Union[dict, None]):
        self.hooks_list = [] # List[L]
        for layer_idx in range(len(self.get_layers())):
            hook_function = self.create_hook_function(intervene_config=intervene_config)
            h = self.get_target_act_module(layer_idx=layer_idx).register_forward_hook(hook_function)
            self.hooks_list.append(h)
    
    def remove_hook(self):
        for h in self.hooks_list:
            h.remove()
    
    def forward(self, input_ids: torch.tensor, attention_mask: torch.tensor, intervene_config: Union[dict, None] = None, labels: Union[torch.tensor, None] = None) -> torch.tensor:
        """intervene_config = {
            "indices": [(i, j) | i: layer index, j: neuron index]
            "value": [i | for each index i in indices])
        }"""
        assert list(input_ids.shape).__len__() == 2, "inputs rank must be 2 and inputs.shape = (b, T)"
        
        if intervene_config is not None:
            self.register_hook(intervene_config=intervene_config)
            prediction_output = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels) # (b, T, c)
            self.remove_hook()
        else:
            prediction_output = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels) # (b, T, c)
        return prediction_output
        
def main_for_int(model_name: str, device: torch.device) -> None:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = ModelForMLMWithSFT(device=device, tokenizer=tokenizer, model_name=model_name, neurons_frac=0.01, quant_config=None).to(device)
    input_ids = torch.randint(low=0, high=100, size=(16, 256)).to(device) # (b, T)
    labels = torch.cat([input_ids[:, 1:], input_ids[:,0:1]], dim=-1) # (b, T)
    attention_mask = torch.ones(size=(16, 256)).to(device) # (b, T)
    print(model)
    model.calc_num_sft_params()
    intervene_config = {
        "indices": torch.tensor([[0,1], [1,2], [2,3], [2,4], [2,5]]),
        "value": torch.tensor([0, 0, 0, 0, 0])
    }
    model.eval()
    with torch.no_grad():
        out1 = model(input_ids=input_ids, attention_mask=attention_mask, intervene_config=None, labels=labels) # (b, T)
        out2 = model(input_ids=input_ids, attention_mask=attention_mask, intervene_config=intervene_config, labels=labels) # (b, T)
    print(out1["pred_labels"].sum(), out1["loss"]) 
    print(out2["pred_labels"].sum(), out2["loss"])

def main_for_sft(model_name: str, device: torch.device) -> None:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = ModelForMLMWithSFT(device=device, tokenizer=tokenizer, model_name=model_name, neurons_frac=0.001, quant_config=None).to(device)
    model.prepare_for_round1_sft()
    
    ds = WikipediaDatasetHF(model_name=model_name, lang="en", max_context_len=256)
    dl = DataLoader(ds, batch_size=4)
    batch = next(iter(dl))
    input_ids = batch["input_ids"].to("cuda")
    attention_mask = batch["attention_mask"].to("cuda")
    labels = batch["labels"].to("cuda")
    
    model.eval()
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask, intervene_config=None, labels=labels) # (b, T)
    print(out["pred_labels"].sum(), out["loss"]) 
    
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}...")
    
    main_for_sft("meta-llama/Llama-2-7b-hf", device=device)
    
    