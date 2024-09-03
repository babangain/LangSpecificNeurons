import torch, json, os, sys
from pathlib import Path
sys.path.append(Path(__file__).parent)
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoModel
from typing import List, Tuple, Union
import pandas as pd
from dataset import WikipediaDataset
from abc import ABC, abstractmethod
from utils import models_map

class LoRALayer(torch.nn.Module):
    def __init__(self, rank: int, alpha: float, d_in: int, d_out: int):  
        super(LoRALayer, self).__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.alpha = alpha
        self.rank = rank
        
        self.A = torch.nn.Parameter(
            data=torch.normal(mean=0, std=0.01, size=(self.d_in, self.rank)), 
            requires_grad=True
        )
        self.B = torch.nn.Parameter(
            data=torch.zeros(size=(self.rank, self.d_out)),
            requires_grad=True
        )
    
    def forward(self, x: torch.tensor) -> torch.tensor:
        assert x.dim() >= 2, "Input tensor must have at least 2 dimensions."
        assert x.shape[-1] == self.d_in, f"Expected the last dimension of input to be {self.d_in}, but got {x.shape[-1]}."
        delta_W = torch.matmul(self.A, self.B) * (self.alpha / self.rank)
        z = torch.matmul(x, delta_W)
        return z
    
class LinearWithLoRA(torch.nn.Module):
    def __init__(self, linear: torch.nn.Linear, rank: int, alpha: float):
        super(LinearWithLoRA, self).__init__()
        self.rank = rank
        self.alpha = alpha
        self.linear = linear
        self.d_in = self.linear.in_features
        self.d_out = self.linear.out_features
        self.lora = LoRALayer(rank=self.rank, alpha=self.alpha, d_in=self.d_in, d_out=self.d_out)

    def forward(self, x: torch.tensor) -> torch.tensor:
        assert x.dim() >= 2, "Input tensor must have at least 2 dimensions."
        assert x.shape[-1] == self.d_in, f"Expected the last dimension of input to be {self.d_in}, but got {x.shape[-1]}."
        z1 = self.linear(x) 
        z2 = self.lora(x) 
        z = z1 + z2
        return z
    
class ModelForCLSWithLoRA(torch.nn.Module):
    def __init__(self, device: torch.device, model_name: str, num_class: int, lora_rank: int, lora_alpha: float):
        super(ModelForCLSWithLoRA, self).__init__()
        self.model = ModelForCLS(device=device, model_name=model_name, num_class=num_class)
        for param in self.model.base.parameters():
            param.requires_grad = False
        self.rank = lora_rank
        self.alpha = lora_alpha
        ModelForCLSWithLoRA.replace_linear_with_lora(model=self.model.base, rank=self.rank, alpha=self.alpha)
    
    @staticmethod
    def replace_linear_with_lora(model: torch.nn.Module, rank: int, alpha: float):
        for name, module in model.named_children():
            if isinstance(module, torch.nn.Linear):
                linear_lora = LinearWithLoRA(module, rank, alpha)
                setattr(model, name, linear_lora) # parent is model, child is module
            else:
                ModelForCLSWithLoRA.replace_linear_with_lora(module, rank, alpha)
       
    def calc_num_lora_params(self) -> None:
        # Check if the requires_grad are set correctly
        train_params = 0
        total_params = 0
        for name, param in self.model.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                train_params += param.numel()
        print(f"Number of total parameters: {total_params}")
        print(f"Number of trainable parameters: {train_params}")
        print(f"LoRA efficiency: {train_params * 100 / total_params:.3f}%")
        
    def forward(self, input_ids: torch.tensor, attention_mask: torch.tensor) -> torch.tensor:
        assert list(input_ids.shape).__len__() == 2, "inputs rank must be 2 and inputs.shape = (b, T)"
        z = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return z
    
class ModelForCLS(torch.nn.Module):
    def __init__(self, device: torch.device, model_name: str, num_class: int):
        super(ModelForCLS, self).__init__()
        self.device = device
        self.model_name = model_name
        self.model_name_srt = model_name.split("/")[-1]
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.base = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.d = self.base.config.hidden_size
        self.c = num_class
        self.head = torch.nn.Linear(in_features=self.d, out_features=self.c).to(self.device)
    
    def forward(self, input_ids: torch.tensor, attention_mask: torch.tensor, labels: Union[None, torch.tensor] = None) -> torch.tensor:
        z = self.base(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state # (b, T, d)
        y = self.head(z.to(self.device)) # (b, T, c)
        out = y[:, -1, :] # (b, c) (last position embedding)
        return out

class AbstractModelForProbing(ABC, torch.nn.Module):
    def __init__(self, device: torch.device, model_name: str, model: torch.nn.Module, tokenizer: Union[AutoTokenizer, None]):
        super(AbstractModelForProbing, self).__init__()
        self.pretrain_model_name = model_name
        self.device = device
        self.model_name = model_name.split("/")[-1]
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.model_type = "ed" if "aya-101" in model_name.lower() else "d"

    @abstractmethod
    def get_layers(self) -> torch.nn.ModuleList:
        """Abstract method to get the layers of the model. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def get_target_module(self, layer_idx: int) -> torch.nn.Module:
        """Abstract method to get the target module for hooking at a given layer index. Must be implemented by subclasses."""
        pass
    
    def create_hook_function(self, intervene_config: Union[dict, None]):
        def hook_function(module: torch.nn.Module, inputs: torch.Tensor, outputs: torch.Tensor):
            if intervene_config is None:
                self.neuron_output_list.append(outputs.detach())
            else:
                indices = intervene_config["indices"] # List[n x Tuple[layer_index, neuron_index]] (n neurons to intervene)
                value = intervene_config["value"] # scalar
                for layer_idx, neuron_idx in indices:
                    if module == self.get_target_module(layer_idx=layer_idx):
                        outputs.index_fill_(dim=-1, index=torch.tensor([neuron_idx]).to(self.device), value=value)
                self.neuron_output_list.append(outputs.detach())       
        return hook_function

    def register_hook(self, intervene_config: Union[dict, None]):
        self.hooks_list = [] # List[L]
        for layer_idx in range(len(self.get_layers())):
            hook_function = self.create_hook_function(intervene_config=intervene_config)
            h = self.get_target_module(layer_idx=layer_idx).register_forward_hook(hook_function)
            self.hooks_list.append(h)
    
    def remove_hook(self):
        for h in self.hooks_list:
            h.remove()
    
    def forward(self, input_ids: torch.tensor, attention_mask: torch.tensor, intervene_config: Union[dict, None]) -> dict:
        """input_ids = attention_mask = token_type_ids = (b, T)
        attention_mask = 0 means padded token and 1 means real token
        intervene_config = {"indices": List[Tuple[i, j]] where i in [0, L-1] and j in [0, 4d-1], "value": float}
        """
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        
        self.neuron_output_list = [] # List[L x (b, T, 4d)]
        self.model.eval()
        with torch.no_grad():
            self.register_hook(intervene_config=intervene_config)
            if self.model_type == "ed":
                decoder_input_ids = input_ids[:, :-1]  # Shifting input IDs by one 
                prediction_output = self.model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids).logits.cpu() # (b, T, d)
            elif self.model_type == "d":
                prediction_output = self.model(input_ids=input_ids, attention_mask=attention_mask).logits.cpu() # (b, T, d)
            else:
                raise ValueError("Sorry! Invalid model type")
            self.remove_hook()
        
        neuron_output = torch.stack(self.neuron_output_list, dim=0) # (L, b, T, 4d)
        neuron_output = neuron_output.permute(1, 2, 0, 3) # (b, T, L, 4d)
        avg_neuron_output = neuron_output.mean(dim=(0, 1)).cpu() # (L, 4d)
        gt_zero_count = (neuron_output > 0) # (b, T, L, 4d)
        gt_zero_count = gt_zero_count.sum(dim=(0, 1)).cpu() # (L, 4d) 
            
        return {"logits": prediction_output, # (b, T, V)
                "tokens_count": prediction_output.shape[0] * prediction_output.shape[1], # scalar: b*T
                "avg_neuron_out": avg_neuron_output, # (L, 4d)
                "neuron_out_gt_zero_count": gt_zero_count # (L, 4d)
        }

class LlamaModelForProbing(AbstractModelForProbing):
    def __init__(self, device: torch.device, model_name: str, tokenizer: Union[AutoTokenizer, None]):
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
        super(LlamaModelForProbing, self).__init__(device=device, model_name=model_name, model=model, tokenizer=tokenizer)
    
    def get_layers(self) -> torch.nn.ModuleList:
        return self.model.model.layers
    
    def get_target_module(self, layer_idx: int) -> torch.nn.Module:
        return self.get_layers()[layer_idx].mlp.act_fn

class BloomModelForProbing(AbstractModelForProbing):
    def __init__(self, device: torch.device, model_name: str, tokenizer: Union[AutoTokenizer, None]):
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
        super(BloomModelForProbing, self).__init__(device=device, model_name=model_name, model=model, tokenizer=tokenizer)
    
    def get_layers(self) -> torch.nn.ModuleList:
        return self.model.transformer.h
    
    def get_target_module(self, layer_idx: int) -> torch.nn.Module:
        return self.get_layers()[layer_idx].mlp.gelu_impl

class MistralModelForProbing(AbstractModelForProbing):
    def __init__(self, device: torch.device, model_name: str, tokenizer: Union[AutoTokenizer, None]):
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
        super(MistralModelForProbing, self).__init__(device=device, model_name=model_name, model=model, tokenizer=tokenizer)
    
    def get_layers(self) -> torch.nn.ModuleList:
        return self.model.model.layers
    
    def get_target_module(self, layer_idx: int) -> torch.nn.Module:
        return self.get_layers()[layer_idx].mlp.act_fn

class SarvamModelForProbing(AbstractModelForProbing):
    def __init__(self, device: torch.device, model_name: str, tokenizer: Union[AutoTokenizer, None]):
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
        super(SarvamModelForProbing, self).__init__(device=device, model_name=model_name, model=model, tokenizer=tokenizer)
    
    def get_layers(self) -> torch.nn.ModuleList:
        return self.model.model.layers
    
    def get_target_module(self, layer_idx: int) -> torch.nn.Module:
        return self.get_layers()[layer_idx].mlp.act_fn

class AyaEncDecModelForProbing(AbstractModelForProbing):
    def __init__(self, device: torch.device, model_name: str, tokenizer: Union[AutoTokenizer, None]):
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
        super(AyaEncDecModelForProbing, self).__init__(device=device, model_name=model_name, model=model, tokenizer=tokenizer)
    
    def get_layers(self) -> torch.nn.ModuleList:
        return self.model.decoder.block
    
    def get_target_module(self, layer_idx: int) -> torch.nn.Module:
        return self.get_layers()[layer_idx].layer[2].DenseReluDense.act

class AyaDecModelForProbing(AbstractModelForProbing):
    def __init__(self, device: torch.device, model_name: str, tokenizer: Union[AutoTokenizer, None]):
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
        super(AyaDecModelForProbing, self).__init__(device=device, model_name=model_name, model=model, tokenizer=tokenizer)
    
    def get_layers(self) -> torch.nn.ModuleList:
        return self.model.model.layers
    
    def get_target_module(self, layer_idx: int) -> torch.nn.Module:
        return self.get_layers()[layer_idx].mlp.act_fn

def get_tokenizer_and_model(model_name: Union[str, None], device: torch.device) -> Tuple[AutoTokenizer, torch.nn.Module]:
    if model_name is None:
        tokenizer = None
        model = None
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if "llama" in model_name.lower():
            model = LlamaModelForProbing(tokenizer=tokenizer, device=device, model_name=model_name)
        elif "bloom" in model_name.lower():
            model = BloomModelForProbing(tokenizer=tokenizer, device=device, model_name=model_name)
        elif "mistral" in model_name.lower():
            model = MistralModelForProbing(tokenizer=tokenizer, device=device, model_name=model_name)
        elif "sarvam" in model_name.lower():
            model = SarvamModelForProbing(tokenizer=tokenizer, device=device, model_name=model_name)
        elif "aya-101" in model_name.lower():
            model = AyaEncDecModelForProbing(tokenizer=tokenizer, device=device, model_name=model_name)
        elif "aya-23" in model_name.lower():
            model = AyaDecModelForProbing(tokenizer=tokenizer, device=device, model_name=model_name)
        else:
            raise NotImplementedError("Invalid model name!")
    return tokenizer, model
        
def main(model_name: str, device: torch.device) -> None:
    tokenizer, model = get_tokenizer_and_model(model_name=model_name, device=device)
    ds = WikipediaDataset(model_name=model_name, lang="en", max_context_len=512)
    dl = ds.prepare_dataloader(batch_size=4, frac=0.001)
    input_dict = next(iter(dl))
    n = 1000
    intervene_config = {
        "indices": list(zip([1]*(n-1), range(n))),
        "value": 0
    }
    print(model)
    out = model(input_dict["input_ids"], input_dict["attention_mask"], intervene_config=intervene_config)
    print(out["logits"].sum())      
    out = model(input_dict["input_ids"], input_dict["attention_mask"], intervene_config=None)
    print(out["logits"].sum())  
    
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "5"
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}...")
    
    main(models_map["llama2"], device=device)
    
    