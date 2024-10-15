import torch, os, sys
from pathlib import Path
sys.path.append(Path(__file__).parent)
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, BitsAndBytesConfig
from typing import List, Tuple, Union

class LoRALayer(torch.nn.Module):
    def __init__(self, rank: int, alpha: float, d_in: int, d_out: int, mask_A: Union[torch.Tensor, None], mask_B: Union[torch.Tensor, None]):  
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
        self.mask_A = mask_A
        self.mask_B = mask_B
    
    def forward(self, x: torch.tensor) -> torch.tensor:
        assert x.dim() >= 2, "Input tensor must have at least 2 dimensions."
        assert x.shape[-1] == self.d_in, f"Expected the last dimension of input to be {self.d_in}, but got {x.shape[-1]}."
        masked_A = self.A.to(self.mask_A.device) * self.mask_A
        masked_B = self.B.to(self.mask_A.device) * self.mask_B
        delta_W = torch.matmul(masked_A, masked_B) * (self.alpha / self.rank)
        z = torch.matmul(x, delta_W)
        return z
    
class LinearWithLoRA(torch.nn.Module):
    def __init__(self, linear: torch.nn.Linear, rank: int, alpha: float, mask_A: Union[torch.Tensor, None], mask_B: Union[torch.Tensor, None]):
        super(LinearWithLoRA, self).__init__()
        self.rank = rank
        self.alpha = alpha
        self.linear = linear
        self.d_in = self.linear.in_features
        self.d_out = self.linear.out_features
        self.lora = LoRALayer(rank=self.rank, alpha=self.alpha, d_in=self.d_in, d_out=self.d_out, mask_A=mask_A, mask_B=mask_B)

    def forward(self, x: torch.tensor) -> torch.tensor:
        assert x.dim() >= 2, "Input tensor must have at least 2 dimensions."
        assert x.shape[-1] == self.d_in, f"Expected the last dimension of input to be {self.d_in}, but got {x.shape[-1]}."
        z1 = self.linear(x) 
        z2 = self.lora(x) 
        z = z1 + z2
        return z

class ModelForMLM(torch.nn.Module):
    def __init__(self, device: torch.device, model_name: str, quant_config: Union[BitsAndBytesConfig, None]):
        super(ModelForMLM, self).__init__()
        self.device = device
        self.model_name = model_name
        self.model_name_srt = model_name.split("/")[-1]
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, quantization_config=quant_config, device_map="auto")
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
    
    def forward(self, input_ids: torch.tensor, attention_mask: torch.tensor, labels: Union[None, torch.tensor]) -> dict:
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)["logits"] # (b, T, V)
        if labels is not None:
            labels = labels.to(self.device)  
            loss = self.loss_fn(out.flatten(0,1), labels.flatten()) 
        else:
            loss = None
        return {"pred_labels": out.argmax(dim=-1), "loss": loss} # (b, T)

class ModelForMLMWithLoRA(torch.nn.Module):
    def __init__(self, device: torch.device, tokenizer: AutoTokenizer, model_name: str, lora_rank: int, lora_alpha: float, apply_only_mlp: bool, quant_config: Union[None, BitsAndBytesConfig]):
        super(ModelForMLMWithLoRA, self).__init__()
        self.model_name = model_name
        self.device = device
        self.tokenizer = tokenizer
        self.model = ModelForMLM(device=self.device, model_name=self.model_name, quant_config=quant_config)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model.model.config.pad_token_id = self.tokenizer.pad_token_id
        
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.rank = lora_rank
        self.alpha = lora_alpha
        self.apply_lora(rank=self.rank, alpha=self.alpha, apply_only_mlp=apply_only_mlp)
    
    def apply_lora(self, rank: int, alpha: float, apply_only_mlp: bool) -> None:
        if apply_only_mlp:
            layers = self.get_layers()
            for layer_idx, layer in enumerate(layers):
                target_linear_name, target_module = self.get_target_linear_module(layer_idx=layer_idx)
                up_proj_linear = getattr(target_module, target_linear_name)
                mask_A, mask_B = ModelForMLMWithLoRA.create_lora_mask(device=device, d_in=up_proj_linear.in_features, d_out=up_proj_linear.out_features, rank=rank, frozen_neuron_ids=None)
                linear_lora = LinearWithLoRA(up_proj_linear, rank, alpha, mask_A, mask_B)
                setattr(target_module, target_linear_name, linear_lora)
        else:
            ModelForMLMWithLoRA.replace_linear_with_lora(device=self.device, model=self.model.model.model, rank=rank, alpha=alpha)            
        
        head_linear = getattr(self.model.model, "lm_head")
        mask_A, mask_B = ModelForMLMWithLoRA.create_lora_mask(device=self.device, d_in=head_linear.in_features, d_out=head_linear.out_features, rank=rank, frozen_neuron_ids=None)
        linear_lora = LinearWithLoRA(head_linear, rank, alpha, mask_A, mask_B)
        setattr(self.model.model, "lm_head", linear_lora)
            
    @staticmethod
    def replace_linear_with_lora(device: torch.device, model: torch.nn.Module, rank: int, alpha: float):
        for name, module in model.named_children():
            if isinstance(module, torch.nn.Linear):
                mask_A, mask_B = ModelForMLMWithLoRA.create_lora_mask(device=device, d_in=module.in_features, d_out=module.out_features, rank=rank, frozen_neuron_ids=None)
                linear_lora = LinearWithLoRA(module, rank, alpha, mask_A, mask_B)
                setattr(model, name, linear_lora) # parent is model, child is module
            else:
                ModelForMLMWithLoRA.replace_linear_with_lora(device, module, rank, alpha)
    
    @staticmethod
    def create_lora_mask(device: torch.device, d_in: int, d_out: int, rank: int, frozen_neuron_ids: Union[None, torch.tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """frozen_neuron_ids = [j | j: neuron index for a layer i]
        """
        mask_A = torch.ones(d_in, rank).to(device)
        mask_B = torch.ones(rank, d_out).to(device)
        if frozen_neuron_ids is not None:
            for neuron_id in frozen_neuron_ids:
                mask_B[:, neuron_id] = 0  # Zero out the entire column for the frozen neurons
        return mask_A, mask_B
     
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
    
    def forward(self, input_ids: torch.tensor, attention_mask: torch.tensor, intervene_config: Union[dict, None] = None, labels: Union[torch.tensor, None] = None) -> dict:
        """intervene_config = {
            "indices": [(i, j) | i: layer index, j: neuron index]
            "value": [i | for each index i in indices])
        }"""
        assert list(input_ids.shape).__len__() == 2, "inputs rank must be 2 and inputs.shape = (b, T)"
        if intervene_config is not None:
            self.register_hook(intervene_config=intervene_config)
            prediction_output = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels) # (b, T)
            self.remove_hook()
        else:
            prediction_output = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels) # (b, T)
        return prediction_output

class ModelForCLS(torch.nn.Module):
    def __init__(self, device: torch.device, model_name: str, num_class: int, quant_config: Union[BitsAndBytesConfig, None]):
        super(ModelForCLS, self).__init__()
        self.device = device
        self.model_name = model_name
        self.model_name_srt = model_name.split("/")[-1]
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.base = AutoModel.from_pretrained(self.model_name, quantization_config=quant_config, device_map="auto")
        self.d = self.base.config.hidden_size
        self.c = num_class
        self.loss_fn = torch.nn.CrossEntropyLoss()

        # Multi-layer classification head with ReLU activation
        self.head = torch.nn.Sequential(
            torch.nn.Linear(self.d, 1024),      
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 256),      
            torch.nn.ReLU(),
            torch.nn.Linear(256, 64),      
            torch.nn.ReLU(),
            torch.nn.Linear(64, 16),      
            torch.nn.ReLU(),                   
            torch.nn.Linear(16, self.c)       
        ).to(self.device)
    
    def forward(self, input_ids: torch.tensor, attention_mask: torch.tensor, labels: Union[None, torch.tensor]) -> torch.tensor:
        z = self.base(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state # (b, T, d)
        y = z[:, -1, :].to(self.device) # (b, d) (last position embedding)
        out = self.head(y) # (b, c)
        if labels is not None:
            labels = labels.to(self.device)  
            loss = self.loss_fn(out, labels) 
        else:
            loss = None
        return {"logits": out, "loss": loss}
 
class ModelForCLSWithLoRA(torch.nn.Module):
    def __init__(self, device: torch.device, tokenizer: AutoTokenizer, model_name: str, num_class: int, lora_rank: int, lora_alpha: float, quant_config: Union[None, BitsAndBytesConfig], frozen_neurons: Union[None, torch.tensor]):
        super(ModelForCLSWithLoRA, self).__init__()
        self.model_name = model_name
        self.device = device
        self.tokenizer = tokenizer
        self.num_class = num_class
        self.model = ModelForCLS(device=self.device, model_name=self.model_name, num_class=self.num_class, quant_config=quant_config)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model.base.config.pad_token_id = self.tokenizer.pad_token_id
        
        for param in self.model.base.parameters():
            param.requires_grad = False
        for param in self.model.head.parameters():
            param.requires_grad = True
        
        self.rank = lora_rank
        self.alpha = lora_alpha
        self.apply_lora(rank=self.rank, alpha=self.alpha, frozen_neurons=frozen_neurons)
    
    def apply_lora(self, rank: int, alpha: float, frozen_neurons: Union[None, torch.tensor]) -> None:
        """frozen_neuron_ids = [(i, j) | j: neuron index for a layer i]
        """
        if frozen_neurons is not None:
            frozen_neurons_dict = {}
            for (layer_idx, neuron_idx) in frozen_neurons:
                layer_idx = layer_idx.item()
                if layer_idx not in frozen_neurons_dict:
                    frozen_neurons_dict[layer_idx] = []
                frozen_neurons_dict[layer_idx].append(neuron_idx.item())

            for layer_idx, layer in enumerate(self.get_layers()):
                target_linear_name, target_module = self.get_target_linear_module(layer_idx=layer_idx)
                if layer_idx in frozen_neurons_dict:
                    frozen_neuron_ids = frozen_neurons_dict[layer_idx]
                    up_proj_linear = getattr(target_module, target_linear_name)
                    mask_A, mask_B = ModelForCLSWithLoRA.create_lora_mask(device=self.device, d_in=up_proj_linear.in_features, d_out=up_proj_linear.out_features, rank=rank, frozen_neuron_ids=frozen_neuron_ids)
                    up_proj_lora = LinearWithLoRA(up_proj_linear, rank, alpha, mask_A, mask_B)
                    setattr(target_module, target_linear_name, up_proj_lora)
                    
                ModelForCLSWithLoRA.replace_linear_with_lora(device=self.device, model=layer, rank=rank, alpha=alpha, name_to_skip="linear")
        else:
            ModelForCLSWithLoRA.replace_linear_with_lora(device=self.device, model=self.model.base, rank=rank, alpha=alpha, name_to_skip="")            

    @staticmethod
    def replace_linear_with_lora(device: torch.device, model: torch.nn.Module, rank: int, alpha: float, name_to_skip: str):
        for name, module in model.named_children():
            if isinstance(module, torch.nn.Linear) and name != name_to_skip:
                mask_A, mask_B = ModelForCLSWithLoRA.create_lora_mask(device=device, d_in=module.in_features, d_out=module.out_features, rank=rank, frozen_neuron_ids=None)
                linear_lora = LinearWithLoRA(module, rank, alpha, mask_A, mask_B)
                setattr(model, name, linear_lora) # parent is model, child is module
            else:
                ModelForCLSWithLoRA.replace_linear_with_lora(device, module, rank, alpha, name_to_skip)
    
    @staticmethod
    def create_lora_mask(device: torch.device, d_in: int, d_out: int, rank: int, frozen_neuron_ids: Union[None, torch.tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """frozen_neuron_ids = [j | j: neuron index for a layer i]
        """
        mask_A = torch.ones(d_in, rank).to(device)
        mask_B = torch.ones(rank, d_out).to(device)
        if frozen_neuron_ids is not None:
            for neuron_id in frozen_neuron_ids:
                mask_B[:, neuron_id] = 0  # Zero out the entire column for the frozen neurons
        return mask_A, mask_B
     
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
    
    def get_layers(self) -> torch.nn.ModuleList:
        m = self.model_name.lower()
        if ("llama" in m) or ("mistral" in m) or ("sarvam" in m) or ("aya-23" in m):
            layers = self.model.base.layers
        elif "bloom" in m:
            layers = self.model.base.transformer.h
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
            prediction_output = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels) # (b, c)
            self.remove_hook()
        else:
            prediction_output = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels) # (b, c)
        return prediction_output
        
def main(model_name: str, device: torch.device) -> None:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model = ModelForCLSWithLoRA(device=device, tokenizer=tokenizer, model_name=model_name, num_class=3, lora_rank=8, lora_alpha=16, quant_config=None, frozen_neurons=torch.tensor([[0,1], [1,2], [2,3], [2,4], [2,5]])).to(device)
    model = ModelForMLMWithLoRA(device=device, tokenizer=tokenizer, model_name=model_name, lora_rank=8, lora_alpha=16, quant_config=None, apply_only_mlp=False).to(device)
    input_ids = torch.randint(low=0, high=100, size=(16, 256)).to(device) # (b, T)
    labels = torch.cat([input_ids[:, 1:], input_ids[:, 0:1]], dim=-1).to(device) # (b, T)
    attention_mask = torch.ones(size=(16, 256)).to(device) # (b, T)
    print(model)
    model.calc_num_lora_params()
    intervene_config = {
        "indices": torch.tensor([[0,1], [1,2], [2,3], [2,4], [2,5]]),
        "value": torch.tensor([0, 0, 0, 0, 0])
    }
    model.eval()
    with torch.no_grad():
        out1 = model(input_ids=input_ids, attention_mask=attention_mask, intervene_config=None, labels=labels) # (b, c)
        out2 = model(input_ids=input_ids, attention_mask=attention_mask, intervene_config=intervene_config, labels=labels) # (b, c)
    print(out1["pred_labels"].sum(), out1["loss"]) 
    print(out2["pred_labels"].sum(), out2["loss"])
    
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}...")
    
    main("meta-llama/Meta-Llama-3.1-8B", device=device)
    
    