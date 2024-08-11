import torch, json, os, sys
from pathlib import Path
sys.path.append(Path(__file__).parent)
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Tuple, Union
import pandas as pd
from dataset import WikipediaDataset
from abc import ABC, abstractmethod

class AbstractModelForProbing(ABC, torch.nn.Module):
    def __init__(self, device: torch.device, model_name: str, model: torch.nn.Module, tokenizer: Union[AutoTokenizer, None]):
        super(AbstractModelForProbing, self).__init__()
        self.pretrain_model_name = model_name
        self.device = device
        self.model_name = model_name.split("/")[-1]
        self.model = model.to(self.device)
        self.tokenizer = tokenizer

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
            prediction_output = self.model(input_ids=input_ids, attention_mask=attention_mask).logits.cpu() # (b, T, d)
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

class BloomzModelForProbing(AbstractModelForProbing):
    def __init__(self, device: torch.device, model_name: str, tokenizer: Union[AutoTokenizer, None]):
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
        super(BloomzModelForProbing, self).__init__(device=device, model_name=model_name, model=model, tokenizer=tokenizer)
    
    def get_layers(self) -> torch.nn.ModuleList:
        return self.model.transformer.h
    
    def get_target_module(self, layer_idx: int) -> torch.nn.Module:
        return self.get_layers()[layer_idx].mlp.gelu_impl
    
def main(model_name: str, device: torch.device) -> None:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if "llama" in model_name.lower():
        model = LlamaModelForProbing(tokenizer=tokenizer, device=device, model_name=model_name)
    elif "bloomz" in model_name.lower():
        model = BloomzModelForProbing(tokenizer=tokenizer, device=device, model_name=model_name)
    else:
        raise NotImplementedError("Invalid model name!")

    ds = WikipediaDataset(tokenizer=tokenizer, lang="en", max_context_len=512)
    dl = ds.prepare_dataloader(batch_size=4, frac=0.001)
    input_dict = next(iter(dl))
    n = 1000
    intervene_config = {
        "indices": list(zip([1]*(n-1), range(n))),
        "value": 0
    }
    out = model(input_dict["input_ids"], input_dict["attention_mask"], intervene_config=intervene_config)
    print(out["logits"].sum())      
    out = model(input_dict["input_ids"], input_dict["attention_mask"], intervene_config=None)
    print(out["logits"].sum())  
    
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    torch.cuda.empty_cache()
    models = ["meta-llama/Llama-2-7b-hf", "bigscience/bloomz-7b1"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}...")
    
    main(models[1], device=device)
    