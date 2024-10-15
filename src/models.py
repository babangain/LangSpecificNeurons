import torch, json, os, sys
from pathlib import Path
sys.path.append(Path(__file__).parent)
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoModel, BitsAndBytesConfig, QuantoConfig, FbgemmFp8Config
from typing import List, Tuple, Union
import pandas as pd
from dataset import WikipediaDataset, WikipediaDatasetHF
from abc import ABC, abstractmethod
from utils import models_map
from torch.utils.data import DataLoader

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
                prediction_output = self.model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids).logits.cpu() # (b, T-1, d)
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

class ModelForMLM(torch.nn.Module):
    def __init__(self, device: torch.device, model_name: str, quant_config: Union[None, BitsAndBytesConfig]):
        super(ModelForMLM, self).__init__()
        self.model_name = model_name
        self.model_name_srt = self.model_name.split("/")[-1]
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.quant_config = quant_config
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, quantization_config=self.quant_config, device_map="auto")
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
    
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        
    def get_layers(self) -> torch.nn.ModuleList:
        m = self.model_name.lower()
        if ("llama" in m) or ("mistral" in m) or ("sarvam" in m) or ("aya-23" in m):
            layers = self.model.model.layers
        elif "bloom" in m:
            layers = self.model.model.transformer.h
        else:
            raise NotImplementedError("Invalid model name!")
        return layers

    def get_target_linear_module(self, layer_idx: int) -> Tuple[str, torch.nn.Linear]:
        m = self.model_name.lower()
        mlp = self.get_layers()[layer_idx].mlp        
        if ("llama" in m) or ("mistral" in m) or ("sarvam" in m) or ("aya-23" in m):
            target_module = mlp.gate_proj
            name = "gate_proj"
        elif "bloom" in m:
            target_module = mlp.dense_h_to_4h
            name = "dense_h_to_4h"
        else:
            raise NotImplementedError("Invalid model name!")
        return name, target_module
    
    def create_hook_function(self, name: str):
        def hook_function(module: torch.nn.Module, inputs: torch.Tensor, outputs: torch.Tensor):
            self.activations[name] = outputs # (b, T, 4d)
            outputs.retain_grad()
        return hook_function

    def register_hook(self):
        self.hooks_list = [] # List[L]
        for layer_idx in range(len(self.get_layers())):
            _, target_module = self.get_target_linear_module(layer_idx=layer_idx)
            hook_function = self.create_hook_function(name=layer_idx)
            h = target_module.register_forward_hook(hook_function)
            self.hooks_list.append(h)
    
    def remove_hook(self):
        for h in self.hooks_list:
            h.remove()
    
    def forward(self, input_ids: torch.tensor, attention_mask: torch.tensor, labels: torch.tensor) -> dict:
        """input_ids = attention_mask = labels = (b, T)
        attention_mask = 0 means padded token and 1 means real token
        """
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        labels = labels.to(self.device)
        
        self.activations = {} # Dict[(b, T, 4d)]
        self.register_hook()
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)["logits"] # (b, T, V)
        self.remove_hook()
        
        loss = self.loss_fn(out.flatten(0,1), labels.flatten()) 
        return {"pred_labels": out.argmax(dim=-1), "loss": loss} # (b, T)
       
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

def main_mlm(model_name: str, device: torch.device) -> None:
    ds = WikipediaDatasetHF(model_name=model_name, lang="en", max_context_len=512)
    dl = DataLoader(dataset=ds, batch_size=4)
    input_dict = next(iter(dl))
    quant_config = QuantoConfig(weights="float8")
    model = ModelForMLM(device=device, model_name=model_name, quant_config=quant_config)
    print(model)
    out = model(**input_dict)
    print(out)   
    print([v.grad for k,v in model.activations.items()])
    out["loss"].backward()   
    print([v.grad.sum() for k,v in model.activations.items()])

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}...")
    
    main_mlm(models_map["llama3"], device=device)
    
    