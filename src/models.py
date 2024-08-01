import torch, json, os
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Tuple
import pandas as pd
from pathlib import Path

class LlamaModelForProbing(torch.nn.Module):
    def __init__(self, tokenizer: AutoTokenizer, device: torch.device):
        super(LlamaModelForProbing, self).__init__()
        self.pretrain_llama_name = tokenizer.name_or_path
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(self.pretrain_llama_name).to(self.device)
        self.tokenizer = tokenizer
    
    def create_hook_function(self, act_fn: torch.nn.Module):
        def hook_function(module: torch.nn.Module, inputs: torch.Tensor, outputs: torch.Tensor):
            self.neuron_output_list.append(act_fn(outputs).detach())
        return hook_function
    
    def register_hook(self):
        self.hooks_list = [] # List[L]
        for L in self.model.model.layers:
            hook_function = self.create_hook_function(L.mlp.act_fn)
            h = L.mlp.gate_proj.register_forward_hook(hook_function)
            self.hooks_list.append(h)
    
    def remove_hook(self):
        for h in self.hooks_list:
            h.remove()
            
    def forward(self, input_ids: torch.tensor, attention_mask: torch.tensor) -> dict:
        """input_ids = attention_mask = token_type_ids = (b, T)
        attention_mask = 0 means padded token and 1 means real token
        """
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        
        self.neuron_output_list = [] # List[L x (b, T, 4d)]
        self.register_hook()
        prediction_output = self.model(input_ids=input_ids, 
                                       attention_mask=attention_mask).logits.cpu() # (b, T, d)
        self.remove_hook()
        
        neuron_output = torch.cat(self.neuron_output_list, dim=0) # (L, b, T, 4d)
        avg_neuron_output = neuron_output.mean(dim=(1,2)).cpu() # (L, 4d)
        gt_zero_count = torch.where(condition=neuron_output > 0, input=1, other=0) # (L, b, T, 4d)
        gt_zero_count = gt_zero_count.sum(dim=(1,2)).cpu() # (L, 4d) 
            
        return {"logits": prediction_output, # (b, T, d)
                "avg_neuron_out": avg_neuron_output, # (L, d)
                "neuron_out_gt_zero_count": gt_zero_count # (L, d)
        } 
         
def main(model_name: str, device: torch.device) -> None:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if "llama" in model_name.lower():
        model = LlamaModelForProbing(tokenizer=tokenizer, device=device)
    else:
        raise NotImplementedError("Invalid model name!")
    
    input_ids = torch.randint(low=0, high=1000, size=(32, 2048))
    attention_mask = torch.ones_like(input_ids)
    out = model(input_ids, attention_mask)
    print(out)      
    
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    models = ["meta-llama/Meta-Llama-3-8B-Instruct"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}...")
    
    main(models[0], device=device)
    