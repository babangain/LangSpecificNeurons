import torch, json, os, sys, tqdm, pickle, datetime
from pathlib import Path
sys.path.append(Path(__file__).parent)
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple
import pandas as pd
from dataset import WikipediaDataset
from models import LlamaModelForProbing
from activation import Activation

class LangNeuron:
    def __init__(self, tokenizer: AutoTokenizer, model: torch.nn.Module, max_context_len: int, batch_size: int):
        self.tokenizer = tokenizer
        self.model = model
        self.cwd = Path.cwd()
        self.Tmax = max_context_len
        self.batch_size = batch_size
        
        self.lang_list = ["en", "fr", "es", "vi", "id", "ja", "zh"]
        self.lang_neuron_frac = 0.01
        self.norm_act_prob = self.get_norm_act_prob()
    
    def info(self) -> str:
        return f"[INFO] {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    
    def get_norm_act_prob(self) -> dict:
        act_prob_dict = {}
        sum_act_prob = 0
        for lang in self.lang_list:
            dataset = WikipediaDataset(tokenizer=self.tokenizer, lang=lang, max_context_len=self.Tmax)   
            act = Activation(tokenizer=self.tokenizer, model=self.model, dataset=dataset)
            act_prob = act.get_activation_probability(batch_size=self.batch_size)["act_prob"].to(self.model.device)
            act_prob_dict[lang] = act_prob
            sum_act_prob += act_prob
        
        norm_act_prob_dict = {}
        for lang, act_prob in act_prob_dict.items():
            norm_act_prob_dict[lang] = act_prob/sum_act_prob
        return norm_act_prob_dict
        
    def calculate_LAPE_score(self) -> torch.tensor:
        norm_act_prob = torch.stack(list(self.norm_act_prob.values()), dim=0) # (k, L, 4d)
        epsilon = 1e-10
        lape = -1 * (norm_act_prob * torch.log2(norm_act_prob + epsilon)).sum(dim=0) # (L, 4d)
        return lape
        
    def identify_lang_neurons(self):
        pass
        
def main(model_name: str, device: torch.device) -> None:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if "llama" in model_name.lower():
        model = LlamaModelForProbing(tokenizer=tokenizer, device=device)
    else:
        raise NotImplementedError("Invalid model name!")
    
    max_context_len = 512
    batch_size = 4
    lang_neuron = LangNeuron(tokenizer=tokenizer, 
                             model=model, 
                             max_context_len=max_context_len, 
                             batch_size=batch_size)
    lape = lang_neuron.calculate_LAPE_score()
    print(lape)
    
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    models = ["meta-llama/Llama-2-7b-hf"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}...")
    
    main(models[0], device=device)
    