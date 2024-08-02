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
        self.threeshold_quantile = 0.95
        self.norm_act_prob = self._get_norm_act_prob()
        self.act_prob_threeshold = self._calculate_act_prob_threeshold()
        
        self.lape = self._calculate_LAPE_score()
        self.L = self.norm_act_prob[self.lang_list[0]].shape[0]
        self.int_d = self.norm_act_prob[self.lang_list[0]].shape[1]
        self.m = int(self.lang_neuron_frac * self.L * self.int_d)
        self.lang_neurons = self._identify_lang_neurons()
    
    def _get_norm_act_prob(self) -> dict:
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
            norm_act_prob = act_prob/(sum_act_prob + 1e-10)
            norm_act_prob_dict[lang] = norm_act_prob
        return norm_act_prob_dict
        
    def _calculate_LAPE_score(self) -> torch.tensor:
        norm_act_prob = torch.stack(list(self.norm_act_prob.values()), dim=0) # (k, L, 4d)
        epsilon = 1e-10
        lape = -1 * (norm_act_prob * torch.log2(norm_act_prob + epsilon)).sum(dim=0) # (L, 4d)
        return lape
        
    def _identify_lang_neurons(self) -> torch.tensor:
        lape_cutoff = self.lape.flatten().sort()[0][:self.m].max() # scalar
        lang_neurons = torch.nonzero(self.lape <= lape_cutoff) # (m, 2)
        return lang_neurons
    
    def _calculate_act_prob_threeshold(self) -> float:
        norm_act_prob = torch.stack(list(self.norm_act_prob.values()), dim=0) # (k, L, 4d)
        threeshold = norm_act_prob.flatten().quantile(self.threeshold_quantile).item()
        return threeshold
    
    def _lang_specific_neuron(self, layer_index: int, dim_index: int) -> List[str]:
        assert layer_index >= 0 and layer_index < self.L
        assert dim_index >= 0 and dim_index < self.int_d
        assert layer_index in self.lang_neurons[:,0] and dim_index in self.lang_neurons[:,1]
        lang_spec_list = []
        for lang, act_prob in self.norm_act_prob.items():
            if act_prob[layer_index, dim_index] > self.act_prob_threeshold:
                lang_spec_list.append(lang)
        return lang_spec_list
    
    def get_lang_specific_neurons_dist(self) -> dict:
        neuron_dist = dict(zip(self.lang_list, [0] * self.lang_list.__len__()))
        for m in range(self.lang_neurons.shape[0]):
            i, j = self.lang_neurons[m]
            lang_spec_list = self._lang_specific_neuron(layer_index=i, dim_index=j)
            for lang in lang_spec_list:
                neuron_dist[lang] += 1
        return neuron_dist
        
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
    neuron_dist = lang_neuron.get_lang_specific_neurons_dist()
    print(neuron_dist)
    
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    models = ["meta-llama/Llama-2-7b-hf"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}...")
    
    main(models[0], device=device)
    