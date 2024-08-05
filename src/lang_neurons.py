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
    def __init__(self, device: torch.device, tokenizer: AutoTokenizer, model: torch.nn.Module, model_name: str, lang_neuron_config: dict):
        self.device = device
        self.tokenizer = tokenizer
        self.model = model
        self.model_name = model_name
        
        self.Tmax = lang_neuron_config["max_context_len"]
        self.batch_size = lang_neuron_config["batch_size"]
        self.lang_list = lang_neuron_config["lang_list"] 
        self.lang_neuron_frac = lang_neuron_config["lang_neuron_frac"]
        self.threshold_quantile = lang_neuron_config["threshold_quantile"] 
        self.data_frac = lang_neuron_config["data_frac"]
        
        self.norm_act_prob, self.act_prob = self._get_act_prob()
        self.act_prob_threshold = self._calculate_act_prob_threshold()
        self.lape = self._calculate_LAPE_score()
        
        self.L = self.norm_act_prob[self.lang_list[0]].shape[0]
        self.int_d = self.norm_act_prob[self.lang_list[0]].shape[1]
        self.m = round(self.lang_neuron_frac * self.L * self.int_d)
        self.lang_neurons = self._identify_lang_neurons()
        self.neuron_to_lang = self._get_neuron_to_lang_map()
        self.lang_to_neuron = self._get_lang_to_neuron_map()
    
    def _get_act_prob(self) -> dict:
        act_prob_dict = {}
        sum_act_prob = 0
        for lang in self.lang_list:
            dataset = WikipediaDataset(tokenizer=self.tokenizer, lang=lang, max_context_len=self.Tmax) if self.model else None
            act = Activation(tokenizer=self.tokenizer, model=self.model, model_name=self.model_name, dataset=dataset, lang=lang)
            act_prob = act.get_activation_probability(batch_size=self.batch_size, data_frac=self.data_frac)["act_prob"].to(self.device)
            act_prob_dict[lang] = act_prob
            sum_act_prob += act_prob
        
        norm_act_prob_dict = {}
        for lang, act_prob in act_prob_dict.items():
            norm_act_prob = act_prob/(sum_act_prob + 1e-10)
            norm_act_prob_dict[lang] = norm_act_prob
        return norm_act_prob_dict, act_prob_dict

    def _calculate_act_prob_threshold(self) -> float:
        act_prob = torch.stack(list(self.act_prob.values()), dim=0) # (k, L, 4d)
        threshold = act_prob.flatten().quantile(self.threshold_quantile).item()
        return threshold
          
    def _calculate_LAPE_score(self) -> torch.tensor:
        norm_act_prob = torch.stack(list(self.norm_act_prob.values()), dim=0) # (k, L, 4d)
        act_prob = torch.stack(list(self.act_prob.values()), dim=0) # (k, L, 4d)
        epsilon = 1e-10
        lape = -1 * (norm_act_prob * torch.log(norm_act_prob + epsilon)).sum(dim=0) # (L, 4d)
        condition = ((act_prob > self.act_prob_threshold).sum(dim=0) == 0) # (L, 4d)
        lape[condition] = torch.inf # Ignore those neurons positions where condition is not satisfied
        return lape       
    
    def _identify_lang_neurons(self) -> torch.tensor:
        lape_cutoff = self.lape.flatten().sort()[0][:self.m].max() # scalar
        lang_neurons = torch.nonzero(self.lape <= lape_cutoff) # (m, 2)
        return lang_neurons
    
    def _get_neuron_to_lang_map(self) -> dict:
        neuron_to_lang = {}
        for m in range(self.lang_neurons.shape[0]):
            i, j = self.lang_neurons[m].tolist()
            lang_spec_list = []
            for lang, act_prob in self.act_prob.items():
                if act_prob[i, j] > self.act_prob_threshold:
                    lang_spec_list.append(lang)
            neuron_to_lang[(i,j)] = lang_spec_list
        return neuron_to_lang
    
    def _get_lang_to_neuron_map(self) -> dict:
        lang_to_neuron = {lang: [] for lang in self.lang_list}
        for neuron, lang_list in self.neuron_to_lang.items():
            for lang in lang_list:
                lang_to_neuron[lang].append(list(neuron))
        for lang, neuron_list in lang_to_neuron.items():
            lang_to_neuron[lang] = torch.tensor(neuron_list)
        return lang_to_neuron
    
    def get_lang_specific_neurons_dist(self) -> dict:
        neuron_dist = {}
        for lang, neuron_tensor in self.lang_to_neuron.items():
            neuron_dist[lang] = neuron_tensor.shape[0]
        return neuron_dist

def main(model_name: str, device: torch.device) -> None:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    lang_neuron_config = {
        "max_context_len": 512,
        "batch_size": 4,
        "lang_list": ["en", "fr", "es", "vi", "id", "ja", "zh"],
        "lang_neuron_frac": 0.01,
        "threshold_quantile": 0.95,
        "data_frac": 1.0
    }
    
    is_act_data_path = []
    for lang in lang_neuron_config["lang_list"]:
        is_act_data_path.append(Path(Path.cwd(), f"outputs/activation/{model_name.split('/')[-1]}/act_{lang}.pkl").exists())
    if all(is_act_data_path):
        model = None
    elif "llama" in model_name.lower():
        model = LlamaModelForProbing(tokenizer=tokenizer, device=device)
    else:
        raise NotImplementedError("Invalid model name!")
        
    lang_neuron = LangNeuron(device=device,
                             tokenizer=tokenizer, 
                             model=model, 
                             model_name=model_name.split("/")[-1],
                             lang_neuron_config=lang_neuron_config)
    print(lang_neuron.get_lang_specific_neurons_dist())
    
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    models = ["meta-llama/Llama-2-7b-hf"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}...")
    
    main(models[0], device=device)
    