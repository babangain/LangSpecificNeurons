import torch, json, os, sys, tqdm, pickle, datetime
from pathlib import Path
sys.path.append(Path(__file__).parent)
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Union
import pandas as pd
from models import get_tokenizer_and_model
from activation import Activation
from utils import lang_map, lang_triplet_map, models_map, token_repr_map, lang_repr_map
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib_venn import venn3, venn3_circles

class LangNeuron:
    def __init__(self, device: Union[torch.device, None], model_name: str, lang_neuron_config: Union[dict, None]):
        self.device = device
        self.model_name = model_name.split("/")[-1] 
        self.cwd = Path.cwd()
        self.lang_set = lang_neuron_config["lang_set"]
        self.lang_neuron_path = Path(self.cwd, f"outputs/lang_neurons/{self.model_name}/{self.lang_set}/lang_neuron_data.pkl")
        self.lang_neuron_path.parent.mkdir(parents=True, exist_ok=True)
        
        if self.lang_neuron_path.exists():
            self.__dict__.update(pickle.load(open(self.lang_neuron_path, "rb")))
            print(f"{self.info()}: The lang neurons data is loaded from {self.lang_neuron_path}")
        else:
            self._init_attr(config=lang_neuron_config)
            pickle.dump(self.__dict__, open(self.lang_neuron_path, "wb"))
            print(f"{self.info()}: The lang neurons data is stored at {self.lang_neuron_path}")
    
    def _init_attr(self, config: dict):
        self.lang_list = lang_map[config["lang_set"]]
        self.lang_neuron_frac = config["lang_neuron_frac"]
        self.threshold_quantile = config["threshold_quantile"]
        
        self.norm_act_prob, self.act_prob = self._get_norm_act_prob()
        self.act_prob_threshold = self._calculate_act_prob_threshold()
        self.lape = self._calculate_LAPE_score()
        
        self.L = self.norm_act_prob[self.lang_list[0]].shape[0]
        self.int_d = self.norm_act_prob[self.lang_list[0]].shape[1]
        self.m = round(self.lang_neuron_frac * self.L * self.int_d)
        self.lang_neurons = self._identify_lang_neurons()
        self.neuron_to_lang = self._get_neuron_to_lang_map()
        self.lang_to_neuron = self._get_lang_to_neuron_map()
    
    def info(self) -> str:
        return f"[INFO] {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
           
    def _get_norm_act_prob(self) -> dict:
        act_prob_dict = {}
        sum_act_prob = 0
        for lang in self.lang_list:
            act = Activation(model=None, model_name=self.model_name, dataset=None, lang=lang)
            act_prob = act.get_activation_data(batch_size=None, data_frac=None)["act_prob"].to(self.device)
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
    
    def get_lang_specific_neurons_dist(self, is_plot: bool) -> dict:
        neuron_dist = {}
        for lang, neuron_tensor in self.lang_to_neuron.items():
            neuron_dist[lang] = neuron_tensor.shape[0]
            
        if is_plot:
            languages = list(neuron_dist.keys())
            num_neurons = list(neuron_dist.values())
            plt.figure(figsize=(len(neuron_dist), 10))
            bars = plt.bar(languages, num_neurons, color='skyblue')
            for bar in bars:
                yval = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2.0, yval, int(yval), ha='center', va='bottom')
            
            plt.xlabel(f'Language of Set {self.lang_set[-1]}')
            plt.ylabel('Number of Language Specific Neurons')
            tokens = {k: v for k, v in token_repr_map[self.model_name].items() if k in lang_map[self.lang_set]}
            lang_repr = {k: v for k, v in lang_repr_map[self.model_name].items() if k in lang_map[self.lang_set]}
            title = f'{self.model_name}: Language Specific Neurons Distribution\n' + f'Tokens seen (in M): {tokens}\n' + f'Lang repr: {lang_repr}'
            
            plt.title(title, wrap=True)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            save_path = Path(self.lang_neuron_path.parent, "lang_neuron_dist.png")
            plt.savefig(str(save_path), format='png', dpi=300)
            plt.clf()
        return neuron_dist
    
    def get_layerwise_neurons_dist(self, is_plot: bool) -> dict:
        layer_neuron_dist = {}
        for lang, neuron_tensor in self.lang_to_neuron.items():
            layer_spec_dist = torch.zeros(size=(self.L,), dtype=torch.int64)
            if len(neuron_tensor) != 0:
                unique_val, counts = neuron_tensor[:, 0].unique(return_counts=True)
                for u, c in zip(unique_val, counts):
                    layer_spec_dist[u] = c
            layer_neuron_dist[lang] = layer_spec_dist
        
        neuron_dist = self.get_lang_specific_neurons_dist(is_plot=False)
        for lang, layer_tensor in layer_neuron_dist.items():
            assert layer_tensor.sum() == neuron_dist[lang], "Layerwise neurons count should match overall count!"

        if is_plot:
            df = pd.DataFrame(layer_neuron_dist, index=[f'{i}' for i in range(self.L)])
            df = df.iloc[::-1] # reverses the order of layers so that bottom (0) appears at bottom of matrix
            plt.figure(figsize=(len(layer_neuron_dist), 10))
            sns.heatmap(df, annot=True, fmt="d", cmap="YlGnBu", cbar=True, linewidths=.5)
            plt.xlabel(f'Language of Set {self.lang_set[-1]}')
            plt.ylabel('Layer Indices (0: bottom)')
            tokens = {k: v for k, v in token_repr_map[self.model_name].items() if k in lang_map[self.lang_set]}
            lang_repr = {k: v for k, v in lang_repr_map[self.model_name].items() if k in lang_map[self.lang_set]}
            title = f'{self.model_name}: Layerwise Neurons Distribution\n' + f'Tokens seen (in M): {tokens}\n' + f'Lang repr: {lang_repr}'
            
            plt.title(title, wrap=True)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            save_path = Path(self.lang_neuron_path.parent, "layerwise_lang_neurons_dist.png")
            plt.savefig(str(save_path), format='png', dpi=300)
            plt.clf()
        return layer_neuron_dist
    
    def get_neurons_overlap(self, is_plot: bool) -> dict:
        keys = list(self.lang_to_neuron.keys())
        matrix = pd.DataFrame(np.zeros((len(keys), len(keys)), dtype=np.int64), index=keys, columns=keys)
        for i, key1 in enumerate(keys):
            for j, key2 in enumerate(keys):
                if i <= j:  
                    data1 = {tuple(i) for i in self.lang_to_neuron[key1].tolist()}
                    data2 = {tuple(i) for i in self.lang_to_neuron[key2].tolist()}
                    common_elements = data1 & data2
                    count_common = len(common_elements)
                    matrix.at[key1, key2] = count_common
                    matrix.at[key2, key1] = count_common 
        
        neurons_dist = self.get_lang_specific_neurons_dist(is_plot=False)
        matrix_dict = matrix.to_dict()
        for key, val in matrix_dict.items():
            assert neurons_dist[key] == val[key], "Diagonal should match with number of lang specific neurons!"

        if is_plot:
            plt.figure(figsize=(len(self.lang_list), len(self.lang_list)))
            sns.heatmap(matrix, annot=True, cmap="YlGnBu", fmt="d", linewidths=.5)
            plt.xlabel(f'Language of Set {self.lang_set[-1]}')
            plt.ylabel(f'Language of Set {self.lang_set[-1]}')
            tokens = {k: v for k, v in token_repr_map[self.model_name].items() if k in lang_map[self.lang_set]}
            lang_repr = {k: v for k, v in lang_repr_map[self.model_name].items() if k in lang_map[self.lang_set]}
            title = f'{self.model_name}: Neurons Overlap Between Languages\n' + f'Tokens seen (in M): {tokens}\n' + f'Lang repr: {lang_repr}'
            
            plt.title(title, wrap=True)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=45, ha='right')
            plt.tight_layout()
            save_path = Path(self.lang_neuron_path.parent, 'lang_neurons_overlap.png')
            plt.savefig(str(save_path), format='png', dpi=300)
            plt.clf()
        return matrix.to_dict()
    
    def plot_3_lang_overlap_venn(self, languages: List[str]) -> None:
        assert len(languages) <= 3, "Venn diagrams only support up to 3 sets (languages)."
        for lang in languages:
            assert lang in self.lang_to_neuron, f"Language {lang} not found in lang neuron set."
            
        data = {}
        for lang in languages:
            data[lang] = {tuple(i) for i in self.lang_to_neuron[lang].tolist()}
        set1, set2, set3 = data[languages[0]], data[languages[1]], data[languages[2]]
        venn = venn3([set1, set2, set3], set_labels=languages)
        venn3_circles([set1, set2, set3])
        tokens = {k: v for k, v in token_repr_map[self.model_name].items() if k in languages}
        lang_repr = {k: v for k, v in lang_repr_map[self.model_name].items() if k in languages}
        title = f'{self.model_name}: Neurons Overlap Between Languages ({languages}) \n' + f'Tokens seen (in M): {tokens}\n' + f'Lang repr: {lang_repr}'
            
        plt.title(title, wrap=True)
        save_path = Path(self.lang_neuron_path.parent, f'neuron_overlap_{"_".join(languages)}.png')
        plt.savefig(str(save_path), format='png', dpi=300)
        plt.clf()
    
def main(model_name: str, lang_set: str, device: torch.device) -> None:
    lang_neuron_config = {
        "lang_set": lang_set,
        "lang_neuron_frac": 0.01,
        "threshold_quantile": 0.95
    }
    lang_neuron = LangNeuron(device=device, model_name=model_name, lang_neuron_config=lang_neuron_config)
    lang_neuron.get_layerwise_neurons_dist(is_plot=True)
    lang_neuron.get_lang_specific_neurons_dist(is_plot=True)
    lang_neuron.get_neurons_overlap(is_plot=True)
    for lang_triplet in lang_triplet_map[lang_set]:
        lang_neuron.plot_3_lang_overlap_venn(languages=lang_triplet)
    
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}...")
    
    for model_key in ["sarvam"]:
        for lang_set in ["set1", "set2", "set3", "set4"]:
            main(model_name=models_map[model_key], lang_set=lang_set, device=device)
            print(f"Model: {model_key}, Lang set: {lang_set} done!")
