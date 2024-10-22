import json, os, sys, tqdm, pickle, datetime, random
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
from pathlib import Path
sys.path.append(Path(__file__).parent)
from transformers import AutoTokenizer, AutoModelForCausalLM, QuantoConfig
from torch.utils.data import Dataset, DataLoader, Subset
from typing import List, Tuple, Union
import pandas as pd
from dataset import WikipediaDatasetHF
from models import ModelForMLM
from utils import lang_map, models_map

class Activation:
    def __init__(self, model: Union[torch.nn.Module, None], model_name: str, dataset: Union[Dataset, None], lang: str):
        self.cwd = Path.cwd()
        self.lang = lang
        self.model_name = model_name.split('/')[-1]
        self.act_data_path = Path(self.cwd, f"outputs/activation/{self.model_name}/act_{self.lang}.pkl")
        self.model = model
        self.dataset = dataset
        Path.mkdir(self.act_data_path.parent, exist_ok=True, parents=True)
    
    def info(self) -> str:
        return f"[INFO] {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    
    def get_activation_data(self, batch_size: Union[int, None], data_frac: Union[float, None]) -> dict:
        if self.act_data_path.exists():
            out = pickle.load(open(self.act_data_path, "rb"))
            print(f"{self.info()}: The activation data is loaded from {self.act_data_path}")
            return out
        
        dl = self.dataset.prepare_dataloader(batch_size=batch_size, frac=data_frac)
        total_tokens = 0
        total_avg_neuron_out = 0
        total_gt_zero_count = 0
        
        with tqdm.tqdm(iterable=dl, 
                       desc=f"Calculating activation for lang: {self.lang}",
                       unit=" batches",
                       colour="green") as pbar:
            for input_dict in pbar:
                out_dict = self.model(input_dict["input_ids"], input_dict["attention_mask"], intervene_config=None)
                total_avg_neuron_out += out_dict["avg_neuron_out"] # (L, 4d) 
                total_gt_zero_count += out_dict["neuron_out_gt_zero_count"] # (L, 4d)
                total_tokens += out_dict["tokens_count"] # scalar
        
        mean_act = total_avg_neuron_out/total_tokens
        act_prob = total_gt_zero_count/total_tokens
        out = {"act_prob": act_prob.cpu(), "mean_act": mean_act.cpu(), "total_tokens": total_tokens}
        pickle.dump(out, open(self.act_data_path, "wb"))
        print(f"{self.info()}: The activation data is stored at {self.act_data_path}")
        return out

class NeuronRelevance:
    def __init__(self, device: torch.device, model_name: str, quant_config: Union[None, QuantoConfig], lang: str, scoring_method: str):
        """scoring_method: Any["act_prob_zero", "act_prob_mean", "act_prob_95p", "grad_act", "act_abs_mean", "act_abs_std"]
        """
        self.cwd = Path.cwd()
        self.device = device
        self.lang = lang
        self.model_name = model_name
        self.model_name_srt = self.model_name.split('/')[-1]
        self.method = scoring_method
        self.rel_data_path = Path(self.cwd, f"outputs/activation/{self.model_name_srt}/{self.method}/rel_{self.lang}.pkl")
        Path.mkdir(self.rel_data_path.parent, exist_ok=True, parents=True)
        
        if not self.rel_data_path.exists():
            self.quant_config = quant_config
            self.model = ModelForMLM(device=self.device, model_name=self.model_name, quant_config=self.quant_config).to(torch.float16)
            self.dataset = WikipediaDatasetHF(model_name=self.model_name, lang=self.lang, max_context_len=512)
        
    def info(self) -> str:
        return f"[INFO] {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    
    def get_relevance_data(self, batch_size: Union[int, None], data_frac: Union[float, None]) -> dict:
        if self.rel_data_path.exists():
            out_obj = pickle.load(open(self.rel_data_path, "rb"))
            print(f"{self.info()}: The relevance {self.method} data is loaded from {self.rel_data_path}")
            return out_obj
        
        ds = Subset(self.dataset, indices=random.sample(range(len(self.dataset)), k=int(len(self.dataset)*data_frac)))
        dl = DataLoader(dataset=ds, batch_size=batch_size, drop_last=True, shuffle=True, num_workers=4)
        
        mean_rel_tensor = torch.zeros(size=(self.model.L, self.model.int_d)).to(self.device)
        mean_mu_tensor = torch.zeros(size=(self.model.L, self.model.int_d)).to(self.device)
        mean_std_tensor = torch.zeros(size=(self.model.L, self.model.int_d)).to(self.device)
        mean_p50_tensor = torch.zeros(size=(self.model.L, self.model.int_d)).to(self.device)
        mean_p75_tensor = torch.zeros(size=(self.model.L, self.model.int_d)).to(self.device)
        mean_p90_tensor = torch.zeros(size=(self.model.L, self.model.int_d)).to(self.device)
        mean_p95_tensor = torch.zeros(size=(self.model.L, self.model.int_d)).to(self.device)
        N = 0
        with tqdm.tqdm(iterable=dl, desc=f"Calculating relevance for lang: {self.lang}", unit="batches", colour="green") as pbar:
            for input_dict in pbar:
                if self.method == "grad_act":
                    self.model.train()
                    out = self.model(**input_dict)
                    out["loss"].backward()
                elif self.method in ["act_abs_mean", "act_abs_std", "act_prob_zero", "act_prob_mean", "act_prob_95p"]:
                    self.model.eval()
                    out = self.model(**input_dict)
                else:
                    raise ValueError(f"Sorry, invalid method! - {self.method}")
                
                theta_list = []
                mu_list = []
                std_list = []
                p90_list = []
                p95_list = []
                p75_list = []
                p50_list = []
                
                for layer_idx in range(len(self.model.activations.keys())):
                    act = self.model.activations[layer_idx] # (b, T, 4d)
                    mu = act.mean(dim=(0,1)) # (4d,)
                    mu_list.append(mu.clone().detach()) # (L, 4d)
                    std_list.append(act.std(dim=(0,1)).clone().detach()) # (L, 4d)
                    p50_list.append(torch.quantile(act.flatten(0,1).to(torch.float32), q=0.50, dim=0).clone().detach()) # (L, 4d)
                    p75_list.append(torch.quantile(act.flatten(0,1).to(torch.float32), q=0.75, dim=0).clone().detach()) # (L, 4d)
                    p90_list.append(torch.quantile(act.flatten(0,1).to(torch.float32), q=0.90, dim=0).clone().detach()) # (L, 4d)
                    p95_list.append(torch.quantile(act.flatten(0,1).to(torch.float32), q=0.95, dim=0).clone().detach()) # (L, 4d)
                    
                    if self.method == "act_abs_mean":
                        rel = torch.abs(act) # (b, T, 4d)
                        theta = rel.mean(dim=(0,1)) # (4d,)
                    elif self.method == "act_abs_std":
                        rel = torch.abs(act) # (b, T, 4d)
                        theta = rel.std(dim=(0,1)) # (4d,)
                    elif self.method == "act_prob_zero":
                        rel = (act > 0).to(torch.float16) # (b, T, 4d)
                        theta = rel.mean(dim=(0,1)) # (4d,)
                    elif self.method == "act_prob_mean":
                        rel = (act > mu).to(torch.float16) # (b, T, 4d)
                        theta = rel.mean(dim=(0,1)) # (4d,)
                    elif self.method == "act_prob_95p":
                        rel = (act > torch.quantile(act.flatten(0,1).to(torch.float32), q=0.95, dim=0)).to(torch.float16) # (b, T, 4d)
                        theta = rel.mean(dim=(0,1)) # (4d,)
                    elif self.method == "grad_act":
                        rel = torch.abs(act.grad * act) # (b, T, 4d)
                        theta = rel.mean(dim=(0,1)) # (4d,)
                    else:
                        raise ValueError(f"Sorry, invalid method! - {self.method}")
                    theta_list.append(theta.clone().detach()) # (L, 4d)
                
                mean_rel_tensor += torch.stack(theta_list, dim=0) # (L, 4d)
                mean_mu_tensor += torch.stack(mu_list, dim=0) # (L, 4d)
                mean_std_tensor += torch.stack(std_list, dim=0) # (L, 4d)
                mean_p50_tensor += torch.stack(p50_list, dim=0) # (L, 4d)
                mean_p75_tensor += torch.stack(p75_list, dim=0) # (L, 4d)
                mean_p90_tensor += torch.stack(p90_list, dim=0) # (L, 4d)
                mean_p95_tensor += torch.stack(p95_list, dim=0) # (L, 4d)
                N += 1
    
        mean_rel_tensor = mean_rel_tensor/N # (L, 4d)
        mean_mu_tensor = mean_mu_tensor/N # (L, 4d)
        mean_std_tensor = mean_std_tensor/N # (L, 4d)
        mean_p50_tensor = mean_p50_tensor/N # (L, 4d)
        mean_p75_tensor = mean_p75_tensor/N # (L, 4d)
        mean_p90_tensor = mean_p90_tensor/N # (L, 4d)
        mean_p95_tensor = mean_p95_tensor/N # (L, 4d)
        
        data = {
            "mean_rel": mean_rel_tensor.cpu(), 
            "mean_mu_act": mean_mu_tensor.cpu(),
            "mean_std_act": mean_std_tensor.cpu(),
            "mean_p50_act": mean_p50_tensor.cpu(),
            "mean_p75_act": mean_p75_tensor.cpu(),
            "mean_p90_act": mean_p90_tensor.cpu(),
            "mean_p95_act": mean_p95_tensor.cpu(),
        }
        pickle.dump(data, open(self.rel_data_path, "wb"))
        print(f"{self.info()}: The relevance {self.method} data is stored at {self.rel_data_path}")
        return data

def main(model_name: str, device: torch.device) -> None:
    methods = ["act_prob_zero", "act_abs_mean", "grad_act", "act_prob_mean", "act_prob_95p", "act_abs_std"]
    for lang in ["id", "ja", "zh"]:
        for method in ["act_prob_mean"]:
            rel = NeuronRelevance(device=device, model_name=model_name, quant_config=None, lang=lang, scoring_method=method)
            out = rel.get_relevance_data(batch_size=4, data_frac=0.25)
            print(out) 
    print("DONE")
    
if __name__ == "__main__":
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}...")
    
    main(models_map["llama3"], device=device)
    
    