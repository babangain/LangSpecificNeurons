import json, os, sys, tqdm, pickle, datetime, random
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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
        """scoring_method: Any["act_prob_zero", "act_prob_mean", "act_prob_95p", "grad_act", "act_abs_mean", "act_abs_std", "act_stat"]
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
        mean_p5_tensor = torch.zeros(size=(self.model.L, self.model.int_d)).to(self.device)
        mean_p10_tensor = torch.zeros(size=(self.model.L, self.model.int_d)).to(self.device)
        mean_p25_tensor = torch.zeros(size=(self.model.L, self.model.int_d)).to(self.device)
        N = 0
        with tqdm.tqdm(iterable=dl, desc=f"Computing activation for lang: {self.lang}", unit="batches", colour="green") as pbar:
            for input_dict in pbar:
                if self.method == "grad_act":
                    self.model.train()
                    out = self.model(**input_dict)
                    out["loss"].backward()
                elif self.method in ["act_abs_mean", "act_abs_std", "act_prob_zero", "act_prob_mean", "act_prob_95p", "act_stat"]:
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
                p25_list = []
                p10_list = []
                p5_list = []
                
                for layer_idx in range(len(self.model.activations.keys())):
                    act = self.model.activations[layer_idx] # (b, T, 4d)
                    mu = act.mean(dim=(0,1)) # (4d,)
                    mu_list.append(mu.clone().detach()) # (L, 4d)
                    std_list.append(act.std(dim=(0,1)).clone().detach()) # (L, 4d)
                    p50_list.append(torch.quantile(act.flatten(0,1).to(torch.float32), q=0.50, dim=0).clone().detach()) # (L, 4d)
                    p75_list.append(torch.quantile(act.flatten(0,1).to(torch.float32), q=0.75, dim=0).clone().detach()) # (L, 4d)
                    p90_list.append(torch.quantile(act.flatten(0,1).to(torch.float32), q=0.90, dim=0).clone().detach()) # (L, 4d)
                    p95_list.append(torch.quantile(act.flatten(0,1).to(torch.float32), q=0.95, dim=0).clone().detach()) # (L, 4d)
                    p5_list.append(torch.quantile(act.flatten(0,1).to(torch.float32), q=0.05, dim=0).clone().detach()) # (L, 4d)
                    p10_list.append(torch.quantile(act.flatten(0,1).to(torch.float32), q=0.10, dim=0).clone().detach()) # (L, 4d)
                    p25_list.append(torch.quantile(act.flatten(0,1).to(torch.float32), q=0.25, dim=0).clone().detach()) # (L, 4d)
                    
                    if self.method == "act_abs_mean":
                        rel = torch.abs(act) # (b, T, 4d)
                        theta = rel.mean(dim=(0,1)) # (4d,)
                    elif self.method == "act_abs_std":
                        rel = torch.abs(act) # (b, T, 4d)
                        theta = rel.std(dim=(0,1)) # (4d,)
                    elif self.method == "act_prob_zero":
                        rel = (act > 0).to(torch.float16) # (b, T, 4d)
                        theta = rel.mean(dim=(0,1)) # (4d,)
                    elif self.method == "act_prob_cont":
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
                    elif self.method == "act_stat":
                        rel = None
                        theta = torch.zeros(size=(self.model.int_d,)).to(self.device) # (4d,)
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
                mean_p5_tensor += torch.stack(p5_list, dim=0) # (L, 4d)
                mean_p10_tensor += torch.stack(p10_list, dim=0) # (L, 4d)
                mean_p25_tensor += torch.stack(p25_list, dim=0) # (L, 4d)
                N += 1
    
        mean_rel_tensor = mean_rel_tensor/N # (L, 4d)
        mean_mu_tensor = mean_mu_tensor/N # (L, 4d)
        mean_std_tensor = mean_std_tensor/N # (L, 4d)
        mean_p50_tensor = mean_p50_tensor/N # (L, 4d)
        mean_p75_tensor = mean_p75_tensor/N # (L, 4d)
        mean_p90_tensor = mean_p90_tensor/N # (L, 4d)
        mean_p95_tensor = mean_p95_tensor/N # (L, 4d)
        mean_p5_tensor = mean_p5_tensor/N # (L, 4d)
        mean_p10_tensor = mean_p10_tensor/N # (L, 4d)
        mean_p25_tensor = mean_p25_tensor/N # (L, 4d)
        
        data = {
            "lang": self.lang,
            "mean_rel": mean_rel_tensor.cpu(), 
            "mean_mu_act": mean_mu_tensor.cpu(),
            "mean_std_act": mean_std_tensor.cpu(),
            "mean_p50_act": mean_p50_tensor.cpu(),
            "mean_p75_act": mean_p75_tensor.cpu(),
            "mean_p90_act": mean_p90_tensor.cpu(),
            "mean_p95_act": mean_p95_tensor.cpu(),
            "mean_p5_act": mean_p5_tensor.cpu(),
            "mean_p10_act": mean_p10_tensor.cpu(),
            "mean_p25_act": mean_p25_tensor.cpu(),
        }
        pickle.dump(data, open(self.rel_data_path, "wb"))
        print(f"{self.info()}: The relevance {self.method} data is stored at {self.rel_data_path}")
        return data

class NeuronRelevanceByContrastingActivation:
    def __init__(self, device: torch.device, model_name: str, quant_config: Union[None, QuantoConfig], lang_list: List[str], num_iterations: int):
        self.cwd = Path.cwd()
        self.device = device
        self.model_name = model_name
        self.model_name_srt = self.model_name.split('/')[-1]
        self.lang_list = lang_list
        self.method = "act_prob_contrast"
        self.N = num_iterations
        self.rel_data_path = Path(self.cwd, f"outputs/activation/{self.model_name_srt}/{self.method}/rel.pkl")
        self.act_stat_data_dict = {lang: pickle.load(open(Path(self.cwd, f"outputs/activation/{self.model_name_srt}/act_stat/rel_{lang}.pkl"), "rb")) for lang in self.lang_list}
        Path.mkdir(self.rel_data_path.parent, exist_ok=True, parents=True)
        
        if not self.rel_data_path.exists():
            self.quant_config = quant_config
            self.model = ModelForMLM(device=self.device, model_name=self.model_name, quant_config=self.quant_config).to(torch.float16)
            self.dataset = {lang: WikipediaDatasetHF(model_name=self.model_name, lang=lang, max_context_len=512) for lang in self.lang_list}
        
    def info(self) -> str:
        return f"[INFO] {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    
    def _get_contrastive_condition(self, batch_size: int) -> dict:
        ds_dict = {lang: Subset(dataset, indices=random.sample(range(len(dataset)), k=batch_size)) for lang, dataset in self.dataset.items()}
        dl_dict = {lang: DataLoader(dataset=ds, batch_size=batch_size) for lang, ds in ds_dict.items()}
        mu_dict = {}
        for lang, dl in dl_dict.items():
            input_dict = next(iter(dl))
            self.model.eval()
            out = self.model(**input_dict)
            mu_list = []
            for layer_idx in range(len(self.model.activations.keys())):
                act = self.model.activations[layer_idx] # (b, T, 4d)
                mu = act.mean(dim=(0,1)) # (4d,)
                mu_list.append(mu.clone().detach()) # (L, 4d)
            mu_dict[lang] = torch.stack(mu_list, dim=0) # (L, 4d)
        
        cond_dict = {}
        for lang1 in self.lang_list:
            lang1_p90 = self.act_stat_data_dict[lang1]["mean_p75_act"].mean(dim=1, keepdim=True).to(self.device)
            mu1 = mu_dict[lang1]
            cond = (mu1 > lang1_p90)
            for lang2 in set(self.lang_list) - set(lang1):
                lang2_p10 = self.act_stat_data_dict[lang2]["mean_p75_act"].mean(dim=1, keepdim=True).to(self.device)
                mu2 = mu_dict[lang2]
                cond = cond & (mu2 < lang2_p10)
            cond_dict[lang1] = cond.to(torch.float32)       
        return cond_dict

    def get_relevance_data(self, batch_size: Union[int, None]) -> dict:
        if self.rel_data_path.exists():
            out_obj = pickle.load(open(self.rel_data_path, "rb"))
            print(f"{self.info()}: The relevance {self.method} data is loaded from {self.rel_data_path}")
            return out_obj
        
        theta_dict = {lang: torch.zeros(size=(self.model.L, self.model.int_d)).to(self.device) for lang in self.lang_list}
        with tqdm.tqdm(range(self.N), desc=f"Contrasting activation: ", unit="iters", colour="green") as pbar:
            for i in pbar:
                cond_dict = self._get_contrastive_condition(batch_size=batch_size)
                for lang, cond in cond_dict.items():
                    theta_dict[lang] += cond
        
        mean_theta_dict = {lang: torch.zeros(size=(self.model.L, self.model.int_d)).to(self.device) for lang in self.lang_list}
        for lang, theta in theta_dict.items():
            mean_theta_dict[lang] = theta/self.N

        for lang, mean_theta in mean_theta_dict.items():
            data = {
                "mean_rel": mean_theta
            }
            rel_data_path = Path(self.rel_data_path.parent, f"rel_{lang}.pkl")
            pickle.dump(data, open(rel_data_path, "wb"))
            print(f"{self.info()}: The relevance {self.method} data is stored at {rel_data_path}")
        
        pickle.dump(mean_theta_dict, open(self.rel_data_path, "wb"))
        return mean_theta_dict
        
def main(model_name: str, device: torch.device) -> None:
    methods = ["act_prob_zero", "act_abs_mean", "grad_act", "act_prob_mean", "act_prob_95p", "act_abs_std"]
    for lang in ["ur"]:
        for method in ["act_prob_zero"]:
            rel = NeuronRelevance(device=device, model_name=model_name, quant_config=None, lang=lang, scoring_method=method)
            out = rel.get_relevance_data(batch_size=4, data_frac=0.5)
            print(out) 
    print("DONE")
    
if __name__ == "__main__":
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}...")
    
    main(models_map["llama3"], device=device)
    
    