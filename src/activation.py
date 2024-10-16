import torch, json, os, sys, tqdm, pickle, datetime, random
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
    def __init__(self, device: torch.device, model_name: str, quant_config: Union[None, QuantoConfig], lang: str):
        self.cwd = Path.cwd()
        self.device = device
        self.lang = lang
        self.model_name = model_name
        self.model_name_srt = self.model_name.split('/')[-1]
        self.rel_data_path = Path(self.cwd, f"outputs/activation/{self.model_name_srt}/rel_grad_act_{self.lang}.pkl")
        Path.mkdir(self.rel_data_path.parent, exist_ok=True, parents=True)
        
        if not self.rel_data_path.exists():
            self.quant_config = quant_config
            self.model = ModelForMLM(device=self.device, model_name=self.model_name, quant_config=self.quant_config)
            self.dataset = WikipediaDatasetHF(model_name=self.model_name, lang=self.lang, max_context_len=512)
        
    def info(self) -> str:
        return f"[INFO] {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    
    def get_relevance_data(self, batch_size: Union[int, None], data_frac: Union[float, None], is_random: bool=False) -> dict:
        if self.rel_data_path.exists():
            out_obj = pickle.load(open(self.rel_data_path, "rb"))
            print(f"{self.info()}: The relevance grad act data is loaded from {self.rel_data_path}")
            return out_obj
        
        if is_random:
            out_obj = pickle.load(open(Path(self.rel_data_path.parent, "rel_grad_act_en.pkl"), "rb"))
            avg_rel_tensor = torch.randn(size=out_obj.shape) * 10**-6
            pickle.dump(avg_rel_tensor, open(self.rel_data_path, "wb"))
            print(f"{self.info()}: The random relevance grad act data is stored at {self.rel_data_path}")
            return avg_rel_tensor
        
        ds = Subset(self.dataset, indices=random.sample(range(len(self.dataset)), k=int(len(self.dataset)*data_frac)))
        dl = DataLoader(dataset=ds, batch_size=batch_size, drop_last=True, shuffle=True, num_workers=4)
        rel_list = []
        with tqdm.tqdm(iterable=dl, desc=f"Calculating relevance for lang: {self.lang}", unit=" batches", colour="green") as pbar:
            for input_dict in pbar:
                out = self.model(**input_dict)
                self.model.zero_grad()
                out["loss"].backward()  
                theta_list = []
                for layer_idx in range(len(self.model.activations.keys())):
                    act = self.model.activations[layer_idx]
                    rel = torch.abs(act.grad * act) # (b, T, 4d)
                    theta = rel.mean(dim=(0,1)) # (4d,)
                    theta_list.append(theta.clone().detach()) # (L, 4d)
                rel_list.append(torch.stack(theta_list, dim=0)) # (N, L, 4d)
    
        rel_tensor = torch.stack(rel_list, dim=0)  # (N, L, 4d)
        avg_rel_tensor = rel_tensor.mean(dim=0) # (L, 4d)
        pickle.dump(avg_rel_tensor, open(self.rel_data_path, "wb"))
        print(f"{self.info()}: The relevance grad act data is stored at {self.rel_data_path}")
        return avg_rel_tensor
   
def main(model_name: str, device: torch.device) -> None:
    quant_config = QuantoConfig(weights="float8")
    for lang in lang_map["set5"]:
        rel = NeuronRelevance(device=device, model_name=model_name, quant_config=quant_config, lang=lang)
        out = rel.get_relevance_data(batch_size=8, data_frac=0.01, is_random=True)
        print(out, out.shape) 
    print("DONE")
    
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}...")
    
    main(models_map["llama3"], device=device)
    
    