import torch, json, os, sys, tqdm, pickle, datetime, math
from pathlib import Path
sys.path.append(Path(__file__).parent)
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Union
import pandas as pd
from dataset import WikipediaDataset
from models import LlamaModelForProbing, BloomzModelForProbing
import matplotlib.pyplot as plt
import seaborn as sns

class Perplexity:
    def __init__(self, device: torch.device, tokenizer: AutoTokenizer, model: torch.nn.Module, model_name: str, ppx_config: dict):
        self.device = device
        self.tokenizer = tokenizer
        self.model_name = model_name.split("/")[-1]
        self.model = model
        self.batch_size = ppx_config["batch_size"]
        self.data_frac = ppx_config["data_frac"]
        self.Tmax = ppx_config["max_context_len"]
        self.lang_list = ppx_config["lang_list"]
        
        self.cwd = Path.cwd()
        self.lang_neuron_path = Path(self.cwd, f"outputs/lang_neurons/{self.model_name}/lang_neuron_data.pkl")
        self.lang_neuron = pickle.load(open(self.lang_neuron_path, "rb"))
        self.ppx_path = Path(self.cwd, f"outputs/perplexity/{self.model_name}/ppx_change_data.pkl")
        self.ppx_path.parent.mkdir(parents=True, exist_ok=True)

    def info(self) -> str:
        return f"[INFO] {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

    def _calc_ppx_for_lang(self, dataset: Dataset, inv_lang: Union[str, None], lang: str) -> float:
        """dataset: for which we want to calculate the perplexity
        inv_lang: where we want to intervene the activation (inv_lang specific neurons), can be None if no intervention is needed
        lang: for which we want to calculate the perplexity score
        """
        dl = dataset.prepare_dataloader(batch_size=self.batch_size, frac=self.data_frac[lang])
        if inv_lang is None:
            desc = f"Calculating perplexity without intervention for lang: {lang}"
            intervene_config = None
        else:
            # lang specific neurons will be deactivated here
            desc = f"Calculating perplexity with intervention in lang: {inv_lang} for lang: {lang}"
            intervene_config = {
                "indices": self.lang_neuron["lang_to_neuron"][inv_lang],
                "value": 0
            }
            
        loss_list = []
        with tqdm.tqdm(iterable=dl, desc=desc, unit=" batches", colour="green") as pbar:
            for input_dict in pbar:
                out_dict = self.model(input_dict["input_ids"], input_dict["attention_mask"], intervene_config=intervene_config)
                logits = out_dict["logits"] # (b, Tmax, V) 
                target_ids = input_dict["target_ids"] # (b, Tmax)
                logits = logits.flatten(start_dim=0, end_dim=1).to(self.device) # (b*Tmax, V)
                target_ids = target_ids.flatten().to(self.device) # (b*Tmax,)
                
                loss = torch.nn.functional.cross_entropy(logits, target_ids, reduction="mean") # scalar
                pbar.set_postfix(loss=f"{loss.item():.3f}")
                loss_list.append(loss.item())  
        
        avg_loss = sum(loss_list)/len(loss_list)
        ppx = math.exp(avg_loss)
        return ppx
    
    def calc_ppx_change(self) -> torch.tensor:
        if self.ppx_path.exists():
            ppx_change = pickle.load(open(self.ppx_path, "rb"))
            print(f"{self.info()}: The perplexity data is loaded from {self.ppx_path}")
        else:
            ppx_1d_list = []
            ppx_2d_list = []
            for src_lang in self.lang_list:
                src_dataset = WikipediaDataset(tokenizer=self.tokenizer, lang=src_lang, max_context_len=self.Tmax) 
                src_ppx = self._calc_ppx_for_lang(dataset=src_dataset, inv_lang=None, lang=src_lang)
                ppx_1d_list.append(src_ppx)
                
                ppx_inv_list = []
                for tgt_lang in self.lang_list:
                    tgt_dataset = WikipediaDataset(tokenizer=self.tokenizer, lang=tgt_lang, max_context_len=self.Tmax)
                    tgt_ppx = self._calc_ppx_for_lang(dataset=tgt_dataset, inv_lang=src_lang, lang=tgt_lang)
                    ppx_inv_list.append(tgt_ppx)
                ppx_2d_list.append(ppx_inv_list)
                
            ppx_1d_tensor = torch.tensor(ppx_1d_list)
            ppx_2d_tensor = torch.tensor(ppx_2d_list)
            ppx_change = ppx_2d_tensor - ppx_1d_tensor
            
            pickle.dump(ppx_change, open(self.ppx_path, "wb"))
            print(f"{self.info()}: The perplexity data is stored at {self.ppx_path}")
        
        self._plot_ppx_change(ppx_change=ppx_change)
        return ppx_change

    def _plot_ppx_change(self, ppx_change: torch.tensor) -> None:
        save_path = str(Path(self.ppx_path.parent, "ppx_change.png"))
        plt.figure(figsize=(6,6))
        ppx_change_np = ppx_change.numpy()
        sns.heatmap(ppx_change_np, annot=True, fmt=".2f", cmap="Reds", xticklabels=self.lang_list, yticklabels=self.lang_list, cbar=False)
        fs = 16
        plt.xlabel("Lang: j", fontsize=fs)
        plt.ylabel("Lang: i", fontsize=fs)
        plt.xticks(fontsize=fs)
        plt.yticks(fontsize=fs)
        plt.title("PPXC(i,j): Perplexity change at j after intervention at i", fontsize=fs)
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        
def main(model_name: str, device: torch.device) -> None:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if "llama" in model_name.lower():
        model = LlamaModelForProbing(tokenizer=tokenizer, device=device, model_name=model_name)
    elif "bloomz" in model_name.lower():
        model = BloomzModelForProbing(tokenizer=tokenizer, device=device, model_name=model_name)
    else:
        raise NotImplementedError("Invalid model name!")
    
    lang_list = ["en", "fr", "es", "hi", "bn", "te", "tn"]
    data_frac_list = [0.0001]*6 + [0.01]
    ppx_config = {
        "lang_list": lang_list,
        "max_context_len": 512,
        "batch_size": 4,
        "data_frac": dict(zip(lang_list, data_frac_list))
    }
     
    ppx = Perplexity(device=device, tokenizer=tokenizer, model=model, model_name=model_name, ppx_config=ppx_config)
    out = ppx.calc_ppx_change()
    print(out)
    
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    models = ["meta-llama/Llama-2-7b-hf", "bigscience/bloomz-7b1"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}...")
    
    main(models[1], device=device)
    