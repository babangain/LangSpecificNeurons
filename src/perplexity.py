import torch, json, os, sys, tqdm, pickle, datetime, math
from pathlib import Path
sys.path.append(Path(__file__).parent)
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Union
import pandas as pd
from dataset import WikipediaDataset
from models import get_tokenizer_and_model
import matplotlib.pyplot as plt
import seaborn as sns
from utils import lang_map, models_map, token_repr_map, lang_repr_map

class Perplexity:
    def __init__(self, device: torch.device, tokenizer: Union[AutoTokenizer, None], model: Union[torch.nn.Module, None], model_name: str, ppx_config: dict):
        self.full_model_name = model_name
        self.model_name = model_name.split("/")[-1]
        self.cwd = Path.cwd()
        self.lang_set = ppx_config["lang_set"]
        self.ppx_path = Path(self.cwd, f"outputs/perplexity/{self.model_name}/{self.lang_set}/ppx_data.pkl")
        self.ppx_path.parent.mkdir(parents=True, exist_ok=True)
        
        if self.ppx_path.exists():
            self.__dict__.update(pickle.load(open(self.ppx_path, "rb")))
            print(f"{self.info()}: The perplexity data is loaded from {self.ppx_path}")
        else:
            self._init_attr(device=device, tokenizer=tokenizer, model=model, config=ppx_config)
            state_dict = {k: v for k, v in self.__dict__.items() if k != "model"}
            pickle.dump(state_dict, open(self.ppx_path, "wb"))
            print(f"{self.info()}: The perplexity data is stored at {self.ppx_path}")
        
        self._plot_change(change=self.ppx_change, is_ppx=True)
        self._plot_change(change=self.loss_change, is_ppx=False)
    
    def _init_attr(self, device: torch.device, tokenizer: AutoTokenizer, model: torch.nn.Module, config: dict):
        self.device = device
        self.tokenizer = tokenizer
        self.model = model
        self.batch_size = config["batch_size"]
        self.data_frac = config["data_frac"]
        self.Tmax = config["max_context_len"]
        self.lang_neuron_path = Path(self.cwd, f"outputs/lang_neurons/{self.model_name}/{self.lang_set}/lang_neuron_data.pkl") 
        
        if self.lang_neuron_path.exists():
            self.lang_neuron = pickle.load(open(self.lang_neuron_path, "rb"))
            print(f"{self.info()}: The lang neurons data is loaded from {self.lang_neuron_path}")
            self.lang_list = self.lang_neuron["lang_list"]
        else:
            raise ValueError(f"{self.lang_neuron_path} doesn't exist!")
        
        self.ppx_change, self.loss_change = self._calc_ppx_change()
        
    def info(self) -> str:
        return f"[INFO] {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

    def _calc_ppx_for_lang(self, dataset: Dataset, inv_lang: Union[str, None], lang: str) -> float:
        """dataset: for which we want to calculate the perplexity
        inv_lang: where we want to intervene the activation (inv_lang specific neurons), can be None if no intervention is needed
        lang: for which we want to calculate the perplexity score
        """
        dl = dataset.prepare_dataloader(batch_size=self.batch_size, frac=self.data_frac)
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
                logits1 = out_dict["logits"] # (b, Tmax, V) 
                target_ids1 = input_dict["target_ids"] # (b, Tmax)
                
                logits = logits1.flatten(start_dim=0, end_dim=1).to(self.device) # (b*Tmax, V)
                target_ids = target_ids1.flatten().to(self.device) # (b*Tmax,)
                loss = torch.nn.functional.cross_entropy(logits, target_ids, reduction="mean") # scalar
                pbar.set_postfix(loss=f"{loss.item():.3f}")
                loss_list.append(loss.item())  
        
        self._print_model_output(logits=logits1, target_ids=target_ids1)        
        avg_loss = sum(loss_list)/len(loss_list)
        ppx = math.exp(avg_loss)
        return ppx, avg_loss
    
    def _print_model_output(self, logits: torch.tensor, target_ids: torch.tensor) -> None:
        logits = logits.argmax(dim=-1) # (b, Tmax, V) -> (b, Tmax)
        decoded_out = self.tokenizer.batch_decode(logits[0,:25].tolist())
        true_out = self.tokenizer.batch_decode(target_ids[0,:25].tolist())
        out = list(zip(decoded_out, true_out))
        print("\n" + "-"*10 + "(Pred token, True token)" + "-"*10 + "\n")
        for i in out:
            print(i)
        
    def _calc_ppx_change(self) -> torch.tensor:
        ppx_1d_list = []
        ppx_2d_list = []
        loss_1d_list = []
        loss_2d_list = []
        
        for src_lang in self.lang_list:
            src_dataset = WikipediaDataset(model_name=self.full_model_name ,lang=src_lang, max_context_len=self.Tmax) 
            src_ppx, src_loss = self._calc_ppx_for_lang(dataset=src_dataset, inv_lang=None, lang=src_lang)
            ppx_1d_list.append(src_ppx)
            loss_1d_list.append(src_loss)
            
            ppx_inv_list = []
            loss_inv_list = []
            for tgt_lang in self.lang_list:
                tgt_dataset = WikipediaDataset(model_name=self.full_model_name, lang=tgt_lang, max_context_len=self.Tmax)
                tgt_ppx, tgt_loss = self._calc_ppx_for_lang(dataset=tgt_dataset, inv_lang=src_lang, lang=tgt_lang)
                ppx_inv_list.append(tgt_ppx)
                loss_inv_list.append(tgt_loss)
            ppx_2d_list.append(ppx_inv_list)
            loss_2d_list.append(loss_inv_list)
            
        ppx_1d_tensor = torch.tensor(ppx_1d_list)
        loss_1d_tensor = torch.tensor(loss_1d_list)
        ppx_2d_tensor = torch.tensor(ppx_2d_list)
        loss_2d_tensor = torch.tensor(loss_2d_list)
        ppx_change = ppx_2d_tensor - ppx_1d_tensor
        loss_change = loss_2d_tensor - loss_1d_tensor
        return ppx_change, loss_change

    def _plot_change(self, change: torch.tensor, is_ppx: bool) -> None:
        name = "ppx" if is_ppx else "loss"
        save_path = str(Path(self.ppx_path.parent, f"{name}_change.png"))
        change_np = change.numpy()
        plt.figure(figsize=(len(self.lang_list), len(self.lang_list)))
        sns.heatmap(change_np, annot=True, cmap="Reds", fmt=".2f", linewidths=.5, xticklabels=self.lang_list, yticklabels=self.lang_list)
        plt.xlabel(f'Language - i of Set {self.lang_set[-1]}')
        plt.ylabel(f'Language - j of Set {self.lang_set[-1]}')
        tokens = {k: v for k, v in token_repr_map[self.model_name].items() if k in lang_map[self.lang_set]}
        lang_repr = {k: v for k, v in lang_repr_map[self.model_name].items() if k in lang_map[self.lang_set]}
        ch_type = "PPXC(i,j): Perplexity" if is_ppx else "CELC(i,j): CE Loss"
        title = f'{self.model_name}: {ch_type} change at language j after intervention at language i\n' + f'Tokens seen (in M): {tokens}\n' + f'Lang repr: {lang_repr}'
            
        plt.title(title, wrap=True)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(str(save_path), format='png', dpi=300)
        plt.clf()
        
def main(model_name: str, lang_set: str, device: torch.device) -> None:
    tokenizer, model = get_tokenizer_and_model(model_name=model_name, device=device)
    ppx_config = {
        "max_context_len": 512,
        "batch_size": 4,
        "lang_set": lang_set,
        "data_frac": 0.001
    }
    ppx = Perplexity(device=device, tokenizer=tokenizer, model=model, model_name=model_name, ppx_config=ppx_config)
    
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "4"
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}...")
    
    for model_key in ["aya101"]:
        for lang_set in ["set3", "set1", "set2", "set4"]:
            main(model_name=models_map[model_key], lang_set=lang_set, device=device)
            print(f"Model: {model_key}, Lang set: {lang_set} done!")
    
