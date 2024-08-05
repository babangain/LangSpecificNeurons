import torch, json, os, sys, tqdm, pickle, datetime, math
from pathlib import Path
sys.path.append(Path(__file__).parent)
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Union
import pandas as pd
from dataset import WikipediaDataset
from models import LlamaModelForProbing
      
class Perplexity:
    def __init__(self, device: torch.device, model: torch.nn.Module, model_name: str, src_lang_dataset: Dataset):
        self.device = device
        self.src_lang_dataset = src_lang_dataset
        self.src_lang = self.src_lang_dataset.lang
        self.model_name = model_name.split("/")[-1]
        self.model = model
        self.cwd = Path.cwd()
        self.lang_neuron_path = Path(self.cwd, f"outputs/lang_neurons/{self.model_name}/lang_neuron_data.pkl")
        self.lang_neuron = pickle.load(open(self.lang_neuron_path, "rb"))
    
    def info(self) -> str:
        return f"[INFO] {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    
    def _calc_ppx(self, dl: DataLoader, lang: str, lang_type: str, intervene_config: Union[dict, None]):
        loss_list = []
        with tqdm.tqdm(iterable=dl, 
                       desc=f"Calculating perplexity for {lang_type} lang: {lang}",
                       unit=" batches",
                       colour="green") as pbar:
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
    
    def _calc_ppx_for_src_lang(self, batch_size: int, data_frac: float) -> float:
        dl = self.src_lang_dataset.prepare_dataloader(batch_size=batch_size, frac=data_frac)
        ppx = self._calc_ppx(dl, lang=self.src_lang, lang_type="source", intervene_config=None)
        return ppx

    def _calc_ppx_for_tgt_lang(self, tgt_lang_dataset: str, batch_size: int, data_frac: float) -> float:
        """src lang neurons will be deactivated here"""
        tgt_lang = tgt_lang_dataset.lang
        tgt_dl = tgt_lang_dataset.prepare_dataloader(batch_size=batch_size, frac=data_frac)
        intervene_config = {
            "indices": self.lang_neuron["lang_to_neuron"][self.src_lang],
            "value": 0
        }
        ppx = self._calc_ppx(tgt_dl, lang=tgt_lang, lang_type="target", intervene_config=intervene_config)
        return ppx
        
def main(model_name: str, device: torch.device) -> None:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if "llama" in model_name.lower():
        model = LlamaModelForProbing(tokenizer=tokenizer, device=device)
    else:
        raise NotImplementedError("Invalid model name!")
    
    max_context_len = 512
    batch_size = 4
    data_frac = 0.001
    
    src_lang = "en"
    tgt_lang = "en"
    src_dataset = WikipediaDataset(tokenizer=tokenizer, lang=src_lang, max_context_len=max_context_len)   
    tgt_dataset = WikipediaDataset(tokenizer=tokenizer, lang=tgt_lang, max_context_len=max_context_len)   
    ppx = Perplexity(device=device, model=model, model_name=model_name, src_lang_dataset=src_dataset)
    print(ppx._calc_ppx_for_src_lang(batch_size, data_frac))
    print(ppx._calc_ppx_for_tgt_lang(tgt_dataset, batch_size, data_frac))
    
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "6"
    models = ["meta-llama/Llama-2-7b-hf"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}...")
    
    main(models[0], device=device)
    