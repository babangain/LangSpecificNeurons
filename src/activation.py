import torch, json, os, sys, tqdm, pickle, datetime
from pathlib import Path
sys.path.append(Path(__file__).parent)
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Union
import pandas as pd
from dataset import WikipediaDataset
from models import get_tokenizer_and_model
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
        
def main(model_name: str, device: torch.device) -> None:
    tokenizer, model = get_tokenizer_and_model(model_name=model_name, device=device)
    max_context_len = 512
    batch_size = 4
    for lang in ["hi", "ta", "te", "ur"]:
        dataset = WikipediaDataset(model_name=model_name, lang=lang, max_context_len=max_context_len)   
        act = Activation(model=model, model_name=model_name, dataset=dataset, lang=lang)
        data_frac = 1.0 if dataset.tokens_count < 75*10**5 else 0.75
        out = act.get_activation_data(batch_size=batch_size, data_frac=data_frac) 
    
    print("DONE")
    
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}...")
    
    main(models_map["mistral-nemo"], device=device)
    
    