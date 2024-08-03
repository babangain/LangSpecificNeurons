import torch, json, os, sys, tqdm, pickle, datetime
from pathlib import Path
sys.path.append(Path(__file__).parent)
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple
import pandas as pd
from dataset import WikipediaDataset
from models import LlamaModelForProbing
      
class Activation:
    def __init__(self, tokenizer: AutoTokenizer, model: torch.nn.Module, dataset: Dataset):
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.cwd = Path.cwd()
        self.lang = self.dataset.lang
        self.act_data_path = Path(self.cwd, f"outputs/activation/act_{self.lang}.pkl")
        self.model = None if self.act_data_path.exists() else model
        Path.mkdir(self.act_data_path.parent, exist_ok=True, parents=True)
    
    def info(self) -> str:
        return f"[INFO] {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    
    def get_activation_probability(self, batch_size: int, data_frac: float) -> dict:
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
                       colour="green",
                       ascii=True) as pbar:
            for input_dict in pbar:
                out_dict = self.model(**input_dict)
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
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if "llama" in model_name.lower():
        model = LlamaModelForProbing(tokenizer=tokenizer, device=device)
    else:
        raise NotImplementedError("Invalid model name!")
    
    max_context_len = 512
    batch_size = 4
    
    for lang in ["en", "fr", "es", "vi", "id", "ja", "zh"]:
        dataset = WikipediaDataset(tokenizer=tokenizer, lang=lang, max_context_len=max_context_len)   
        act = Activation(tokenizer=tokenizer, model=model, dataset=dataset)
        out = act.get_activation_probability(batch_size=batch_size, data_frac=1.0) 
    
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    models = ["meta-llama/Llama-2-7b-hf"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}...")
    
    main(models[0], device=device)
    