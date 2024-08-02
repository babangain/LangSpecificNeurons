import torch, tqdm, os, sys, pickle, datetime
from pathlib import Path
from torch.utils.data import Dataset
from datasets import load_dataset
from typing import Tuple, List
from transformers import AutoTokenizer

class WikipediaDataset(Dataset):
    def __init__(self, tokenizer: AutoTokenizer, lang: str, max_context_len: int=512) -> None:
        super(WikipediaDataset, self).__init__()
        self.cwd = Path.cwd()
        self.lang = lang
        self.ds_file_name = Path(self.cwd, f"data/{self.lang}.pkl")
        self.tokenizer = tokenizer
        self.Tmax = max_context_len
        
        if self.ds_file_name.exists():
            self.ds = pickle.load(open(self.ds_file_name, "rb"))
            print(f"{self.info()}: The dataset is loaded from {self.ds_file_name}")
        else:
            self.ds = self.get_dataset()
    
    def info(self) -> str:
        return f"[INFO] {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    
    def get_dataset(self) -> List[int]:
        ds = load_dataset("graelo/wikipedia", f"20230901.{self.lang}")
        shuffled_ds = ds["train"].shuffle(seed=42)
        token_ids = []
        count = 0
        with tqdm.tqdm(iterable=range(len(shuffled_ds)), 
                       desc=f"Creating dataset for lang: {self.lang} (to be stopped after 100 M tokens)",
                       unit=" articles",
                       colour="green") as pbar:
            for i in pbar:
                ids  = self.tokenizer(shuffled_ds[i]["text"])["input_ids"]
                token_ids +=  ids # List[1M]
                count += len(ids)
                pbar.set_postfix(tokens_seen=f"{count/(10**6):.2f} M")
                if count > 10**8 and count % 4096 == 0:
                    break
        
        pickle.dump(token_ids, open(self.ds_file_name, "wb"))
        print(f"{self.info()}: The dataset is stored at {self.ds_file_name}")
        return token_ids
    
    def __len__(self) -> int:
        return int(len(self.ds)/self.Tmax)
    
    def __getitem__(self, index: int) -> torch.tensor:
        assert index < len(self) and index >= 0, f"Index must be in between 0 to {len(self)-1}"
        start = self.Tmax * index
        end = start + self.Tmax
        return torch.tensor(self.ds[start:end])

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    ds = WikipediaDataset(tokenizer=tokenizer, lang="en")
