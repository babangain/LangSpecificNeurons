import torch, tqdm, os, sys, pickle, datetime
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, Subset
from datasets import load_dataset
from typing import Tuple, List
from transformers import AutoTokenizer

class WikipediaDataset(Dataset):
    def __init__(self, tokenizer: AutoTokenizer, lang: str, max_context_len: int) -> None:
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
            Path.mkdir(self.ds_file_name.parent, exist_ok=True, parents=True)
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
        return torch.tensor(self.ds[start:end]) # (Tmax,)
    
    def collate_function(self, data: List[torch.tensor]) -> dict:
        input_ids = torch.stack(data, dim=0) # (b, Tmax)
        attention_mask = torch.ones_like(input_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask}
    
    def prepare_dataloader(self, batch_size: int, frac: float) -> DataLoader:
        indices = list(range(0, int(len(self) * frac)))
        subset = Subset(self, indices=indices)
        dl = DataLoader(subset, batch_size=batch_size, shuffle=True, collate_fn=self.collate_function, drop_last=True)
        return dl

def main():
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    ds = WikipediaDataset(tokenizer=tokenizer, lang="en", max_context_len=512)
    ds = WikipediaDataset(tokenizer=tokenizer, lang="fr", max_context_len=512)
    ds = WikipediaDataset(tokenizer=tokenizer, lang="es", max_context_len=512)
    ds = WikipediaDataset(tokenizer=tokenizer, lang="vi", max_context_len=512)
    ds = WikipediaDataset(tokenizer=tokenizer, lang="id", max_context_len=512)
    ds = WikipediaDataset(tokenizer=tokenizer, lang="ja", max_context_len=512)
    ds = WikipediaDataset(tokenizer=tokenizer, lang="zh", max_context_len=512)

if __name__ == "__main__":
    main()