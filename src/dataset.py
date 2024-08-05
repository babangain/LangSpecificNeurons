import torch, tqdm, os, sys, pickle, datetime
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, Subset
from datasets import load_dataset
from typing import Tuple, List, Union
from transformers import AutoTokenizer

class WikipediaDataset(Dataset):
    def __init__(self, tokenizer: Union[AutoTokenizer, None], lang: str, max_context_len: int) -> None:
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
        
        self.ds.append(0) # last token ID must be eot token ID
    
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
        return int((len(self.ds)-1)/self.Tmax)
    
    def __getitem__(self, index: int) -> Tuple[torch.tensor, torch.tensor]:
        assert index < len(self) and index >= 0, f"Index must be in between 0 to {len(self)-1}"
        start = self.Tmax * index
        end = start + self.Tmax
        x = torch.tensor(self.ds[start:end]) # (Tmax,)
        y = torch.tensor(self.ds[start+1:end+1]) # (Tmax,)
        return (x, y)
    
    def collate_function(self, data: List[Tuple[torch.tensor, torch.tensor]]) -> dict:
        data_x = [x for x, y in data]
        data_y = [y for x, y in data]
        input_ids = torch.stack(data_x, dim=0) # (b, Tmax)
        target_ids = torch.stack(data_y, dim=0) # (b, Tmax)
        attention_mask = torch.ones_like(input_ids)
        return {"input_ids": input_ids, "target_ids": target_ids, "attention_mask": attention_mask}
    
    def prepare_dataloader(self, batch_size: int, frac: float) -> DataLoader:
        indices = list(range(0, int(len(self) * frac)))
        subset = Subset(self, indices=indices)
        dl = DataLoader(subset, batch_size=batch_size, shuffle=True, collate_fn=self.collate_function, drop_last=True)
        return dl

def main():
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    lang_list = ["en", "fr", "es", "vi", "id", "ja", "zh"]
    for lang in lang_list:
        ds = WikipediaDataset(tokenizer=tokenizer, lang=lang, max_context_len=512)

if __name__ == "__main__":
    main()