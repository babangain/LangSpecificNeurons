import torch, tqdm, os, sys, pickle, datetime
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, Subset
from datasets import load_dataset
from typing import Tuple, List, Union
from transformers import AutoTokenizer
from utils import lang_map, models_map

class WikipediaDataset(Dataset):
    def __init__(self, model_name: str, lang: str, max_context_len: int) -> None:
        super(WikipediaDataset, self).__init__()
        self.cwd = Path.cwd()
        self.lang = lang
        self.model_name = model_name.split("/")[-1]
        self.ds_file_name = Path(self.cwd, f"data/{self.model_name}/{self.lang}.pkl")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.Tmax = max_context_len
        
        if self.ds_file_name.exists():
            self.__dict__.update(pickle.load(open(self.ds_file_name, "rb")))
            print(f"{self.info()}: The dataset is loaded from {self.ds_file_name}")
        else:
            Path.mkdir(self.ds_file_name.parent, exist_ok=True, parents=True)
            self.ds, self.tokens_count = self.get_dataset()
            self.ds.append(0) # last token ID must be eot token ID
            pickle.dump(self.__dict__, open(self.ds_file_name, "wb"))
            print(f"{self.info()}: The dataset is stored at {self.ds_file_name}")
    
    def info(self) -> str:
        return f"[INFO] {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    
    def get_dataset(self) -> List[int]:
        ds = load_dataset("graelo/wikipedia", f"20230901.{self.lang}")
        shuffled_ds = ds["train"].shuffle(seed=42)
        token_ids = []
        count = 0
        with tqdm.tqdm(iterable=range(len(shuffled_ds)), 
                       desc=f"Creating dataset for lang: {self.lang} (to be stopped after 100M tokens)",
                       unit=" articles",
                       colour="green") as pbar:
            for i in pbar:
                ids  = self.tokenizer(shuffled_ds[i]["text"])["input_ids"]
                token_ids +=  ids # List[100M]
                count += len(ids)
                pbar.set_postfix(tokens_seen=f"{count/(10**6):.2f}M")
                if count > 10**8:
                    break
        return token_ids, count
    
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
        dl = DataLoader(subset, batch_size=batch_size, shuffle=False, collate_fn=self.collate_function, drop_last=True)
        return dl

class XNLIDataset(Dataset):
    def __init__(self, model_name: str, lang: str, max_context_len: int, frac: float, is_train: bool) -> None:
        super(XNLIDataset, self).__init__()
        self.cwd = Path.cwd()
        self.lang = lang
        self.frac = frac
        self.is_train = is_train
        self.model_name = model_name.split("/")[-1]
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.Tmax = max_context_len
        self.ds = self.get_dataset()
        
    def get_dataset(self) -> Dataset:
        ds_dict = load_dataset("xnli", self.lang)
        key = "train" if self.is_train else "test"
        ds = ds_dict[key]
        size = int(len(ds) * self.frac)
        subset_ds = ds.select(range(size)) # do not apply shuffling
        dsl = [subset_ds[i] for i in range(len(subset_ds))]
        
        filter_dsl = []
        with tqdm.tqdm(iterable=range(len(dsl)), desc="Preparing dataset...", unit="example", colour="green") as pbar:
            for index in pbar:
                inputs = [dsl[index]["premise"] + f" {self.tokenizer.eos_token} " + dsl[index]["hypothesis"]]
                outputs = self.tokenizer(inputs, padding="max_length", truncation=True, max_length=512, return_tensors="pt") # (1, Tmax)
                label = torch.tensor([dsl[index]["label"]]) # (1,)
                outputs["seq_len"] = outputs["attention_mask"].sum().item()
                if outputs["seq_len"] < self.Tmax:
                    outputs["input_ids"] = outputs["input_ids"][:, :self.Tmax]
                    outputs["attention_mask"] = outputs["attention_mask"][:, :self.Tmax]
                    filter_dsl.append((outputs, label))
        return filter_dsl
    
    def __len__(self) -> int:
        return len(self.ds)
    
    def __getitem__(self, index: int) -> Tuple[dict, torch.tensor]:
        return self.ds[index]
    
    @staticmethod
    def collate_function(data: List[Tuple[dict, torch.tensor]]) -> dict:
        input_ids_list = []
        attention_mask_list = []
        seq_len_list = []
        label_list = []
        for x, y in data:
            input_ids_list.append(x["input_ids"][0, :])
            attention_mask_list.append(x["attention_mask"][0, :])
            label_list.append(y[0])
        
        input_ids = torch.stack(input_ids_list, dim=0) # (b, Tmax)
        attention_mask = torch.stack(attention_mask_list, dim=0) # (b, Tmax)
        labels = torch.stack(label_list, dim=0) # (b,)
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}
    
    def prepare_dataloader(self, batch_size: int) -> DataLoader:
        dl = DataLoader(self, batch_size=batch_size, shuffle=self.is_train, collate_fn=XNLIDataset.collate_function, drop_last=True)
        return dl

class XCOPADataset(Dataset):
    def __init__(self, model_name: str, lang: str, max_context_len: int, frac: float, is_train: bool) -> None:
        super(XCOPADataset, self).__init__()
        self.cwd = Path.cwd()
        self.lang = lang
        self.frac = frac
        self.is_train = is_train
        self.model_name = model_name.split("/")[-1]
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.Tmax = max_context_len
        self.ds = self.get_dataset()
        
    def get_dataset(self) -> Dataset:
        if self.lang == "en":
            ds_dict = load_dataset("pkavumba/balanced-copa")
            ds_train = ds_dict["train"]
            ds_test = ds_dict["test"]
            ds = ds_train if self.is_train else ds_test
        else:
            ds_dict = load_dataset("xcopa", self.lang)
            ds_train = ds_dict["test"]
            ds_test = ds_dict["validation"]
            ds = ds_train if self.is_train else ds_test

        size = int(len(ds) * self.frac)
        subset_ds = ds.select(range(size)) # do not apply shuffling
        dsl = [subset_ds[i] for i in range(len(subset_ds))]
        
        filter_dsl = []
        with tqdm.tqdm(iterable=range(len(dsl)), desc="Preparing dataset...", unit="example", colour="green") as pbar:
            for index in pbar:
                inputs = [dsl[index]["premise"] + f" {self.tokenizer.eos_token} " + dsl[index]["choice1"] + f" {self.tokenizer.eos_token} " + dsl[index]["choice2"]]
                outputs = self.tokenizer(inputs, padding="max_length", truncation=True, max_length=512, return_tensors="pt") # (1, Tmax)
                label = torch.tensor([dsl[index]["label"]]) # (1,)
                outputs["seq_len"] = outputs["attention_mask"].sum().item()
                if outputs["seq_len"] < self.Tmax:
                    outputs["input_ids"] = outputs["input_ids"][:, :self.Tmax]
                    outputs["attention_mask"] = outputs["attention_mask"][:, :self.Tmax]
                    filter_dsl.append((outputs, label))
        return filter_dsl
    
    def __len__(self) -> int:
        return len(self.ds)
    
    def __getitem__(self, index: int) -> Tuple[dict, torch.tensor]:
        return self.ds[index]
    
    @staticmethod
    def collate_function(data: List[Tuple[dict, torch.tensor]]) -> dict:
        input_ids_list = []
        attention_mask_list = []
        label_list = []
        for x, y in data:
            input_ids_list.append(x["input_ids"][0, :])
            attention_mask_list.append(x["attention_mask"][0, :])
            label_list.append(y[0])
        
        input_ids = torch.stack(input_ids_list, dim=0) # (b, Tmax)
        attention_mask = torch.stack(attention_mask_list, dim=0) # (b, Tmax)
        labels = torch.stack(label_list, dim=0) # (b,)
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}
    
    def prepare_dataloader(self, batch_size: int) -> DataLoader:
        dl = DataLoader(self, batch_size=batch_size, shuffle=self.is_train, collate_fn=XNLIDataset.collate_function, drop_last=True)
        return dl

class XNLIDatasetHF(Dataset):
    def __init__(self, model_name: str, lang: str, max_context_len: int, frac: float, is_train: bool) -> None:
        super(XNLIDatasetHF, self).__init__()
        self.cwd = Path.cwd()
        self.lang = lang
        self.frac = frac
        self.is_train = is_train
        self.model_name = model_name.split("/")[-1]
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.Tmax = max_context_len
        self.ds = self.get_dataset()
        
    def get_dataset(self) -> Dataset:
        ds_dict = load_dataset("xnli", self.lang)
        key = "train" if self.is_train else "test"
        ds = ds_dict[key]
        size = int(len(ds) * self.frac)
        subset_ds = ds.select(range(size)) # do not apply shuffling
        dsl = [subset_ds[i] for i in range(len(subset_ds))]
        
        filter_dsl = []
        with tqdm.tqdm(iterable=range(len(dsl)), desc="Preparing dataset...", unit="example", colour="green") as pbar:
            for index in pbar:
                inputs = [dsl[index]["premise"] + f" {self.tokenizer.eos_token} " + dsl[index]["hypothesis"]]
                outputs = self.tokenizer(inputs, padding="max_length", truncation=True, max_length=512, return_tensors="pt") # (1, Tmax)
                labels = torch.tensor([dsl[index]["label"]]) # (1,)
                seq_len = outputs["attention_mask"].sum().item()
                if seq_len < self.Tmax:
                    outputs["input_ids"] = outputs["input_ids"][0, :self.Tmax] # (Tmax,)
                    outputs["attention_mask"] = outputs["attention_mask"][0, :self.Tmax] # (Tmax,)
                    outputs["labels"] = labels[0] # (scalar)
                    filter_dsl.append(outputs)
        return filter_dsl
    
    def __len__(self) -> int:
        return len(self.ds)
    
    def __getitem__(self, index: int) -> dict:
        return self.ds[index]

class WikipediaDatasetHF(Dataset):
    def __init__(self, model_name: str, lang: str, max_context_len: int) -> None:
        super(WikipediaDatasetHF, self).__init__()
        self.cwd = Path.cwd()
        self.lang = lang
        self.model_name = model_name
        self.model_name_srt = self.model_name.split("/")[-1]
        self.ds_file_name = Path(self.cwd, f"data/{self.model_name_srt}/{self.lang}.pkl")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.Tmax = max_context_len
        
        if self.ds_file_name.exists():
            self.__dict__.update(pickle.load(open(self.ds_file_name, "rb")))
            print(f"{self.info()}: The dataset is loaded from {self.ds_file_name}")
        else:
            Path.mkdir(self.ds_file_name.parent, exist_ok=True, parents=True)
            self.ds, self.tokens_count = self.get_dataset()
            self.ds.append(0) # last token ID must be eot token ID
            pickle.dump(self.__dict__, open(self.ds_file_name, "wb"))
            print(f"{self.info()}: The dataset is stored at {self.ds_file_name}")
    
    def info(self) -> str:
        return f"[INFO] {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    
    def get_dataset(self) -> List[int]:
        ds = load_dataset("graelo/wikipedia", f"20230901.{self.lang}")
        shuffled_ds = ds["train"].shuffle(seed=42)
        token_ids = []
        count = 0
        with tqdm.tqdm(iterable=range(len(shuffled_ds)), 
                       desc=f"Creating dataset for lang: {self.lang} (to be stopped after 100M tokens)",
                       unit=" articles",
                       colour="green") as pbar:
            for i in pbar:
                ids  = self.tokenizer(shuffled_ds[i]["text"])["input_ids"]
                token_ids +=  ids # List[100M]
                count += len(ids)
                pbar.set_postfix(tokens_seen=f"{count/(10**6):.2f}M")
                if count > 10**8:
                    break
        return token_ids, count
    
    def __len__(self) -> int:
        return int((len(self.ds)-1)/self.Tmax)
    
    def __getitem__(self, index: int) -> dict:
        assert index < len(self) and index >= 0, f"Index must be in between 0 to {len(self)-1}"
        start = self.Tmax * index
        end = start + self.Tmax
        x = torch.tensor(self.ds[start:end]) # (Tmax,)
        y = torch.tensor(self.ds[start+1:end+1]) # (Tmax,)
        item = {
            "input_ids": x[:self.Tmax], # (Tmax,)
            "attention_mask": torch.ones_like(x[:self.Tmax]), # (Tmax,)
            "labels": y[:self.Tmax] # (Tmax,)
        }         
        return item

def main_wiki(model_name: str):
    ds = WikipediaDatasetHF(model_name=model_name, lang="en", max_context_len=256)
    print(len(ds))

def main_xnli(model_name: str):
    ds = XNLIDatasetHF(model_name=model_name, lang="fr", max_context_len=256, frac=0.01, is_train=True)
    print(len(ds))
    print("DONE")
    
def main_xcopa(model_name: str):
    ds = XCOPADataset(model_name=model_name, lang="en", max_context_len=256, frac=1.0, is_train=True)
    print(len(ds))
    print("DONE")

if __name__ == "__main__":
    ml = ["llama2"]
    for model_key in ml:
        main_wiki(model_name=models_map[model_key])
        print(f"Model: {model_key} done!")
