import wandb, torch, tqdm, sys, os, json, math, gc
from pathlib import Path
sys.path.append(Path(__file__).parent.parent.__str__())
from typing import List, Tuple, Union, Any
from dataset import XNLIDataset
from lora_models import ModelForCLSWithLoRA
from utils import models_map
from datasets import load_dataset
from torch.utils.data import Dataset

class Evaluator:
    def __init__(self, device: torch.device, config: dict):
        self.device = device
        self.config = config
        self.model_name_srt = config["model_name"].split("/")[-1]
        self.train_lang = config["train_lang"]
        self.eval_lang = config["eval_lang"]
        self.project_name = f"{self.model_name_srt}-finetune-XNLI-{self.train_lang}"
        self.checkpoint_dir = Path(Path.cwd(), f"outputs/ckpt/{self.project_name}")
        
        self.model = ModelForCLSWithLoRA(device=self.device, model_name=self.config["model_name"], num_class=self.config["num_class"], lora_rank=self.config["lora_rank"], lora_alpha=self.config["lora_alpha"]).to(self.device)
        self.print_cuda_memory_usage("0")
        self._load_checkpoint()
        self.print_cuda_memory_usage("1")
        self.eval_ds = XNLIDataset(model_name=config["model_name"], lang=self.eval_lang, max_context_len=config["max_seq_len"], frac=config["eval_frac"], is_train=False)
        self.print_cuda_memory_usage("2")
        self.eval_dl = self.eval_ds.prepare_dataloader(batch_size=config["batch_size"])
        self.print_cuda_memory_usage("3")
        
    def print_cuda_memory_usage(self, step_description=""):
        print(f"\n--- {step_description} ---")
        print(f"Allocated Memory: {torch.cuda.memory_allocated() / (1024 ** 3):.2f} GB")
        print(f"Reserved Memory: {torch.cuda.memory_reserved() / (1024 ** 3):.2f} GB")
        print(f"Max Allocated Memory: {torch.cuda.max_memory_allocated() / (1024 ** 3):.2f} GB")
        print(f"Max Reserved Memory: {torch.cuda.max_memory_reserved() / (1024 ** 3):.2f} GB")

        gpu_tensors = []
        for obj in gc.get_objects():
            if torch.is_tensor(obj):
                if obj.device.type == 'cuda':
                    gpu_tensors.append(obj)
        size = 0
        for tensor in gpu_tensors:
            size += tensor.numel() * tensor.element_size() / 1024**3
        print(f"Occupied Memory by Tensors: {size:.2f} GB\n")
   
    def _load_checkpoint(self, name: str = "ckpt_ep_0.pth") -> None:
        checkpoint_path = Path(self.checkpoint_dir, name)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        print(f"[LOAD] ep: {checkpoint['epoch']}, checkpoint loaded from: {checkpoint_path}")
    
    def _forward_batch(self, batch: dict) -> torch.tensor:
        input_ids = batch["input_ids"].to(self.device) # (b, T)
        attention_mask = batch["attention_mask"].to(self.device) # (b, T)
        labels = batch["labels"].to(self.device) # (b, T)
        self.model.eval()
        with torch.no_grad():
            out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return out # (b, c)
    
    def _calc_acc_batch(self, pred_outputs: torch.tensor, true_outputs: torch.tensor) -> torch.tensor:
        pred_outputs = pred_outputs.to(self.device)
        true_outputs = true_outputs.to(self.device)
        assert pred_outputs.dim() == 2, f"pred_outputs.shape = {pred_outputs.shape} must be (b, c)"
        assert true_outputs.dim() == 1, f"true_outputs.shape = {true_outputs.shape} must be (b,)"
        acc = (pred_outputs.argmax(dim=-1) == true_outputs).to(torch.float32).mean()
        return torch.tensor(acc.item()) # returns the tensor as a scalar number

    def evaluate_on_eval_lang(self) -> float:
        with tqdm.tqdm(iterable=self.eval_dl, desc=f"[EVAL] on lang {self.eval_lang}", total=len(self.eval_dl), unit="step", colour="green") as pbar:
            acc_list = []
            for i, batch in enumerate(pbar):    
                pred_out = self._forward_batch(batch=batch, is_train=False) # (b, c)  
                true_out = batch["labels"] # (b,)   
                acc = self._calc_acc_batch(pred_outputs=pred_out, true_outputs=true_out)
                acc_list.append(acc)
                pbar.set_postfix({"acc": f"{acc:.3f}"})  
                self.print_cuda_memory_usage(step_description=i)
        eval_acc = sum(acc_list)/len(acc_list)
        print(f"[RESULT] Evaluation accuracy: {eval_acc} on eval lang: {self.eval_lang}")
        return eval_acc                      
         
def main(model_name: str, device: torch.device) -> None:
    config = {
        "model_name": model_name,
        "train_lang": "en",
        "eval_lang": "vi",
        "num_class": 3,
        "lora_rank": 8,
        "lora_alpha": 16,
        "eval_frac": 0.001,
        "max_seq_len": 256,
        "batch_size": 4
    }
    evaluator = Evaluator(device=device, config=config)
    evaluator.evaluate_on_eval_lang()
    print("DONE")
    
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "4"
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}...")
    
    main(models_map["llama2"], device=device)
        

