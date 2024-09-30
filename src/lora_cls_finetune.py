import wandb, torch, tqdm, sys, os, json, math, gc
from pathlib import Path
sys.path.append(Path(__file__).parent.parent.__str__())
from typing import List, Tuple, Union, Any
from dataset import XNLIDataset, XCOPADataset
from lora_models import ModelForCLSWithLoRA
from utils import models_map
from datasets import load_dataset
from torch.utils.data import Dataset
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class LoRAFineTuner:
    def __init__(self, device: torch.device, config: dict):
        self.device = device
        self.config = config
        self.model_name_srt = config["model_name"].split("/")[-1]
        self.lang = config["lang"]
        self.num_epochs = config["num_epochs"]
        self.wandb_log = config["wandb_log"]
        self.calc_norm = config["calc_norm"]
        self.acc_grad_steps = config["acc_grad_steps"]
        self.project_name = f"{self.model_name_srt}-finetune-{self.config['task_name']}"
        self.checkpoint_dir = Path(Path.cwd(), f"outputs/ckpt/{self.project_name}/{self.lang}")
        
        if self.config["task_name"] == "XNLI":
            self.train_ds = XNLIDataset(model_name=self.config["model_name"], lang=self.lang, max_context_len=self.config["max_seq_len"], frac=self.config["train_frac"], is_train=True)
            self.val_ds = XNLIDataset(model_name=self.config["model_name"], lang=self.lang, max_context_len=self.config["max_seq_len"], frac=self.config["val_frac"], is_train=False)
        elif self.config["task_name"] == "XCOPA":
            self.train_ds = XCOPADataset(model_name=self.config["model_name"], lang=self.lang, max_context_len=self.config["max_seq_len"], frac=self.config["train_frac"], is_train=True)
            self.val_ds = XCOPADataset(model_name=self.config["model_name"], lang=self.lang, max_context_len=self.config["max_seq_len"], frac=self.config["val_frac"], is_train=False)
        else:
            raise ValueError("Invalid task name!")
        
        self.train_dl = self.train_ds.prepare_dataloader(batch_size=self.config["batch_size"])
        self.val_dl = self.val_ds.prepare_dataloader(batch_size=self.config["batch_size"])
        self.model = ModelForCLSWithLoRA(device=self.device, model_name=self.config["model_name"], num_class=self.config["num_class"], lora_rank=self.config["lora_rank"], lora_alpha=self.config["lora_alpha"]).to(self.device)
        self.optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=self.config["initial_learning_rate"], weight_decay=self.config["weight_decay"], betas=(0.95, 0.99))
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=1, eta_min=self.config["final_learning_rate"])

        if self.wandb_log:
            run_name = f"{self.lang}_{self.config['train_frac']:.2f}_{self.config['initial_learning_rate']:.1e}_{self.config['final_learning_rate']:.1e}_r{self.config['lora_rank']}"
            wandb.init(project=self.project_name, name=run_name, config=config)
            wandb.watch(self.model, log="all")
            wandb.define_metric("train/step")
            wandb.define_metric("val/step")
            wandb.define_metric("train/*", step_metric="train/step")
            wandb.define_metric("val/*", step_metric="val/step")
    
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

    def _find_norm(self, is_grad: bool) -> float:
        norm = 0
        for val in self.model.parameters():
            if val.requires_grad:
                if is_grad:
                    k = val
                else:
                    k = val.grad if val.grad is not None else torch.tensor(0.0, device=self.device)
                norm += (k ** 2).sum().item()
        norm = norm ** 0.5  
        return norm
    
    def _save_checkpoint(self, ep: int, is_latest_ckpt: bool) -> None:
        checkpoint = {"epoch": ep, 
                      "model_state": self.model.state_dict(), 
                      "opt_state": self.optimizer.state_dict(),
                      "config": self.config}   
        if not Path.exists(self.checkpoint_dir):
            Path.mkdir(self.checkpoint_dir, parents=True, exist_ok=True)
        if is_latest_ckpt:
            checkpoint_path = Path(self.checkpoint_dir, f"ckpt_ep_latest.pth")
        else:
            checkpoint_path = Path(self.checkpoint_dir, f"ckpt_ep_{ep}.pth")            
        torch.save(checkpoint, checkpoint_path)
        print(f"[SAVE] ep: {ep}/{self.num_epochs-1}, checkpoint saved at: {checkpoint_path}")
    
    def _forward_batch(self, batch: dict, is_train: bool) -> torch.tensor:
        input_ids = batch["input_ids"].to(self.device) # (b, T)
        attention_mask = batch["attention_mask"].to(self.device) # (b, T)
        if is_train:
            self.model.train()
            out = self.model(input_ids=input_ids, attention_mask=attention_mask, intervene_config=None)
            out.requires_grad_(True)
            assert out.requires_grad == True
        else:
            self.model.eval()
            with torch.no_grad():
                out = self.model(input_ids=input_ids, attention_mask=attention_mask, intervene_config=None)
        return out # (b, c)
        
    def _calc_loss_batch(self, pred_outputs: torch.tensor, true_outputs: torch.tensor) -> torch.tensor:
        pred_outputs = pred_outputs.to(self.device)
        true_outputs = true_outputs.to(self.device)
        assert pred_outputs.dim() == 2, f"pred_outputs.shape = {pred_outputs.shape} must be (b, c)"
        assert true_outputs.dim() == 1, f"true_outputs.shape = {true_outputs.shape} must be (b,)"
        loss = torch.nn.functional.cross_entropy(input=pred_outputs, target=true_outputs)
        return loss # returns the computational graph also along with it
        
    def _calc_acc_batch(self, pred_outputs: torch.tensor, true_outputs: torch.tensor) -> torch.tensor:
        pred_outputs = pred_outputs.to(self.device)
        true_outputs = true_outputs.to(self.device)
        assert pred_outputs.dim() == 2, f"pred_outputs.shape = {pred_outputs.shape} must be (b, c)"
        assert true_outputs.dim() == 1, f"true_outputs.shape = {true_outputs.shape} must be (b,)"
        acc = (pred_outputs.argmax(dim=-1) == true_outputs).to(torch.float32).mean()
        return torch.tensor(acc.item()) # returns the tensor as a scalar number

    def _optimize_batch(self, pred_outputs: torch.tensor, true_outputs: torch.tensor, ep: int, batch_index: int) -> Tuple[float, float, float, float]:  
        loss = self._calc_loss_batch(pred_outputs=pred_outputs, true_outputs=true_outputs) / self.acc_grad_steps
        loss.backward()       
        if (batch_index+1) % self.acc_grad_steps == 0:
            gn = self._find_norm(True) if self.calc_norm else -1
            pn = self._find_norm(False) if self.calc_norm else -1 
            lr = self.optimizer.param_groups[0]['lr'] 
            torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=self.config["clip_grad_norm_value"], norm_type=2.0)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.scheduler.step(ep + batch_index/len(self.train_dl))
        else:
            gn, pn, lr = -1, -1, -1
        return loss.item() * self.acc_grad_steps, gn, pn, lr
    
    def _optimize_dataloader(self, ep: int) -> None:  
        with tqdm.tqdm(iterable=self.train_dl, desc=f"[TRAIN] ep: {ep}/{self.num_epochs-1}", total=len(self.train_dl), unit="step", colour="green") as pbar:
            loss_list, acc_list = [], []
            for i, batch in enumerate(pbar):    
                pred_out = self._forward_batch(batch=batch, is_train=True) # (b, c)  
                true_out = batch["labels"] # (b,)   
                loss, gn, pn, lr = self._optimize_batch(pred_outputs=pred_out, true_outputs=true_out, ep=ep, batch_index=i)
                acc = self._calc_acc_batch(pred_outputs=pred_out, true_outputs=true_out)
                loss_list.append(loss) 
                acc_list.append(acc)
                
                if ((i+1) % self.acc_grad_steps == 0):
                    loss_avg, acc_avg = sum(loss_list)/len(loss_list), sum(acc_list)/len(acc_list)
                    if self.wandb_log:
                        wandb.log({"train/loss": loss_avg, "train/accuracy": acc_avg, "train/learning_rate": lr, "train/grad_norm": gn, "train/param_norm": pn, "train/epoch": ep, "train/step": self.train_step})
                        self.train_step += 1
                    loss_list, acc_list = [], []
                    torch.cuda.empty_cache()
                    pbar.set_postfix({"loss_avg": f"{loss_avg:.3f}", "acc_avg": f"{acc_avg:.3f}", "lr": f"{lr:.3e}", "gn": f"{gn:.3f}", "pn": f"{pn:.3f}"})
                else:
                    pbar.set_postfix({"loss": f"{loss:.3f}", "acc": f"{acc:.3f}", "lr": f"{lr:.3e}", "gn": f"{gn:.3f}", "pn": f"{pn:.3f}"})                        
    
    def _validate_dataloader(self, ep: int) -> None:
        with tqdm.tqdm(iterable=self.val_dl, desc=f"[VAL] ep: {ep}/{self.num_epochs-1}", total=len(self.val_dl), unit="step", colour="green") as pbar:
            loss_list, acc_list = [], []
            for i, batch in enumerate(pbar):    
                pred_out = self._forward_batch(batch=batch, is_train=False) # (b, c)  
                true_out = batch["labels"] # (b,)   
                loss = self._calc_loss_batch(pred_outputs=pred_out, true_outputs=true_out).item()
                acc = self._calc_acc_batch(pred_outputs=pred_out, true_outputs=true_out)
                loss_list.append(loss) 
                acc_list.append(acc)
                
                if ((i+1) % self.acc_grad_steps == 0):
                    loss_avg, acc_avg = sum(loss_list)/len(loss_list), sum(acc_list)/len(acc_list)
                    if self.wandb_log:
                        wandb.log({"val/loss": loss_avg, "val/accuracy": acc_avg, "val/epoch": ep, "val/step": self.val_step})
                        self.val_step += 1
                    loss_list, acc_list = [], []
                    torch.cuda.empty_cache()
                    pbar.set_postfix({"loss_avg": f"{loss_avg:.3f}", "acc_avg": f"{acc_avg:.3f}"})
                else:
                    pbar.set_postfix({"loss": f"{loss:.3f}", "acc": f"{acc:.3f}"})                        
    
    def train(self) -> None:
        self.model.calc_num_lora_params()
        self.train_step = 0
        self.val_step = 0
        for ep in range(self.num_epochs):
            self._optimize_dataloader(ep=ep)
            self._validate_dataloader(ep=ep)
            self._save_checkpoint(ep=ep, is_latest_ckpt=self.config["is_latest_ckpt"])
        if self.wandb_log:
            wandb.finish()
    
    def _load_checkpoint(self, name: str = "ckpt_ep_0.pth") -> None:
        checkpoint_path = Path(self.checkpoint_dir, name)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["opt_state"])
        print(f"[LOAD] ep: {checkpoint['epoch']}, checkpoint loaded from: {checkpoint_path}")

def main(model_name: str, device: torch.device) -> None:
    config = {
        "model_name": model_name,
        "task_name": "XNLI",
        "lang": "en",
        "num_epochs": 1, 
        "batch_size": 4,
        "max_seq_len": 256,
        "train_frac": 0.1,
        "val_frac": 0.01,
        "num_class": 3,
        "lora_rank": 4,
        "lora_alpha": 8,
        "clip_grad_norm_value": 50.0,
        "initial_learning_rate": 5e-4,
        "final_learning_rate": 1e-9, 
        "weight_decay": 0.1,
        "acc_grad_steps": 16,
        "is_latest_ckpt": True,
        "calc_norm": True,
        "wandb_log": True
    }
    trainer = LoRAFineTuner(device=device, config=config)
    trainer.train()
    print("DONE")

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}...")
    
    main(models_map["sarvam"], device=device)
        
