import os, json, pickle, random
from pathlib import Path
import wandb
wandb.login()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from transformers import AutoTokenizer, TrainingArguments, Trainer, DefaultDataCollator, BitsAndBytesConfig, TrainerCallback, get_linear_schedule_with_warmup
from dataset import WikipediaDatasetHF
from utils import models_map
from lora_models import ModelForMLMWithLoRA
import evaluate, torch
import numpy as np
from typing import List, Tuple, Union, Any
import bitsandbytes as bnb
from torch.utils.data import Subset

class LoRAFineTuner:
    def __init__(self, device: torch.device, config: dict):
        self.config = config
        self.device = device
        self.quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',  
            bnb_4bit_compute_dtype=torch.bfloat16,  
            bnb_4bit_use_double_quant=True,  
        ) if self.config["is_quant"] else None
        
        self.model_name = config["model_name"]
        self.model_name_srt = config["model_name"].split("/")[-1]
        self.lang = config["lang"]
        ds = WikipediaDatasetHF(model_name=self.model_name, lang=self.lang, max_context_len=self.config["max_context_length"])
        self.train_ds = Subset(ds, indices=random.sample(range(len(ds)), k=int(len(ds)*self.config["data_frac"])))
        self.eval_ds = Subset(ds, indices=random.sample(range(len(ds)), k=int(len(self.train_ds)*0.1)))
        self.data_collator = DefaultDataCollator(return_tensors="pt")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        self.batch_size = self.config["batch_size"]
        self.config["num_steps"] = (len(self.train_ds)//self.batch_size)//(self.config["grad_acc_steps"])
        self.num_steps = self.config["num_steps"]
        self.num_epochs = self.config["num_epochs"]
        self.project_name = f"{self.model_name.split('/')[-1]}_finetune_{self.config['task_name']}"
        os.environ["WANDB_PROJECT"] = self.project_name
        self.run_name = f"data_{self.lang}_{self.config['data_frac']:.2f}_{self.config['initial_lr']:.1e}_{self.config['min_lr']:.1e}_r{self.config['lora_rank']}"
        self.output_dir = f"outputs/ckpt/{self.project_name}/{self.run_name}"

        self.model = ModelForMLMWithLoRA(device=self.device, tokenizer=self.tokenizer, model_name=self.model_name, lora_rank=self.config["lora_rank"], lora_alpha=self.config["lora_alpha"], quant_config=self.quant_config, apply_only_mlp=self.config["apply_only_mlp"])
        self.optimizer = bnb.optim.AdamW8bit(params=self.model.parameters(), lr=self.config['initial_lr'], weight_decay=self.config["weight_decay"], betas=self.config["adam_betas"])
        self.scheduler = get_linear_schedule_with_warmup(optimizer=self.optimizer,
                                                        num_warmup_steps=int(0.01 * self.num_steps), 
                                                        num_training_steps=int(self.num_epochs * self.num_steps),
                                                        min_lr_ratio=self.config["min_lr"]/self.config["initial_lr"])

        self.train_arg_config = {
            "output_dir": self.output_dir,
            "eval_strategy": "steps",
            "eval_steps": max(self.batch_size, self.num_steps//self.config["num_ckpt_per_epoch"]),
            "per_device_train_batch_size": self.batch_size,
            "per_device_eval_batch_size": self.batch_size,
            "gradient_accumulation_steps": self.config["grad_acc_steps"],
            "max_grad_norm": self.config["max_grad_norm"],
            "num_train_epochs": self.num_epochs,
            "logging_strategy": "steps",
            "logging_first_step": True,
            "logging_steps": self.batch_size,
            "save_strategy": "steps",
            "save_steps": max(self.batch_size, self.num_steps//self.config["num_ckpt_per_epoch"]),
            "save_safetensors": False, 
            "save_total_limit": self.num_epochs * self.config["num_ckpt_per_epoch"] + 1,
            "save_only_model": False,
            "fp16": self.config["fp16"],
            "bf16": self.config["bf16"],
            "dataloader_drop_last": True,
            "run_name": self.run_name,
            "report_to": "wandb" if self.config["wandb_log"] else "none",
            "eval_on_start": False
        }

        self.training_args = TrainingArguments(**self.train_arg_config)
        self.trainer_config = {
            "model": self.model,
            "args": self.training_args,
            "data_collator": self.data_collator,
            "train_dataset": self.train_ds,
            "eval_dataset": self.eval_ds,
            "tokenizer": self.tokenizer,
            "optimizers": (self.optimizer, self.scheduler),
            "compute_metrics": self.compute_metrics
        }
        self.trainer = Trainer(**self.trainer_config)
    
    @staticmethod
    def compute_metrics(eval_pred: Tuple[np.ndarray, np.ndarray]) -> dict:
        predictions, labels = eval_pred
        min_len = min(len(predictions), len(labels))
        predictions = predictions[:min_len]
        labels = labels[:min_len]
        accuracy = (predictions == labels).mean()
        output = {"accuracy": accuracy} # keys: {"accuracy"}
        return output
    
    @staticmethod
    def compute_loss(self, model: torch.nn.Module, inputs: dict, outputs: torch.tensor = None):
        """Not to be used by LoRAFineTuner.
        Go to: ~/miniconda3/envs/env_name/lib/python3.8/site-packages/transformers/trainer.py
        Modify Trainer.compute_loss() function.
        Add this code snippet just before the return statement
        """
        # Custom: Begin.
        if model.training:
            total_norm = 0.0
            for p in model.parameters():
                if p.requires_grad:
                    param_norm = p.norm(2).item()
                    total_norm += param_norm ** 2
            total_norm = total_norm ** 0.5

            preds = outputs["pred_labels"].detach() # outputs is already calculate before
            acc = (preds == inputs["labels"]).to(torch.float).mean().item()
            if self.state.global_step % self.args.logging_steps == 0:
                self.log({"accuracy": acc, "param_norm": total_norm})
        # Custom: End.
    
    def _save_config(self) -> None: 
        config_data = {
            "config": self.config,
            "output_dir": self.output_dir,
            "project_name": self.project_name,
            "run_name": self.run_name,
            "train_arg_config": self.train_arg_config,
            "quant_config": self.quant_config,
        }
        with open(self.output_dir + "/master_config.pkl", 'wb') as f:
            pickle.dump(config_data, f)
    
    @staticmethod
    def load_model(device: torch.device, config_path: Path, checkpoint_name: str) -> dict:
        with open(config_path, 'rb') as f:
            config_data = pickle.load(f)
        config = config_data["config"]
        model_name = config["model_name"]
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = ModelForMLMWithLoRA(device=device, tokenizer=tokenizer, model_name=model_name, lora_rank=config["lora_rank"], lora_alpha=config["lora_alpha"], quant_config=config_data["quant_config"], apply_only_mlp=config["apply_only_mlp"])
        checkpoint_path = Path(config_path.parent, checkpoint_name)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint, strict=False)
        
        print(f"Model loaded from checkpoint: {checkpoint_path}")
        return {"model": model, "config_data": config_data}

    def train(self) -> None:
        self._save_config()
        with torch.amp.autocast(device_type="cuda"):
            self.trainer.train(resume_from_checkpoint=False)

def main(model_name: str, device: torch.device) -> None:
    config = {
        "model_name": model_name, "task_name": "LoRA", "apply_only_mlp": False,
        "lang": "en", 
        "num_epochs": 1, "num_steps": None, "batch_size": 8, "max_context_length": 512, # steps are auto calculated
        "data_frac": 0.25,
        "initial_lr": 5e-4, "min_lr": 1e-5, "lora_rank": 8, "lora_alpha": 16, "max_grad_norm": 10.0, "weight_decay": 0.01,
        "adam_betas": (0.95, 0.999), "grad_acc_steps": 8, "num_ckpt_per_epoch": 4, "is_quant": True, "fp16": False, "bf16": True,
        "wandb_log": True
    }
    trainer = LoRAFineTuner(device=device, config=config)
    print(trainer.model)
    print(trainer.model.calc_num_lora_params())
    trainer.train()
    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}...")
    
    main(models_map["llama3"], device=device)