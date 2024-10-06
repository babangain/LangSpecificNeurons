import os, json, pickle
from pathlib import Path
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from transformers import AutoTokenizer, TrainingArguments, Trainer, DefaultDataCollator, get_cosine_with_hard_restarts_schedule_with_warmup, BitsAndBytesConfig, TrainerCallback
from dataset import XNLIDatasetHF
from utils import models_map
from lora_models import ModelForCLSWithLoRA
import evaluate, torch
import numpy as np
from typing import List, Tuple, Union, Any

class LoRAFineTuner:
    def __init__(self, device: torch.device, config: dict):
        self.config = config
        self.device = device
        self.quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',  
            bnb_4bit_compute_dtype=torch.bfloat16,  
            bnb_4bit_use_double_quant=True,  
        ) if self.config["is_4bit_quant"] else None
        
        self.model_name = config["model_name"]
        self.lang = config["lang"]
        self.train_ds = XNLIDatasetHF(model_name=self.model_name, lang=self.lang, max_context_len=self.config["max_context_length"], frac=self.config["train_frac"], is_train=True)
        self.eval_ds = XNLIDatasetHF(model_name=self.model_name, lang=self.lang, max_context_len=self.config["max_context_length"], frac=self.config["eval_frac"], is_train=False)
        self.data_collator = DefaultDataCollator(return_tensors="pt")
        self.tokenizer = self.train_ds.tokenizer

        self.batch_size = self.config["batch_size"]
        self.config["num_steps"] = len(self.train_ds)//self.batch_size
        self.num_steps = self.config["num_steps"]
        self.num_epochs = self.config["num_epochs"]
        self.run_name = f"{self.model_name.split('/')[-1]}_{self.config['task_name']}_{self.lang}_{self.config['train_frac']:.2f}_{self.config['initial_lr']:.1e}_r{self.config['lora_rank']}"
        self.output_dir = f"outputs/ckpt/{self.run_name}"

        self.model = ModelForCLSWithLoRA(device=self.device, tokenizer=self.tokenizer, model_name=self.model_name, num_class=self.config["num_class"], lora_rank=self.config["lora_rank"], lora_alpha=self.config["lora_alpha"], quant_config=self.quant_config)
        self.optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=self.config['initial_lr'], weight_decay=self.config["weight_decay"], betas=self.config["adam_betas"])
        self.scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer=self.optimizer, 
                                                                            num_warmup_steps=int(0.05 * self.num_steps), 
                                                                            num_training_steps=int(self.num_epochs * self.num_steps), 
                                                                            num_cycles=self.num_epochs)

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
            "eval_on_start": True
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
        predictions = np.argmax(predictions[:min_len], axis=1)
        labels = labels[:min_len]
        accuracy = evaluate.load("accuracy")
        output = accuracy.compute(predictions=predictions, references=labels) # keys: {"accuracy"}
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

            preds = outputs["logits"].detach() # outputs is already calculate before
            acc = (preds.argmax(axis=1) == inputs["labels"]).to(torch.float).mean().item()
            if self.state.global_step % self.args.logging_steps == 0:
                self.log({"accuracy": acc, "param_norm": total_norm})
        # Custom: End.
    
    def _save_config(self) -> None: 
        config_data = {
            "config": self.config,
            "output_dir": self.output_dir,
            "run_name": self.run_name,
            "train_arg_config": self.train_arg_config,
            "quant_config": self.quant_config,
        }
        with open(self.output_dir + "/master_config.pkl", 'wb') as f:
            pickle.dump(config_data, f)
    
    def train(self) -> None:
        self.trainer.train(resume_from_checkpoint=False)
        self._save_config()

    @staticmethod
    def load_model_and_trainer(config_path: str, checkpoint_name: str, device: torch.device) -> dict:
        with open(config_path, 'rb') as f:
            config_data = pickle.load(f)
        config = config_data["config"]
        model_name = config["model_name"]
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = ModelForCLSWithLoRA(device=device, tokenizer=tokenizer, model_name=model_name, num_class=config["num_class"], lora_rank=config["lora_rank"], lora_alpha=config["lora_alpha"], quant_config=config_data["quant_config"])
        checkpoint_path = Path(config_data["output_dir"], checkpoint_name)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint)
        
        training_args = TrainingArguments(**config_data["train_arg_config"])
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=DefaultDataCollator(return_tensors="pt"),
            tokenizer=tokenizer,
            compute_metrics=LoRAFineTuner.compute_metrics
        )
        print(f"Model loaded from checkpoint: {checkpoint_path}")
        return {"model": model, "trainer": trainer, "config_data": config_data}

def main(model_name: str, device: torch.device) -> None:
    config = {
        "model_name": model_name,
        "task_name": "XNLI",
        "lang": "en",
        "num_epochs": 2,
        "num_steps": None, # Auto calculated 
        "batch_size": 8,
        "max_context_length": 256,
        "train_frac": 0.5,
        "eval_frac": 1.0,
        "num_class": 3,
        "lora_rank": 8,
        "lora_alpha": 16,
        "max_grad_norm": 10.0,
        "initial_lr": 5e-5,
        "weight_decay": 0.1,
        "adam_betas": (0.95, 0.999),
        "grad_acc_steps": 1,
        "num_ckpt_per_epoch": 2,
        "is_4bit_quant": True,
        "fp16": False,
        "bf16": True,
        "wandb_log": False
    }
    trainer = LoRAFineTuner(device=device, config=config)
    trainer.train()
    # out = LoRAFineTuner.load_model_and_trainer(config_path="/raid/speech/soumen/MS_Research/LangSpecificNeurons/outputs/ckpt/Llama-2-7b-hf_XNLI_en_0.00_5.0e-05_r8/master_config.pkl",
    #                                                       checkpoint_name="checkpoint-98/pytorch_model.bin",
    #                                                       device=device)
    # model = out["model"]
    # trainer = out["trainer"]
    # config_data = out["config_data"]
    
    # eval_ds = XNLIDatasetHF(model_name=config_data["config"]["model_name"], lang=config_data["config"]["lang"], max_context_len=config_data["config"]["max_context_length"], frac=config_data["config"]["eval_frac"], is_train=False)
    # val = trainer.predict(eval_ds)
    # print(val)
    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}...")
    
    main(models_map["llama3"], device=device)