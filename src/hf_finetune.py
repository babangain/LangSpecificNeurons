import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from transformers import TrainingArguments, Trainer, DefaultDataCollator, AdamW, get_cosine_with_hard_restarts_schedule_with_warmup, BitsAndBytesConfig
from dataset import XNLIDatasetHF
from utils import models_map
from lora_models import ModelForCLSWithLoRA
import evaluate, torch
import numpy as np

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    min_len = min(len(predictions), len(labels))
    predictions = np.argmax(predictions[:min_len], axis=1)
    labels = labels[:min_len]
    accuracy = evaluate.load("accuracy")
    return accuracy.compute(predictions=predictions, references=labels)

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',  
    bnb_4bit_compute_dtype=torch.float16,  
    bnb_4bit_use_double_quant=True,  
)

model_name = models_map["llama2"]
train_ds = XNLIDatasetHF(model_name=model_name, lang="en", max_context_len=256, frac=0.1, is_train=True)
eval_ds = XNLIDatasetHF(model_name=model_name, lang="en", max_context_len=256, frac=0.1, is_train=False)
data_collator = DefaultDataCollator(return_tensors="pt")
tokenizer = train_ds.tokenizer

batch_size = 8
num_steps = len(train_ds)//batch_size
num_epochs = 2
initial_lr = 5e-5
name = "llama2-XNLI-en-finetune"

model = ModelForCLSWithLoRA(device="cuda", tokenizer=tokenizer, model_name=model_name, num_class=3, lora_rank=8, lora_alpha=16, quant_config=quant_config).to(torch.float16)
optimizer = AdamW(params=model.parameters(), lr=initial_lr, weight_decay=0.01, betas=(0.9, 0.999))
scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer=optimizer, 
                                                               num_warmup_steps=(0.1 * num_steps), 
                                                               num_training_steps=int(num_epochs * num_steps), 
                                                               num_cycles=num_epochs)

train_arg_config = {
    "output_dir": "results",
    "eval_strategy": "epoch",
    "per_device_train_batch_size": batch_size,
    "per_device_eval_batch_size": batch_size,
    "gradient_accumulation_steps": 1,
    "torch_empty_cache_steps": None,
    "max_grad_norm": 1.0,
    "num_train_epochs": num_epochs,
    "logging_strategy": "steps",
    "logging_first_step": True,
    "logging_steps": batch_size,
    "save_strategy": "epoch",
    "save_total_limit": 1,
    "save_only_model": True,
    "fp16": False,
    "dataloader_drop_last": True,
    "run_name": name,
    "report_to": "wandb",
    "eval_on_start": True
}

training_args = TrainingArguments(**train_arg_config)
trainer_config = {
    "model": model,
    "args": training_args,
    "data_collator": data_collator,
    "train_dataset": train_ds,
    "eval_dataset": eval_ds,
    "tokenizer": tokenizer,
    "optimizers": (optimizer, scheduler),
    "compute_metrics": compute_metrics,   
}

trainer = Trainer(**trainer_config)
trainer.train()