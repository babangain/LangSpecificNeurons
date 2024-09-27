import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_metric
from torch.utils.data import DataLoader
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm
import numpy as np
from dataset import XNLIDataset
import os
os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from transformers import DataCollatorWithPadding


# Load Llama2 Model for Sequence Classification
model_name = "meta-llama/Llama-2-7b-hf"  # Adjust the model name as required
tokenizer = AutoTokenizer.from_pretrained(model_name)
data_collator = DataCollatorWithPadding(tokenizer)

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)  # XNLI has 3 labels
print(model)
# Freeze all layers except the last layer
def freeze_model_except_last_layer(model):
    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze only the classifier layer (the last layer used for sequence classification)
    for param in model.score.parameters():
        param.requires_grad = True

# Call the function to freeze all layers except the last one
freeze_model_except_last_layer(model)

# Check the number of parameters being trained (should be only the last layer)
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params}")
print(f"Trainable parameters: {trainable_params}")  # Should be significantly less now

# Define the dataset
dataset = XNLIDataset(
    model_name=model_name,
    lang="en",  # Specify the language for XNLI
    max_context_len=512,  # Set max context length
    frac=0.1,  # Use the full dataset (adjust frac as needed)
    is_train=True  # Set to False for evaluation
)

# For evaluation, create a separate dataset instance with `is_train=False`
eval_dataset = XNLIDataset(
    model_name=model_name,
    lang="en",
    max_context_len=512,
    frac=0.1,  # Evaluate on 20% of the test set
    is_train=False
)

# Define the metric
accuracy_metric = load_metric("accuracy")

# Function to compute metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
    return accuracy

# TrainingArguments for Hugging Face Trainer
training_args = TrainingArguments(
    output_dir="./results",  # Output directory for saving model checkpoints
    evaluation_strategy="steps",  # Evaluate during training at every `eval_steps`
    eval_steps=100,  # Evaluate every 100 steps
    logging_dir="./logs",  # Logging directory
    logging_steps=100,  # Log every 100 steps
    save_steps=500,  # Save model every 500 steps
    per_device_train_batch_size=8,  # Adjust based on your available GPU memory
    per_device_eval_batch_size=8,  # Adjust as needed
    num_train_epochs=3,  # Number of training epochs
    warmup_steps=500,  # Warmup steps for learning rate
    learning_rate=5e-5,  # Learning rate
    load_best_model_at_end=True,  # Load best model at the end of training
    metric_for_best_model="accuracy",  # Metric for selecting the best model
    weight_decay=0.01,  # Weight decay for optimizer
    save_total_limit=2,  # Limit the number of saved checkpoints
    fp16=True  # Use mixed precision training for faster performance (if available)
)

# Trainer class for Hugging Face model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    data_collator=data_collator,  # Use the built-in data collator
)

# Fine-tuning the model
trainer.train()

# Save the final model
trainer.save_model("./final_model")

# Evaluate the model on the evaluation set
results = trainer.evaluate()
print(results)
