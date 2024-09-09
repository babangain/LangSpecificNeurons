import torch, json, os, sys
from pathlib import Path
sys.path.append(Path(__file__).parent)
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoModel
from typing import List, Tuple, Union
import pandas as pd
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
device = "cuda"

# model = AutoModelForSeq2SeqLM.from_pretrained("CohereForAI/aya-101", torch_dtype=torch.bfloat16).to(device=device)
# input_ids = torch.randint(low=0, high=100, size=(4, 512))
# attention_mask = torch.ones(size=(4, 512), dtype="torch.int64")
# input_ids = input_ids.to(device)
# attention_mask = attention_mask.to(device)

# model.eval()
# with torch.no_grad():
#     decoder_input_ids = input_ids[:, :-1]  # Shifting input IDs by one 
#     prediction_output = model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids).logits.cpu() # (b, T-1, d)
#     print("DONE")

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("CohereForAI/aya-101")
model = AutoModelForSeq2SeqLM.from_pretrained("CohereForAI/aya-101")

# Function to perform inference
def make_inference(input_text):
    # Encode the input text
    inputs = tokenizer(input_text, return_tensors="pt").input_ids

    # Prepare attention mask (optional, but good for performance)
    attention_mask = torch.ones(inputs.shape, dtype=torch.long)  # assuming all inputs are non-padded

    # Initialize decoder input as start token
    decoder_input = torch.tensor([[tokenizer.eos_token_id]], dtype=torch.long)

    # Container for generated sequence
    generated_sequence = []

    # Maximum sequence length to prevent infinite loops
    max_length = 50

    # Generate sequence without using .generate()
    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(input_ids=inputs, decoder_input_ids=decoder_input, attention_mask=attention_mask)
            logits = outputs.logits
            next_token_id = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(0)
            generated_sequence.append(next_token_id.item())

            # Update the decoder_input to include the predicted next token
            decoder_input = torch.cat([decoder_input, next_token_id], dim=-1)

            # Check if eos_token was generated
            if next_token_id == tokenizer.eos_token_id:
                break

    # Decode the generated sequence
    decoded_sequence = tokenizer.decode(generated_sequence)

    return decoded_sequence

# Example usage
input_text = "Translate this text to French:"
output_text = make_inference(input_text)
print(f"Translated text: {output_text}")