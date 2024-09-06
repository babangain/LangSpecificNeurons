import torch, os, sys
from pathlib import Path
sys.path.append(Path(__file__).parent)
from transformers import AutoTokenizer, AutoModel
from typing import List, Tuple, Union

class LoRALayer(torch.nn.Module):
    def __init__(self, rank: int, alpha: float, d_in: int, d_out: int):  
        super(LoRALayer, self).__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.alpha = alpha
        self.rank = rank
        
        self.A = torch.nn.Parameter(
            data=torch.normal(mean=0, std=0.01, size=(self.d_in, self.rank)), 
            requires_grad=True
        )
        self.B = torch.nn.Parameter(
            data=torch.zeros(size=(self.rank, self.d_out)),
            requires_grad=True
        )
    
    def forward(self, x: torch.tensor) -> torch.tensor:
        assert x.dim() >= 2, "Input tensor must have at least 2 dimensions."
        assert x.shape[-1] == self.d_in, f"Expected the last dimension of input to be {self.d_in}, but got {x.shape[-1]}."
        delta_W = torch.matmul(self.A, self.B) * (self.alpha / self.rank)
        z = torch.matmul(x, delta_W)
        return z
    
class LinearWithLoRA(torch.nn.Module):
    def __init__(self, linear: torch.nn.Linear, rank: int, alpha: float):
        super(LinearWithLoRA, self).__init__()
        self.rank = rank
        self.alpha = alpha
        self.linear = linear
        self.d_in = self.linear.in_features
        self.d_out = self.linear.out_features
        self.lora = LoRALayer(rank=self.rank, alpha=self.alpha, d_in=self.d_in, d_out=self.d_out)

    def forward(self, x: torch.tensor) -> torch.tensor:
        assert x.dim() >= 2, "Input tensor must have at least 2 dimensions."
        assert x.shape[-1] == self.d_in, f"Expected the last dimension of input to be {self.d_in}, but got {x.shape[-1]}."
        z1 = self.linear(x) 
        z2 = self.lora(x) 
        z = z1 + z2
        return z
    
class ModelForCLSWithLoRA(torch.nn.Module):
    def __init__(self, device: torch.device, model_name: str, num_class: int, lora_rank: int, lora_alpha: float):
        super(ModelForCLSWithLoRA, self).__init__()
        self.model = ModelForCLS(device=device, model_name=model_name, num_class=num_class)
        for param in self.model.base.parameters():
            param.requires_grad = False
        self.rank = lora_rank
        self.alpha = lora_alpha
        ModelForCLSWithLoRA.replace_linear_with_lora(model=self.model.base, rank=self.rank, alpha=self.alpha)
    
    @staticmethod
    def replace_linear_with_lora(model: torch.nn.Module, rank: int, alpha: float):
        for name, module in model.named_children():
            if isinstance(module, torch.nn.Linear):
                linear_lora = LinearWithLoRA(module, rank, alpha)
                setattr(model, name, linear_lora) # parent is model, child is module
            else:
                ModelForCLSWithLoRA.replace_linear_with_lora(module, rank, alpha)
       
    def calc_num_lora_params(self) -> None:
        # Check if the requires_grad are set correctly
        train_params = 0
        total_params = 0
        for name, param in self.model.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                train_params += param.numel()
        print(f"Number of total parameters: {total_params}")
        print(f"Number of trainable parameters: {train_params}")
        print(f"LoRA efficiency: {train_params * 100 / total_params:.3f}%")
        
    def forward(self, input_ids: torch.tensor, attention_mask: torch.tensor) -> torch.tensor:
        assert list(input_ids.shape).__len__() == 2, "inputs rank must be 2 and inputs.shape = (b, T)"
        z = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return z
    
class ModelForCLS(torch.nn.Module):
    def __init__(self, device: torch.device, model_name: str, num_class: int):
        super(ModelForCLS, self).__init__()
        self.device = device
        self.model_name = model_name
        self.model_name_srt = model_name.split("/")[-1]
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.base = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.d = self.base.config.hidden_size
        self.c = num_class
        self.head = torch.nn.Linear(in_features=self.d, out_features=self.c).to(self.device)
    
    def forward(self, input_ids: torch.tensor, attention_mask: torch.tensor, labels: Union[None, torch.tensor] = None) -> torch.tensor:
        z = self.base(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state # (b, T, d)
        y = self.head(z.to(self.device)) # (b, T, c)
        out = y[:, -1, :] # (b, c) (last position embedding)
        return out
        
def main(model_name: str, device: torch.device) -> None:
    model = ModelForCLSWithLoRA(device=device, model_name=model_name, num_class=3, lora_rank=8, lora_alpha=16).to(device)
    input_ids = torch.randint(low=0, high=100, size=(16, 256)).to(device) # (b, T)
    attention_mask = torch.ones(size=(16, 256)).to(device) # (b, T)
    print(model)
    model.calc_num_lora_params()
    model.eval()
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)
    print(out.shape) # (b, c)
    
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "4"
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}...")
    
    main("meta-llama/Llama-2-7b-hf", device=device)
    
    