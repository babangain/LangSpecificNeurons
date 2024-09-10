import wandb, torch, tqdm, sys, os, json, math, gc, pickle
from pathlib import Path
sys.path.append(Path(__file__).parent.parent.__str__())
from typing import List, Tuple, Union, Any
from dataset import XNLIDataset
from lora_models import ModelForCLSWithLoRA
from utils import models_map
from datasets import load_dataset
from torch.utils.data import Dataset

class NLIEvaluator:
    def __init__(self, device: torch.device, config: dict):
        self.device = device
        self.config = config
        self.model_name_srt = config["model_name"].split("/")[-1]
        self.train_lang = config["train_lang"]
        self.eval_lang = config["eval_lang"]
        self.eval_res_path = Path(Path.cwd(), f"outputs/task_eval/{self.config['task_name']}/{self.model_name_srt}/eval_result.json")
        
    def _get_intervene_config(self) -> dict:
        lang_neuron_path = Path(Path.cwd(), f"outputs/lang_neurons/{self.model_name_srt}/set1/lang_neuron_data.pkl")
        if lang_neuron_path.exists():
            lang_neuron = pickle.load(open(lang_neuron_path, "rb"))
            print(f"The lang neurons data is loaded from {lang_neuron_path}")
        else:
            raise ValueError(f"{lang_neuron_path} doesn't exist!")

        act_data_path = Path(Path.cwd(), f"outputs/activation/{self.model_name_srt}/act_{self.eval_lang}.pkl")
        if act_data_path.exists():
            act_data = pickle.load(open(act_data_path, "rb"))
            print(f"The activation data is loaded from {act_data_path}")
        else:
            raise ValueError(f"{act_data_path} doesn't exist!")
        
        mean_act = act_data["mean_act"] # (L, 4d)
        index = lang_neuron["lang_to_neuron"][self.eval_lang] # (N, 2)
        value = mean_act[index[:, 0], index[:, 1]] # (N,)
        intervene_config = {
            "indices": index,
            "value": value
        }
        return intervene_config
     
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
   
    def _load_checkpoint(self, checkpoint_path: Path) -> None:
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        print(f"[LOAD] ep: {checkpoint['epoch']}, checkpoint loaded from: {checkpoint_path}")
    
    def _forward_batch(self, batch: dict, intervene_config: Union[dict, None]) -> torch.tensor:
        input_ids = batch["input_ids"].to(self.device) # (b, T)
        attention_mask = batch["attention_mask"].to(self.device) # (b, T)
        self.model.eval()
        with torch.no_grad():
            out = self.model(input_ids=input_ids, attention_mask=attention_mask, intervene_config=intervene_config)
        return out # (b, c)
    
    def _calc_acc_batch(self, pred_outputs: torch.tensor, true_outputs: torch.tensor) -> torch.tensor:
        pred_outputs = pred_outputs.to(self.device)
        true_outputs = true_outputs.to(self.device)
        assert pred_outputs.dim() == 2, f"pred_outputs.shape = {pred_outputs.shape} must be (b, c)"
        assert true_outputs.dim() == 1, f"true_outputs.shape = {true_outputs.shape} must be (b,)"
        acc = (pred_outputs.argmax(dim=-1) == true_outputs).to(torch.float32).mean()
        return torch.tensor(acc.item()) # returns the tensor as a scalar number 
    
    def _evaluate_dataloader(self, intervene_config: Union[dict, None]) -> float:
        with tqdm.tqdm(iterable=self.eval_dl, desc=f"[EVAL] on lang {self.eval_lang}", total=len(self.eval_dl), unit="batch", colour="green") as pbar:
            acc_list = []
            for i, batch in enumerate(pbar):   
                pred_out = self._forward_batch(batch=batch, intervene_config=intervene_config) # (b, c)  
                true_out = batch["labels"] # (b,)   
                acc = self._calc_acc_batch(pred_outputs=pred_out, true_outputs=true_out)
                acc_list.append(acc)
                pbar.set_postfix({"acc": f"{acc:.3f}"})  
        eval_acc = sum(acc_list)/len(acc_list)
        return float(eval_acc)
    
    def evaluate(self) -> dict:
        keys1 = {f"train_lang_{i}_zero_shot" for i in self.train_lang}
        keys2 = {f"train_lang_{i}_intervene" for i in self.train_lang}
        all_keys = keys1.union(keys2)
        
        if self.eval_res_path.exists():
            eval_res = json.load(open(self.eval_res_path, "r")) 
            avail_keys = set(eval_res.keys())
            eval_keys = all_keys.difference(avail_keys)
        else:  
            self.eval_res_path.parent.mkdir(parents=True, exist_ok=True)    
            avail_keys = {}
            eval_keys = all_keys
            eval_res = {}
        
        if len(eval_keys) == 0:
            print(f"[LOAD] The keys ({avail_keys}) are already available and taken from {self.eval_res_path}")
            return eval_res
        else:
            print(f"[LOAD] The keys ({avail_keys}) are already available and taken from {self.eval_res_path}")
            print(f"[RESULT] The keys ({eval_keys}) would be calculated")
            self.eval_ds = XNLIDataset(model_name=self.config["model_name"], lang=self.eval_lang, max_context_len=self.config["max_seq_len"], frac=self.config["data_frac"], is_train=False)
            self.eval_dl = self.eval_ds.prepare_dataloader(batch_size=10)
            result_dict = {}
            intervene_config = self._get_intervene_config()
            
            for train_lang, ckpt_name in zip(self.train_lang, self.config["ckpt_name"]):
                checkpoint_path = Path(Path.cwd(), f"outputs/ckpt/{self.model_name_srt}-finetune-{self.config['task_name']}-{train_lang}/{ckpt_name}")
                if not checkpoint_path.exists():
                    print(f"[WARNING] The checkpoint path ({checkpoint_path}) does not exist! Continuing to next iteration...")
                    continue
                self.model = ModelForCLSWithLoRA(device=self.device, model_name=self.config["model_name"], num_class=self.config["num_class"], lora_rank=self.config["lora_rank"], lora_alpha=self.config["lora_alpha"]).to(self.device)
                self._load_checkpoint(checkpoint_path=checkpoint_path)
            
                if f"train_lang_{self.train_lang}_zero_shot" not in eval_keys:
                    zero_shot_acc = self._evaluate_dataloader(intervene_config=None)
                    result_dict[f"train_lang_{train_lang}_zero_shot"] = {"eval_lang": self.eval_lang, "acc": zero_shot_acc}
                    print(f"[RESULT] Train lang: {train_lang}, Eval lang: {self.eval_lang}, Zero shot acc: {zero_shot_acc}")

                if f"train_lang_{self.train_lang}_intervene" not in eval_keys:
                    int_acc = self._evaluate_dataloader(intervene_config=intervene_config)
                    result_dict[f"train_lang_{train_lang}_intervene"] = {"eval_lang": self.eval_lang, "acc": int_acc}
                    print(f"[RESULT] Train lang: {train_lang}, Eval lang: {self.eval_lang}, Int acc: {int_acc}")
            
            eval_res.update(result_dict)    
            json.dump(eval_res, open(self.eval_res_path, "w"), indent=4)
            print(f"[SAVE] The result is saved at {self.eval_res_path}")
            return eval_res
                         
def main(model_name: str, device: torch.device) -> None:
    config = {
        "task_name": "XNLI",
        "model_name": model_name,
        "train_lang": ["en", "vi", "en-vi"],
        "ckpt_name": ["ckpt_ep_0.pth", "ckpt_ep_0.pth", "ckpt_ep_0.pth"],
        "eval_lang": "vi",
        "num_class": 3,
        "lora_rank": 8,
        "lora_alpha": 16,
        "data_frac": 1.0,
        "max_seq_len": 256,
    }
    evaluator = NLIEvaluator(device=device, config=config)
    evaluator.evaluate()
    print("DONE")
    
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "5"
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}...")
    
    main(models_map["llama2"], device=device)
        

