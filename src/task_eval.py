import wandb, torch, tqdm, sys, os, json, math, gc, pickle
from pathlib import Path
sys.path.append(Path(__file__).parent.parent.__str__())
from typing import List, Tuple, Union, Any
from dataset import XNLIDatasetHF
from lora_models import ModelForCLSWithLoRA
from utils import models_map
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from lora_cls_finetune_hf import LoRAFineTuner

class Evaluator:
    def __init__(self, device: torch.device, config: dict):
        self.device = device
        self.config = config
        self.config_path = self.config["config_path"]
        out = LoRAFineTuner.load_model(config_path=self.config_path, checkpoint_name=self.config["ckpt_name"], device=self.device)
        self.model = out["model"]
        self.config_data = out["config_data"]
        
        self.model_name = self.config_data["config"]["model_name"]
        self.model_name_srt = self.model_name.split("/")[-1]
        self.task_name = self.config_data["config"]["task_name"]
        self.train_lang = self.config_path.parent.name.split("_")[1]
        self.frozen_lang = self.config_path.parent.name.split("_")[3:5] if self.config_path.parent.name.split("_")[3] else ""
        self.eval_lang = self.config["eval_lang"]
        
        self.eval_path = Path(Path.cwd(), f"outputs/task_eval/{self.model_name_srt}_finetune_{self.task_name}")
        if not self.eval_path.exists():
            Path.mkdir(self.eval_path, parents=True, exist_ok=True)
        
    def _get_intervene_config(self, intervene_lang: str, is_activate: bool) -> dict:
        """intervene_lang = setX_yy"""
        lang_set = intervene_lang.split('_')[0]
        lang = intervene_lang.split('_')[1]
        lang_neuron_path = Path(Path.cwd(), f"outputs/lang_neurons/{self.model_name_srt}/{lang_set}/lang_neuron_data.pkl")
        if lang_neuron_path.exists():
            lang_neuron = pickle.load(open(lang_neuron_path, "rb"))
            print(f"The lang neurons data is loaded from {lang_neuron_path}")
        else:
            raise ValueError(f"{lang_neuron_path} doesn't exist!")

        act_data_path = Path(Path.cwd(), f"outputs/activation/{self.model_name_srt}/act_{lang}.pkl")
        if act_data_path.exists():
            act_data = pickle.load(open(act_data_path, "rb"))
            print(f"The activation data is loaded from {act_data_path}")
        else:
            raise ValueError(f"{act_data_path} doesn't exist!")
        
        mean_act = act_data["mean_act"] # (L, 4d)
        index = lang_neuron["lang_to_neuron"][lang] # (N, 2)
        value = mean_act[index[:, 0], index[:, 1]] # (N,)
        intervene_config = {
            "indices": index,
            "value": value if is_activate else torch.zeros_like(value)
        }
        return intervene_config
   
    def _forward_batch(self, batch: dict, intervene_config: Union[dict, None]) -> torch.tensor:
        input_ids = batch["input_ids"].to(self.device) # (b, T)
        attention_mask = batch["attention_mask"].to(self.device) # (b, T)
        self.model.eval()
        with torch.no_grad(), torch.amp.autocast(device_type="cuda"):
            out = self.model(input_ids=input_ids, attention_mask=attention_mask, intervene_config=intervene_config)["logits"]
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
    
    def evaluate(self) -> None:
        lang_set = self.eval_lang.split("_")[0]
        lang = self.eval_lang.split("_")[1]
        intervene_config = self._get_intervene_config(intervene_lang=self.eval_lang, is_activate=True)

        self.eval_ds = XNLIDatasetHF(model_name=self.model_name, lang=lang, max_context_len=self.config_data["config"]["max_context_length"], frac=self.config["eval_frac"], is_train=False)
        self.eval_dl = DataLoader(self.eval_ds, batch_size=self.config["batch_size"], shuffle=False, drop_last=True)
        
        acc = self._evaluate_dataloader(intervene_config=None)
        if self.train_lang == lang:
            res1 = f"[RESULT] Train lang: {self.train_lang}, Frozen lang: {self.frozen_lang}, Eval lang: {self.eval_lang}, Direct acc: {acc}"
        else:
            res1 = f"[RESULT] Train lang: {self.train_lang}, Frozen lang: {self.frozen_lang}, Eval lang: {self.eval_lang}, Zero shot acc: {acc}"
        
        print(res1)
        int_acc = self._evaluate_dataloader(intervene_config=intervene_config)
        res2 = f"[RESULT] Train lang: {self.train_lang}, Frozen lang: {self.frozen_lang}, Eval lang: {self.eval_lang}, Intervene acc: {int_acc}"
        print(res2)
        
        with open(Path(self.eval_path, f"train_{self.train_lang}_frozen_{self.frozen_lang}_eval_{self.eval_lang}_result.txt"), "w") as f:
            f.writelines("\n".join([res1, res2]))
              
def main(device: torch.device) -> None:
    config = {
        "config_path": Path(Path.cwd(), "outputs/ckpt/Meta-Llama-3.1-8B_finetune_XNLI/data_en_frozen_set1_vi_0.25_5.0e-05_r8/master_config.pkl"),
        "ckpt_name": "checkpoint-12269/pytorch_model.bin",
        "eval_lang": "set1_vi", # "setX_yy"
        "batch_size": 8,
        "eval_frac": 0.1,
    }
    evaluator = Evaluator(device=device, config=config)
    evaluator.evaluate()
    print("DONE")
    
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}...")
    
    main(device=device)
        

