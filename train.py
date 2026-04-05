import os
import torch
import pandas as pd
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer
from transformers import (
    AutoProcessor, 
    BitsAndBytesConfig, 
    Qwen2_5_VLForConditionalGeneration,
    Trainer
)
from parameter_search import (
    prepare_drama_x_dataset, 
    flat_drama_collator
)

# Custom Trainer to weight the loss
class WeightedSFTTrainer(SFTTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        logits = outputs.get("logits")
        labels = inputs.get("labels")
        
        loss_fct = torch.nn.CrossEntropyLoss(label_smoothing=0.1) 
        
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return (loss, outputs) if return_outputs else loss

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
OUTPUT_DIR = "./hazard_vqa_final_model"
CSV_PATH = "../drama_subset_8/train/annotations.csv"

# Adjusted for Generalization
BEST_LR = 2e-5   
BEST_RANK = 16
MAX_STEPS = 200  

csv = pd.read_csv(CSV_PATH)
flat_data_list = prepare_drama_x_dataset(csv)
train_dataset = Dataset.from_list(flat_data_list)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map={"": 0},
    torch_dtype=torch.bfloat16
)

lora_config = LoraConfig(
    r=BEST_RANK,
    lora_alpha=BEST_RANK, 
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
processor = AutoProcessor.from_pretrained(MODEL_ID)

training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    max_steps=MAX_STEPS,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=BEST_LR,
    bf16=True, 
    logging_steps=10,
    save_steps=50,
    save_total_limit=3,
    optim="adamw_bnb_8bit",
    weight_decay=0.01,  # Added weight decay to address possible overfitting
    remove_unused_columns=False,
    dataset_kwargs={"skip_prepare_dataset": True}
)

# Use weighted trainer
trainer = WeightedSFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=flat_drama_collator
)

trainer.train()
trainer.save_model(OUTPUT_DIR)