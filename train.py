import os
import torch

from datasets import Dataset
import pandas as pd

from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer
from transformers import (
    AutoProcessor, 
    BitsAndBytesConfig, 
    Qwen2_5_VLForConditionalGeneration
)
import parameter_search
from parameter_search import (
    prepare_drama_x_dataset, 
    flat_drama_collator
)

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
OUTPUT_DIR = "./hazard_vqa_final_model"
DATA_PATH = "./../drama_subset_5/drama_subset_5"
CSV_PATH = "./../drama_subset_5/drama_subset_5/subset_annotations.csv"

# Optimized Hyperparameters (From Phase 2 Results)
BEST_LR = 5e-5
BEST_RANK = 16
MAX_STEPS = 150  # Increased for final convergence

os.environ["MAX_WORKERS"] = "4"  
csv = pd.read_csv(CSV_PATH)
flat_data_list = prepare_drama_x_dataset(csv)
train_dataset = Dataset.from_list(flat_data_list)

# Load model with 4-bit quantization
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

# Apply LoRA with chosen Rank
lora_config = LoraConfig(
    r=BEST_RANK,
    lora_alpha=BEST_RANK * 2,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
parameter_search.processor = processor

# Training Arguments
# Added checkpointing to save progress in case of system suspend
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    max_steps=MAX_STEPS,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=BEST_LR,
    bf16=True, 
    logging_steps=5,
    save_steps=50,  # Save a checkpoint every 50 steps
    save_total_limit=2,
    optim="adamw_bnb_8bit",
    remove_unused_columns=False,
    dataset_kwargs={"skip_prepare_dataset": True}
)

# Start Training
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=flat_drama_collator
)

print(f"Starting Final Training: LR={BEST_LR}, Rank={BEST_RANK}")
last_checkpoint = None

if os.path.exists(OUTPUT_DIR):
    checkpoints = [os.path.join(OUTPUT_DIR, d) for d in os.listdir(OUTPUT_DIR) if "checkpoint-" in d]
    if checkpoints:
        last_checkpoint = max(checkpoints, key=os.path.getmtime)

# If checkpoint exists, resume from checkpoint
trainer.train(resume_from_checkpoint=last_checkpoint)
trainer.train()

# Save Final Adapters
trainer.save_model(OUTPUT_DIR)
print(f"Model saved to {OUTPUT_DIR}")