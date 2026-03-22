import gc
import json
import os
import time

import cv2
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from PIL import Image, ImageChops

from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer
from transformers import (
    AutoProcessor, 
    BitsAndBytesConfig, 
    Qwen2_5_VLForConditionalGeneration
)

if os.getcwd() != '420 Project/Hazard-Aware-VQA':
    if os.path.exists("420 Project/Hazard-Aware-VQA"):
        os.chdir("420 Project/Hazard-Aware-VQA")
csv = pd.read_csv("../drama_subset_8/train/annotations.csv")
data_path = "../drama_subset_8/train"
work_dir = "./temp_frames" 
os.makedirs(work_dir, exist_ok=True)

results_file = "hpo_results.json"
hpo_log = []

def log_result(lr, rank, loss, status="success"):
    entry = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "learning_rate": lr,
        "lora_rank": rank,
        "loss": loss if status == "success" else None,
        "status": status
    }
    hpo_log.append(entry)
    with open(results_file, "w") as f:
        json.dump(hpo_log, f, indent=4)
    print(f"Result logged to {results_file}")

def get_clipped_gif_frames(gif_path, key_image_path, radius):
    """
    Extracts frames from a GIF, aligns them with a primary PNG, 
    and returns a symmetric list clipped to the GIF's boundaries.
    """
    # 1. Load the GIF and the Key PNG
    if not os.path.exists(gif_path) or not os.path.exists(key_image_path):
        print("Error: Path not found.")
        return []

    gif = Image.open(gif_path)
    # Convert PNG to RGB to match GIF frame mode for comparison
    key_frame = Image.open(key_image_path).convert("RGB")
    total_frames = gif.n_frames
    
    # 2. Locate the "Match" within the GIF
    matched_idx = -1
    # Optimization: Resize for faster pixel comparison
    search_size = (128, 128) 
    key_frame_small = key_frame.resize(search_size)
    
    for i in range(total_frames):
        gif.seek(i)
        current_gif_frame = gif.convert("RGB").resize(search_size)
        
        # Calculate pixel-wise difference
        diff = ImageChops.difference(current_gif_frame, key_frame_small)
        if not diff.getbbox() or np.mean(np.array(diff)) < 8: # Threshold for match
            matched_idx = i
            break
            
    if matched_idx == -1:
        print("Match not found; defaulting to middle frame.")
        matched_idx = total_frames // 2

    # 3. Sample Symmetrically with Hard Clipping
    theoretical_start = matched_idx - radius
    theoretical_end = matched_idx + radius
    
    final_frames = []
    
    for i in range(theoretical_start, theoretical_end + 1):
        if 0 <= i < total_frames:
            if i == matched_idx:
                # Use the key frame for the center
                final_frames.append(f"file://{os.path.abspath(key_image_path)}")
            else:
                gif.seek(i)
                # Save locally as JPG so the Qwen Processor can load it
                frame_path = os.path.join(work_dir, f"gif_frame_{i}.jpg")
                gif.convert("RGB").save(frame_path)
                final_frames.append(f"file://{os.path.abspath(frame_path)}")
        else:
            # Index is out of bounds; just skip (no padding)
            continue
            
    return final_frames


def prepare_drama_x_dataset(df):
    flat_data = [] 
    TARGET_SIZE = (640, 480) 
    
    # 5-Class Intent Taxonomy & Semantic Prompt Construction
    # Proposal Section III-A and III-B
    # PROMPT_TEXT: Standardized instruction prompt from the DRAMA-X benchmark [Godbole et al., 2025].
    # Adopting this prompt ensures alignment with the 5-class intent taxonomy 
    # and allows for direct performance comparison with the baseline paper. 
    PROMPT_TEXT = (
        """
        For the provided image and question, generate a object-intent JSON which includes the following: 
        1. AT MOST 5 objects from the scene including Pedestrians and Cylists.
        2. Predicted intent for every object. Intent should be one of these values:
        2.1 Lateral (Sideways) Intent Options (has to be from these two options): - “goes to the left” - “goes
        to the right”
        2.2 Vertical Intent Options: -“moves away from ego vehicle” - “moves towards ego vehicle” -
        “stationary”
        3. Risk score for this prediction (Yes or No). Risk is defined as a hazardous scenario that poses
        danger to the ego vehicle.
        4. Bounding box of each object. these should be with respect to orginal image dimensions.
        5. Suggested action given the scene and risk score.
        An example structure would look like this:
        {
        "Risk": "Yes/No",
        "Suggested_action": "suggested action for ego vehicle",
        "pedestrian": {
        "Intent": ["predicted lateral intent", "predicted vertical intent"],
        "Reason": "reason for this prediction",
        "Bounding_box": [x1, y1, x2, y2]
        },
        "car": {
        "Intent": ["predicted lateral intent", "predicted vertical intent"],
        "Reason": "reason for this prediction",
        "Bounding_box": [x1, y1, x2, y2]
        }
        ... (for all objects and NOT a list)
        }
        The Intent field list should ALWAYS have two values: one for lateral and one for vertical. Strictly
        output in valid JSON format.
        """
    )

    print(f"Starting dataset preparation for {len(df)} rows...")

    for i, row in df.iterrows():
        vid_path = os.path.join(data_path, row['video_path'])
        img_path = os.path.join(data_path, row['image_path'])
        
        # Found appropriate frame limits (Proposal Section V-A)
        # After testing in Colab, gifs on average have about 8-10 frames. Settle on 5 frame limit.
        raw_paths = get_clipped_gif_frames(vid_path, img_path, radius=2) 
        if not raw_paths: continue

        processed_pil_images = []
        for f_path in raw_paths:
            clean_path = f_path.replace("file://", "")
            try:
                img = Image.open(clean_path).convert("RGB")
                img = img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
                processed_pil_images.append(img)
            except Exception as e:
                print(f"Row {i} | Skipping frame {clean_path}: {e}")

        if not processed_pil_images: continue
        
        # Verification of spatial consistency
        unique_shapes = {img.size for img in processed_pil_images}
        if len(unique_shapes) > 1: continue

        # Hierarchical Intent Taxonomy mapping (Proposal Section III-A)
        target_json = {
            "Risk": row.get('risk_label', 'No'),
            "Suggested_action": row.get('action_suggestion', 'continue'),
            f"{row.get('object_type', 'object')}_0": {
                "Intent": [row.get('lateral_intent'), row.get('vertical_intent')],
                "Reason": row.get('reasoning', 'trajectory analysis'),
                "Bounding_box": [row.get('x1'), row.get('y1'), row.get('x2'), row.get('y2')]
            }
        }

    
        flat_data.append({
            "frames": processed_pil_images,
            "instruction": PROMPT_TEXT,
            "target": json.dumps(target_json)
        })
    
    print(f"\nSuccessfully processed {len(flat_data)} samples into flat format.")
    return flat_data

def flat_drama_collator(batch):
    all_texts = []
    all_videos = []
    
    for item in batch:
        # Build the full text template including vision placeholders
        # Qwen2.5-VL needs these tags to know where to insert the visual features
        full_text = (
            f"<|im_start|>user\n<|vision_start|><|video_pad|><|vision_end|>{item['instruction']}<|im_end|>\n"
            f"<|im_start|>assistant\n{item['target']}<|im_end|>"
        )
        all_texts.append(full_text)
        
        # Extract and clean frames
        raw_frames = item['frames']
        pil_frames = [f['image'] if isinstance(f, dict) and 'image' in f else f for f in raw_frames]
        all_videos.append(pil_frames)


    inputs = processor(
        text=all_texts,
        videos=all_videos,
        padding=True,
        return_tensors="pt",
        max_pixels=313600 
    )

    # Apply Label Masking (Standard -100 for the user prompt part)
    labels = inputs["input_ids"].clone()
    assistant_marker = "<|im_start|>assistant\n"
    
    for i, text in enumerate(all_texts):
        parts = text.split(assistant_marker)
        if len(parts) > 1:
            user_tokens_len = len(processor.tokenizer(parts[0] + assistant_marker, add_special_tokens=False).input_ids)
            labels[i, :user_tokens_len] = -100
                
    inputs["labels"] = labels
    
    return {k: v for k, v in inputs.items()} 


def run_experiment(lr_val, rank_val, dataset, is_final=False):
    """Function to initialize and train for a specific HPO config."""
    
    # Reset VRAM and Garbage Collector
    gc.collect()
    torch.cuda.empty_cache()

    # Re-initialize Quantization for fresh start
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 # Optimized for Ada GPU
    )

    # Load Model (must be re-loaded to clear previous adapters)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct",
        quantization_config=bnb_config,
        device_map={"": 0}, # Force to primary GPU
        torch_dtype=torch.bfloat16
    )

    # Apply Lora with SEARCH RANK (Proposed: 4, 8, 16)
    # Strictly focus on Attention blocks for proposal alignment
    lora_config = LoraConfig(
        r=rank_val, 
        lora_alpha=rank_val * 2,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

    # Config matching Table I of Proposal
    training_args = SFTConfig(
        output_dir=f"./results/lr_{lr_val}_r_{rank_val}",
        max_steps=25 if not is_final else 100, # Short runs for HPO
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16, 
        learning_rate=lr_val, 
        bf16=True, # Native Ada support
        optim="adamw_bnb_8bit",
        remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True}
    )

    trainer = SFTTrainer(
        model=model, args=training_args,
        train_dataset=dataset, data_collator=flat_drama_collator
    )

    history = trainer.train()
    return history.training_loss

# Search for Best Parameters
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
if __name__ == "__main__":
    os.environ["MAX_WORKERS"] = "4" 
        
    flat_data_list = prepare_drama_x_dataset(csv)
    train_dataset = Dataset.from_list(flat_data_list)
    print(f"Dataset ready. Columns: {train_dataset.column_names}")


    # Sweep Learning Rate (Proposed: 1e-5, 2e-5, 5e-5)
    best_lr = 1e-5
    lowest_lr_loss = float('inf')

    for lr in [1e-5, 2e-5, 5e-5]: 
        print(f"\n Starting Phase 2.1 | Testing LR: {lr}, Rank: 8")
        try:
            # run_experiment handles the VRAM reset, BF16 config, and training
            loss = run_experiment(lr, 8, train_dataset) 
            log_result(lr, 8, loss)
            
            if loss < lowest_lr_loss:
                lowest_lr_loss = loss
                best_lr = lr
        except Exception as e:
            print(f"Run Failed for LR {lr}: {e}")
            log_result(lr, 8, str(e), status="failed")

    # Sweep LoRA Rank (Values from Proposal Table I)
    # Locking in the best_lr found above to optimize the rank-r manifold.
    best_rank = 8
    lowest_rank_loss = lowest_lr_loss

    for r in [4, 8, 16]: # 
        print(f"\n Starting Phase 2.2 | Testing Rank: {r} with Best LR: {best_lr}")
        try:
            loss = run_experiment(best_lr, r, train_dataset)
            log_result(best_lr, r, loss)
            
            if loss < lowest_rank_loss:
                lowest_rank_loss = loss
                best_rank = r
        except Exception as e:
            print(f"Run Failed for Rank {r}: {e}")
            log_result(best_lr, r, str(e), status="failed")

    print(f"\nPHASE 2 COMPLETE")
    print(f"Optimized Configuration: LR={best_lr}, Rank={best_rank}")