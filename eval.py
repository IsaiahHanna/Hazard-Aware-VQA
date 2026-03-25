import os
import ast
import json
import torch
import pandas as pd
from PIL import Image
from PIL import Image
from bert_score import score
from sklearn.metrics import f1_score
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import PeftModel
from parameter_search import get_clipped_gif_frames

# 1. Configuration
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
ADAPTER_PATH = "./hazard_vqa_final_model"
CSV_PATH = "../drama_subset_8/test/annotations.csv"
DATA_PATH = "../drama_subset_8/test"

def extract_intent(text):
    """Maps raw text to the 5-Class Intent Taxonomy (Towards, Away, Left, Right, Stationary)"""
    if not isinstance(text, str):
        return ["stationary"]
    
    text = text.lower()
    intents = []
    if "towards" in text: intents.append("moves towards ego vehicle")
    if "away" in text: intents.append("moves away from ego vehicle")
    if "left" in text: intents.append("goes to the left")
    if "right" in text: intents.append("goes to the right")
    
    if not intents: intents.append("stationary")
    return list(set(intents))

def prepare_eval_dataset(df, path):
    """Handles Pathing and Multi-VRU JSON strings"""
    flat_data = []
    TARGET_SIZE = (640, 480)
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

    print(f"Starting Phase 3 dataset preparation for {len(df)} rows...")

    for i, row in df.iterrows():
        # Fix paths using the DATA_PATH prefix
        vid_path = os.path.join(path, row['video_path'])
        img_path = os.path.join(path, row['image_path'])
        
        raw_paths = get_clipped_gif_frames(vid_path, img_path, radius=2)
        if not raw_paths: continue

        processed_pil_images = []
        for f_path in raw_paths:
            clean_path = f_path.replace("file://", "")
            try:
                img = Image.open(clean_path).convert("RGB").resize(TARGET_SIZE)
                processed_pil_images.append(img)
            except: continue

        if not processed_pil_images: continue

        try:
            peds = ast.literal_eval(row['Pedestrians']) if pd.notna(row['Pedestrians']) else {}
            cycs = ast.literal_eval(row['Cyclists']) if pd.notna(row['Cyclists']) else {}
        except:
            peds, cycs = {}, {}

        # Build Ground Truth labels
        target_dict = {
            "Risk": row.get('Risk', 'No'),
            "Suggested_action": row.get('suggested_action', 'continue')
        }

        vru_count = 0
        for cat_name, cat_data in [("pedestrian", peds), ("cyclist", cycs)]:
            for _, obj_data in cat_data.items():
                if obj_data and vru_count < 5:
                    target_dict[f"{cat_name}_{vru_count}"] = {
                        "Intent": obj_data.get('Intent', ["stationary"]),
                        "Reason": obj_data.get('Description', "Hazard detected"),
                        "Bounding_box": obj_data.get('Box', [0, 0, 0, 0])
                    }
                    vru_count += 1

        flat_data.append({
            "frames": processed_pil_images,
            "instruction": PROMPT_TEXT,
            "target": json.dumps(target_dict)
        })
    return flat_data

def get_model_prediction(model, processor, frames, instruction):
    """Extracts risk, action, and aggregated intents for up to 5 VRUs"""
    full_prompt = f"<|im_start|>user\n<|vision_start|><|video_pad|><|vision_end|>{instruction}<|im_end|>\n<|im_start|>assistant\n"
    
    # 2. Process with the new prompt
    inputs = processor(
        text=[full_prompt], 
        videos=[frames], 
        padding=True, 
        return_tensors="pt"
    ).to("cuda")
    
    # 3. Generate
    output_ids = model.generate(**inputs, max_new_tokens=512)
    response = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
    
    risk = "yes" if "risk\": \"yes" in response.lower() else "no"
    action = response
    all_intents = []
    
    try:
        clean_json = response.split("assistant\n")[-1].strip()
        res_json = json.loads(clean_json)
        action = res_json.get("Suggested_action", response)
        
        vru_keywords = ["pedestrian", "cyclist", "motorcyclist"]
        for key, value in res_json.items():
            if any(vru in key.lower() for vru in vru_keywords) and isinstance(value, dict):
                raw_intent = value.get("Intent", "")
                # Handle both string and list outputs from model
                if isinstance(raw_intent, list):
                    for sub_intent in raw_intent:
                        all_intents.extend(extract_intent(sub_intent))
                else:
                    all_intents.extend(extract_intent(raw_intent))
    except:
        all_intents = extract_intent(response)
        
    if not all_intents: all_intents = ["stationary"]
    if not all_intents: all_intents = ["stationary"]
    return {"intents": sorted(list(set(all_intents))), "action": action, "risk": risk}

def evaluate():
    print("Initializing Phase 3 Evaluation on 91 Unseen Samples...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto"
    )
    ft_model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    
    df = pd.read_csv(CSV_PATH) 
    test_data = prepare_eval_dataset(df,DATA_PATH)
    if not test_data:
        print("Fatal: No test data loaded. Check sibling directory paths.")
        return
    
    metrics = {
        "base": {"intents": [], "actions": [], "consistency": []},
        "ft": {"intents": [], "actions": [], "consistency": []},
        "gt": {"intents": [], "actions": []}
    }

    for i, item in enumerate(test_data):
        print(f"Processing Sample {i+1}/91...")
        
        # Ground Truth Multi-Object Extraction
        gt_json = json.loads(item['target'])
        gt_intents_list = []
        for key, value in gt_json.items():
            if isinstance(value, dict) and "Intent" in value:
                # Handle cases where Intent might be a list or a string
                intent_val = value["Intent"]
                if isinstance(intent_val, list):
                    gt_intents_list.extend(intent_val)
                else:
                    gt_intents_list.extend(extract_intent(intent_val))
        
        # Intent for F1-score
        metrics["gt"]["intents"].append(", ".join(sorted(list(set(gt_intents_list)))))
        # Action for BERT-Score
        metrics["gt"]["actions"].append(gt_json.get("Suggested_action", ""))

        for mode, model in [("base", base_model), ("ft", ft_model)]:
            
            pred = get_model_prediction(model, processor, item['frames'], item['instruction'])
            
            # Aggregate all detected VRU intents into one string for F1
            metrics[mode]["intents"].append(", ".join(pred["intents"]))
            metrics[mode]["actions"].append(pred["action"])

            # Temporal Consistency 
            if len(item['frames']) > 1:
                # Get the 'Global' prediction for the entire video clip
                global_risk = pred["risk"]
                
                # Check every other frame for agreement
                frame_matches = 0
                checked_frames = 0
                
                # Using a stride of 2 to check every other frame
                for f_idx in range(0, len(item['frames']), 2):
                    f_pred = get_model_prediction(model, processor, [item['frames'][f_idx]], item['instruction'])
                    
                    if f_pred["risk"] == global_risk:
                        frame_matches += 1
                    checked_frames += 1
                
                row_consistency = frame_matches / checked_frames
                metrics[mode]["consistency"].append(row_consistency)
        
        torch.cuda.empty_cache()

    # Calculate Metrics
    for mode in ["base", "ft"]:
        y_true = metrics["gt"]["intents"]
        y_pred = metrics[mode]["intents"]
        f1 = f1_score(y_true, y_pred, average='weighted') 

        clean_preds = [str(a) for a in metrics[mode]["actions"]]
        clean_gts = [str(a) for a in metrics["gt"]["actions"]]
        
        _, _, f1_bert = score(clean_preds, clean_gts, lang="en", verbose=False)
        
        consistency = sum(metrics[mode]["consistency"]) / len(metrics[mode]["consistency"]) if metrics[mode]["consistency"] else 0
        
        print(f"\n--- FINAL {mode.upper()} MODEL RESULTS ---")
        print(f"Intent Prediction (Mean F1): {f1:.4f}")
        print(f"Action Similarity (BERTScore): {f1_bert.mean().item():.4f}")
        print(f"Temporal Consistency (Agreement): {consistency:.4f}")

if __name__ == "__main__":
    evaluate()