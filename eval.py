import torch
import json
import pandas as pd
from bert_score import score
from sklearn.metrics import f1_score
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import PeftModel
from parameter_search import prepare_drama_x_dataset

# 1. Configuration
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
ADAPTER_PATH = "./hazard_vqa_final_model"
CSV_PATH = "../drama_subset_8/test/annotations.csv"

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

def get_model_prediction(model, processor, frames, instruction):
    """Extracts risk, action, and aggregated intents for up to 5 VRUs"""
    inputs = processor(text=[instruction], videos=[frames], return_tensors="pt").to("cuda")
    output_ids = model.generate(**inputs, max_new_tokens=512)
    response = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
    
    risk = "yes" if "risk\": \"yes" in response.lower() else "no"
    action = response
    all_intents = []
    
    try:
        clean_json = response.split("assistant\n")[-1].strip()
        res_json = json.loads(clean_json)
        action = res_json.get("Suggested_action", response)
        
        # Identify keys for up to 5 VRUs
        vru_keywords = ["pedestrian", "cyclist", "motorcyclist"]
        for key, value in res_json.items():
            if any(vru in key.lower() for vru in vru_keywords) and isinstance(value, dict):
                raw_intent = value.get("Intent", "")
                all_intents.extend(extract_intent(raw_intent))
    except:
        # Fallback to general text extraction if JSON parsing fails
        all_intents = extract_intent(response)
        
    if not all_intents: 
        all_intents = ["stationary"]
        
    return {"intents": sorted(list(set(all_intents))), "action": action, "risk": risk}

def evaluate():
    print("Initializing Phase 3 Evaluation on 91 Unseen Samples...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto"
    )
    ft_model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    
    df = pd.read_csv(CSV_PATH) 
    test_data = prepare_drama_x_dataset(df)
    
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
            if mode == "base": model.disable_adapter_layers()
            else: model.enable_adapter_layers()
            
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
        
        _, _, f1_bert = score(metrics[mode]["actions"], metrics["gt"]["actions"], lang="en", verbose=False)
        
        consistency = sum(metrics[mode]["consistency"]) / len(metrics[mode]["consistency"]) if metrics[mode]["consistency"] else 0
        
        print(f"\n--- FINAL {mode.upper()} MODEL RESULTS ---")
        print(f"Intent Prediction (Mean F1): {f1:.4f}")
        print(f"Action Similarity (BERTScore): {f1_bert.mean().item():.4f}")
        print(f"Temporal Consistency (Agreement): {consistency:.4f}")

if __name__ == "__main__":
    evaluate()