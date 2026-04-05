import json
import torch
import pandas as pd
from bert_score import score
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import PeftModel
from sklearn.preprocessing import MultiLabelBinarizer

# Config
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
ADAPTER_PATH = "./hazard_vqa_final_model2/checkpoint-150"
CSV_PATH = "../drama_subset_8/test/annotations.csv"
DATA_PATH = "../drama_subset_8/test"

INTENT_KEYWORDS = ["moves towards ego vehicle", "moves away from ego vehicle", "goes to the left", "goes to the right", "stationary"]

def extract_intent(text):
    """Revised: Explicitly searches for all taxonomy keywords independently."""
    if not isinstance(text, str): return ["stationary"]
    text = text.lower()
    intents = []
    if "towards" in text: intents.append("moves towards ego vehicle")
    if "away" in text: intents.append("moves away from ego vehicle")
    if "left" in text: intents.append("goes to the left")
    if "right" in text: intents.append("goes to the right")
    if "stationary" in text: intents.append("stationary")
    return list(set(intents)) if intents else ["stationary"]

def get_model_prediction(model, processor, frames, instruction):
    """Extracts paired agent-intent labels and action text."""
    full_prompt = f"<|im_start|>user\n<|vision_start|><|video_pad|><|vision_end|>{instruction}<|im_end|>\n<|im_start|>assistant\n"
    inputs = processor(text=[full_prompt], videos=[frames], return_tensors="pt").to("cuda")
    output_ids = model.generate(**inputs, max_new_tokens=512)
    response = processor.batch_decode(output_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
    
    risk = "yes" if "risk\": \"yes" in response.lower() else "no"
    action = response
    paired_labels = set()
    
    try:
        clean_json = response.replace("```json", "").replace("```", "").strip()
        res_json = json.loads(clean_json)
        action = res_json.get("Suggested_action", response)
        
        for key, value in res_json.items():
            if any(vru in key.lower() for vru in ["pedestrian", "cyclist", "car", "vru", "object"]) and isinstance(value, dict):
                # Determine base class for pairing
                obj_class = "vru"
                for cls in ["pedestrian", "cyclist", "car"]:
                    if cls in key.lower():
                        obj_class = cls
                        break
                
                raw_intent = value.get("Intent", "")
                # Parse all intents in the string/list and pair with class
                found_intents = []
                if isinstance(raw_intent, list):
                    for sub in raw_intent: found_intents.extend(extract_intent(str(sub)))
                else:
                    found_intents.extend(extract_intent(str(raw_intent)))
                
                for i in found_intents:
                    paired_labels.add(f"{obj_class}_{i}")
    except:
        pass # Fallback to empty if JSON fails

    if not paired_labels: paired_labels.add("vru_stationary")
    return {"paired_labels": list(paired_labels), "action": str(action), "risk": risk}

def evaluate():
    print("Initializing Strict Object-Aware Evaluation...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH, adapter_name="hazard_vqa")
    
    from eval import prepare_eval_dataset
    df = pd.read_csv(CSV_PATH) 
    test_data = prepare_eval_dataset(df, DATA_PATH)
    
    # Define all possible Agent_Action pairs for Binarizer
    all_agents = ["pedestrian", "cyclist", "car", "vru"]
    all_pairs = [f"{a}_{i}" for a in all_agents for i in INTENT_KEYWORDS]
    mlb = MultiLabelBinarizer(classes=all_pairs)

    results_store = {
        "base": {"preds": [], "actions": [], "risks": [], "consistency": []},
        "ft": {"preds": [], "actions": [], "risks": [], "consistency": []},
        "gt": {"preds": [], "actions": []}
    }

    for i, item in enumerate(test_data):
        print(f"Processing Sample {i+1}/91...")
        gt_json = json.loads(item['target'])
        
        # Build Paired Ground Truth
        gt_pairs = set()
        for k, v in gt_json.items():
            if any(x in k.lower() for x in ["pedestrian", "cyclist", "car", "vru"]):
                obj_class = "vru"
                for cls in ["pedestrian", "cyclist", "car"]:
                    if cls in k.lower(): obj_class = cls; break
                
                intent_val = v.get("Intent", ["stationary"])
                found = []
                if isinstance(intent_val, list):
                    for sub in intent_val: found.extend(extract_intent(str(sub)))
                else:
                    found.extend(extract_intent(str(intent_val)))
                for intent in found:
                    gt_pairs.add(f"{obj_class}_{intent}")
        
        if not gt_pairs: gt_pairs.add("vru_stationary")
        results_store["gt"]["preds"].append(list(gt_pairs))
        results_store["gt"]["actions"].append(gt_json.get("Suggested_action", ""))

        for mode in ["base", "ft"]:
            if mode == "base": model.base_model.disable_adapter_layers()
            else: 
                model.base_model.enable_adapter_layers()
                model.set_adapter("hazard_vqa")
            
            pred = get_model_prediction(model, processor, item['frames'], item['instruction'])
            results_store[mode]["preds"].append(pred["paired_labels"])
            results_store[mode]["actions"].append(pred["action"])
            results_store[mode]["risks"].append(pred["risk"])

            # Temporal Consistency Logic
            if len(item['frames']) > 1:
                matches = 0
                checked = 0
                for f_idx in range(0, len(item['frames']), 2):
                    f_pred = get_model_prediction(model, processor, [item['frames'][f_idx]], item['instruction'])
                    if f_pred["risk"] == pred["risk"]: matches += 1
                    checked += 1
                results_store[mode]["consistency"].append(matches / checked)

    # Final Scoring
    y_true = mlb.fit_transform(results_store["gt"]["preds"])
    for mode in ["base", "ft"]:
        y_pred = mlb.transform(results_store[mode]["preds"])
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        
        _, _, b_f1 = score([str(a) for a in results_store[mode]["actions"]], 
                           [str(a) for a in results_store["gt"]["actions"]], lang="en", verbose=False)
        action_acc = (b_f1 >= 0.8).float().mean().item()
        consistency = sum(results_store[mode]["consistency"]) / len(results_store[mode]["consistency"]) if results_store[mode]["consistency"] else 0

        print(f"\n--- FINAL {mode.upper()} MODEL RESULTS (STRICT) ---")
        print(f"Intent F1: {f1:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f}")
        print(f"Action Acc (>0.8): {action_acc:.2%}")
        print(f"Temporal Consistency: {consistency:.4f}")

if __name__ == "__main__":
    evaluate()
