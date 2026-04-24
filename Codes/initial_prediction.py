import os
import re
import math
import jsonlines
import torch
import numpy as np
from PIL import Image
from datasets import load_dataset
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, 
    recall_score, confusion_matrix, classification_report,
    balanced_accuracy_score, precision_recall_fscore_support
)

# ============================================================
# 1. CONFIGURATION & PATHS
# ============================================================
os.environ["HF_HOME"] = "/home/s4577663/NLP/hf_cache"

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
DATASET_ID = "emily49/hateful_memes_train_dev"

BASE_OUTPUT = "/home/s4577663/NLP/Output"
IMAGE_SAVE_DIR = os.path.join(BASE_OUTPUT, "downloaded_images")
LOG_FILE = os.path.join(BASE_OUTPUT, "Step1_hate_meme_initial_predictions.jsonl")

# Ensure Output directories exist
os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

# Control parameters
MAX_RECORDS = 100
PROGRESS_EVERY = 20 
LABELS_TO_SCORE = ["yes", "no"]

# ============================================================
# 2. MODEL INITIALIZATION
# ============================================================
print(f"Initializing {MODEL_ID} on {DEVICE}...")
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_ID, torch_dtype=DTYPE, device_map="auto", trust_remote_code=True
).eval()

# ============================================================
# 3. CORE LOGIC
# ============================================================

def get_simple_hate_prompt(meme_text: str) -> str:
    """YYama (2024) optimized zero-shot prompt."""
    return (
        f"Does this image contain hate speech?\n"
        f"Text in image: {meme_text}\n"
        f"Answer with 'Yes' or 'No' and provide a brief reason."
    )

def get_label_likelihoods(image_obj, prompt_text):
    """Calculates model confidence for 'yes' vs 'no'."""
    messages = [{"role": "user", "content": [{"type": "image", "image": image_obj}, {"type": "text", "text": prompt_text}]}]
    rendered_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    
    label_scores = {}
    for label in LABELS_TO_SCORE:
        full_text = rendered_prompt + label
        inputs = processor(text=[full_text], images=image_inputs, return_tensors="pt").to(model.device)
        prefix_inputs = processor(text=[rendered_prompt], images=image_inputs, return_tensors="pt").to(model.device)
        
        prefix_len = prefix_inputs.input_ids.shape[1]
        labels = inputs.input_ids.clone()
        labels[:, :prefix_len] = -100 
        
        with torch.no_grad():
            outputs = model(**inputs, labels=labels)
            # Log-likelihood = -loss * num_tokens_in_label
            label_scores[label] = -float(outputs.loss.item()) * (inputs.input_ids.shape[1] - prefix_len)

    max_val = max(label_scores.values())
    probs = {k: math.exp(v - max_val) for k, v in label_scores.items()}
    total = sum(probs.values())
    return {k: round(probs[k]/total, 4) for k in LABELS_TO_SCORE}

def run_prediction(image_obj, meme_text):
    prompt = get_simple_hate_prompt(meme_text)
    probs = get_label_likelihoods(image_obj, prompt)
    
    messages = [{"role": "user", "content": [{"type": "image", "image": image_obj}, {"type": "text", "text": prompt}]}]
    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(text=[text_prompt], images=image_inputs, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        gen_ids = model.generate(**inputs, max_new_tokens=60, do_sample=False)
    
    reason = processor.batch_decode([gen_ids[0][inputs.input_ids.shape[1]:]], skip_special_tokens=True)[0].strip()
    # Apply YYama Rule: Classify based on whether the output starts with 'Yes'
    pred = "yes" if reason.lower().startswith("yes") else "no"
    return pred, reason, probs

# ============================================================
# 4. EXECUTION LOOP & DETAILED REPORTING
# ============================================================

def main():
    print(f"Streaming {MAX_RECORDS} records from {DATASET_ID}...")
    dataset = load_dataset(DATASET_ID, split="train", streaming=True)
    results = []
    
    with jsonlines.open(LOG_FILE, mode="w") as writer:
        for i, item in enumerate(dataset):
            if i >= MAX_RECORDS: break

            image = item.get("image")
            text = item.get("text", "")
            gt = "yes" if item.get("label") == 1 else "no"
            
            # Save image for verification
            image.save(os.path.join(IMAGE_SAVE_DIR, f"{item['id']}.png"))
            
            # Process Prediction
            pred, reason, probs = run_prediction(image, text)
            
            row = {
                "id": item['id'], "meme_text": text, "ground_truth": gt,
                "prediction": pred, "initial_reason": reason,
                "likelihoods": probs, "is_correct": (pred == gt)
            }
            writer.write(row)
            results.append(row)

            # Progress Tracking
            current_num = i + 1
            if current_num % PROGRESS_EVERY == 0 or current_num == MAX_RECORDS:
                print(f"Progress: {current_num}/{MAX_RECORDS} completed.")

    # --- METRICS CALCULATION ---
    y_true = [1 if r['ground_truth'] == 'yes' else 0 for r in results]
    y_pred = [1 if r['prediction'] == 'yes' else 0 for r in results]
    
    total_rows = len(results)
    unique_ids = len(set([r['id'] for r in results]))
    hate_support = sum(y_true)
    non_hate_support = total_rows - hate_support

    # --- PRINTING ACADEMIC REPORT ---
    print("\n" + "="*140)
    print("DATASET OVERVIEW")
    print("="*140)
    print(f"{'Full rows':<35} : {total_rows}")
    print(f"{'Full unique examples':<35} : {unique_ids}")
    print(f"{'Hate records (Ground Truth)':<35} : {hate_support}")
    print(f"{'Non-Hate records (Ground Truth)':<35} : {non_hate_support}")
    print(f"{'Full duplicate key rows':<35} : {total_rows - unique_ids}")

    print("\nAtomic - Structured Soft")
    print("-" * 140)
    print(f"{'Accuracy':<20}: {accuracy_score(y_true, y_pred):.4f}")
    print(f"{'Macro Precision':<20}: {precision_score(y_true, y_pred, average='macro'):.4f}")
    print(f"{'Macro Recall':<20}: {recall_score(y_true, y_pred, average='macro'):.4f}")
    print(f"{'Macro F1':<20}: {f1_score(y_true, y_pred, average='macro'):.4f}")
    print(f"{'Weighted Precision':<20}: {precision_score(y_true, y_pred, average='weighted'):.4f}")
    print(f"{'Weighted Recall':<20}: {recall_score(y_true, y_pred, average='weighted'):.4f}")
    print(f"{'Weighted F1':<20}: {f1_score(y_true, y_pred, average='weighted'):.4f}")
    print(f"{'Balanced Accuracy':<20}: {balanced_accuracy_score(y_true, y_pred):.4f}")

    print("\n" + "="*140)
    print("CLASS-WISE METRICS")
    print("="*140)
    print("Baseline - Structured Direct")
    print("-" * 140)
    print(f"{'Class':<20} {'Precision':<15} {'Recall':<15} {'F1':<15} {'Support':<10}")
    
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1])
    class_names = ['non-hateful', 'hateful']
    for idx, name in enumerate(class_names):
        print(f"{name:<20} {p[idx]:<15.4f} {r[idx]:<15.4f} {f[idx]:<15.4f} {s[idx]:<10}")
    
    print("="*140)
    print(f"Log file saved to: {LOG_FILE}\n")

if __name__ == "__main__":
    main()