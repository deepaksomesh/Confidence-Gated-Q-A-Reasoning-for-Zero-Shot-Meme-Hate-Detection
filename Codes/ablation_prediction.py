import os
import math
import jsonlines
import torch
from datasets import load_dataset
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, 
    recall_score, balanced_accuracy_score, precision_recall_fscore_support
)

# ============================================================
# 1. ABLATION SETTINGS (CHANGE THESE FOR YOUR EXPERIMENTS)
# ============================================================
# Choose one: "atomic", "dynamic_qa", "static_qa", "cot"
TECHNIQUE = "dynamic_qa"  

# Pushing the threshold higher to trigger more refinements!
CONFIDENCE_THRESHOLD = 0.95  

# ============================================================
# 2. PATHS & CONFIGURATION
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
os.environ["HF_HOME"] = os.path.join(PROJECT_ROOT, "hf_cache")

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
DATASET_ID = "emily49/hateful_memes_train_dev"

BASE_OUTPUT = os.path.join(PROJECT_ROOT, "Output")
IMAGE_SAVE_DIR = os.path.join(BASE_OUTPUT, "downloaded_images")
LOG_FILE = os.path.join(BASE_OUTPUT, f"Step2_{TECHNIQUE}_t{CONFIDENCE_THRESHOLD}_predictions.jsonl")

os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)
os.makedirs(os.environ["HF_HOME"], exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

MAX_RECORDS = 100
PROGRESS_EVERY = 20 
LABELS_TO_SCORE = ["yes", "no"]

# ============================================================
# 3. MODEL INITIALIZATION
# ============================================================
print(f"Initializing {MODEL_ID} on {DEVICE.upper()}...")
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_ID, torch_dtype=DTYPE, device_map="auto", trust_remote_code=True
).eval()

# ============================================================
# 4. CORE LOGIC
# ============================================================
def get_simple_hate_prompt(meme_text: str) -> str:
    return f"Does this image contain hate speech?\nText in image: {meme_text}\nAnswer with 'Yes' or 'No' and provide a brief reason."

def get_label_likelihoods(image_obj, prompt_text):
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
            label_scores[label] = -float(outputs.loss.item()) * (inputs.input_ids.shape[1] - prefix_len)

    max_val = max(label_scores.values())
    probs = {k: math.exp(v - max_val) for k, v in label_scores.items()}
    total = sum(probs.values())
    return {k: round(probs[k]/total, 4) for k in LABELS_TO_SCORE}

def get_refinement_prompt(technique, meme_text):
    if technique == "atomic":
        return (
            f"Meme text: '{meme_text}'.\n"
            "STEP 1: Extract up to 3 objective 'Atoms' (Visual/Textual facts). Do not judge hatefulness yet.\n"
            "STEP 2: Bridge/Connection. Do these atoms invoke a known slur or stereotype against a protected group?\n"
            "STEP 3: Answer strictly 'FINAL VERDICT: Yes' or 'FINAL VERDICT: No'."
        )
    elif technique == "dynamic_qa":
        return (
            f"Meme text: '{meme_text}'.\n"
            "STEP 1: Generate 3 targeted questions to help determine if this meme contains hate speech. Focus on who is targeted, the cross-modal meaning, and stereotypes.\n"
            "STEP 2: Answer each of your generated questions based on the image and text.\n"
            "STEP 3: Based on your answers, conclude strictly with 'FINAL VERDICT: Yes' or 'FINAL VERDICT: No'."
        )
    elif technique == "static_qa":
        return (
            f"Meme text: '{meme_text}'.\n"
            "Answer these 3 specific questions carefully:\n"
            "Q1. Target: Who or what is the primary subject or target of this meme?\n"
            "Q2. Interaction: Does the text change the meaning of the image to imply something offensive?\n"
            "Q3. Stereotype: Does this combination rely on a harmful stereotype, slur, or express hostility toward a protected group?\n"
            "Based on your answers, conclude strictly with 'FINAL VERDICT: Yes' or 'FINAL VERDICT: No'."
        )
    elif technique == "cot":
        return (
            f"Meme text: '{meme_text}'.\n"
            "Let's think step by step to determine if this meme contains hate speech. Consider the image and text together. "
            "Conclude strictly with 'FINAL VERDICT: Yes' or 'FINAL VERDICT: No'."
        )

def run_prediction(image_obj, meme_text):
    prompt = get_simple_hate_prompt(meme_text)
    probs = get_label_likelihoods(image_obj, prompt)
    
    messages = [{"role": "user", "content": [{"type": "image", "image": image_obj}, {"type": "text", "text": prompt}]}]
    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(text=[text_prompt], images=image_inputs, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        gen_ids = model.generate(**inputs, max_new_tokens=60, do_sample=False)
    
    initial_reason = processor.batch_decode([gen_ids[0][inputs.input_ids.shape[1]:]], skip_special_tokens=True)[0].strip()
    initial_pred = "yes" if initial_reason.lower().startswith("yes") else "no"
    confidence_score = probs[initial_pred]

    if confidence_score >= CONFIDENCE_THRESHOLD:
        return initial_pred, initial_reason, probs, False, None, initial_pred

    refinement_prompt = get_refinement_prompt(TECHNIQUE, meme_text)
    
    ref_messages = [{"role": "user", "content": [{"type": "image", "image": image_obj}, {"type": "text", "text": refinement_prompt}]}]
    ref_text_prompt = processor.apply_chat_template(ref_messages, tokenize=False, add_generation_prompt=True)
    ref_image_inputs, _ = process_vision_info(ref_messages)
    ref_inputs = processor(text=[ref_text_prompt], images=ref_image_inputs, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        # Allowed up to 400 tokens because Dynamic Q&A takes a lot of text to generate
        ref_gen_ids = model.generate(**ref_inputs, max_new_tokens=400, do_sample=False)
        
    ref_reasoning = processor.batch_decode([ref_gen_ids[0][ref_inputs.input_ids.shape[1]:]], skip_special_tokens=True)[0].strip()
    
    if "FINAL VERDICT: Yes" in ref_reasoning or "final verdict: yes" in ref_reasoning.lower():
        final_pred = "yes"
    elif "FINAL VERDICT: No" in ref_reasoning or "final verdict: no" in ref_reasoning.lower():
        final_pred = "no"
    else:
        final_pred = initial_pred 

    return final_pred, initial_reason, probs, True, ref_reasoning, initial_pred

def main():
    print(f"--- ABLATION EXPERIMENT ---")
    print(f"Technique: {TECHNIQUE.upper()}")
    print(f"Threshold: {CONFIDENCE_THRESHOLD}")
    print(f"---------------------------\n")
    
    dataset = load_dataset(DATASET_ID, split="train", streaming=True)
    results = []
    
    with jsonlines.open(LOG_FILE, mode="w") as writer:
        for i, item in enumerate(dataset):
            if i >= MAX_RECORDS: break

            image = item.get("image")
            text = item.get("text", "")
            gt = "yes" if item.get("label") == 1 else "no"
            
            final_pred, initial_reason, probs, was_refined, ref_reason, initial_pred = run_prediction(image, text)
            
            conf_score = probs[initial_pred]
            print(f"[{i+1}/{MAX_RECORDS}] ID: {item['id']} | Init Pred: {initial_pred.upper()} (Conf: {conf_score:.4f})", end="")
            
            if was_refined:
                print(f" -> ⚠️ Refined ({TECHNIQUE}) -> Final: {final_pred.upper()} | GT: {gt.upper()}")
            else:
                print(f" -> ✅ Skipped | GT: {gt.upper()}")
            
            row = {
                "id": item['id'], "meme_text": text, "ground_truth": gt,
                "initial_prediction": initial_pred, "final_prediction": final_pred, 
                "was_refined": was_refined, "confidence_score": conf_score,
                "likelihoods": probs, "initial_reason": initial_reason,
                "refined_reasoning": ref_reason if was_refined else "N/A - High Confidence",
                "is_correct": (final_pred == gt)
            }
            writer.write(row)
            results.append(row)

    y_true = [1 if r['ground_truth'] == 'yes' else 0 for r in results]
    y_pred = [1 if r['final_prediction'] == 'yes' else 0 for r in results]
    refined_count = sum([1 for r in results if r['was_refined']])

    print("\n" + "="*140)
    print("ABLATION RESULTS")
    print("="*140)
    print(f"{'Technique':<35} : {TECHNIQUE}")
    print(f"{'Threshold':<35} : {CONFIDENCE_THRESHOLD}")
    print(f"{'Records Refined':<35} : {refined_count} / {MAX_RECORDS}")
    print("-" * 140)
    print(f"{'Accuracy':<20}: {accuracy_score(y_true, y_pred):.4f}")
    print(f"{'Macro F1':<20}: {f1_score(y_true, y_pred, average='macro'):.4f}")

    print("\n" + "="*140)
    print("CLASS-WISE METRICS")
    print("="*140)
    print(f"{'Class':<20} {'Precision':<15} {'Recall':<15} {'F1':<15} {'Support':<10}")
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1])
    class_names = ['non-hateful', 'hateful']
    for idx, name in enumerate(class_names):
        print(f"{name:<20} {p[idx]:<15.4f} {r[idx]:<15.4f} {f[idx]:<15.4f} {s[idx]:<10}")
    print(f"Log file saved to: {LOG_FILE}\n")

if __name__ == "__main__":
    main()