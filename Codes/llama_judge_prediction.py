import os
import math
import jsonlines
import torch
from datasets import load_dataset
from transformers import (
    Qwen3VLForConditionalGeneration, 
    AutoProcessor,
    AutoModelForCausalLM,
    AutoTokenizer
)
from qwen_vl_utils import process_vision_info
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, 
    recall_score, precision_recall_fscore_support
)

# ============================================================
# 1. CONFIGURATION
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
os.environ["HF_HOME"] = os.path.join(PROJECT_ROOT, "hf_cache")

MODEL_ID_QWEN = "Qwen/Qwen3-VL-8B-Instruct"
MODEL_ID_LLAMA = "meta-llama/Meta-Llama-3-8B-Instruct"
DATASET_ID = "emily49/hateful_memes_train_dev"

BASE_OUTPUT = os.path.join(PROJECT_ROOT, "Output")
IMAGE_SAVE_DIR = os.path.join(BASE_OUTPUT, "downloaded_images")
LOG_FILE = os.path.join(BASE_OUTPUT, "Step3_Llama_Judge_predictions.jsonl")

os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)
os.makedirs(os.environ["HF_HOME"], exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

MAX_RECORDS = 100
CONFIDENCE_THRESHOLD = 0.95  
LABELS_TO_SCORE = ["yes", "no"]

# ============================================================
# 2. MODEL INITIALIZATION (LOADING BOTH AI's!)
# ============================================================
print(f"Loading Qwen3-VL (The Eyes) on {DEVICE.upper()}...")
qwen_processor = AutoProcessor.from_pretrained(MODEL_ID_QWEN, trust_remote_code=True)
qwen_model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_ID_QWEN, torch_dtype=DTYPE, device_map="auto", trust_remote_code=True
).eval()

print(f"Loading Llama-3-8B (The Judge) on {DEVICE.upper()}...")
llama_tokenizer = AutoTokenizer.from_pretrained(MODEL_ID_LLAMA)
llama_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID_LLAMA, torch_dtype=DTYPE, device_map="auto"
).eval()

# ============================================================
# 3. PIPELINE FUNCTIONS
# ============================================================
def get_qwen_likelihoods(image_obj, prompt_text):
    messages = [{"role": "user", "content": [{"type": "image", "image": image_obj}, {"type": "text", "text": prompt_text}]}]
    rendered_prompt = qwen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    
    label_scores = {}
    for label in LABELS_TO_SCORE:
        full_text = rendered_prompt + label
        inputs = qwen_processor(text=[full_text], images=image_inputs, return_tensors="pt").to(qwen_model.device)
        prefix_inputs = qwen_processor(text=[rendered_prompt], images=image_inputs, return_tensors="pt").to(qwen_model.device)
        
        prefix_len = prefix_inputs.input_ids.shape[1]
        labels = inputs.input_ids.clone()
        labels[:, :prefix_len] = -100 
        
        with torch.no_grad():
            outputs = qwen_model(**inputs, labels=labels)
            label_scores[label] = -float(outputs.loss.item()) * (inputs.input_ids.shape[1] - prefix_len)

    max_val = max(label_scores.values())
    probs = {k: math.exp(v - max_val) for k, v in label_scores.items()}
    total = sum(probs.values())
    return {k: round(probs[k]/total, 4) for k in LABELS_TO_SCORE}

def run_pipeline(image_obj, meme_text):
    # --- STAGE 1: Qwen Direct Prediction ---
    simple_prompt = f"Does this image contain hate speech?\nText in image: {meme_text}\nAnswer with 'Yes' or 'No' and provide a brief reason."
    probs = get_qwen_likelihoods(image_obj, simple_prompt)
    
    messages = [{"role": "user", "content": [{"type": "image", "image": image_obj}, {"type": "text", "text": simple_prompt}]}]
    text_prompt = qwen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = qwen_processor(text=[text_prompt], images=image_inputs, return_tensors="pt").to(qwen_model.device)
    
    with torch.no_grad():
        gen_ids = qwen_model.generate(**inputs, max_new_tokens=60, do_sample=False)
    
    initial_reason = qwen_processor.batch_decode([gen_ids[0][inputs.input_ids.shape[1]:]], skip_special_tokens=True)[0].strip()
    initial_pred = "yes" if initial_reason.lower().startswith("yes") else "no"
    confidence_score = probs[initial_pred]

    # Confidence Gate!
    if confidence_score >= CONFIDENCE_THRESHOLD:
        return initial_pred, initial_reason, probs, False, None, None, initial_pred

    # --- STAGE 2: Qwen generates Dynamic Q&A ---
    qa_prompt = (
        f"Meme text: '{meme_text}'.\n"
        "Generate 3 targeted questions about this meme's target, cross-modal meaning, and stereotypes. Then, answer them objectively based on the image."
    )
    
    ref_messages = [{"role": "user", "content": [{"type": "image", "image": image_obj}, {"type": "text", "text": qa_prompt}]}]
    ref_text_prompt = qwen_processor.apply_chat_template(ref_messages, tokenize=False, add_generation_prompt=True)
    ref_image_inputs, _ = process_vision_info(ref_messages)
    ref_inputs = qwen_processor(text=[ref_text_prompt], images=ref_image_inputs, return_tensors="pt").to(qwen_model.device)
    
    with torch.no_grad():
        ref_gen_ids = qwen_model.generate(**ref_inputs, max_new_tokens=300, do_sample=False)
    qwen_qa_output = qwen_processor.batch_decode([ref_gen_ids[0][ref_inputs.input_ids.shape[1]:]], skip_special_tokens=True)[0].strip()

    # --- STAGE 3: Llama-3 Acts as the Judge ---
    llama_system_prompt = "You are an expert, impartial hate speech moderator."
    llama_user_prompt = f"""Another AI analyzed a potentially hateful meme and generated these facts:
    
Meme Text: "{meme_text}"
Visual AI Analysis:
{qwen_qa_output}

Definition: Hate speech targets a protected group (race, religion, gender, sexual orientation, disability) with hostility, slurs, or dehumanizing stereotypes. General insults, edgy jokes, or mockery of terrorists are NOT hate speech.

Based strictly on the definition and the provided facts, does this meme constitute hate speech?
Reason step-by-step, then end your response with exactly 'FINAL VERDICT: Yes' or 'FINAL VERDICT: No'."""

    llama_messages = [
        {"role": "system", "content": llama_system_prompt},
        {"role": "user", "content": llama_user_prompt}
    ]
    
    llama_inputs = llama_tokenizer.apply_chat_template(llama_messages, add_generation_prompt=True, return_tensors="pt").to(llama_model.device)
    
    # Generate Llama's verdict
    with torch.no_grad():
        llama_outputs = llama_model.generate(llama_inputs, max_new_tokens=250, do_sample=False, pad_token_id=llama_tokenizer.eos_token_id)
        
    # Decode only the newly generated tokens
    llama_response = llama_tokenizer.decode(llama_outputs[0][llama_inputs.shape[1]:], skip_special_tokens=True).strip()

    # Parse final verdict
    if "FINAL VERDICT: Yes" in llama_response or "final verdict: yes" in llama_response.lower():
        final_pred = "yes"
    elif "FINAL VERDICT: No" in llama_response or "final verdict: no" in llama_response.lower():
        final_pred = "no"
    else:
        final_pred = initial_pred # Fallback

    return final_pred, initial_reason, probs, True, qwen_qa_output, llama_response, initial_pred

# ============================================================
# 4. EXECUTION LOOP
# ============================================================
def main():
    print(f"\n🚀 STARTING DUAL-MODEL PIPELINE (QWEN + LLAMA JUDGE)")
    dataset = load_dataset(DATASET_ID, split="train", streaming=True)
    results = []
    
    with jsonlines.open(LOG_FILE, mode="w") as writer:
        for i, item in enumerate(dataset):
            if i >= MAX_RECORDS: break

            image = item.get("image")
            text = item.get("text", "")
            gt = "yes" if item.get("label") == 1 else "no"
            
            final_pred, initial_reason, probs, was_refined, qwen_qa, llama_reason, initial_pred = run_pipeline(image, text)
            
            conf_score = probs[initial_pred]
            print(f"[{i+1}/{MAX_RECORDS}] ID: {item['id']} | Init: {initial_pred.upper()} (Conf: {conf_score:.4f})", end="")
            
            if was_refined:
                print(f" -> ⚖️ Llama Judged -> Final: {final_pred.upper()} | GT: {gt.upper()}")
            else:
                print(f" -> ✅ Skipped | GT: {gt.upper()}")
            
            row = {
                "id": item['id'], "meme_text": text, "ground_truth": gt,
                "initial_prediction": initial_pred, "final_prediction": final_pred, 
                "was_refined": was_refined, "confidence_score": conf_score,
                "likelihoods": probs, 
                "qwen_qa_output": qwen_qa if was_refined else "N/A",
                "llama_reasoning": llama_reason if was_refined else "N/A",
                "is_correct": (final_pred == gt)
            }
            writer.write(row)
            results.append(row)

    y_true = [1 if r['ground_truth'] == 'yes' else 0 for r in results]
    y_pred = [1 if r['final_prediction'] == 'yes' else 0 for r in results]
    refined_count = sum([1 for r in results if r['was_refined']])

    print("\n" + "="*140)
    print("LLAMA JUDGE RESULTS")
    print("="*140)
    print(f"{'Records Sent to Llama':<35} : {refined_count} / {MAX_RECORDS}")
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