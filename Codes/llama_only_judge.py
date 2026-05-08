import os
import re
import jsonlines
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
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

MODEL_ID_LLAMA = "meta-llama/Meta-Llama-3-8B-Instruct"

# We read the output from your best ablation study!
INPUT_LOG_FILE = os.path.join(PROJECT_ROOT, "Output", "Step2_dynamic_qa_t0.95_predictions.jsonl")
FINAL_OUTPUT_FILE = os.path.join(PROJECT_ROOT, "Output", "Step3_Final_Llama_Judged.jsonl")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

# ============================================================
# 2. LOAD LLAMA-3
# ============================================================
print(f"Loading Llama-3-8B (The Judge) on {DEVICE.upper()}...")
llama_tokenizer = AutoTokenizer.from_pretrained(MODEL_ID_LLAMA)
llama_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID_LLAMA, torch_dtype=DTYPE, device_map="auto"
).eval()

# ============================================================
# 3. LLAMA JUDGE LOGIC
# ============================================================
def get_llama_verdict(meme_text, qwen_qa_facts, qwen_pred):
    llama_system_prompt = "You are an expert, impartial hate speech moderator."
    llama_user_prompt = f"""Another AI analyzed a potentially hateful meme and generated these facts:
    
Meme Text: "{meme_text}"
Visual AI Analysis:
{qwen_qa_facts}

Definition: Hate speech targets a protected group (race, religion, gender, sexual orientation, disability) with hostility, slurs, or dehumanizing stereotypes. General insults, edgy jokes, or mockery of terrorists are NOT hate speech.

Based strictly on the definition and the provided facts, does this meme constitute hate speech?
Reason step-by-step, then end your response with exactly 'FINAL VERDICT: Yes' or 'FINAL VERDICT: No'."""

    messages = [
        {"role": "system", "content": llama_system_prompt},
        {"role": "user", "content": llama_user_prompt}
    ]
    
    prompt_str = llama_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = llama_tokenizer(prompt_str, return_tensors="pt").to(llama_model.device)
    
    with torch.no_grad():
        outputs = llama_model.generate(**inputs, max_new_tokens=250, do_sample=False, pad_token_id=llama_tokenizer.eos_token_id)
        
    llama_response = llama_tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

    # MORE FLEXIBLE PARSING & SAFE FALLBACK
    llama_lower = llama_response.lower()
    if "verdict: yes" in llama_lower:
        return "yes", llama_response
    elif "verdict: no" in llama_lower:
        return "no", llama_response
    else:
        # If Llama fails to format exactly, fallback safely to Qwen's original prediction
        return qwen_pred, llama_response

# ============================================================
# 4. EXECUTION LOOP
# ============================================================
def main():
    print(f"\n🚀 STARTING LLAMA-3 JUDGE PASS")
    
    if not os.path.exists(INPUT_LOG_FILE):
        print(f"ERROR: Cannot find {INPUT_LOG_FILE}. Did you run the ablation study?")
        return

    # Read the previous predictions
    with jsonlines.open(INPUT_LOG_FILE, mode="r") as reader:
        records = list(reader)

    updated_records = []
    judged_count = 0

    with jsonlines.open(FINAL_OUTPUT_FILE, mode="w") as writer:
        for i, row in enumerate(records):
            
            # If Qwen refined it, let Llama judge it!
            if row["was_refined"]:
                judged_count += 1
                meme_text = row["meme_text"]
                qwen_facts_raw = row["refined_reasoning"]
                
                # HIDE Qwen's final verdict so Llama doesn't cheat!
                # This uses regex to split the text at 'FINAL VERDICT' regardless of capitalization
                qwen_facts_clean = re.split(r'(?i)final verdict', qwen_facts_raw)[0].strip()
                
                # Pass the CLEANED facts to Llama
                llama_pred, llama_reason = get_llama_verdict(meme_text, qwen_facts_clean, row["final_prediction"])
                
                print(f"ID {row['id']}: Qwen said {row['final_prediction'].upper()} -> Llama says {llama_pred.upper()} | GT: {row['ground_truth'].upper()}")
                
                # Overwrite Qwen's prediction with Llama's
                row["final_prediction"] = llama_pred
                row["llama_reasoning"] = llama_reason
                row["is_correct"] = (row["final_prediction"] == row["ground_truth"])
            else:
                row["llama_reasoning"] = "N/A - Skipped by Confidence Gate"

            writer.write(row)
            updated_records.append(row)

    # Calculate final metrics
    y_true = [1 if r['ground_truth'] == 'yes' else 0 for r in updated_records]
    y_pred = [1 if r['final_prediction'] == 'yes' else 0 for r in updated_records]

    print("\n" + "="*140)
    print("FINAL PIPELINE RESULTS (QWEN EXTRACTOR + BLIND LLAMA JUDGE)")
    print("="*140)
    print(f"{'Records Judged by Llama':<35} : {judged_count} / {len(records)}")
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
    print(f"Log file saved to: {FINAL_OUTPUT_FILE}\n")

if __name__ == "__main__":
    main()