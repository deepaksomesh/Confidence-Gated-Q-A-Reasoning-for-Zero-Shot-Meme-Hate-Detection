import os
import re
import json
import jsonlines
import torch

from datasets import load_dataset
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

if os.path.basename(SCRIPT_DIR).lower() in {"scripts", "src", "codes"}:
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
else:
    PROJECT_ROOT = SCRIPT_DIR

HF_CACHE_DIR = os.path.join(PROJECT_ROOT, "hf_cache")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "Codes/Output")

INITIAL_FILE = os.path.join(OUTPUT_DIR, "Step1_qwen3_initial_predictions.jsonl")

PREDICTION_DIR = os.path.join(OUTPUT_DIR, "Step2_qwen3_ablation_predictions")
METRICS_DIR = os.path.join(OUTPUT_DIR, "Step2_qwen3_ablation_metrics")
SUMMARY_FILE = os.path.join(OUTPUT_DIR, "Step2_qwen3_ablation_summary.json")

os.makedirs(HF_CACHE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PREDICTION_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

os.environ["HF_HOME"] = HF_CACHE_DIR

MODEL_NAME = "Qwen3-VL-8B"
MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
DATASET_ID = "emily49/hateful_memes_train_dev"

MAX_RECORDS = 3000
PROGRESS_EVERY = 100

TECHNIQUES = ["atomic", "static_qa", "dynamic_qa", "cot"]
CONFIDENCE_THRESHOLDS = [0.60, 0.75, 0.85, 0.95]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32


# ============================================================
# 2. MODEL LOADING
# ============================================================

print(f"Model        : {MODEL_ID}")
print(f"Dataset      : {DATASET_ID}")
print(f"Device       : {DEVICE}")
print(f"Max records  : {MAX_RECORDS}")
print(f"Initial file : {INITIAL_FILE}")

processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
)

model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=DTYPE,
    device_map="auto",
    trust_remote_code=True,
).eval()


# ============================================================
# 3. LOADING STEP 1 OUTPUT
# ============================================================

def to_key(value) -> str:
    """
    Converts dataset IDs into a stable string key.
    """
    return str(value)


def load_initial_predictions() -> list:
    """
    Loads the saved Qwen3 initial predictions from Step 1.
    These predictions provide the initial label and confidence score.
    """
    if not os.path.exists(INITIAL_FILE):
        raise FileNotFoundError(f"Missing Step 1 file: {INITIAL_FILE}")

    records = []

    with jsonlines.open(INITIAL_FILE, mode="r") as reader:
        for row in reader:
            records.append(row)

    if len(records) > MAX_RECORDS:
        records = records[:MAX_RECORDS]

    return records


# ============================================================
# 4. REFINEMENT PROMPTS
# ============================================================

def get_refinement_prompt(technique: str, meme_text: str) -> str:
    """
    Creates the refinement prompt for one ablation strategy.
    Each strategy must end with a final verdict in the same format.
    """
    if technique == "atomic":
        return (
            f"Meme text: '{meme_text}'.\n\n"
            "We need to decide whether this meme contains hate speech.\n"
            "Use the following steps:\n"
            "Step 1: Extract up to 3 objective visual or textual facts from the meme.\n"
            "Step 2: Explain how these facts interact with each other.\n"
            "Step 3: Decide whether the meme targets a protected group with hostility, slurs, "
            "dehumanization, or harmful stereotypes.\n\n"
            "Protected groups include race, religion, gender, sexual orientation, disability, "
            "nationality, or similar identity groups. General insults, criticism of individuals, "
            "or mockery of terrorists are not hate speech.\n\n"
            "End with exactly one line:\n"
            "FINAL VERDICT: Yes\n"
            "or\n"
            "FINAL VERDICT: No"
        )

    if technique == "static_qa":
        return (
            f"Meme text: '{meme_text}'.\n\n"
            "Answer the following questions using the image and text together:\n"
            "Q1. Who or what is the main target of the meme?\n"
            "Q2. Does the image-text combination express hostility, ridicule, dehumanization, "
            "or a harmful stereotype?\n"
            "Q3. Is the target a protected group such as race, religion, gender, sexual orientation, "
            "disability, nationality, or a similar identity group?\n\n"
            "Based on these answers, decide whether the meme contains hate speech.\n\n"
            "End with exactly one line:\n"
            "FINAL VERDICT: Yes\n"
            "or\n"
            "FINAL VERDICT: No"
        )

    if technique == "dynamic_qa":
        return (
            f"Meme text: '{meme_text}'.\n\n"
            "Generate 3 useful questions that help decide whether this meme contains hate speech. "
            "The questions should focus on the target, the image-text interaction, and whether a "
            "protected group is attacked through hostility, slurs, dehumanization, or stereotypes.\n\n"
            "Answer each question based only on the meme image and text. Then make a final decision.\n\n"
            "End with exactly one line:\n"
            "FINAL VERDICT: Yes\n"
            "or\n"
            "FINAL VERDICT: No"
        )

    if technique == "cot":
        return (
            f"Meme text: '{meme_text}'.\n\n"
            "Decide whether this meme contains hate speech. Consider the image and text together, "
            "identify the target, and check whether the meme attacks a protected group through "
            "hostility, slurs, dehumanization, or harmful stereotypes.\n\n"
            "General insults, criticism of individuals, offensive jokes without protected-group "
            "targeting, or mockery of terrorists are not hate speech.\n\n"
            "End with exactly one line:\n"
            "FINAL VERDICT: Yes\n"
            "or\n"
            "FINAL VERDICT: No"
        )

    raise ValueError(f"Unknown technique: {technique}")


def parse_final_verdict(text: str, fallback: str) -> str:
    """
    Extracts the final Yes/No verdict from the refinement output.
    If no valid verdict is found, the initial prediction is kept.
    """
    lower_text = text.lower()

    if re.search(r"final\s+verdict\s*:\s*yes", lower_text):
        return "yes"

    if re.search(r"final\s+verdict\s*:\s*no", lower_text):
        return "no"

    return fallback


# ============================================================
# 5. REFINEMENT
# ============================================================

def run_refinement(image_obj, meme_text: str, technique: str, initial_prediction: str):
    """
    Runs one refinement strategy for one low-confidence example.
    The refined prediction is parsed from the final verdict line.
    """
    prompt = get_refinement_prompt(technique, meme_text)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_obj},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text_prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    image_inputs, _ = process_vision_info(messages)

    inputs = processor(
        text=[text_prompt],
        images=image_inputs,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=350,
            do_sample=False,
        )

    refined_reasoning = processor.batch_decode(
        [generated_ids[0][inputs.input_ids.shape[1]:]],
        skip_special_tokens=True,
    )[0].strip()

    refined_prediction = parse_final_verdict(
        refined_reasoning,
        fallback=initial_prediction,
    )

    return refined_prediction, refined_reasoning


def build_refinement_cache(initial_records: list, technique: str) -> dict:
    """
    Runs refinement once per candidate example for a given strategy.

    Candidates are examples with confidence below the largest threshold.
    The cached refinements are reused for all lower thresholds.
    """
    max_threshold = max(CONFIDENCE_THRESHOLDS)

    candidate_ids = {
        to_key(row["id"])
        for row in initial_records
        if row["confidence_score"] < max_threshold
    }

    initial_by_id = {
        to_key(row["id"]): row
        for row in initial_records
    }

    print("\n" + "=" * 100)
    print(f"BUILDING REFINEMENT CACHE: {technique}")
    print("=" * 100)
    print(f"Candidate examples below threshold {max_threshold}: {len(candidate_ids)}")

    cache = {}
    dataset = load_dataset(DATASET_ID, split="train", streaming=True)

    processed = 0
    refined_count = 0

    for index, item in enumerate(dataset):
        if index >= MAX_RECORDS:
            break

        item_id = to_key(item["id"])

        if item_id not in candidate_ids:
            continue

        initial_row = initial_by_id[item_id]

        image = item["image"]
        meme_text = item.get("text", initial_row.get("meme_text", ""))
        initial_prediction = initial_row["prediction"]

        refined_prediction, refined_reasoning = run_refinement(
            image_obj=image,
            meme_text=meme_text,
            technique=technique,
            initial_prediction=initial_prediction,
        )

        cache[item_id] = {
            "refined_prediction": refined_prediction,
            "refined_reasoning": refined_reasoning,
        }

        refined_count += 1
        processed += 1

        if refined_count % PROGRESS_EVERY == 0:
            print(f"Refined {refined_count}/{len(candidate_ids)} for {technique}")

    print(f"Completed refinement cache for {technique}: {len(cache)} records")

    return cache


# ============================================================
# 6. METRICS
# ============================================================

def calculate_metrics(records: list, technique: str, threshold: float) -> dict:
    """
    Calculates performance and refinement statistics for one run.
    """
    y_true = [1 if row["ground_truth"] == "yes" else 0 for row in records]
    y_pred = [1 if row["final_prediction"] == "yes" else 0 for row in records]

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1],
        zero_division=0,
    )

    refined_count = sum(1 for row in records if row["was_refined"])
    changed_count = sum(
        1 for row in records
        if row["initial_prediction"] != row["final_prediction"]
    )

    initial_wrong_final_correct = sum(
        1 for row in records
        if (not row["initial_is_correct"]) and row["is_correct"]
    )

    initial_correct_final_wrong = sum(
        1 for row in records
        if row["initial_is_correct"] and (not row["is_correct"])
    )

    conf_matrix = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()

    metrics = {
        "model_name": MODEL_NAME,
        "model_id": MODEL_ID,
        "dataset_id": DATASET_ID,
        "technique": technique,
        "confidence_threshold": threshold,
        "rows_processed": len(records),
        "refined_count": refined_count,
        "refinement_rate": round(refined_count / len(records), 6),
        "changed_prediction_count": changed_count,
        "initial_wrong_final_correct": initial_wrong_final_correct,
        "initial_correct_final_wrong": initial_correct_final_wrong,
        "net_correction_gain": initial_wrong_final_correct - initial_correct_final_wrong,
        "accuracy": round(accuracy_score(y_true, y_pred), 6),
        "balanced_accuracy": round(balanced_accuracy_score(y_true, y_pred), 6),
        "macro_precision": round(precision_score(y_true, y_pred, average="macro", zero_division=0), 6),
        "macro_recall": round(recall_score(y_true, y_pred, average="macro", zero_division=0), 6),
        "macro_f1": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 6),
        "weighted_precision": round(precision_score(y_true, y_pred, average="weighted", zero_division=0), 6),
        "weighted_recall": round(recall_score(y_true, y_pred, average="weighted", zero_division=0), 6),
        "weighted_f1": round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 6),
        "class_metrics": {
            "non_hateful": {
                "precision": round(float(precision[0]), 6),
                "recall": round(float(recall[0]), 6),
                "f1": round(float(f1[0]), 6),
                "support": int(support[0]),
            },
            "hateful": {
                "precision": round(float(precision[1]), 6),
                "recall": round(float(recall[1]), 6),
                "f1": round(float(f1[1]), 6),
                "support": int(support[1]),
            },
        },
        "confusion_matrix": {
            "labels": ["non_hateful", "hateful"],
            "matrix": conf_matrix,
        },
    }

    return metrics


def print_metrics(metrics: dict):
    """
    Prints a compact result summary for one ablation run.
    """
    print("\n" + "-" * 100)
    print(f"Technique: {metrics['technique']} | Threshold: {metrics['confidence_threshold']}")
    print("-" * 100)
    print(f"{'Rows processed':<30}: {metrics['rows_processed']}")
    print(f"{'Refined count':<30}: {metrics['refined_count']}")
    print(f"{'Refinement rate':<30}: {metrics['refinement_rate']:.4f}")
    print(f"{'Changed predictions':<30}: {metrics['changed_prediction_count']}")
    print(f"{'Net correction gain':<30}: {metrics['net_correction_gain']}")
    print(f"{'Accuracy':<30}: {metrics['accuracy']:.4f}")
    print(f"{'Macro F1':<30}: {metrics['macro_f1']:.4f}")
    print(f"{'Hateful F1':<30}: {metrics['class_metrics']['hateful']['f1']:.4f}")
    print(f"{'Hateful recall':<30}: {metrics['class_metrics']['hateful']['recall']:.4f}")


def save_metrics(metrics: dict, file_path: str):
    """
    Saves one metric file as JSON.
    """
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)


# ============================================================
# 7. OUTPUT CREATION
# ============================================================

def make_output_paths(technique: str, threshold: float):
    """
    Builds prediction and metric file paths for one ablation run.
    """
    threshold_tag = f"t{threshold:.2f}"

    prediction_file = os.path.join(
        PREDICTION_DIR,
        f"Step2_qwen3_{technique}_{threshold_tag}_predictions.jsonl",
    )

    metrics_file = os.path.join(
        METRICS_DIR,
        f"Step2_qwen3_{technique}_{threshold_tag}_metrics.json",
    )

    return prediction_file, metrics_file


def create_run_records(initial_records: list, refinement_cache: dict, technique: str, threshold: float) -> list:
    """
    Creates final predictions for one technique-threshold combination.
    High-confidence examples keep the initial prediction.
    Low-confidence examples use the cached refinement output.
    """
    run_records = []

    for row in initial_records:
        item_id = to_key(row["id"])
        confidence = row["confidence_score"]
        was_refined = confidence < threshold

        if was_refined and item_id in refinement_cache:
            final_prediction = refinement_cache[item_id]["refined_prediction"]
            refined_reasoning = refinement_cache[item_id]["refined_reasoning"]
        else:
            was_refined = False
            final_prediction = row["prediction"]
            refined_reasoning = "N/A - skipped by confidence gate"

        output_row = {
            "id": row["id"],
            "meme_text": row["meme_text"],
            "ground_truth": row["ground_truth"],
            "initial_prediction": row["prediction"],
            "final_prediction": final_prediction,
            "confidence_score": confidence,
            "likelihoods": row.get("likelihoods", {}),
            "technique": technique,
            "confidence_threshold": threshold,
            "was_refined": was_refined,
            "initial_reason": row.get("initial_reason", ""),
            "refined_reasoning": refined_reasoning,
            "initial_is_correct": row["prediction"] == row["ground_truth"],
            "is_correct": final_prediction == row["ground_truth"],
        }

        run_records.append(output_row)

    return run_records


def write_predictions(records: list, file_path: str):
    """
    Writes one JSONL prediction file for one ablation run.
    """
    with jsonlines.open(file_path, mode="w") as writer:
        for row in records:
            writer.write(row)


# ============================================================
# 8. MAIN LOOP
# ============================================================

def main():
    """
    Runs the full Step 2 ablation over refinement strategies and
    confidence thresholds using Qwen3-VL as the selected backbone.
    """
    initial_records = load_initial_predictions()
    summary = []

    print("\n" + "=" * 100)
    print("STEP 2 QWEN3 REFINEMENT ABLATION")
    print("=" * 100)
    print(f"Initial records loaded: {len(initial_records)}")
    print(f"Techniques: {TECHNIQUES}")
    print(f"Thresholds: {CONFIDENCE_THRESHOLDS}")

    for technique in TECHNIQUES:
        refinement_cache = build_refinement_cache(
            initial_records=initial_records,
            technique=technique,
        )

        for threshold in CONFIDENCE_THRESHOLDS:
            prediction_file, metrics_file = make_output_paths(
                technique=technique,
                threshold=threshold,
            )

            run_records = create_run_records(
                initial_records=initial_records,
                refinement_cache=refinement_cache,
                technique=technique,
                threshold=threshold,
            )

            metrics = calculate_metrics(
                records=run_records,
                technique=technique,
                threshold=threshold,
            )

            write_predictions(run_records, prediction_file)
            save_metrics(metrics, metrics_file)
            print_metrics(metrics)

            summary.append(
                {
                    "technique": technique,
                    "confidence_threshold": threshold,
                    "prediction_file": prediction_file,
                    "metrics_file": metrics_file,
                    "accuracy": metrics["accuracy"],
                    "macro_f1": metrics["macro_f1"],
                    "balanced_accuracy": metrics["balanced_accuracy"],
                    "hateful_f1": metrics["class_metrics"]["hateful"]["f1"],
                    "hateful_recall": metrics["class_metrics"]["hateful"]["recall"],
                    "refined_count": metrics["refined_count"],
                    "refinement_rate": metrics["refinement_rate"],
                    "changed_prediction_count": metrics["changed_prediction_count"],
                    "net_correction_gain": metrics["net_correction_gain"],
                }
            )

    summary = sorted(
        summary,
        key=lambda row: (row["accuracy"], row["macro_f1"]),
        reverse=True,
    )

    with open(SUMMARY_FILE, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("\n" + "=" * 100)
    print("BEST STEP 2 RUNS BY ACCURACY")
    print("=" * 100)
    print(f"{'Rank':<6}{'Technique':<15}{'Threshold':<12}{'Accuracy':<12}{'Macro F1':<12}{'Refined':<10}")

    for rank, row in enumerate(summary[:10], start=1):
        print(
            f"{rank:<6}"
            f"{row['technique']:<15}"
            f"{row['confidence_threshold']:<12.2f}"
            f"{row['accuracy']:<12.4f}"
            f"{row['macro_f1']:<12.4f}"
            f"{row['refined_count']:<10}"
        )

    print(f"\nSaved summary to: {SUMMARY_FILE}")


if __name__ == "__main__":
    main()
