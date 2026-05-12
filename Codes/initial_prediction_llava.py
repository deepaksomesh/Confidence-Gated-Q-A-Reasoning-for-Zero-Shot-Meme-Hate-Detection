import os
import json
import math
import jsonlines
import torch

from datasets import load_dataset
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
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

if os.path.basename(SCRIPT_DIR).lower() in {"scripts", "src"}:
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
else:
    PROJECT_ROOT = SCRIPT_DIR

HF_CACHE_DIR = os.path.join(PROJECT_ROOT, "hf_cache")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "Output")

LOG_FILE = os.path.join(OUTPUT_DIR, "Step1_llava_onevision_initial_predictions.jsonl")
METRICS_FILE = os.path.join(OUTPUT_DIR, "Step1_llava_onevision_initial_metrics.json")

os.makedirs(HF_CACHE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

os.environ["HF_HOME"] = HF_CACHE_DIR

MODEL_NAME = "LLaVA-OneVision-Qwen2-7B"
MODEL_ID = "llava-hf/llava-onevision-qwen2-7b-ov-hf"
DATASET_ID = "emily49/hateful_memes_train_dev"

MAX_RECORDS = 3000
PROGRESS_EVERY = 100
LABELS_TO_SCORE = ["yes", "no"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

# ============================================================
# 2. MODEL LOADING
# ============================================================

print(f"Model      : {MODEL_ID}")
print(f"Dataset    : {DATASET_ID}")
print(f"Device     : {DEVICE}")
print(f"Max records: {MAX_RECORDS}")
print(f"Output     : {LOG_FILE}")

processor = AutoProcessor.from_pretrained(MODEL_ID)

model = LlavaOnevisionForConditionalGeneration.from_pretrained(
    MODEL_ID,
    dtype=DTYPE,
    device_map="auto",
).eval()


# ============================================================
# 3. PREDICTION FUNCTIONS
# ============================================================

def build_messages(meme_text: str):
    """
    Builds the chat-style input expected by LLaVA-OneVision.
    The image is provided as a separate content item, and the meme text
    is included inside the user instruction.
    """
    prompt_text = (
        "Does this image contain hate speech?\n"
        f"Text in image: {meme_text}\n"
        "Answer with 'Yes' or 'No' and provide a brief reason."
    )

    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]


def build_prompt(meme_text: str) -> str:
    """
    Converts the chat message into the final text prompt used by the model.
    This keeps the prompt format consistent with the model processor.
    """
    messages = build_messages(meme_text)

    return processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def get_label_likelihoods(image_obj, prompt_text: str) -> dict:
    """
    Scores the labels 'yes' and 'no' using the model loss.

    Each candidate label is appended to the same prompt, and only
    the appended label tokens are used when calculating the loss.
    """
    label_scores = {}

    for label in LABELS_TO_SCORE:
        full_text = prompt_text + " " + label

        full_inputs = processor(
            text=full_text,
            images=image_obj,
            return_tensors="pt",
        ).to(model.device)

        prefix_inputs = processor(
            text=prompt_text,
            images=image_obj,
            return_tensors="pt",
        ).to(model.device)

        prefix_len = prefix_inputs.input_ids.shape[1]
        target_labels = full_inputs.input_ids.clone()
        target_labels[:, :prefix_len] = -100

        with torch.no_grad():
            outputs = model(**full_inputs, labels=target_labels)

        target_len = full_inputs.input_ids.shape[1] - prefix_len
        label_scores[label] = -float(outputs.loss.item()) * target_len

    max_score = max(label_scores.values())
    exp_scores = {
        label: math.exp(score - max_score)
        for label, score in label_scores.items()
    }

    total = sum(exp_scores.values())

    return {
        label: round(exp_scores[label] / total, 4)
        for label in LABELS_TO_SCORE
    }


def run_prediction(image_obj, meme_text: str):
    """
    Runs one complete initial prediction.

    The function first calculates yes/no likelihood scores, then generates
    the model's answer. The final class is based on whether the generated
    response starts with Yes or No.
    """
    prompt = build_prompt(meme_text)
    likelihoods = get_label_likelihoods(image_obj, prompt)

    inputs = processor(
        text=prompt,
        images=image_obj,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=60,
            do_sample=False,
        )

    generated_text = processor.batch_decode(
        generated_ids[:, inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    prediction = "yes" if generated_text.lower().startswith("yes") else "no"

    return prediction, generated_text, likelihoods


# ============================================================
# 4. METRICS
# ============================================================

def calculate_metrics(results: list) -> dict:
    """
    Calculates dataset statistics, overall metrics, class-wise metrics,
    and the confusion matrix.
    """
    y_true = [1 if row["ground_truth"] == "yes" else 0 for row in results]
    y_pred = [1 if row["prediction"] == "yes" else 0 for row in results]

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1],
        zero_division=0,
    )

    conf_matrix = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()

    metrics = {
        "model_name": MODEL_NAME,
        "model_id": MODEL_ID,
        "dataset_id": DATASET_ID,
        "max_records": MAX_RECORDS,
        "rows_processed": len(results),
        "unique_examples": len({row["id"] for row in results}),
        "hateful_examples": int(sum(y_true)),
        "non_hateful_examples": int(len(results) - sum(y_true)),
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

    metrics["duplicate_id_rows"] = metrics["rows_processed"] - metrics["unique_examples"]

    return metrics


def print_metrics(metrics: dict):
    """
    Prints the metrics in a readable format.
    """
    print("\n" + "=" * 100)
    print("DATASET OVERVIEW")
    print("=" * 100)
    print(f"{'Rows processed':<30}: {metrics['rows_processed']}")
    print(f"{'Unique examples':<30}: {metrics['unique_examples']}")
    print(f"{'Hateful examples':<30}: {metrics['hateful_examples']}")
    print(f"{'Non-hateful examples':<30}: {metrics['non_hateful_examples']}")
    print(f"{'Duplicate id rows':<30}: {metrics['duplicate_id_rows']}")

    print("\n" + "=" * 100)
    print("INITIAL PREDICTION PERFORMANCE")
    print("=" * 100)
    print(f"{'Accuracy':<30}: {metrics['accuracy']:.4f}")
    print(f"{'Balanced accuracy':<30}: {metrics['balanced_accuracy']:.4f}")
    print(f"{'Macro precision':<30}: {metrics['macro_precision']:.4f}")
    print(f"{'Macro recall':<30}: {metrics['macro_recall']:.4f}")
    print(f"{'Macro F1':<30}: {metrics['macro_f1']:.4f}")
    print(f"{'Weighted precision':<30}: {metrics['weighted_precision']:.4f}")
    print(f"{'Weighted recall':<30}: {metrics['weighted_recall']:.4f}")
    print(f"{'Weighted F1':<30}: {metrics['weighted_f1']:.4f}")

    print("\n" + "=" * 100)
    print("CLASS-WISE METRICS")
    print("=" * 100)
    print(f"{'Class':<20} {'Precision':<15} {'Recall':<15} {'F1':<15} {'Support':<10}")

    for class_name, values in metrics["class_metrics"].items():
        print(
            f"{class_name:<20} "
            f"{values['precision']:<15.4f} "
            f"{values['recall']:<15.4f} "
            f"{values['f1']:<15.4f} "
            f"{values['support']:<10}"
        )

    print("\n" + "=" * 100)
    print("CONFUSION MATRIX")
    print("=" * 100)
    print("Rows = ground truth, columns = prediction")
    print("Labels: [non-hateful, hateful]")
    print(metrics["confusion_matrix"]["matrix"])

    print(f"\nSaved predictions to: {LOG_FILE}")
    print(f"Saved metrics to    : {METRICS_FILE}")


def save_metrics(metrics: dict):
    """
    Saves metrics as JSON so that plots can be created later.
    """
    with open(METRICS_FILE, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)


# ============================================================
# 5. MAIN LOOP
# ============================================================

def main():
    """
    Streams the dataset, runs initial prediction, writes JSONL predictions,
    and saves a separate metrics JSON file.
    """
    print("\nStarting LLaVA-OneVision initial prediction")

    dataset = load_dataset(DATASET_ID, split="train", streaming=True)
    results = []

    with jsonlines.open(LOG_FILE, mode="w") as writer:
        for index, item in enumerate(dataset):
            if index >= MAX_RECORDS:
                break

            image = item["image"]
            meme_text = item.get("text", "")
            ground_truth = "yes" if item.get("label") == 1 else "no"

            prediction, reason, likelihoods = run_prediction(image, meme_text)

            row = {
                "id": item["id"],
                "meme_text": meme_text,
                "ground_truth": ground_truth,
                "prediction": prediction,
                "confidence_score": likelihoods[prediction],
                "likelihoods": likelihoods,
                "initial_reason": reason,
                "is_correct": prediction == ground_truth,
            }

            writer.write(row)
            results.append(row)

            current = index + 1
            if current % PROGRESS_EVERY == 0:
                print(f"Processed {current}/{MAX_RECORDS}")

    metrics = calculate_metrics(results)
    save_metrics(metrics)
    print_metrics(metrics)


if __name__ == "__main__":
    main()
