import os
import json

import matplotlib.pyplot as plt


# ============================================================
# 1. CONFIGURATION
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

if os.path.basename(SCRIPT_DIR).lower() in {"scripts", "src", "codes"}:
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
else:
    PROJECT_ROOT = SCRIPT_DIR

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "Codes/Output")

METRIC_FILES = [
    os.path.join(OUTPUT_DIR, "Step1_qwen3_initial_metrics.json"),
    os.path.join(OUTPUT_DIR, "Step1_llava_onevision_initial_metrics.json"),
    os.path.join(OUTPUT_DIR, "Step1_internvl_initial_metrics.json"),
]

RANKING_FILE = os.path.join(OUTPUT_DIR, "Step1_model_comparison_ranking.json")
BAR_CHART_FILE = os.path.join(OUTPUT_DIR, "Step1_model_comparison_bar_chart.png")


# ============================================================
# 2. LOADING
# ============================================================

def load_metrics(metric_files):
    """
    Loads the saved metric JSON files from the Step 1 model runs.
    """
    records = []

    for file_path in metric_files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Missing metric file: {file_path}")

        with open(file_path, "r", encoding="utf-8") as file:
            metrics = json.load(file)

        records.append(metrics)

    return records


# ============================================================
# 3. COMPARISON
# ============================================================

def rank_models(records):
    """
    Ranks models using Macro F1 first, then accuracy as a tie-breaker.
    """
    return sorted(
        records,
        key=lambda row: (
            row["macro_f1"],
            row["accuracy"],
            row["balanced_accuracy"],
        ),
        reverse=True,
    )


def print_dataset_statistics(records):
    """
    Prints dataset statistics from the first metric file.
    All Step 1 models should have been evaluated on the same records.
    """
    first = records[0]

    print("\n" + "=" * 100)
    print("DATASET STATISTICS")
    print("=" * 100)
    print(f"{'Rows processed':<30}: {first['rows_processed']}")
    print(f"{'Unique examples':<30}: {first['unique_examples']}")
    print(f"{'Hateful examples':<30}: {first['hateful_examples']}")
    print(f"{'Non-hateful examples':<30}: {first['non_hateful_examples']}")
    print(f"{'Duplicate id rows':<30}: {first['duplicate_id_rows']}")


def print_ranking_table(ranked_records):
    """
    Prints a compact ranking table for model selection.
    """
    print("\n" + "=" * 100)
    print("STEP 1 MODEL RANKING")
    print("=" * 100)

    header = (
        f"{'Rank':<6}"
        f"{'Model':<30}"
        f"{'Accuracy':<12}"
        f"{'Macro F1':<12}"
        f"{'Bal. Acc':<12}"
        f"{'Hateful F1':<12}"
        f"{'Hateful Recall':<15}"
    )
    print(header)
    print("-" * 100)

    for rank, row in enumerate(ranked_records, start=1):
        hateful = row["class_metrics"]["hateful"]

        print(
            f"{rank:<6}"
            f"{row['model_name']:<30}"
            f"{row['accuracy']:<12.4f}"
            f"{row['macro_f1']:<12.4f}"
            f"{row['balanced_accuracy']:<12.4f}"
            f"{hateful['f1']:<12.4f}"
            f"{hateful['recall']:<15.4f}"
        )

    best = ranked_records[0]

    print("\n" + "=" * 100)
    print("SELECTED MODEL")
    print("=" * 100)
    print(f"Best model based on Macro F1: {best['model_name']}")
    print(f"Model ID: {best['model_id']}")


def save_ranking(ranked_records):
    """
    Saves the ranking table as JSON for later reporting.
    """
    ranking = []

    for rank, row in enumerate(ranked_records, start=1):
        ranking.append(
            {
                "rank": rank,
                "model_name": row["model_name"],
                "model_id": row["model_id"],
                "accuracy": row["accuracy"],
                "macro_f1": row["macro_f1"],
                "balanced_accuracy": row["balanced_accuracy"],
                "hateful_f1": row["class_metrics"]["hateful"]["f1"],
                "hateful_recall": row["class_metrics"]["hateful"]["recall"],
            }
        )

    with open(RANKING_FILE, "w", encoding="utf-8") as file:
        json.dump(ranking, file, indent=2)

    print(f"\nSaved ranking to: {RANKING_FILE}")


# ============================================================
# 4. PLOTTING
# ============================================================

def create_bar_chart(ranked_records):
    """
    Creates a grouped bar chart comparing Accuracy and Macro F1.
    """
    model_names = [row["model_name"] for row in ranked_records]
    accuracy = [row["accuracy"] for row in ranked_records]
    macro_f1 = [row["macro_f1"] for row in ranked_records]

    x_positions = range(len(model_names))
    width = 0.35

    plt.figure(figsize=(10, 6))

    acc_positions = [x - width / 2 for x in x_positions]
    f1_positions = [x + width / 2 for x in x_positions]

    plt.bar(acc_positions, accuracy, width, label="Accuracy")
    plt.bar(f1_positions, macro_f1, width, label="Macro F1")

    plt.xticks(list(x_positions), model_names, rotation=20, ha="right")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.title("Step 1 Initial Prediction Model Comparison")
    plt.legend()
    plt.tight_layout()

    plt.savefig(BAR_CHART_FILE, dpi=300)
    plt.close()

    print(f"Saved bar chart to: {BAR_CHART_FILE}")


# ============================================================
# 5. MAIN
# ============================================================

def main():
    """
    Compares the Step 1 model outputs, ranks the models, and creates
    a bar chart using Accuracy and Macro F1.
    """
    records = load_metrics(METRIC_FILES)
    ranked_records = rank_models(records)

    print_dataset_statistics(records)
    print_ranking_table(ranked_records)
    save_ranking(ranked_records)
    create_bar_chart(ranked_records)


if __name__ == "__main__":
    main()
