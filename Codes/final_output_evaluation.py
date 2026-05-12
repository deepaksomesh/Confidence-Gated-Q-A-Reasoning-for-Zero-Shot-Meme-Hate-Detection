import os
import json
import pandas as pd
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

STEP1_METRICS_FILE = os.path.join(
    OUTPUT_DIR,
    "Step1_qwen3_initial_metrics.json"
)

STEP2_SUMMARY_FILE = os.path.join(
    OUTPUT_DIR,
    "Step2_qwen3_ablation_summary.json"
)

RESULTS_DIR = os.path.join(OUTPUT_DIR, "Step3_final_results")
TABLE_DIR = os.path.join(RESULTS_DIR, "tables")
PLOT_DIR = os.path.join(RESULTS_DIR, "plots")

os.makedirs(TABLE_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)


# ============================================================
# 2. LOADING
# ============================================================

def load_json(file_path: str):
    """
    Loads a JSON file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Missing file: {file_path}")

    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


# ============================================================
# 3. TABLE CREATION
# ============================================================

def create_ablation_table(summary: list) -> pd.DataFrame:
    """
    Creates the full refinement ablation table.
    """
    rows = []

    for item in summary:
        rows.append(
            {
                "Technique": item["technique"],
                "Threshold": item["confidence_threshold"],
                "Accuracy": item["accuracy"],
                "Macro F1": item["macro_f1"],
                "Hateful F1": item["hateful_f1"],
                "Hateful Recall": item["hateful_recall"],
                "Refined Count": item["refined_count"],
                "Refinement Rate": item["refinement_rate"],
                "Changed Predictions": item["changed_prediction_count"],
                "Net Correction Gain": item["net_correction_gain"],
            }
        )

    table = pd.DataFrame(rows)

    table = table.sort_values(
        by=["Accuracy", "Macro F1"],
        ascending=False,
    ).reset_index(drop=True)

    table.insert(0, "Rank", range(1, len(table) + 1))

    return table


def create_initial_vs_best_table(step1_metrics: dict, ablation_table: pd.DataFrame) -> pd.DataFrame:
    """
    Compares the initial Qwen3 result with the best refinement result.
    """
    best = ablation_table.iloc[0]

    rows = [
        {
            "System": "Initial Qwen3-VL",
            "Technique": "Direct Prompt",
            "Threshold": "N/A",
            "Accuracy": step1_metrics["accuracy"],
            "Macro F1": step1_metrics["macro_f1"],
            "Hateful F1": step1_metrics["class_metrics"]["hateful"]["f1"],
            "Hateful Recall": step1_metrics["class_metrics"]["hateful"]["recall"],
            "Refined Count": 0,
            "Refinement Rate": 0.0,
            "Net Correction Gain": 0,
        },
        {
            "System": "Best Confidence-Gated",
            "Technique": best["Technique"],
            "Threshold": best["Threshold"],
            "Accuracy": best["Accuracy"],
            "Macro F1": best["Macro F1"],
            "Hateful F1": best["Hateful F1"],
            "Hateful Recall": best["Hateful Recall"],
            "Refined Count": best["Refined Count"],
            "Refinement Rate": best["Refinement Rate"],
            "Net Correction Gain": best["Net Correction Gain"],
        },
    ]

    table = pd.DataFrame(rows)

    return table


def save_tables(ablation_table: pd.DataFrame, comparison_table: pd.DataFrame):
    """
    Saves result tables as CSV and LaTeX.
    """
    ablation_csv = os.path.join(TABLE_DIR, "Step3_refinement_ablation_table.csv")
    comparison_csv = os.path.join(TABLE_DIR, "Step3_initial_vs_best_table.csv")

    ablation_tex = os.path.join(TABLE_DIR, "Step3_refinement_ablation_table.tex")
    comparison_tex = os.path.join(TABLE_DIR, "Step3_initial_vs_best_table.tex")

    ablation_table.to_csv(ablation_csv, index=False)
    comparison_table.to_csv(comparison_csv, index=False)

    ablation_table.to_latex(
        ablation_tex,
        index=False,
        float_format="%.4f",
    )

    comparison_table.to_latex(
        comparison_tex,
        index=False,
        float_format="%.4f",
    )

    print(f"Saved ablation CSV   : {ablation_csv}")
    print(f"Saved comparison CSV : {comparison_csv}")
    print(f"Saved ablation LaTeX : {ablation_tex}")
    print(f"Saved comparison LaTeX: {comparison_tex}")


# ============================================================
# 4. PLOT CREATION
# ============================================================

def create_main_bar_chart(comparison_table: pd.DataFrame):
    """
    Creates one compact bar chart comparing initial and best refinement.
    """
    plot_data = comparison_table[["System", "Accuracy", "Macro F1"]].copy()

    x = range(len(plot_data))
    width = 0.35

    plt.figure(figsize=(7, 4))
    plt.bar(
        [i - width / 2 for i in x],
        plot_data["Accuracy"],
        width,
        label="Accuracy",
    )
    plt.bar(
        [i + width / 2 for i in x],
        plot_data["Macro F1"],
        width,
        label="Macro F1",
    )

    plt.xticks(x, plot_data["System"], rotation=0)
    plt.ylim(0.65, 0.75)
    plt.ylabel("Score")
    plt.title("Initial Prediction vs. Best Confidence-Gated Refinement")
    plt.legend()
    plt.tight_layout()

    output_file = os.path.join(PLOT_DIR, "Step3_initial_vs_best_bar_chart.png")
    plt.savefig(output_file, dpi=300)
    plt.close()

    print(f"Saved bar chart      : {output_file}")


def create_threshold_plot(ablation_table: pd.DataFrame):
    """
    Creates one line plot showing accuracy across thresholds for each refinement strategy.
    """
    plt.figure(figsize=(7, 4))

    for technique in sorted(ablation_table["Technique"].unique()):
        subset = ablation_table[ablation_table["Technique"] == technique].sort_values("Threshold")
        plt.plot(
            subset["Threshold"],
            subset["Accuracy"],
            marker="o",
            label=technique,
        )

    plt.xlabel("Confidence Threshold")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Across Confidence Thresholds")
    plt.legend()
    plt.tight_layout()

    output_file = os.path.join(PLOT_DIR, "Step3_threshold_accuracy_plot.png")
    plt.savefig(output_file, dpi=300)
    plt.close()

    print(f"Saved threshold plot : {output_file}")


# ============================================================
# 5. PRINT SUMMARY
# ============================================================

def print_summary(step1_metrics: dict, ablation_table: pd.DataFrame):
    """
    Prints the main results needed for the report.
    """
    best = ablation_table.iloc[0]

    initial_acc = step1_metrics["accuracy"]
    initial_f1 = step1_metrics["macro_f1"]

    best_acc = best["Accuracy"]
    best_f1 = best["Macro F1"]

    print("\n" + "=" * 100)
    print("FINAL RESULT SUMMARY")
    print("=" * 100)

    print(f"{'Initial Qwen3 Accuracy':<35}: {initial_acc:.4f}")
    print(f"{'Initial Qwen3 Macro F1':<35}: {initial_f1:.4f}")

    print("\nBest confidence-gated run")
    print("-" * 100)
    print(f"{'Technique':<35}: {best['Technique']}")
    print(f"{'Threshold':<35}: {best['Threshold']:.2f}")
    print(f"{'Accuracy':<35}: {best_acc:.4f}")
    print(f"{'Macro F1':<35}: {best_f1:.4f}")
    print(f"{'Hateful F1':<35}: {best['Hateful F1']:.4f}")
    print(f"{'Hateful Recall':<35}: {best['Hateful Recall']:.4f}")
    print(f"{'Refined Count':<35}: {int(best['Refined Count'])}")
    print(f"{'Refinement Rate':<35}: {best['Refinement Rate']:.4f}")
    print(f"{'Net Correction Gain':<35}: {int(best['Net Correction Gain'])}")

    print("\nImprovement over initial Qwen3")
    print("-" * 100)
    print(f"{'Accuracy gain':<35}: {best_acc - initial_acc:+.4f}")
    print(f"{'Macro F1 gain':<35}: {best_f1 - initial_f1:+.4f}")


# ============================================================
# 6. MAIN
# ============================================================

def main():
    """
    Creates final tables and plots for the report.
    """
    print("Loading Step 1 and Step 2 results")

    step1_metrics = load_json(STEP1_METRICS_FILE)
    step2_summary = load_json(STEP2_SUMMARY_FILE)

    ablation_table = create_ablation_table(step2_summary)
    comparison_table = create_initial_vs_best_table(step1_metrics, ablation_table)

    save_tables(ablation_table, comparison_table)

    create_main_bar_chart(comparison_table)
    create_threshold_plot(ablation_table)

    print_summary(step1_metrics, ablation_table)


if __name__ == "__main__":
    main()
