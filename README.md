# Confidence-Gated Q&A Reasoning for Zero-Shot Meme Hate Detection

This project investigates a novel approach to zero-shot meme hate detection by utilizing a **Confidence-Gated Reasoning Pipeline**. Given the subjective and nuanced nature of multimodal memes (image + text), this method first generates an initial zero-shot prediction using Vision-Language Models (VLMs) and evaluates its confidence. For predictions falling below a set confidence threshold, a refinement step is triggered where the model engages in structured reasoning strategies (e.g., Chain-of-Thought, Atomic Fact extraction, QA) to reassess the meme.

## Project Structure

The repository is organized into the following key directories and files:

```text
Confidence-Gated-Q-A-Reasoning-for-Zero-Shot-Meme-Hate-Detection/
├── Codes/                              
│   ├── initial_prediction_internvl.py   # Step 1 script using InternVL3
│   ├── initial_prediction_llava.py      # Step 1 script using LLaVA-OneVision
│   ├── initial_prediction_qwen.py       # Step 1 script using Qwen3-VL
│   ├── vlm_comparison.py                # Compares initial VLM predictions
│   ├── confidence_guided_refinement.py  # Step 2 script for confidence-gated reasoning ablation
│   ├── final_output_evaluation.py       # Step 3 script to generate final tables and plots
│   └── Output/                          # Auto-generated outputs, logs, metrics, and plots
├── requirements.txt                     
└── README.md                            
```

## Setup Instructions

### Prerequisites
- Python 3.9+
- An environment with GPU support is highly recommended for running Vision-Language Models.

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd Confidence-Gated-Q-A-Reasoning-for-Zero-Shot-Meme-Hate-Detection
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate      # On Windows: venv\Scripts\activate
   ```

3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *Note: This will install necessary packages like `torch`, `transformers`, `accelerate`, `datasets`, and `qwen-vl-utils`.*

## Pipeline Overview & Usage

The evaluation pipeline is split into three main steps:

### Step 1: Initial Predictions
The first step evaluates several VLMs on the `emily49/hateful_memes_train_dev` dataset in a direct zero-shot manner, calculating likelihood-based confidence scores.

To run the initial predictions:
```bash
python Codes/initial_prediction_qwen.py
python Codes/initial_prediction_internvl.py
python Codes/initial_prediction_llava.py
```
After executing the models, you can compare their initial baseline performance:
```bash
python Codes/vlm_comparison.py
```

### Step 2: Confidence-Gated Refinement
Based on the best-performing model, Qwen3-VL has the best accuracy. This step runs an ablation over different **confidence thresholds** (e.g., 0.60, 0.75, 0.85, 0.95) and **reasoning techniques** using the Qwen3-VL model:
- `atomic`: Extracting visual/textual facts before deciding.
- `static_qa`: Answering predefined target/hostility questions.
- `dynamic_qa`: Dynamically generating relevant questions to answer.
- `cot`: Chain-of-Thought reasoning.

To run the refinement step:
```bash
python Codes/confidence_guided_refinement.py
```
*This script will use predictions with confidence below the threshold and apply the reasoning prompts to refine the final output.*

### Step 3: Final Output Evaluation
This step aggregates the outputs from Step 1 and Step 2 to generate comparative plots, accuracy/macro-F1 tables, and CSV tables for reporting.

To generate the final results:
```bash
python Codes/final_output_evaluation.py
```

## Output & Results
Running the pipeline will populate the `Codes/Output` directory with:
- `jsonl` files containing the predictions, confidence scores, and reasoning text for each meme.
- `json` files containing detailed class-wise metrics, macro-F1, and accuracy scores.
- Summarized tables inside `Codes/Output/Step3_final_results/tables/`.
- Visualizations inside `Codes/Output/Step3_final_results/plots/`.
