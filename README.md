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

## Results

Evaluated on a fixed 3,000-example English subset of the Hateful Memes benchmark
(1,064 hateful, 1,936 non-hateful). Strict zero-shot throughout — no fine-tuning,
no parameter updates.

**Full write-up, with the prompts, ablations and qualitative analysis:**
[`report.pdf`](report.pdf)


### Stage 1: which VLM

| Model | Accuracy | Macro F1 | Hateful F1 | Hateful recall |
|---|---|---|---|---|
| Qwen3-VL-8B-Instruct | **0.7177** | 0.7027 | 0.6360 | 0.6955 |
| InternVL3-8B | 0.7057 | **0.7031** | **0.6755** | **0.8637** |
| LLaVA-OneVision-Qwen2-7B | 0.6497 | 0.6366 | 0.5677 | 0.6485 |

The two leading models trade off against each other rather than one dominating.
Qwen3-VL wins on accuracy; InternVL3 is better on every hateful-class metric,
with recall 17 points higher. Qwen3-VL was selected because the class imbalance
(65% non-hateful) mirrors real moderation traffic, where overall reliability
matters alongside catching the positive class — but InternVL3 would likely be
the better base model if hateful-class recall were the priority.

### Stage 2: which refinement strategy

Best threshold per strategy. *Refined* is how many of the 3,000 examples were
routed to refinement; *gain* is net corrections minus new errors introduced.

| Strategy | Threshold | Accuracy | Macro F1 | Refined | Net gain |
|---|---|---|---|---|---|
| Atomic Fact Extraction | 0.95 | **0.7230** | **0.7109** | 653 | **+16** |
| Chain-of-Thought | 0.60 | 0.7187 | 0.7049 | 371 | +3 |
| Dynamic QA | 0.85 | 0.7173 | 0.7063 | 495 | −1 |
| Static QA | 0.75 | 0.7150 | 0.7070 | 426 | −8 |
| Direct prompting (no refinement) | — | 0.7177 | 0.7027 | 0 | — |

Atomic Fact Extraction is the only strategy that beats the baseline on both
metrics, and the only one with a clearly positive net gain. It leads at every
threshold tested.

**Three of the four strategies fail to help, and two actively hurt.** That is
the more useful finding. Static QA reaches the highest hateful-class recall in
the whole study (0.7961 at t=0.95) while posting the worst net gain (−17): it
changes 231 predictions and gets more of them wrong than right. Repeatedly
asking a model whether something is hostile appears to push it toward answering
yes. Dynamic QA and Chain-of-Thought stay close to the baseline and mostly add
computation.

The difference seems to be what the strategy asks for. Atomic Fact Extraction
separates observation from judgment — list what is in the meme, then explain how
those facts interact, then decide. The QA strategies ask about hostility
directly at every step, which frames the task as looking for hate rather than
looking at the meme.

### Stage 3: where the threshold should sit

More refinement is not better. At the best-performing threshold (t=0.95) only
21.77% of examples are routed to refinement, and the remaining 78% keep their
direct prediction. For the weaker strategies, raising the threshold changes more
predictions without improving accuracy — Static QA gets monotonically worse as
it refines more.

The confidence gate therefore does two jobs: it saves computation, and it
prevents reasoning from overturning predictions that were already right.

### A case the refinement gets right

One hateful meme relies on a slur that reads as an ordinary technical term in
context, so the direct prediction interprets it as harmless mechanical wordplay
and returns non-hateful with low confidence (0.148). Atomic Fact Extraction
lists the three panel contents as separate facts, which surfaces that the final
panel shows a named person, and the model then connects the wordplay to a
gender-identity slur and flips to hateful.

The gate matters here: at 0.148 confidence this example was always going to be
refined. That is the pattern the pipeline is built around — the model's
uncertainty is a usable signal for where extra reasoning pays off.

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
