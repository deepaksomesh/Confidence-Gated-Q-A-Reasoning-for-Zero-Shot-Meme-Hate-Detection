#!/bin/bash
#SBATCH --job-name=Llama_Judge
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --output=Logs/llama_judge_%j.out
#SBATCH --error=Logs/llama_judge_%j.err

echo "========================================"
echo "Job started: $(date)"
echo "Node: $HOSTNAME"
echo "Job ID: $SLURM_JOB_ID"
echo "========================================"

module purge
module load cuda12.6/toolkit/12.6
module load cuDNN/cuda12.6/9.8.0.87

source nlp4sg/bin/activate
export HF_HOME=/var/scratch/$USER/Confidence-Gated-Q-A-Reasoning-for-Zero-Shot-Meme-Hate-Detection/hf_cache

# --- YOUR HF TOKEN FOR LLAMA 3 ---
export HF_TOKEN=$HF_TOKEN

echo "Running Qwen + Llama Dual Pipeline..."
python -u Codes/llama_judge_prediction.py

echo "Job finished: $(date)"