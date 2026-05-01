#!/bin/bash
#SBATCH --job-name=Ablation_Meme
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --output=Logs/ablation_prediction_%j.out
#SBATCH --error=Logs/ablation_prediction_%j.err

echo "========================================"
echo "Job started: $(date)"
echo "Node: $HOSTNAME"
echo "Job ID: $SLURM_JOB_ID"
echo "========================================"

# 1. Load the exact CUDA and cuDNN modules for DAS-6
module purge
module load cuda12.6/toolkit/12.6
module load cuDNN/cuda12.6/9.8.0.87

# 2. Activate your private virtual environment
source nlp4sg/bin/activate
export HF_HOME=/var/scratch/$USER/Confidence-Gated-Q-A-Reasoning-for-Zero-Shot-Meme-Hate-Detection/hf_cache

# 3. Prove the GPU is awake!
echo "--- GPU CHECK ---"
nvidia-smi
python -c "import torch; print('PyTorch CUDA Available:', torch.cuda.is_available())"
echo "-----------------"

# 4. Run the Python script unbuffered
echo "Running Confidence-Gated Ablation Pipeline..."
python -u Codes/ablation_prediction.py

echo "Job finished: $(date)"