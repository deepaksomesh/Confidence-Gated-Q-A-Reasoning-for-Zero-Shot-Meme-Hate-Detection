#!/bin/bash
#SBATCH --job-name=initial_prediction
#SBATCH --output=/home/s4577663/NLP/Logs/initial_prediction_%j.out
#SBATCH --error=/home/s4577663/NLP/Logs/initial_prediction_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --partition=gpu-mig-40g
#SBATCH --gres=gpu:4g.40gb:1

echo "========================================"
echo "Job started: $(date)"
echo "Node: $HOSTNAME"
echo "Job ID: $SLURM_JOB_ID"
echo "========================================"

module purge
module load ALICE/default
module load CUDA/12.4.0
module load Miniconda3/24.7.1-0

# 1. Install GPU dependencies
source /easybuild/software/Miniconda3/24.7.1-0/etc/profile.d/conda.sh
conda activate base
conda activate NLP_env
export PATH="/home/s4577663/.conda/envs/NLP_env/bin:$PATH"

echo "Python: $(which python)"

cd /home/s4577663/NLP/Codes

# 1. Code execution
echo "Running initial Hate meme prediction .."
python -u initial_prediction.py

echo "Job finished: $(date)"

