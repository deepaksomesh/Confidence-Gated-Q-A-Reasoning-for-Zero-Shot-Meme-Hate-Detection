#!/bin/bash
#SBATCH --job-name=v3_initial_prediction
#SBATCH --output=/home/s4374827/NLP/Logs/initial_prediction_internvl_l4_%j.out
#SBATCH --error=/home/s4374827/NLP/Logs/initial_prediction_internvl_l4_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=12:00:00
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
conda activate nlp_env
export PATH="/home/s4374827/.conda/envs/nlp_env/bin:$PATH"

echo "Python: $(which python)"

cd /home/s4374827/NLP/Codes

# 1. Code execution
echo "Running initial Hate meme prediction using InternVL .."
python -u initial_prediction_internvl.py

echo "Job finished: $(date)"

