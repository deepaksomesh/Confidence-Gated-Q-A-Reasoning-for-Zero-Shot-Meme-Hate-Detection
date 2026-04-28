#!/bin/bash
#SBATCH --job-name=Atomic_Meme
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --output=Logs/atomic_prediction_%j.out
#SBATCH --error=Logs/atomic_prediction_%j.err

# Load the required modules on DAS-6
module load python/3.10.10
module load cuda/12.6

# Activate your specific virtual environment
source nlp4sg/bin/activate

# Execute the script from the root of the repo
python Codes/atomic_prediction.py