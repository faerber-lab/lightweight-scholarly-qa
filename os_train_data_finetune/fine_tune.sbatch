#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=12G
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH -A p_scads_llm_secrets
#SBATCH --time=12:00:00

module load release/24.04  GCC/12.3.0  OpenMPI/4.1.5 PyTorch-bundle

source ~/s9650707-llm_workspace/capella_env/bin/activate

python finetune_llama.py
