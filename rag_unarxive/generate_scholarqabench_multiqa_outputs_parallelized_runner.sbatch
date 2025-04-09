#!/bin/bash

#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=12G
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH -A p_scads_llm_secrets
#SBATCH --time=2:00:00

source ~/.bashrc
set_capella_env

python -u generate_scholarqabench_multiqa_outputs_parallelized_runner.py