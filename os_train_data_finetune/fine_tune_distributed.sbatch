#!/bin/bash

#SBATCH --ntasks-per-node=1
#SBATCH --nodes=8
#SBATCH --mem-per-cpu=12G
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:4
#SBATCH -A p_scads_llm_secrets
#SBATCH --time=12:00:00

module load release/24.04  GCC/12.3.0  OpenMPI/4.1.5 PyTorch-bundle

source ~/s9650707-llm_workspace/capella_env/bin/activate


python list_gpus.py

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=29501  # Choose a free port


WORLD_SIZE=$(( SLURM_NNODES * SLURM_GPUS_ON_NODE ))


# Launch training
srun torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=$SLURM_GPUS_ON_NODE \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    finetune_llama_distributed.py




#python finetune_llama.py
