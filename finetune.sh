#!/bin/bash
#!/bin/bash -l
#
#SBATCH -e test.err
#SBATCH -o test.out

#SBATCH -n 1 # 1 process
#SBATCH -c 4 # 4 CPU cores per process

#SBATCH --time=03:00:00

#SBATCH --gres=gpu:1
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CUDA_NUM_DEVICES=$SLURM_GPUS_ON_NODE

# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source ~/miniconda3/bin/activate
conda activate ecot-lite


nvidia-smi

export PRISMATIC_DATA_ROOT=/mnt/beegfs/hdd/mirror/home/cb14syta/ecot-lite/data/embodied_features_and_demos_libero

echo "HF_TOKEN" > .hf_token
chmod 600 .hf_token

torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/train.py \
  --vla.type "prism-qwen25-dinosiglip-224px-wrist+0_5b+mx-libero-90" \
  --data_root_dir /mnt/beegfs/hdd/mirror/home/cb14syta/ecot-lite/data/embodied_features_and_demos_libero \
  --run_root_dir /mnt/beegfs/hdd/mirror/home/cb14syta/ecot-lite/out \
  --vla.expected_world_size 1 \
  --vla.action_tokenizer "libero_vq_extra_action_tokenizer" \
  --vla.per_device_batch_size 2 \
  --vla.global_batch_size 2 \
  --vla.data_mix "libero_lm_90" \
  --vla.use_wrist_image True \
  --wandb_project "ecot-lite" \
  --wandb_entity "christian-bialas-tu-darmstadt"

conda deactivate