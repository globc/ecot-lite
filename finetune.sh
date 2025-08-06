# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
. ~/.bashrc

source ~/miniconda3/bin/activate
conda activate ecot-lite


nvidia-smi

export PRISMATIC_DATA_ROOT=/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/cb14syta/ecot-lite/data/embodied_features_and_demos_libero

export WANDB_API_KEY="WANDB_API_KEY"

# echo "HF_TOKEN" > .hf_token
# chmod 600 .hf_token

torchrun --nproc-per-node 8 vla-scripts/train.py \
  --vla.type "prism-qwen25-dinosiglip-224px-wrist+0_5b+mx-libero-90" \
  --data_root_dir /pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/cb14syta/ecot-lite/data/embodied_features_and_demos_libero \
  --run_root_dir /pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/cb14syta/ecot-lite/out \
  --vla.expected_world_size 8 \
  --vla.action_tokenizer "libero_vq_extra_action_tokenizer" \
  --vla.per_device_batch_size 16 \
  --vla.global_batch_size 128 \
  --vla.max_steps 100000 \
  --vla.data_mix "libero_lm_90" \
  --vla.use_wrist_image True \
  --wandb_project "ecot-lite" \
  --wandb_entity "christian-bialas-tu-darmstadt"

conda deactivate