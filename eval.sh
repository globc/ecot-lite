# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
. ~/.bashrc

source ~/miniconda3/bin/activate
conda activate ecot-lite

nvidia-smi

export PRISMATIC_DATA_ROOT=/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/cb14syta/ecot-lite/data/embodied_features_and_demos_libero

export WANDB_API_KEY="WANDB_API_KEY"
export MUJOCO_GL=glx

Xvfb :99 -screen 0 1920x1080x24 &
export DISPLAY=:99

torchrun --nproc-per-node 4 experiments/robot/libero/run_libero_eval.py \
  --model_family prismatic \
  --pretrained_checkpoint /pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/cb14syta/ecot-lite/out/prism-qwen25-dinosiglip-224px-wrist+0_5b+mx-libero-90+n1+b16+x7/checkpoints/latest-checkpoint.pt \
  --task_suite_name libero_10 \
  --center_crop True \
  --use_wrist_image True \
  --num_trials_per_task 1 \
  --distributed True \
#  --subset_size 30 \

conda deactivate