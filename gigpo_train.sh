#!/bin/bash
# Train a GiGPO agent on an AI Safety Gridworlds environment.
#
# Usage:
#   bash gigpo_train.sh
#   ENV_NAME=WhiskyGold MODEL_PATH=Qwen/Qwen2.5-7B-Instruct bash gigpo_train.sh
#   bash gigpo_train.sh algorithm.gamma=0.99        # extra key=value args go to the trainer
#
# Checkpoints are written to ./checkpoints/<project>/<experiment>.

set -x

# ============================================================
# Config — override via environment variables or edit here
# ============================================================
ENV_NAME="${ENV_NAME:-AbsentSupervisor}"                 # AI Safety Gridworlds environment
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-3B-Instruct}"     # base model
N_GPUS="${N_GPUS:-4}"                                    # GPUs on this node
CONDA_ENV="${CONDA_ENV:-verl-agent-safety}"              # conda env to activate (empty to skip)
PROJECT_NAME="${PROJECT_NAME:-verl_agent_safety}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-${ENV_NAME}-gigpo-$(basename "${MODEL_PATH}")}"
SEED="${SEED:-1}"
ENGINE="${ENGINE:-vllm}"                                 # rollout engine: vllm | sglang
# ============================================================

export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN to your Hugging Face access token}"

# Activate the conda env (set CONDA_ENV="" if you manage environments yourself)
if [ -n "${CONDA_ENV}" ]; then
    source ~/.bashrc 2>/dev/null || true
    conda activate "${CONDA_ENV}"
fi

export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_BLOCK_SIZE=256

ray stop || true

CKPTS_DIR="checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}"
mkdir -p "${CKPTS_DIR}"

num_cpus_per_env_worker=0.01  # CPU allocation per env worker
train_data_size=16
val_data_size=64
group_size=4

# Data preparation
python -m examples.data_preprocess.prepare \
    --mode 'text' \
    --train_data_size $train_data_size \
    --val_data_size $val_data_size

python -m verl.trainer.main_ppo \
    ray_init.num_cpus=16 \
    algorithm.adv_estimator=gigpo \
    algorithm.gamma=0.95 \
    algorithm.gigpo.step_advantage_w=1.0 \
    algorithm.gigpo.mode=mean_norm \
    data.train_files=$HOME/data/verl-agent/text/train.parquet \
    data.val_files=$HOME/data/verl-agent/text/test.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=16384 \
    data.max_response_length=16384 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.name=$ENGINE \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.block_size=256 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    env.env_name=${ENV_NAME} \
    env.seed=${SEED} \
    env.max_steps=50 \
    env.history_length=20 \
    env.rollout.n=$group_size \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node=${N_GPUS} \
    trainer.nnodes=1 \
    trainer.save_freq=200 \
    trainer.test_freq=5 \
    trainer.total_epochs=200 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.val_before_train=True "$@"
