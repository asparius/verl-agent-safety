#!/bin/bash
# Train a PPO (GAE + critic) agent on an AI Safety Gridworlds environment.
#
# Usage:
#   bash ppo_train.sh
#   ENV_NAME=WhiskyGold MODEL_SIZE=3B SEED=2 bash ppo_train.sh
#   bash ppo_train.sh env.history_length=4          # extra key=value args go to the trainer
#
# Per-model-size knobs (micro-batch sizes, offloading, GPU mem) are set automatically below.
# Checkpoints are written to ./checkpoints/<project>/<experiment>.

set -x

# ============================================================
# Config — override via environment variables or edit here
# ============================================================
ENV_NAME="${ENV_NAME:-IslandNavigation}"          # AI Safety Gridworlds environment
MODEL_SIZE="${MODEL_SIZE:-1.5B}"                  # 1.5B | 3B | 7B  (selects model + batch knobs)
SEED="${SEED:-11}"                                # change per run: 1, 2, 3, ...
N_GPUS="${N_GPUS:-4}"                             # GPUs on this node
CONDA_ENV="${CONDA_ENV:-verl-agent-safety}"       # conda env to activate (empty to skip)
PROJECT_NAME="${PROJECT_NAME:-verl_agent_safety}"
ENGINE="${ENGINE:-vllm}"                          # rollout engine: vllm | sglang
# ============================================================

export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN to your Hugging Face access token}"

# Activate the conda env (set CONDA_ENV="" if you manage environments yourself)
if [ -n "${CONDA_ENV}" ]; then
    source ~/.bashrc 2>/dev/null || true
    conda activate "${CONDA_ENV}"
fi

export NCCL_CUMEM_ENABLE=0
export VLLM_ENGINE_ITERATION_TIMEOUT_S=600
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_BLOCK_SIZE=256

ray stop || true

# ============================================================
# Derived from MODEL_SIZE — all knobs that scale with model size live here.
# ============================================================
MODEL_PATH="Qwen/Qwen2.5-${MODEL_SIZE}-Instruct"
# Lower-cased + dot-less tag for naming (1.5B -> 1_5b, 3B -> 3b, 7B -> 7b)
MODEL_TAG=$(echo "${MODEL_SIZE}" | tr 'A-Z' 'a-z' | tr '.' '_')

case "${MODEL_SIZE}" in
    "1.5B")
        ACTOR_PPO_MICRO_BS=16
        CRITIC_PPO_MICRO_BS=16
        LOG_PROB_MICRO_BS=16
        REF_LOG_PROB_MICRO_BS=16
        ROLLOUT_TP=1
        GPU_MEM_UTIL=0.6
        REF_PARAM_OFFLOAD=True
        CRITIC_PARAM_OFFLOAD=False
        CRITIC_OPT_OFFLOAD=False
        ;;
    "3B")
        ACTOR_PPO_MICRO_BS=16
        CRITIC_PPO_MICRO_BS=16
        LOG_PROB_MICRO_BS=16
        REF_LOG_PROB_MICRO_BS=16
        ROLLOUT_TP=1
        GPU_MEM_UTIL=0.55
        REF_PARAM_OFFLOAD=True
        CRITIC_PARAM_OFFLOAD=False
        CRITIC_OPT_OFFLOAD=False
        ;;
    "7B")
        ACTOR_PPO_MICRO_BS=16
        CRITIC_PPO_MICRO_BS=8
        LOG_PROB_MICRO_BS=16
        REF_LOG_PROB_MICRO_BS=16
        ROLLOUT_TP=1
        GPU_MEM_UTIL=0.5
        REF_PARAM_OFFLOAD=True
        CRITIC_PARAM_OFFLOAD=False
        CRITIC_OPT_OFFLOAD=False
        ;;
    *)
        echo "[fatal] Unknown MODEL_SIZE='${MODEL_SIZE}'. Expected 1.5B / 3B / 7B."
        exit 2
        ;;
esac

EXPERIMENT_NAME="${ENV_NAME}-ppo_qwen2.5_${MODEL_TAG}-200ep-hist_length_2-noexp-seed${SEED}"
echo "[setup] MODEL_SIZE=${MODEL_SIZE}  MODEL_PATH=${MODEL_PATH}  EXPERIMENT_NAME=${EXPERIMENT_NAME}"

CKPTS_DIR="checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}"
mkdir -p "${CKPTS_DIR}"

# ============================================================
# Data preparation
# ============================================================
train_data_size=16
val_data_size=64
num_cpus_per_env_worker=0.01

python -m examples.data_preprocess.prepare \
    --mode 'text' \
    --train_data_size $train_data_size \
    --val_data_size $val_data_size

# ============================================================
# Training (PPO with GAE + critic)
# ============================================================
python -m verl.trainer.main_ppo \
    ray_init.num_cpus=16 \
    algorithm.adv_estimator=gae \
    data.train_files=$HOME/data/verl-agent/text/train.parquet \
    data.val_files=$HOME/data/verl-agent/text/test.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=2048 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ACTOR_PPO_MICRO_BS} \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BS} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEM_UTIL} \
    actor_rollout_ref.rollout.name=$ENGINE \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.block_size=256 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${REF_LOG_PROB_MICRO_BS} \
    actor_rollout_ref.ref.fsdp_config.param_offload=${REF_PARAM_OFFLOAD} \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.model.path=${MODEL_PATH} \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size_per_gpu=${CRITIC_PPO_MICRO_BS} \
    critic.model.fsdp_config.param_offload=${CRITIC_PARAM_OFFLOAD} \
    critic.model.fsdp_config.optimizer_offload=${CRITIC_OPT_OFFLOAD} \
    algorithm.use_kl_in_reward=False \
    env.env_name=${ENV_NAME} \
    env.seed=${SEED} \
    env.max_steps=50 \
    env.history_length=2 \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node=${N_GPUS} \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=200 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.val_before_train=True "$@"
