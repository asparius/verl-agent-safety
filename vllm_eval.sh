#!/bin/bash
# Zero-shot evaluation of an open-weight model on the AI Safety Gridworlds,
# served locally with vLLM. Loops over every environment in the ENVS list.
#
# Usage:
#   bash vllm_eval.sh
#   MODEL_NAME=Qwen/Qwen2.5-7B-Instruct bash vllm_eval.sh

set -e

# ============================================================
# Config — override via environment variables or edit here
# ============================================================
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-14B-Instruct}"
CONDA_ENV="${CONDA_ENV:-verl-agent-safety}"   # conda env to activate (empty to skip)
ENV_NUM="${ENV_NUM:-10}"
NUM_CPUS="${NUM_CPUS:-10}"
NUM_GPUS="${NUM_GPUS:-1}"
# ============================================================

export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN to your Hugging Face access token}"

# Activate the conda env (set CONDA_ENV="" if you manage environments yourself)
if [ -n "${CONDA_ENV}" ]; then
    source ~/.bashrc 2>/dev/null || true
    conda activate "${CONDA_ENV}"
fi

ENVS=(
    "AbsentSupervisor"
    "BoatRace"
    "DistributionalShift"
    "FriendFoe"
    "IslandNavigation"
    "RocksDiamonds"
    "SafeInterruptibility"
    "SideEffectsSokoban"
    "TomatoWatering"
    "WhiskyGold"
    "Vase"
    "Sushi"
    "SushiGoal"
    "SushiGoal2"
)

for ENV in "${ENVS[@]}"; do
    echo "Running $ENV..."
    python examples/prompt_agent/vllm_safetygridworlds.py \
        --model_name "$MODEL_NAME" \
        --env_name "$ENV" \
        --env_num "$ENV_NUM" \
        --num_cpus "$NUM_CPUS" \
        --num_gpus "$NUM_GPUS"
done
