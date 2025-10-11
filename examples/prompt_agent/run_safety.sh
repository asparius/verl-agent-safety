#!/bin/bash

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
    python examples/prompt_agent/vllm_safetygridworlds.py --env_name "$ENV"
done
