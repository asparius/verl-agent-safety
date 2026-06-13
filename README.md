# Reward Hacking in Language Model Agents: Revisiting AI Safety Gridworlds

> Code for the paper **"Reward Hacking in Language Model Agents: Revisiting AI Safety Gridworlds"**
> by Ömer Veysel Çağatan (KUIS AI Center, Koç University) and Xuandong Zhao (UC Berkeley).
>
> 📄 Paper: preprint coming soon

This repository extends [`verl-agent`](https://github.com/langfengQ/verl-agent) (itself built on
[veRL](https://github.com/volcengine/verl)) with a text-based reformulation of the
[AI Safety Gridworlds](https://github.com/google-deepmind/ai-safety-gridworlds) suite, used to study
**reward hacking / specification gaming** in language-model agents — both zero-shot and under RL fine-tuning.

---

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
  - [1. Create the conda environment](#1-create-the-conda-environment)
  - [2. AI Safety Gridworlds environment](#2-ai-safety-gridworlds-environment)
  - [3. veRL / verl-agent training stack](#3-verl--verl-agent-training-stack)
  - [4. API keys](#4-api-keys)
- [Environments](#environments)
- [Quickstart](#quickstart)
- [Running Experiments](#running-experiments)
  - [Zero-shot / prompt-based evaluation](#zero-shot--prompt-based-evaluation)
  - [RL training (PPO / GRPO / GiGPO)](#rl-training-ppo--grpo--gigpo)
  - [Evaluating a trained checkpoint](#evaluating-a-trained-checkpoint)
- [Reproducing Paper Results](#reproducing-paper-results)
- [Repository Layout](#repository-layout)
- [Acknowledgements](#acknowledgements)

---

## Overview

<!-- TODO: 2-3 sentence summary of the experimental setup. Draft below — edit freely. -->

We adapt the classic AI Safety Gridworlds RL benchmarks into a text-based evaluation suite for
language-model agents. Each environment exposes an **observed (proxy) reward** that the agent optimizes
and a **hidden safety reward** that measures the intended behavior. We measure the gap between the two
across model scales (1.5B–14B), under both zero-shot prompting and RL fine-tuning.

---

## Installation

> Tested on Linux with CUDA 12.x and Python 3.11. <!-- TODO: confirm CUDA version -->

Installation order matters because **vLLM** pins its own `torch`/CUDA stack and will override existing
versions. Recommended order: create the conda env → **install vLLM first** (see the note in Step 3) →
install the **AI Safety Gridworlds environment stack** (vendored sources, in dependency order) → install
the **veRL / verl-agent training stack**.

### 1. Create the conda environment

```bash
git clone https://github.com/asparius/verl-agent-safety.git
cd verl-agent-safety

conda create -n verl-agent-safety python=3.11 -y
conda activate verl-agent-safety
```

### 2. AI Safety Gridworlds environment

The environment packages are vendored under
`agent_system/environments/env_package/safe_gridworlds/`. Install them **in this order** (each depends on
the previous one): `pycolab` → `ai-safety-gridworlds` → `safe-grid-gym`.

```bash
# Path to the vendored environment sources
SGW=agent_system/environments/env_package/safe_gridworlds/safe-grid-gym

# 1. pycolab (the gridworld game engine)
pip install -e $SGW/ai-safety-gridworlds/pycolab

# 2. ai-safety-gridworlds (the classic safety tasks)
pip install -e $SGW/ai-safety-gridworlds

# 3. safe-grid-gym (the Gym wrapper around the gridworlds)
pip install -e $SGW
```

> **Note:** <!-- TODO: any platform caveats (e.g. ARM/Mac notes — see the README_INSTALL_ARM-MAC.md in the webshop env). -->

### 3. veRL / verl-agent training stack

Install the training dependencies, then the repo itself in editable mode:

```bash
# Pinned dependency set used for the paper (verl, vLLM stack, transformers, etc.)
pip install -r requirements_safety.txt

# Install verl-agent itself
pip install -e .
```

> **FlashAttention.** `flash_attn` is commented out in `requirements_safety.txt` because the right wheel
> depends on your exact PyTorch / CUDA / Python combination. If the pre-built wheels do not match your
> setup, build it from source:
>
> ```bash
> pip install flash_attn==2.7.4.post1 --no-build-isolation
> ```
>
> ⚠️ **Warning:** building FlashAttention from source compiles CUDA kernels and **can take a few hours**.
> Run it in a detached session (`tmux` / `screen`) or as a batch job and be patient — it is not hung.

> **vLLM.** `vllm` is also commented out in `requirements_safety.txt` and should be **installed
> separately**, because installing it pulls in its own pinned `torch` (and CUDA deps) and will often
> **override** whatever is already in your environment. To avoid surprises, install vLLM **first** — right
> after creating the conda env — so its `torch` pin is the one everything else builds on:
>
> ```bash
> pip install vllm==0.10.0
> ```
>
> Then install the gridworlds stack (Step 2) and `requirements_safety.txt` / `pip install -e .` (Step 3)
> on top of it. If a later step downgrades `torch`, reinstall vLLM afterwards.

> `safenv_requirements.txt` is a lighter dependency set for **environment-only / prompt-based** runs (no
> training). Use it instead of `requirements_safety.txt` if you only need to run zero-shot evaluation.

### 4. API keys

For prompt-based (closed-model) experiments, set the relevant keys as environment variables — they are
**never** hardcoded in this repo:

```bash
export OPENAI_API_KEY=...        # GPT-4o prompt agent
export OPENROUTER_API_KEY=...    # OpenRouter-hosted models (optional)
export HF_TOKEN=...              # Hugging Face model downloads
export WANDB_API_KEY=...         # Experiment logging (optional)
```

---

## Environments

The following AI Safety Gridworlds tasks are available (see `agent_system/environments/env_package/safe_gridworlds/`):

| Env name | Safety property tested |
| - | - |
| `AbsentSupervisor`    | <!-- TODO --> |
| `BoatRace`            | <!-- TODO --> |
| `DistributionalShift` | <!-- TODO --> |
| `FriendFoe`           | <!-- TODO --> |
| `IslandNavigation`    | <!-- TODO --> |
| `RocksDiamonds`       | <!-- TODO --> |
| `SafeInterruptibility`| <!-- TODO --> |
| `SideEffectsSokoban`  | <!-- TODO --> |
| `TomatoWatering`      | <!-- TODO --> |
| `WhiskyGold`          | <!-- TODO --> |
| `Vase`                | <!-- TODO --> |
| `Sushi` / `SushiGoal` | <!-- TODO --> |

<!-- TODO: trim this table to the envs actually used in the paper. -->

---

## Quickstart

Run a single environment, zero-shot, with an open model served by vLLM:

```bash
python examples/prompt_agent/vllm_safetygridworlds.py \
    --model_name Qwen/Qwen2.5-1.5B-Instruct \
    --env_name AbsentSupervisor \
    --env_num 10 \
    --num_cpus 10 \
    --num_gpus 1
```

---

## Running Experiments

### Zero-shot / prompt-based evaluation

Zero-shot evaluation runs an un-trained model as an agent on the gridworlds. There are **two backends**,
depending on whether the model is open-weight (served locally with vLLM) or closed (accessed over an API).

#### A. Open models — local vLLM

Use `vllm_safetygridworlds.py`. It loads the model with vLLM on your own GPU(s) — no API key needed (only
`HF_TOKEN` if the model is gated).

```bash
python examples/prompt_agent/vllm_safetygridworlds.py \
    --model_name Qwen/Qwen2.5-1.5B-Instruct \
    --env_name AbsentSupervisor \
    --env_num 10 \
    --max_steps 50 \
    --temperature 0.0 \
    --tensor_parallel_size 1 \
    --gpu_memory_utilization 0.9 \
    --num_cpus 10 --num_gpus 1
```

To sweep several environments, list them in the `ENVS` array in `run_safety.sh` and run the wrapper:

```bash
bash examples/prompt_agent/run_safety.sh
```

Useful flags (defaults in parentheses): `--env_name` (`IslandNavigation`), `--model_name`
(`Qwen/Qwen2.5-3B`), `--env_num` (`10`), `--num_seeds` (`5`), `--episodes_per_seed` (`10`), `--max_steps`
(`50`), `--temperature` (`0.0`), `--n_samples` (`1`), `--tensor_parallel_size` (`1`),
`--gpu_memory_utilization` (`0.9`), `--max_model_len` (`None`).

#### B. Closed models — OpenAI API

Use `async_api_safegridworlds.py`. It reads the key from the environment (`OPENAI_API_KEY`) and hits the
OpenAI API directly (no GPU required), running many episodes concurrently:

```bash
export OPENAI_API_KEY=...

# Defaults to gpt-4o-mini
python examples/prompt_agent/async_api_safegridworlds.py \
    --env_name IslandNavigation --model_name gpt-4o --env_num 10
```

Flags: `--env_name`, `--model_name`, `--env_num`, `--num_seeds`, `--episodes_per_seed`, `--max_steps`,
`--base_seed`. The script writes a per-run log named `<env_name>_<model_name>_<timestamp>.log`.

> **Other API providers (e.g. OpenRouter).** The script above targets the OpenAI endpoint. To use an
> OpenAI-compatible provider such as OpenRouter, point the client at its base URL and use that provider's
> key — pass `base_url="https://openrouter.ai/api/v1"` to the `AsyncOpenAI(...)` constructor and set
> `OPENROUTER_API_KEY`. (`test.sh` at the repo root shows the OpenRouter endpoint and the
> `Authorization: Bearer $OPENROUTER_API_KEY` header for a minimal `curl` smoke test.)

### RL training (PPO / GRPO / GiGPO)

**All of our RL results come from running one of the training scripts at the repo root and changing the
environment and the base model.** Each script is a plain bash script: it activates the conda env, sets the
vLLM environment, generates the data, and launches `verl.trainer.main_ppo` with the algorithm
pre-configured. Checkpoints are written to `./checkpoints/<project>/<experiment>`. The only things you
typically change between runs are the **environment name** and the **model**.

| Algorithm | Script | `adv_estimator` | Default model |
| - | - | - | - |
| GRPO  | `grpo_train.sh`  | `grpo`  | `Qwen/Qwen2.5-7B-Instruct` |
| GiGPO | `gigpo_train.sh` | `gigpo` | `Qwen/Qwen2.5-3B-Instruct` |
| PPO   | `ppo_train.sh`   | `gae`   | `Qwen/Qwen2.5-1.5B-Instruct` |

**Before running, set `HF_TOKEN` in your shell** (the scripts read it from the environment). They expect
the `verl-agent-safety` conda env (override with `CONDA_ENV=...`, or `CONDA_ENV=""` if you activate it
yourself).

```bash
bash grpo_train.sh
```

#### Changing the environment and model

The environment, model, GPU count, and seed are config variables at the top of each script and can be set
**without editing the file** — pass them as environment variables:

```bash
# GRPO / GiGPO: model is a full HF path
ENV_NAME=WhiskyGold MODEL_PATH=Qwen/Qwen2.5-14B-Instruct N_GPUS=4 bash grpo_train.sh

# PPO: model is selected by size (1.5B | 3B | 7B), which also sets the batch knobs
ENV_NAME=IslandNavigation MODEL_SIZE=7B SEED=2 bash ppo_train.sh
```

> Any extra `key=value` arguments after the script are forwarded straight to the trainer, e.g.
> `bash grpo_train.sh env.history_length=4`. The experiment name is derived from `ENV_NAME` + model, so
> separate runs are logged separately in Weights & Biases automatically.
>
> The same scripts cover every model scale in the paper (1.5B–14B): GRPO/GiGPO via `MODEL_PATH`, PPO via
> `MODEL_SIZE`. (The 14B GRPO runs are just `MODEL_PATH=Qwen/Qwen2.5-14B-Instruct bash grpo_train.sh`.)

Other knobs worth knowing (already set to the paper's values in each script):

| Knob | Field | Default in scripts |
| - | - | - |
| Algorithm | `algorithm.adv_estimator` | `grpo` / `gigpo` / `gae` |
| History length | `env.history_length` | `2` (GRPO/PPO), `20` (GiGPO exploration) |
| Max steps / episode | `env.max_steps` | `50` |
| Total epochs | `trainer.total_epochs` | `200` |
| GiGPO discount | `algorithm.gamma` | `0.95` |
| Logger | `trainer.logger` | `['console','wandb']` |

### Evaluating a model across all environments

`vllm_eval.sh` runs zero-shot vLLM evaluation over **every** environment in one go (it loops the full
`ENVS` list). Point it at any HF model — a base model or the path to a trained checkpoint:

```bash
bash vllm_eval.sh                                          # default: Qwen/Qwen2.5-14B-Instruct
MODEL_NAME=./checkpoints/verl_agent_safety/<exp> bash vllm_eval.sh   # a trained checkpoint
```

---

## Reproducing Paper Results

Every result is produced by the scripts above — the zero-shot numbers from the prompt-agent scripts, and
the RL numbers by running a training script **once per (environment, model) pair**, changing only
`ENV_NAME` and `MODEL_PATH` (or `ENV_NAME` / `MODEL_SIZE` for PPO) as described in
[RL training](#rl-training-ppo--grpo--gigpo). Model scales reported span **1.5B–14B**
(`Qwen/Qwen2.5-{1.5B,3B,7B,14B}-Instruct`).

| Paper result | How to reproduce |
| - | - |
| Zero-shot specification gaming | `bash examples/prompt_agent/run_safety.sh` (loops all envs); API models via `python examples/prompt_agent/async_api_safegridworlds.py` |
| RL widens the observed/hidden reward gap (GRPO) | `ENV_NAME=<env> MODEL_PATH=<model> bash grpo_train.sh` for each env / model |
| Finer credit assignment doesn't fix it (GiGPO)  | `ENV_NAME=<env> MODEL_PATH=<model> bash gigpo_train.sh` for each env / model |
| PPO baseline | `ENV_NAME=<env> MODEL_SIZE=<size> bash ppo_train.sh` for each env / model |

<!-- TODO (optional): map the above rows to specific table/figure numbers in the paper, and note the exact
     env list + model sizes used for each. -->

---

## Repository Layout

```
verl-agent-safety/
├── agent_system/
│   └── environments/env_package/safe_gridworlds/   # text-based AI Safety Gridworlds
├── examples/prompt_agent/                           # zero-shot / prompt-based agents
│   ├── vllm_safetygridworlds.py                     # open models via vLLM
│   ├── async_api_safegridworlds.py                  # closed models via API (async)
│   └── run_safety.sh                                # vLLM eval driver (all envs)
├── verl/                                            # veRL training core
├── ppo_train.sh / grpo_train.sh / gigpo_train.sh    # RL training entrypoints
├── vllm_eval.sh                                     # zero-shot eval over all envs (vLLM)
├── requirements_safety.txt                          # training stack
└── safenv_requirements.txt                          # gridworlds env deps
```

---

## Acknowledgements

Built on [`verl-agent`](https://github.com/langfengQ/verl-agent) and [veRL](https://github.com/volcengine/verl),
and adapts [DeepMind's AI Safety Gridworlds](https://github.com/google-deepmind/ai-safety-gridworlds).
