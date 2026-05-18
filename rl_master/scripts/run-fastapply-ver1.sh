#!/usr/bin/env bash
set -euo pipefail

# Image-faithful entrypoint for the RL-master / verl FastApply run.
# The real cluster command would call:
#   python -m verl.trainer.main_ppo ...
# This repository keeps the same knobs but routes to the offline runnable smoke.

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B-Instruct-2507}"
CKPTS_DIR="${CKPTS_DIR:-/home/ma-user/work/checkpoints/fastapply}"
TRAIN_FILE="${TRAIN_FILE:-/home/ma-user/work/datasets/fastapply/train_with_verl.parquet}"
TEST_FILE="${TEST_FILE:-/home/ma-user/work/datasets/fastapply/test_with_verl.parquet}"
CUSTOM_REWARD_PATH="${CUSTOM_REWARD_PATH:-pkg://applications.fastapply.reward.score}"
CUSTOM_REWARD_NAME="${CUSTOM_REWARD_NAME:-compute_score}"
WORKSPACE_PATH="${WORKSPACE_PATH:-/home/ma-user/work/workspace}"

export PYTHONPATH="${PYTHONPATH:-}:${WORKSPACE_PATH}"
export VLLM_ASCEND_ENABLE_V1="${VLLM_ASCEND_ENABLE_V1:-0}"
export VLLM_ASCEND_ENABLE_PATH_ALLREDUCE="${VLLM_ASCEND_ENABLE_PATH_ALLREDUCE:-1}"
export SWANLAB_MODE="${SWANLAB_MODE:-local}"
export SHANLAB_LOG_DIR="${SHANLAB_LOG_DIR:-./output}"

GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-8384}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-8000}"
MAX_SEQUENCE_LENGTH="${MAX_SEQUENCE_LENGTH:-16384}"
ROLLOUT_N="${ROLLOUT_N:-8}"
TEMPERATURE="${TEMPERATURE:-0.99}"
TOP_P="${TOP_P:-0.99}"
TOP_K="${TOP_K:-50}"

echo "FastApply GRPO reproduction"
echo "  model=${MODEL_PATH}"
echo "  train=${TRAIN_FILE}"
echo "  test=${TEST_FILE}"
echo "  reward=${CUSTOM_REWARD_PATH}:${CUSTOM_REWARD_NAME}"
echo "  batch=${GLOBAL_BATCH_SIZE} prompt=${MAX_PROMPT_LENGTH} response=${MAX_RESPONSE_LENGTH} seq=${MAX_SEQUENCE_LENGTH}"
echo "  rollout_n=${ROLLOUT_N} temperature=${TEMPERATURE} top_p=${TOP_P} top_k=${TOP_K}"

"${PYTHON:-python3}" -m rl_master.scripts.run_fastapply_grpo_smoke
