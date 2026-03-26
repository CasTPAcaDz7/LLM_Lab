#!/bin/bash
#SBATCH --job-name=lab3_infer
#SBATCH --account=ugiedahpc4
#SBATCH --partition=gpu-l20
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --output=lab3_infer_%j.out
#SBATCH --error=lab3_infer_%j.err

set -euo pipefail

COURSE_ENV="${COURSE_ENV:-ieda4000i}"
MODEL_PATH="${MODEL_PATH:-/project/ugiedahpc4/ieda4000i/models/Qwen3-4B}"
PROMPT="${PROMPT:-Give me three bullet points about operations research and LLMs.}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.9}"
DEVICE="${DEVICE:-cuda}"

source "/opt/shared/.spack-edge/dist/bin/setup-env.sh" -y

echo "[INFO] Job id: ${SLURM_JOB_ID:-unknown}"
echo "[INFO] Hostname: $(hostname)"
echo "[INFO] Environment: ${COURSE_ENV}"
echo "[INFO] Model path: ${MODEL_PATH}"

nvidia-smi || true

${HOME}/.conda/envs/${COURSE_ENV}/bin/python "./run_infer.py" \
  --model_path "${MODEL_PATH}" \
  --prompt "${PROMPT}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --temperature "${TEMPERATURE}" \
  --top_p "${TOP_P}" \
  --device "${DEVICE}"
