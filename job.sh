#!/usr/bin/env bash
#SBATCH --account=def-chenh
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_2g.20gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
# ^ Fallback wall-time only. The real per-model limit is set at submit time by
#   the self-submit block below (sssd-ecg=24h, tts-gan=10h). A command-line
#   --time always overrides both.
#SBATCH --job-name=synth_benchmark
# ^ Fallback name only, used when this file is submitted with `sbatch` directly.
#   The self-submit block below sets a per-run name like tts-gan_ptbxl_NORM.
#SBATCH --output=logs/job-%j.out
#SBATCH --error=logs/job-%j.err

set -euo pipefail

usage() {
  echo "Usage: $(basename "$0") <model> [dataset]"
  echo "  model:   model name (sssd-ecg | tts-gan)"
  echo "  dataset: optional dataset override (tts-gan: unimib | ptbxl; default unimib)"
  exit 1
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
  usage
fi

MODEL="$1"
DATASET="${2:-}"

# tts-gan reads its dataset choice from TTS_GAN_DATASET (see
# generate_scripts/generate_ttsgan.sh); the optional positional arg just sets
# it, so both `TTS_GAN_DATASET=ptbxl ./job.sh tts-gan` and
# `./job.sh tts-gan ptbxl` work.
if [[ -n "${DATASET}" ]]; then
  if [[ "${MODEL}" != "tts-gan" ]]; then
    echo "Error: ${MODEL} takes no dataset argument (got '${DATASET}')" >&2
    usage
  fi
  export TTS_GAN_DATASET="${DATASET}"
fi

# Per-model wall-time. SBATCH headers are static (parsed before the script
# runs), so we can't branch on ${MODEL} inside them. Instead: when run directly
# (not yet under Slurm), pick the right --time for the model and submit
# ourselves. When the submitted job re-runs this file, SLURM_JOB_ID is set and
# we skip straight to the work below.
#
# Usage:  ./job.sh <model>       (NOT `sbatch job.sh <model>`, which would use
#                                 the fallback header time instead of per-model)
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  case "${MODEL}" in
    sssd-ecg) TIME_LIMIT="24:00:00" ;;
    tts-gan)
      # Measured on nibi: PTB-XL NORM at TTS_GAN_MAX_ITER=100000 finished in
      # 1:53:48 (~14.6 it/s), so 3h covers it with margin. The 1000-step
      # sequences cost far less than the 150-step UniMiB ones would suggest
      # because the generator's embed_dim is only 10.
      # NOTE: 3h is calibrated for TTS_GAN_MAX_ITER=100000. A much larger value
      # (e.g. 500000, roughly 10h) needs an explicit override:
      #   sbatch --time=<HH:MM:SS> --export=ALL,TTS_GAN_DATASET=ptbxl,... job.sh tts-gan
      if [[ "${TTS_GAN_DATASET:-unimib}" == "ptbxl" ]]; then
        TIME_LIMIT="03:00:00"
      else
        TIME_LIMIT="10:00:00"
      fi
      ;;
    *)
      echo "Unknown model: ${MODEL}" >&2
      usage
      ;;
  esac
  # Name the job after what it actually runs, so `sq` / `sacct` can tell several
  # concurrent runs apart (the static header above cannot depend on ${MODEL}).
  JOB_NAME="${MODEL}"
  if [[ "${MODEL}" == "tts-gan" ]]; then
    JOB_NAME="${JOB_NAME}_${TTS_GAN_DATASET:-unimib}${TTS_GAN_CLASS:+_${TTS_GAN_CLASS}}"
  fi

  echo "Submitting ${JOB_NAME} with --time=${TIME_LIMIT}"
  # --export=ALL forwards the current environment (e.g. TTS_GAN_CLASS,
  # TTS_GAN_DATASET) into the submitted job so per-model options set on the
  # command line still apply.
  exec sbatch --time="${TIME_LIMIT}" --job-name="${JOB_NAME}" --export=ALL "${BASH_SOURCE[0]}" "${MODEL}"
fi

# 1. Load environment modules
module --force purge
module load StdEnv/2023
module load python/3.11
module load cuda/12.2
module load scipy-stack

# 2. Export CUDA paths
export CUDA_HOME="${EBROOTCUDA}"
export LD_LIBRARY_PATH="${EBROOTCUDA}/lib64:${LD_LIBRARY_PATH:-}"
export PATH="${EBROOTCUDA}/bin:${PATH}"

# 3. Set up and activate model virtual environment
VENV_ROOT="${HOME}/venv"
VENV_DIR="${VENV_ROOT}/${MODEL}"

if [[ ! -d "${VENV_ROOT}" ]]; then
  mkdir -p "${VENV_ROOT}"
fi

if [[ ! -d "${VENV_DIR}/bin" ]]; then
  virtualenv --no-download "${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

# 4. Run generation
case "${MODEL}" in
  sssd-ecg)
    ./generate_scripts/generate_sssdecg.sh
    ;;
  tts-gan)
    ./generate_scripts/generate_ttsgan.sh
    ;;
  *)
    echo "Unknown model: ${MODEL}" >&2
    usage
    ;;
esac
