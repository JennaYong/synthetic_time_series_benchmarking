#!/usr/bin/env bash
#SBATCH --account=def-chenh
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_2g.20gb:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --job-name=sssd_benchmark
#SBATCH --output=logs/job-%j.out
#SBATCH --error=logs/job-%j.err

set -euo pipefail

usage() {
  echo "Usage: $(basename "$0") <model>"
  echo "  model: model name (e.g. sssd-ecg)"
  exit 1
}

if [[ $# -ne 1 ]]; then
  usage
fi

MODEL="$1"

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
  *)
    echo "Unknown model: ${MODEL}" >&2
    usage
    ;;
esac
