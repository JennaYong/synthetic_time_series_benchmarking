#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $(basename "$0") <model>"
  echo "Available models: sssd-ecg, tts-gan, doppelganger"
  exit 1
}

if [[ $# -ne 1 ]]; then
  usage
fi

MODEL="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_BASE_DIR="${SCRIPT_DIR}/model"

case "${MODEL}" in
  sssd-ecg)
    MODEL_DIR="${MODEL_BASE_DIR}/SSSD-ECG"
    MODEL_REPO="https://github.com/AI4HealthUOL/SSSD-ECG.git"
    ;;
  tts-gan)
    MODEL_DIR="${MODEL_BASE_DIR}/tts-gan"
    MODEL_REPO="https://github.com/imics-lab/tts-gan.git"
    ;;
  doppelganger)
    # Gretel's PyTorch implementation of DoppelGANger (Lin et al., IMC 2020);
    # the authors' release is TensorFlow 1.4 code that no longer installs on
    # current clusters. Only src/gretel_synthetics/timeseries_dgan is used.
    MODEL_DIR="${MODEL_BASE_DIR}/doppelganger"
    MODEL_REPO="https://github.com/gretelai/gretel-synthetics.git"
    ;;
  *)
    echo "Unknown model: ${MODEL}" >&2
    usage
    ;;
esac

mkdir -p "${MODEL_BASE_DIR}"

if [[ -d "${MODEL_DIR}/.git" ]]; then
  git -C "${MODEL_DIR}" pull --ff-only
elif [[ -e "${MODEL_DIR}" ]]; then
  echo "Error: ${MODEL_DIR} exists but is not a git repository" >&2
  exit 1
else
  git clone "${MODEL_REPO}" "${MODEL_DIR}"
fi
