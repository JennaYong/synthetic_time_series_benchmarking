#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_BASE_DIR="${PROJECT_DIR}/model"
DATASET_PATH="${PROJECT_DIR}/Dataset"

zip_path="${DATASET_PATH}/UniMiB-SHAR.zip"

model_repo_dir=""
if [[ -d "${MODEL_BASE_DIR}/tts-gan/tts-gan" ]]; then
  model_repo_dir="${MODEL_BASE_DIR}/tts-gan/tts-gan"
elif [[ -d "${MODEL_BASE_DIR}/tts-gan" ]]; then
  model_repo_dir="${MODEL_BASE_DIR}/tts-gan"
else
  echo "Error: tts-gan model not found at ${MODEL_BASE_DIR}/tts-gan. Run ./load_model.sh tts-gan first." >&2
  exit 1
fi

if [[ ! -f "${zip_path}" ]]; then
  echo "Error: UniMiB-SHAR.zip not found at ${zip_path}" >&2
  echo "Download it first:" >&2
  echo "  wget -O ${zip_path} https://www.dropbox.com/s/raw/x2fpfqj0bpf8ep6/UniMiB-SHAR.zip" >&2
  exit 1
fi

cp "${zip_path}" "${model_repo_dir}/UniMiB-SHAR.zip"

echo "Relocated UniMiB-SHAR.zip to ${model_repo_dir}"
