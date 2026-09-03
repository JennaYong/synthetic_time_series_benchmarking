#!/usr/bin/env bash
# Prepare the tts-gan model dir for PTB-XL training:
#   1. copy the SSSD-ECG preprocessed PTB-XL npys from Dataset/ into
#      model/tts-gan/ptbxl/
#   2. copy the tracked adapter code (preprocess/ttsgan/*.py) into
#      model/tts-gan/
# Everything lands inside model/, which sync.sh already ships to the remote
# server unchanged. Re-run this script after editing preprocess/ttsgan/*.py.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_BASE_DIR="${PROJECT_DIR}/model"
DATASET_PATH="${PROJECT_DIR}/Dataset"
ADAPTER_DIR="${PROJECT_DIR}/preprocess/ttsgan"

model_repo_dir=""
if [[ -d "${MODEL_BASE_DIR}/tts-gan/tts-gan" ]]; then
  model_repo_dir="${MODEL_BASE_DIR}/tts-gan/tts-gan"
elif [[ -d "${MODEL_BASE_DIR}/tts-gan" ]]; then
  model_repo_dir="${MODEL_BASE_DIR}/tts-gan"
else
  echo "Error: tts-gan model not found at ${MODEL_BASE_DIR}/tts-gan. Run ./load_model.sh tts-gan first." >&2
  exit 1
fi

data_dir="${DATASET_PATH}/data"
labels_dir="${DATASET_PATH}/labels"
required_files=(
  "${data_dir}/ptbxl_train_data.npy"
  "${data_dir}/ptbxl_validation_data.npy"
  "${data_dir}/ptbxl_test_data.npy"
  "${labels_dir}/ptbxl_train_labels.npy"
  "${labels_dir}/ptbxl_validation_labels.npy"
  "${labels_dir}/ptbxl_test_labels.npy"
)

for file_path in "${required_files[@]}"; do
  if [[ ! -f "${file_path}" ]]; then
    echo "Error: Missing required file ${file_path}" >&2
    echo "Expected the SSSD-ECG preprocessed PTB-XL layout under ${DATASET_PATH} (see README Step 1)" >&2
    exit 1
  fi
done

adapter_files=(
  "${ADAPTER_DIR}/ptbxl_dataLoader.py"
  "${ADAPTER_DIR}/train_ptbxl_GAN.py"
)
for file_path in "${adapter_files[@]}"; do
  if [[ ! -f "${file_path}" ]]; then
    echo "Error: Missing adapter file ${file_path}" >&2
    exit 1
  fi
done

mkdir -p "${model_repo_dir}/ptbxl"
cp "${required_files[@]}" "${model_repo_dir}/ptbxl/"
cp "${adapter_files[@]}" "${model_repo_dir}/"

echo "Relocated PTB-XL npys to ${model_repo_dir}/ptbxl/"
echo "Copied TTS-GAN PTB-XL adapters to ${model_repo_dir}/"
