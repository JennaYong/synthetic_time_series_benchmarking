#!/usr/bin/env bash
# Prepare the tts-gan model dir for MIMIC-IV training:
#   1. copy the preprocessed MIMIC-IV tensors (written by
#      preprocess/mimic_pyhealth.py) into model/tts-gan/mimic/
#   2. copy the tracked adapter code (preprocess/ttsgan/*mimic*.py) into
#      model/tts-gan/
# Everything lands inside model/, which sync.sh already ships to the remote
# server unchanged. Re-run this script after editing the adapters or after the
# shared preprocessing is regenerated.
#
# The source directory defaults to processed_mimic_pyhealth/ at the project
# root; set MIMIC_DATA_DIR to read it from somewhere else. That directory is
# gitignored: MIMIC-IV is credentialed data and must not be committed.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_BASE_DIR="${PROJECT_DIR}/model"
MIMIC_DIR="${MIMIC_DATA_DIR:-${PROJECT_DIR}/processed_mimic_pyhealth}"
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

# stays_*.npy and annotations.json carry the per-admission mortality label the
# class filter uses; channels.json names the tensor's channels.
required_files=(
  "${MIMIC_DIR}/train.npy"
  "${MIMIC_DIR}/test.npy"
  "${MIMIC_DIR}/stays_train.npy"
  "${MIMIC_DIR}/stays_test.npy"
  "${MIMIC_DIR}/channels.json"
  "${MIMIC_DIR}/annotations.json"
)

for file_path in "${required_files[@]}"; do
  if [[ ! -f "${file_path}" ]]; then
    echo "Error: Missing required file ${file_path}" >&2
    echo "Expected the output of preprocess/mimic_pyhealth.py under ${MIMIC_DIR} (override with MIMIC_DATA_DIR)" >&2
    exit 1
  fi
done

adapter_files=(
  "${ADAPTER_DIR}/mimic_dataLoader.py"
  "${ADAPTER_DIR}/train_mimic_GAN.py"
  "${ADAPTER_DIR}/generate_mimic.py"
)
for file_path in "${adapter_files[@]}"; do
  if [[ ! -f "${file_path}" ]]; then
    echo "Error: Missing adapter file ${file_path}" >&2
    exit 1
  fi
done

mkdir -p "${model_repo_dir}/mimic"
cp "${required_files[@]}" "${model_repo_dir}/mimic/"
cp "${adapter_files[@]}" "${model_repo_dir}/"

# Several versions of the preprocessed data exist side by side, so say which
# one was copied.
summary="$(python3 - "${MIMIC_DIR}/channels.json" 2>/dev/null <<'PY' || echo "channels.json unreadable"
import json, sys
c = json.load(open(sys.argv[1]))
fill = 'present' if c.get('leading_fill') else 'absent'
clip = 'present' if c.get('clip_bounds') else 'absent'
print('%d channels, %s train / %s test stays, leading_fill %s, clip_bounds %s'
      % (len(c['channels']), c.get('n_train'), c.get('n_test'), fill, clip))
PY
)"
echo "Source: ${MIMIC_DIR} (${summary})"
echo "Relocated MIMIC-IV tensors to ${model_repo_dir}/mimic/"
echo "Copied TTS-GAN MIMIC-IV adapters to ${model_repo_dir}/"
