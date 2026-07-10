#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_BASE_DIR="${PROJECT_DIR}/model"
DATASET_PATH="${PROJECT_DIR}/Dataset"
model_dir="${MODEL_BASE_DIR}/SSSD-ECG"
sssd_src_dir="${model_dir}/src/sssd"
dest_dir="${sssd_src_dir}"
default_config="${sssd_src_dir}/config/config_SSSD_ECG.json"
inference_py="${dest_dir}/inference.py"
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

if [[ ! -d "${model_dir}" ]]; then
  echo "Error: SSSD-ECG model not found at ${model_dir}. Run ./load_model.sh sssd-ecg first." >&2
  exit 1
fi

if [[ ! -d "${data_dir}" || ! -d "${labels_dir}" ]]; then
  echo "Error: Expected dataset layout with data/ and labels/ under ${DATASET_PATH}" >&2
  exit 1
fi

for file_path in "${required_files[@]}"; do
  if [[ ! -f "${file_path}" ]]; then
    echo "Error: Missing required file ${file_path}" >&2
    exit 1
  fi
done

if [[ ! -f "${default_config}" ]]; then
  echo "Error: SSSD-ECG config not found at ${default_config}" >&2
  exit 1
fi

if [[ ! -f "${inference_py}" ]]; then
  echo "Error: SSSD-ECG inference.py not found at ${inference_py}" >&2
  exit 1
fi

mkdir -p "${dest_dir}"
cp "${data_dir}"/*.npy "${dest_dir}/"
cp "${labels_dir}"/*.npy "${dest_dir}/"

python3 - "${default_config}" "${dest_dir}" <<'PY'
import json
import re
import sys
from pathlib import Path

config_path = Path(sys.argv[1])
dest_dir = sys.argv[2].rstrip("/") + "/"

config_text = config_path.read_text()
config_text = re.sub(r",(\s*[}\]])", r"\1", config_text)
config = json.loads(config_text)
config["gen_config"]["data_path"] = dest_dir
config["gen_config"]["num_samples"] = 400
config_path.write_text(json.dumps(config, indent=4) + "\n")
PY

perl -pi -e "s/labels = np\.load\('ptbxl_test_labels\.npy'\)/labels = np.load(data_path + 'ptbxl_test_labels.npy')/" "${inference_py}"

echo "Relocated PTB-XL dataset to ${dest_dir}"
echo "Updated ${default_config} with gen_config.data_path"
echo "Updated ${inference_py} to load labels from data_path"
