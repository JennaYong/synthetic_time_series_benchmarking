#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_BASE_DIR="${PROJECT_DIR}/model"

model_repo_dir=""
sssd_src_dir=""
default_config=""
training_date=""
trained_dir=""
synthesis_dir=""

if [[ -d "${MODEL_BASE_DIR}/SSSD-ECG/SSSD-ECG" ]]; then
  model_repo_dir="${MODEL_BASE_DIR}/SSSD-ECG/SSSD-ECG"
elif [[ -d "${MODEL_BASE_DIR}/SSSD-ECG" ]]; then
  model_repo_dir="${MODEL_BASE_DIR}/SSSD-ECG"
else
  echo "Error: SSSD-ECG repository not found. Run ./load_model.sh sssd-ecg first." >&2
  exit 1
fi

sssd_src_dir="${model_repo_dir}/src/sssd"
default_config="${sssd_src_dir}/config/config_SSSD_ECG.json"
training_date="$(date +%Y-%m-%d)"
trained_dir="${model_repo_dir}/trained/${training_date}"
synthesis_dir="${PROJECT_DIR}/synthesis/SSSD-ECG/${training_date}"

if [[ ! -d "${sssd_src_dir}" ]]; then
  echo "Error: SSSD source directory not found at ${sssd_src_dir}" >&2
  exit 1
fi

if [[ ! -f "${default_config}" ]]; then
  echo "Error: SSSD-ECG config not found at ${default_config}" >&2
  exit 1
fi

prepare_training_config() {
  local config_path="${trained_dir}/config.json"
  mkdir -p "${trained_dir}"
  cp "${default_config}" "${config_path}"

  python3 - "${config_path}" "${trained_dir}" "${sssd_src_dir}" <<'PY'
import json
import re
import sys
from pathlib import Path

config_path = Path(sys.argv[1])
trained_dir = Path(sys.argv[2]).resolve()
sssd_src_dir = Path(sys.argv[3]).resolve()

config_text = config_path.read_text()
# Upstream config may include trailing commas (invalid strict JSON).
config_text = re.sub(r",(\s*[}\]])", r"\1", config_text)
config = json.loads(config_text)
config["train_config"]["output_directory"] = str(trained_dir)
config["gen_config"]["output_directory"] = str(trained_dir)
config["gen_config"]["ckpt_path"] = f"{trained_dir}/"
# data_path must point to wherever this job actually runs (the nibi src dir),
# not whatever absolute path relocate baked in on another machine. inference.py
# does data_path + 'ptbxl_test_labels.npy', so keep the trailing slash.
config["gen_config"]["data_path"] = str(sssd_src_dir) + "/"
config_path.write_text(json.dumps(config, indent=4) + "\n")
PY

  echo "${config_path}"
}

run_training() {
  local config_path="$1"
  echo "Training SSSD-ECG with ${config_path}"
  (
    cd "${sssd_src_dir}"
    python train.py -c "${config_path}"
  )
}

run_inference() {
  local config_path="$1"
  echo "Generating synthetic ECG with latest checkpoint from ${trained_dir}"
  (
    cd "${sssd_src_dir}"
    python3 - "${config_path}" <<'PY'
import json
import sys
from pathlib import Path

from utils.util import calc_diffusion_hyperparams
import inference as sssd_inference

config_path = Path(sys.argv[1])
config = json.loads(config_path.read_text())

sssd_inference.model_config = config["wavenet_config"]
sssd_inference.diffusion_config = config["diffusion_config"]
sssd_inference.diffusion_hyperparams = calc_diffusion_hyperparams(**config["diffusion_config"])

sssd_inference.generate(
    **config["gen_config"],
    ckpt_iter="max"
)
PY
  )
}

collect_synthesis_outputs() {
  local config_path="$1"
  local ckpt_subdir
  ckpt_subdir="$(python3 - "${config_path}" <<'PY'
import json
import sys
from pathlib import Path

config = json.loads(Path(sys.argv[1]).read_text())
model_config = config["wavenet_config"]
diffusion_config = config["diffusion_config"]
print(
    "ch{res}_T{T}_betaT{beta}".format(
        res=model_config["res_channels"],
        T=diffusion_config["T"],
        beta=diffusion_config["beta_T"],
    )
)
PY
)"

  local ckpt_dir="${trained_dir}/${ckpt_subdir}"
  mkdir -p "${synthesis_dir}"

  shopt -s nullglob
  local output_files=("${ckpt_dir}"/*_samples.npy "${ckpt_dir}"/*_labels.npy)
  shopt -u nullglob

  if [[ ${#output_files[@]} -eq 0 ]]; then
    echo "Error: No generated .npy files found in ${ckpt_dir}" >&2
    exit 1
  fi

  cp "${output_files[@]}" "${synthesis_dir}/"
  cp "${config_path}" "${synthesis_dir}/config_SSSD_ECG.json"
  echo "Saved synthetic ECG outputs and config to ${synthesis_dir}"
}

training_config="$(prepare_training_config)"
run_training "${training_config}"
run_inference "${training_config}"
collect_synthesis_outputs "${training_config}"
