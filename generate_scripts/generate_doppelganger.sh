#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${PROJECT_DIR}/model/doppelganger"

# DoppelGANger on MIMIC-IV, its only dataset. Configurable via environment
# variables at submit time, e.g.:
#   ./job.sh doppelganger
#   DGAN_EPOCHS=200 DGAN_NUM_SAMPLES=21860 ./job.sh doppelganger
# The DGAN_* options are documented in preprocess/doppelganger/mimic_dgan.py;
# the defaults are repeated here only to name the output directory.
EPOCHS="${DGAN_EPOCHS:-400}"
BATCH_SIZE="${DGAN_BATCH_SIZE:-100}"
SAMPLE_LEN="${DGAN_SAMPLE_LEN:-6}"
LOG_LABS="${DGAN_MIMIC_LOG:-1}"
LR="${DGAN_LR:-0.0001}"
EXAMPLE_SCALING="${DGAN_EXAMPLE_SCALING:-1}"
export DGAN_EPOCHS="${EPOCHS}" DGAN_BATCH_SIZE="${BATCH_SIZE}" DGAN_SAMPLE_LEN="${SAMPLE_LEN}" \
       DGAN_MIMIC_LOG="${LOG_LABS}" DGAN_LR="${LR}" DGAN_EXAMPLE_SCALING="${EXAMPLE_SCALING}" \
       DGAN_MIMIC_DATA="./mimic/"
OUTPUT_PREFIX="doppelganger_mimic_all"

# 1. Check the code and the data (compute nodes have no internet access, so
#    both must already be in model/doppelganger/: run ./load_model.sh
#    doppelganger and ./relocate_scripts/relocate_doppelganger_mimic.sh
#    locally, then sync).
for f in src/gretel_synthetics/timeseries_dgan/dgan.py mimic_dgan.py \
         mimic/train.npy mimic/channels.json; do
  if [[ ! -e "${MODEL_DIR}/${f}" ]]; then
    echo "Error: ${MODEL_DIR}/${f} not found." >&2
    echo "Run ./load_model.sh doppelganger and ./relocate_scripts/relocate_doppelganger_mimic.sh locally and re-sync." >&2
    exit 1
  fi
done

# 2. Install python dependencies into the active venv (offline, from the
#    Alliance wheelhouse). numpy, scipy and pandas come from the scipy-stack
#    module job.sh loads; gretel-synthetics itself is not installed, only its
#    timeseries_dgan package is imported from source. If a package is missing
#    from the wheelhouse, run the same pip install without --no-index on a
#    login node once; the venv in ~/venv/doppelganger persists across jobs.
install_dependencies() {
  pip install --no-index --upgrade pip
  if ! pip install --no-index torch category_encoders; then
    echo "Error: offline pip install failed. Pre-install missing packages from a login node:" >&2
    echo "  source ~/venv/doppelganger/bin/activate && pip install <missing packages>" >&2
    exit 1
  fi
}

# 3. Train and generate. The driver writes the samples, the config with the
#    health check and the model straight into the synthesis directory.
run_training_and_generation() {
  local run_tag="ep${EPOCHS}_mimic_b${BATCH_SIZE}_s${SAMPLE_LEN}"
  if [[ "${LOG_LABS}" == "0" ]]; then
    run_tag="${run_tag}_nolog"
  fi
  if [[ "${LR}" != "0.0001" ]]; then
    run_tag="${run_tag}_lr${LR}"
  fi
  if [[ "${EXAMPLE_SCALING}" == "0" ]]; then
    run_tag="${run_tag}_noexscale"
  fi
  local synthesis_dir="${PROJECT_DIR}/synthesis/DoppelGANger/$(date +%Y-%m-%d)_${run_tag}${SLURM_JOB_ID:+_${SLURM_JOB_ID}}"
  echo "Training DoppelGANger on MIMIC-IV (epochs=${EPOCHS}, batch=${BATCH_SIZE}, sample_len=${SAMPLE_LEN}, lr=${LR}, log labs=${LOG_LABS}, example scaling=${EXAMPLE_SCALING})"
  (
    cd "${MODEL_DIR}"
    PYTHONPATH="${MODEL_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}" \
      python3 mimic_dgan.py "${synthesis_dir}" "${OUTPUT_PREFIX}"
  )
  echo "Outputs in ${synthesis_dir}"
}

install_dependencies
run_training_and_generation
