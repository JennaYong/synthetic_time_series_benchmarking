#!/usr/bin/env bash
# Prepare the doppelganger model dir for MIMIC-IV training:
#   1. copy the preprocessed MIMIC-IV tensors (written by
#      preprocess/mimic_pyhealth.py) into model/doppelganger/mimic/
#   2. copy the tracked driver (preprocess/doppelganger/mimic_dgan.py) into
#      model/doppelganger/
# Everything lands inside model/, which sync.sh already ships to the remote
# server unchanged. Re-run this script after editing the driver or after the
# shared preprocessing is regenerated.
#
# The source directory defaults to processed_mimic_pyhealth/ at the project
# root; set MIMIC_DATA_DIR to read it from somewhere else. That directory is
# gitignored: MIMIC-IV is credentialed data and must not be committed.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${PROJECT_DIR}/model/doppelganger"
MIMIC_DIR="${MIMIC_DATA_DIR:-${PROJECT_DIR}/processed_mimic_pyhealth}"
DRIVER="${PROJECT_DIR}/preprocess/doppelganger/mimic_dgan.py"

if [[ ! -d "${MODEL_DIR}/src/gretel_synthetics/timeseries_dgan" ]]; then
  echo "Error: DoppelGANger code not found at ${MODEL_DIR}. Run ./load_model.sh doppelganger first." >&2
  exit 1
fi

# The model trains on the value channels of train.npy only; channels.json
# names the channels and holds the clip bounds the samples are clipped to.
required_files=(
  "${MIMIC_DIR}/train.npy"
  "${MIMIC_DIR}/channels.json"
)
for file_path in "${required_files[@]}"; do
  if [[ ! -f "${file_path}" ]]; then
    echo "Error: Missing required file ${file_path}" >&2
    echo "Expected the output of preprocess/mimic_pyhealth.py under ${MIMIC_DIR} (override with MIMIC_DATA_DIR)" >&2
    exit 1
  fi
done
if [[ ! -f "${DRIVER}" ]]; then
  echo "Error: Missing driver ${DRIVER}" >&2
  exit 1
fi

mkdir -p "${MODEL_DIR}/mimic"
cp "${required_files[@]}" "${MODEL_DIR}/mimic/"
cp "${DRIVER}" "${MODEL_DIR}/"
# sync.sh skips .git, so record which gretel-synthetics commit is shipped; the
# driver copies it into every run's config file.
git -C "${MODEL_DIR}" rev-parse HEAD > "${MODEL_DIR}/UPSTREAM_COMMIT"

# Several versions of the preprocessed data exist side by side, so say which
# one was copied.
summary="$(python3 - "${MIMIC_DIR}/channels.json" 2>/dev/null <<'PY' || echo "channels.json unreadable"
import json, sys
c = json.load(open(sys.argv[1]))
fill = 'present' if c.get('leading_fill') else 'absent'
clip = 'present' if c.get('clip_bounds') else 'absent'
print('%d channels, %s train stays, leading_fill %s, clip_bounds %s'
      % (len(c['channels']), c.get('n_train'), fill, clip))
PY
)"
echo "Source: ${MIMIC_DIR} (${summary})"
echo "Relocated MIMIC-IV tensors to ${MODEL_DIR}/mimic/"
echo "Copied the DoppelGANger MIMIC-IV driver to ${MODEL_DIR}/"
