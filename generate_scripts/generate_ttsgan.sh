#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_BASE_DIR="${PROJECT_DIR}/model"

# Configurable via environment variables at sbatch time, e.g.:
#   TTS_GAN_CLASS=Jumping TTS_GAN_MAX_ITER=1000 sbatch job.sh tts-gan
#   TTS_GAN_DATASET=ptbxl TTS_GAN_CLASS=MI ./job.sh tts-gan ptbxl
# Datasets:
#   unimib (default) - original TTS-GAN motion data, classes e.g. Running/Jumping
#   ptbxl            - PTB-XL 12-lead ECG, classes NORM/MI/STTC/CD/HYP
#                      (superclass filtering; see preprocess/ttsgan/)
DATASET="${TTS_GAN_DATASET:-unimib}"
MAX_ITER="${TTS_GAN_MAX_ITER:-500000}"
NUM_SAMPLES="${TTS_GAN_NUM_SAMPLES:-1000}"
# The generator's self-attention is seq_len x seq_len, so PTB-XL (1000 steps)
# needs far more memory per sample than UniMiB (150). Lower this if a job
# OOMs; the default keeps the original UniMiB setting.
BATCH_SIZE="${TTS_GAN_BATCH_SIZE:-16}"
# Set TTS_GAN_LR_DECAY=1 to decay both learning rates linearly to 0 over
# TTS_GAN_MAX_ITER (upstream's --lr_decay, off by default). PTB-XL runs stay
# healthy for ~60 epochs and then collapse into the frozen-0.25 state at every
# length beyond that; a constant learning rate for 187 epochs is the usual
# suspect for that shape of late-stage GAN failure.
LR_DECAY_FLAG=""
if [[ -n "${TTS_GAN_LR_DECAY:-}" && "${TTS_GAN_LR_DECAY}" != "0" ]]; then
  LR_DECAY_FLAG="--lr_decay"
fi

case "${DATASET}" in
  unimib)
    CLASS_NAME="${TTS_GAN_CLASS:-Running}"
    EXP_NAME="${CLASS_NAME}"
    TRAIN_ENTRY="train_GAN.py"
    OUTPUT_PREFIX="ttsgan_$(echo "${CLASS_NAME}" | tr '[:upper:]' '[:lower:]')"
    ;;
  ptbxl)
    CLASS_NAME="${TTS_GAN_CLASS:-NORM}"
    case "${CLASS_NAME}" in
      NORM|MI|STTC|CD|HYP) ;;
      *)
        echo "Error: for ptbxl, TTS_GAN_CLASS must be one of NORM MI STTC CD HYP (got ${CLASS_NAME})" >&2
        exit 1
        ;;
    esac
    EXP_NAME="ptbxl_${CLASS_NAME}"
    TRAIN_ENTRY="train_ptbxl_GAN.py"
    OUTPUT_PREFIX="ttsgan_ptbxl_$(echo "${CLASS_NAME}" | tr '[:upper:]' '[:lower:]')"
    ;;
  *)
    echo "Error: unknown TTS_GAN_DATASET '${DATASET}' (expected unimib or ptbxl)" >&2
    exit 1
    ;;
esac

model_repo_dir=""
training_date="$(date +%Y-%m-%d)"
synthesis_dir="${PROJECT_DIR}/synthesis/TTS-GAN/${training_date}"

# 1. Check for repository existence
if [[ -d "${MODEL_BASE_DIR}/tts-gan/tts-gan" ]]; then
  model_repo_dir="${MODEL_BASE_DIR}/tts-gan/tts-gan"
elif [[ -d "${MODEL_BASE_DIR}/tts-gan" ]]; then
  model_repo_dir="${MODEL_BASE_DIR}/tts-gan"
else
  echo "Error: tts-gan repository not found. Run ./load_model.sh tts-gan first." >&2
  exit 1
fi

if [[ ! -f "${model_repo_dir}/train_GAN.py" ]]; then
  echo "Error: train_GAN.py not found at ${model_repo_dir}" >&2
  exit 1
fi

# 2. Check dataset availability (compute nodes have no internet access, so the
#    dataLoader's runtime download from Dropbox would fail; the zip must
#    already be in the repo dir. Run ./relocate_scripts/relocate_ttsgan.sh or
#    wget it on a login node.)
if [[ "${DATASET}" == "unimib" ]]; then
  if [[ ! -f "${model_repo_dir}/UniMiB-SHAR.zip" && ! -d "${model_repo_dir}/UniMiB-SHAR" ]]; then
    echo "Error: UniMiB dataset not found in ${model_repo_dir}." >&2
    echo "Place UniMiB-SHAR.zip there first (compute nodes cannot download it):" >&2
    echo "  wget -O ${model_repo_dir}/UniMiB-SHAR.zip https://www.dropbox.com/s/raw/x2fpfqj0bpf8ep6/UniMiB-SHAR.zip" >&2
    exit 1
  fi
else
  # ptbxl: npys and adapter code are copied in by relocate_ttsgan_ptbxl.sh.
  # All three splits are required: train_GAN.py builds the test set at startup
  # too, so a missing test npy would crash after the job has already started.
  for f in ptbxl/ptbxl_train_data.npy ptbxl/ptbxl_train_labels.npy \
           ptbxl/ptbxl_validation_data.npy ptbxl/ptbxl_validation_labels.npy \
           ptbxl/ptbxl_test_data.npy ptbxl/ptbxl_test_labels.npy \
           ptbxl_dataLoader.py train_ptbxl_GAN.py; do
    if [[ ! -e "${model_repo_dir}/${f}" ]]; then
      echo "Error: ${model_repo_dir}/${f} not found." >&2
      echo "Run ./relocate_scripts/relocate_ttsgan_ptbxl.sh locally and re-sync." >&2
      exit 1
    fi
  done
fi

# 3. Install python dependencies into the active venv (offline, from the
#    Alliance wheelhouse). If a package is missing from the wheelhouse, run
#    the same pip install without --no-index on a login node once; the venv
#    in ~/venv/tts-gan persists across jobs.
#    NOTE: opencv is intentionally NOT installed here. On Alliance clusters
#    opencv-python is a dummy wheel that fails on purpose (OpenCV is a module,
#    not a pip package). functions.py imports cv2 but never uses it, so we
#    strip that import in patch_sources() below instead of installing opencv.
install_dependencies() {
  pip install --no-index --upgrade pip
  if ! pip install --no-index \
    torch torchvision tensorboard einops torchsummary \
    tsaug tabulate imageio tqdm requests pillow; then
    echo "Error: offline pip install failed. Pre-install missing packages from a login node:" >&2
    echo "  source ~/venv/tts-gan/bin/activate && pip install <missing packages>" >&2
    exit 1
  fi
}

# Patch upstream source so it runs on Alliance and survives long runs.
# Idempotent: each patch skips if already applied.
patch_sources() {
  local functions_py="${model_repo_dir}/functions.py"
  local train_gan_py="${model_repo_dir}/train_GAN.py"

  # 1. functions.py imports cv2 but never uses it; opencv can't be pip-installed
  #    on Alliance (dummy wheel). Comment the import out.
  if grep -q '^import cv2' "${functions_py}"; then
    perl -pi -e 's/^import cv2/#import cv2  # removed: unused, opencv unavailable on Alliance/' "${functions_py}"
    echo "Patched ${functions_py}: disabled unused 'import cv2'"
  fi

  # 2. gen_plot() builds a matplotlib figure every epoch but never closes it, so
  #    RAM grows unbounded and long runs get OOM-killed (~epoch 3279 at 32G).
  #    Close the figure before returning the buffer.
  if ! grep -q 'plt.close(fig)' "${train_gan_py}"; then
    perl -0pi -e 's/    buf\.seek\(0\)\n    return buf/    buf.seek(0)\n    plt.close(fig)\n    return buf/' "${train_gan_py}"
    echo "Patched ${train_gan_py}: close matplotlib figure in gen_plot (fixes memory leak)"
  fi
}

# 4. Train
run_training() {
  echo "Training TTS-GAN (dataset=${DATASET}, class=${CLASS_NAME}, max_iter=${MAX_ITER})"
  (
    cd "${model_repo_dir}"
    python "${TRAIN_ENTRY}" \
      -gen_bs "${BATCH_SIZE}" \
      -dis_bs "${BATCH_SIZE}" \
      --dist-url 'tcp://localhost:4321' \
      --dist-backend 'nccl' \
      --world-size 1 \
      --rank 0 \
      --dataset "$([[ "${DATASET}" == "unimib" ]] && echo UniMiB || echo "${DATASET}")" \
      --bottom_width 8 \
      --max_iter "${MAX_ITER}" \
      --img_size 32 \
      --gen_model my_gen \
      --dis_model my_dis \
      --df_dim 384 \
      --d_heads 4 \
      --d_depth 3 \
      --g_depth 5,4,2 \
      --dropout 0 \
      --latent_dim 100 \
      --gf_dim 1024 \
      --num_workers "${SLURM_CPUS_PER_TASK:-8}" \
      --g_lr 0.0001 \
      --d_lr 0.0003 \
      --optimizer adam \
      --loss lsgan \
      --wd 1e-3 \
      --beta1 0.9 \
      --beta2 0.999 \
      --phi 1 \
      --batch_size "${BATCH_SIZE}" \
      --num_eval_imgs 50000 \
      --init_type xavier_uniform \
      --n_critic 1 \
      --val_freq 20 \
      --print_freq 50 \
      --grow_steps 0 0 \
      --fade_in 0 \
      --patch_size 2 \
      --ema_kimg 500 \
      --ema_warmup 0.1 \
      --ema 0.9999 \
      --diff_aug translation,cutout,color \
      --class_name "${CLASS_NAME}" \
      ${LR_DECAY_FLAG} \
      --exp_name "${EXP_NAME}"
  )
}

# 5. Generate synthetic samples from the newest checkpoint and collect outputs
collect_synthesis_outputs() {
  local latest_ckpt
  latest_ckpt="$(ls -t "${model_repo_dir}/logs/${EXP_NAME}"_*/Model/checkpoint 2>/dev/null | head -1)"

  if [[ -z "${latest_ckpt}" ]]; then
    echo "Error: no checkpoint found under ${model_repo_dir}/logs/${EXP_NAME}_*/Model/" >&2
    exit 1
  fi

  echo "Generating ${NUM_SAMPLES} synthetic samples from ${latest_ckpt}"
  mkdir -p "${synthesis_dir}"

  (
    cd "${model_repo_dir}"
    python3 - "${latest_ckpt}" "${synthesis_dir}" "${CLASS_NAME}" "${NUM_SAMPLES}" "${DATASET}" "${OUTPUT_PREFIX}" <<'PY'
import os
import sys
from pathlib import Path

import numpy as np
import torch

from GANModels import Generator

ckpt_path = Path(sys.argv[1])
synthesis_dir = Path(sys.argv[2])
class_name = sys.argv[3]
num_samples = int(sys.argv[4])
dataset = sys.argv[5]
output_prefix = sys.argv[6]

# Must match the training-time instantiation (train_GAN.py defaults for
# UniMiB; train_ptbxl_GAN.py dimensions for PTB-XL)
if dataset == "ptbxl":
    # Must mirror train_ptbxl_GAN.py: same embed_dim as the checkpoint was
    # trained with, or load_state_dict fails (loudly) on shape mismatch.
    embed_dim = int(os.environ.get("TTS_GAN_PTBXL_EMBED_DIM", "40"))
    gen_net = Generator(seq_len=1000, channels=12, embed_dim=embed_dim)
else:
    gen_net = Generator()
checkpoint = torch.load(ckpt_path, map_location="cpu")
gen_net.load_state_dict(checkpoint["gen_state_dict"])
gen_net.eval()

# Chunked generation: a single 1000-sample batch through the seq_len=1000
# attention would need tens of GB; identical output, bounded memory.
z = torch.FloatTensor(np.random.normal(0, 1, (num_samples, 100)))
chunks = []
with torch.no_grad():
    for start in range(0, num_samples, 50):
        chunks.append(gen_net(z[start:start + 50]).numpy())
synthetic = np.concatenate(chunks, axis=0)

if dataset == "ptbxl":
    # train_ptbxl_GAN.ScaleNormalizedGenerator z-normalizes the generator
    # output per (sample, lead); apply the identical transform here so
    # generation matches training. It has no parameters, which is why the
    # plain Generator above can load the checkpoint either way.
    synthetic = ((synthetic - synthetic.mean(axis=3, keepdims=True))
                 / (synthetic.std(axis=3, keepdims=True) + 1e-8))

output_path = synthesis_dir / f"{output_prefix}_samples.npy"
np.save(output_path, synthetic)

# Health check, printed so a failed run is obvious from the job log alone.
# Amplitude on its own is not enough for PTB-XL now that the output is
# normalized above, so also report temporal smoothness: the std of the first
# difference over the std of the signal. Pointwise white noise gives
# sqrt(2) ~ 1.41; real PTB-XL leads give ~0.66. The run that collapsed scored
# 1.413, i.e. it had no temporal structure at all.
sample_std = float(synthetic.std())
diff_ratio = float(np.diff(synthetic, axis=3).std() / (synthetic.std() + 1e-12))
print(f"Synthetic sample std: {sample_std:.4f}")
print(f"Synthetic diff/signal std ratio: {diff_ratio:.3f} "
      f"(real PTB-XL ~0.66, pointwise white noise ~1.41)")
if not 0.01 < sample_std < 10.0:
    print("WARNING: amplitude far outside the expected range -- training diverged.")
if diff_ratio > 1.2:
    print("WARNING: output is close to pointwise white noise -- training collapsed.")

if dataset == "ptbxl":
    # One-hot superclass labels (NORM,MI,STTC,CD,HYP) for downstream evaluation
    from ptbxl_dataLoader import SUPERCLASSES
    labels = np.zeros((num_samples, len(SUPERCLASSES)), dtype=np.float32)
    labels[:, SUPERCLASSES.index(class_name)] = 1.0
    labels_path = synthesis_dir / f"{output_prefix}_labels.npy"
    np.save(labels_path, labels)
    print(f"Saved {labels.shape} one-hot superclass labels to {labels_path}")

meta_path = synthesis_dir / f"{output_prefix}_config.txt"
meta_path.write_text(
    f"Dataset: {dataset}\n"
    f"Class: {class_name}\n"
    f"Checkpoint: {ckpt_path}\n"
    f"Epoch: {checkpoint['epoch']}\n"
    f"Num_samples: {num_samples}\n"
    f"Shape: {synthetic.shape}\n"
    f"Sample_std: {sample_std:.4f}\n"
    + (f"Embed_dim: {embed_dim}\n"
       f"Patch_size: {os.environ.get('TTS_GAN_PTBXL_PATCH_SIZE', '100')}\n"
       if dataset == "ptbxl" else "")
)

print(f"Saved {synthetic.shape} synthetic samples to {output_path}")
PY
  )

  cp "${latest_ckpt}" "${synthesis_dir}/${EXP_NAME}_checkpoint"
  echo "Saved synthetic TTS-GAN outputs to ${synthesis_dir}"
}

install_dependencies
patch_sources
run_training
collect_synthesis_outputs
