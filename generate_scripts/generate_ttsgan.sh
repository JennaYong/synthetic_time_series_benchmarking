#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_BASE_DIR="${PROJECT_DIR}/model"

# Configurable via environment variables at sbatch time, e.g.:
#   TTS_GAN_CLASS=Jumping TTS_GAN_MAX_ITER=1000 sbatch job.sh tts-gan
CLASS_NAME="${TTS_GAN_CLASS:-Running}"
MAX_ITER="${TTS_GAN_MAX_ITER:-500000}"
NUM_SAMPLES="${TTS_GAN_NUM_SAMPLES:-1000}"

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
if [[ ! -f "${model_repo_dir}/UniMiB-SHAR.zip" && ! -d "${model_repo_dir}/UniMiB-SHAR" ]]; then
  echo "Error: UniMiB dataset not found in ${model_repo_dir}." >&2
  echo "Place UniMiB-SHAR.zip there first (compute nodes cannot download it):" >&2
  echo "  wget -O ${model_repo_dir}/UniMiB-SHAR.zip https://www.dropbox.com/s/raw/x2fpfqj0bpf8ep6/UniMiB-SHAR.zip" >&2
  exit 1
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
  echo "Training TTS-GAN (class=${CLASS_NAME}, max_iter=${MAX_ITER})"
  (
    cd "${model_repo_dir}"
    python train_GAN.py \
      -gen_bs 16 \
      -dis_bs 16 \
      --dist-url 'tcp://localhost:4321' \
      --dist-backend 'nccl' \
      --world-size 1 \
      --rank 0 \
      --dataset UniMiB \
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
      --batch_size 16 \
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
      --exp_name "${CLASS_NAME}"
  )
}

# 5. Generate synthetic samples from the newest checkpoint and collect outputs
collect_synthesis_outputs() {
  local latest_ckpt
  latest_ckpt="$(ls -t "${model_repo_dir}/logs/${CLASS_NAME}"_*/Model/checkpoint 2>/dev/null | head -1)"

  if [[ -z "${latest_ckpt}" ]]; then
    echo "Error: no checkpoint found under ${model_repo_dir}/logs/${CLASS_NAME}_*/Model/" >&2
    exit 1
  fi

  echo "Generating ${NUM_SAMPLES} synthetic samples from ${latest_ckpt}"
  mkdir -p "${synthesis_dir}"

  (
    cd "${model_repo_dir}"
    python3 - "${latest_ckpt}" "${synthesis_dir}" "${CLASS_NAME}" "${NUM_SAMPLES}" <<'PY'
import sys
from pathlib import Path

import numpy as np
import torch

from GANModels import Generator

ckpt_path = Path(sys.argv[1])
synthesis_dir = Path(sys.argv[2])
class_name = sys.argv[3]
num_samples = int(sys.argv[4])

# Must match the training-time instantiation in train_GAN.py (defaults)
gen_net = Generator()
checkpoint = torch.load(ckpt_path, map_location="cpu")
gen_net.load_state_dict(checkpoint["gen_state_dict"])
gen_net.eval()

z = torch.FloatTensor(np.random.normal(0, 1, (num_samples, 100)))
with torch.no_grad():
    synthetic = gen_net(z).numpy()

output_path = synthesis_dir / f"ttsgan_{class_name.lower()}_samples.npy"
np.save(output_path, synthetic)

meta_path = synthesis_dir / f"ttsgan_{class_name.lower()}_config.txt"
meta_path.write_text(
    f"Class: {class_name}\n"
    f"Checkpoint: {ckpt_path}\n"
    f"Epoch: {checkpoint['epoch']}\n"
    f"Num_samples: {num_samples}\n"
    f"Shape: {synthetic.shape}\n"
)

print(f"Saved {synthetic.shape} synthetic samples to {output_path}")
PY
  )

  cp "${latest_ckpt}" "${synthesis_dir}/${CLASS_NAME}_checkpoint"
  echo "Saved synthetic TTS-GAN outputs to ${synthesis_dir}"
}

install_dependencies
patch_sources
run_training
collect_synthesis_outputs
