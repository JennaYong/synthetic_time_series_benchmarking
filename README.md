# Synthetic Time-Series Data Benchmarking
This repo implements the unified synthetic time-series data benchmarking framework, covering dataset downloading, generation model installation, and synthesis evaluation. The current progress is denoted as below. 
### Available Datasets
✅ PTB-XL \
More datasets to come...
### Available Models
✅ SSSD-ECG \
✅ TTS-GAN — works out of the box on UniMiB; PTB-XL needed four adaptations and
has a training-length limit, see [TTS-GAN on PTB-XL](#tts-gan-on-ptb-xl) \
More models to come...
### Evaluation - Developing...

# File Structure
```text
synthetic_time_series_benchmarking/
|--model/                   # Store the generation model source codes. Not tracked by git
   |--model1/
      |--config.json        # Model configuration
      |--requirements.txt   # Evironment dependencies
      |--src/               # model source code
|
|--evaluation/              # Evaluation methods
|--synthesis/               # Generated data stored in .npy format. Not tracked by git
|--results/                 # Evaluation results. Not tracked by git. Not tracked by git
|
|--preprocess/              # Data preprocessing methods
|   |--ttsgan/              # PTB-XL adapter for TTS-GAN. Upstream model code is
|      |--ptbxl_dataLoader.py   #   never edited; the driver rebinds names at runtime
|      |--train_ptbxl_GAN.py    #   so the UniMiB path stays unchanged
|--generate_scripts/        # Model-specific training and generation scripts
|   |--generate_sssdecg.sh
|   |--generate_ttsgan.sh
|--relocate_scripts/        # Model-specific dataset relocation scripts
|   |--relocate_sssdecg.sh
|   |--relocate_ttsgan.sh         # UniMiB motion data
|   |--relocate_ttsgan_ptbxl.sh   # PTB-XL ECG data
|
|--load_model.sh            # Download specified model to model/
|--evaluate.sh              # Evaluation script
|--sync.sh                  # Local-remote synchronization script
|--job.sh                   # The remote server job script
|
|--.gitignore
|--readme.md
```

# Execution Procedure

## Synthesis Generation
### Step 1: Download Dataset
Since remote server has no outward internet connection, dataset downloads must first be done locally. Dataset downloads will be done manually. 
#### SSSD-ECG
Pre-processed dataset can be downloaded from https://figshare.com/s/43df16e4a50e4dd0a0c5?file=38890965

Expected layout after extraction:
```text
Dataset/
├── data/
│   ├── ptbxl_train_data.npy
│   ├── ptbxl_validation_data.npy
│   └── ptbxl_test_data.npy
└── labels/
    ├── ptbxl_train_labels.npy
    ├── ptbxl_validation_labels.npy
    └── ptbxl_test_labels.npy
```

### Step 2: Download Model
Model downloading also needs to be done locally. To download a model, run
```
./load_model.sh <model>
```
If no arguments are provided, the script will print available models. 

### Step 3: Relocate Dataset for Model Usage
Place the extracted dataset under `Dataset/` at the project root, then run
```
./relocate_scripts/relocate_<model>.sh
```
For TTS-GAN there are two relocate scripts, one per dataset:
- `relocate_ttsgan.sh` — UniMiB motion data (copies `UniMiB-SHAR.zip` into the model dir)
- `relocate_ttsgan_ptbxl.sh` — PTB-XL ECG data (copies the six `ptbxl_*.npy` files into `model/tts-gan/ptbxl/` and the adapter code from `preprocess/ttsgan/` into the model dir; re-run it after editing the adapters)

### Step 4: Sync Local Setup with Remote Server
To sync local setup with remote server, run
```
./sync.sh local-to-remote <destination>
```
`<destination>` is the full path (`userid@remote-server:path_to_dest`) to the target location on the remote server. 

### Step 5: Synthesis Generation on Remote Server
To train the model and generate synthetic data with it, SSH to the remote server and run
```
./job.sh <model> [dataset]
```
Run `job.sh` **directly** — do NOT use `sbatch ./job.sh <model>`. SBATCH `--time` headers are static (parsed before the script runs), so `job.sh` self-submits: it picks a per-model wall-time and submits itself via `sbatch`. Running it under `sbatch` yourself bypasses this and falls back to the static 24h header time.

Per-model wall-time:

| Model    | Dataset | `--time` | Basis |
|----------|---------|----------|-------|
| sssd-ecg | ptbxl   | 24:00:00 | `n_iters=100000` in `config_SSSD_ECG.json` |
| tts-gan  | unimib  | 10:00:00 | |
| tts-gan  | ptbxl   | 03:00:00 | measured 1:53:48 at `TTS_GAN_MAX_ITER=100000` |

**TTS-GAN + PTB-XL timing (measured on nibi, H100 MIG 20GB, batch 16):** ~14.6
it/s, so 100k iterations take under 2 hours wall-clock; the allocated 3h leaves
margin for the per-epoch checkpoint writes and the generation step. This is
faster than the 1000-step sequence length suggests because the generator stays
small even at `embed_dim=40`. The 3h budget assumes `TTS_GAN_MAX_ITER=100000`;
the recommended PTB-XL setting of ~56 epochs (about 30k iterations) finishes in
roughly 35 minutes, so requesting less wall-time explicitly clears the queue faster.

To override the wall-time manually, submit explicitly: `sbatch --time=<HH:MM:SS> job.sh <model>`.

`job.sh` handles job details and computing resource allocation, so double-check before submitting. For SSSD-ECG it runs `./generate_scripts/generate_sssdecg.sh`; for TTS-GAN, `./generate_scripts/generate_ttsgan.sh`. The generated synthesis is stored under `synthesis/<MODEL>/` (TTS-GAN directory naming is described below).

TTS-GAN trains one unconditional model per class; select dataset and class via arguments/environment variables:
```
# UniMiB motion data (default). Classes: Running (default), Jumping, ...
TTS_GAN_CLASS=Jumping ./job.sh tts-gan

# PTB-XL ECG. Classes are the 5 diagnostic superclasses: NORM (default), MI, STTC, CD, HYP
TTS_GAN_CLASS=MI ./job.sh tts-gan ptbxl
```
PTB-XL records are filtered by diagnostic superclass, derived from the 71-dim
multi-hot SCP-statement labels. Environment variables (rationale and evidence
for every default are in *TTS-GAN on PTB-XL* below — **the defaults are the
verified-good configuration; change them only deliberately**):

| Variable | Default | Meaning |
|---|---|---|
| `TTS_GAN_MAX_ITER` | 500000 | Training iterations. **For PTB-XL use ~56 epochs' worth** (per-class table below); longer runs collapse. |
| `TTS_GAN_NUM_SAMPLES` | 1000 | Samples generated from the final checkpoint. |
| `TTS_GAN_BATCH_SIZE` | 16 | Lower it only if a job runs out of GPU memory. |
| `TTS_GAN_LR_DECAY` | off | `1` decays both learning rates linearly to zero over the run. |
| `TTS_GAN_PTBXL_LABEL_MODE` | `any` | `any` = record contains the class; `exclusive` = record has only that superclass. |
| `TTS_GAN_PTBXL_NORMALIZE` | `per_sample` | Per-record, per-lead z-normalization. |
| `TTS_GAN_PTBXL_PATCH_SIZE` | 100 | Discriminator patch size; must divide 1000. |
| `TTS_GAN_PTBXL_EMBED_DIM` | 40 | Generator width per timestep; must be a multiple of 5. |

Outputs land in `synthesis/TTS-GAN/<date>_i<max_iter>[_e<embed>_p<patch>][_lrdecay]/`.
The configuration is part of the directory name because the filenames inside
only distinguish the class: two runs of the same class on the same day with
different settings would otherwise overwrite each other's samples, labels and
checkpoint silently. Runs that differ only in class still share one directory,
so submitting all five classes with the same settings keeps them together.

Outputs per run: `ttsgan_ptbxl_<class>_samples.npy` with shape `(N, 12, 1, 1000)`,
`ttsgan_ptbxl_<class>_labels.npy` with one-hot superclass rows `(N, 5)` in the
order `NORM, MI, STTC, CD, HYP`, plus the checkpoint and a config txt recording
`embed_dim` / `patch_size`. The job log prints `Synthetic sample std` and
`Synthetic diff/signal std ratio`, so a failed run is visible without copying
the npy back.

**Important: cd to the root directory (where `job.sh` is located) before submission so relative paths resolve correctly.**

### Step 6: Acquire Generated Synthesis from Remote Server
To acquire the generated synthetic data from remote server to local, run 
```
./sync.sh remote-to-local <path_to_synthesis_dir>
```
`<path_to_synthesis_dir>` is the full path (`userid@remote-server:path_to_synthesis_dir`) to the `synthesis/` directory
## Synthesis Evaluation - Developing...

# TTS-GAN on PTB-XL

TTS-GAN was published on UniMiB SHAR: 3-axis accelerometer data, 151 timesteps,
9 activity classes. PTB-XL is a harder target for the same architecture —
12 leads, 1000 timesteps, and physiological structure (regular beats, a fixed
linear relationship between leads). Everything below is what that gap cost, and
what is known to work. All adapter code lives in `preprocess/ttsgan/`; upstream
`GANModels.py` and `train_GAN.py` are never edited — the driver rebinds names at
runtime, so the UniMiB path is bit-for-bit unchanged.

## Verified-good configuration

`embed_dim=40`, `patch_size=100`, `per_sample` normalization, and **56 epochs**.
Because a fixed iteration count means different epoch counts per class, set
`TTS_GAN_MAX_ITER` per class:

| Class | Train records | iters/epoch (bs 16) | `TTS_GAN_MAX_ITER` for 56 epochs | Runtime |
|---|---|---|---|---|
| NORM | 8564 | 536 | 30016 | ~35 min |
| MI | 4933 | 309 | 17304 | ~20 min |
| STTC | 4727 | 296 | 16576 | ~19 min |
| CD | 4409 | 276 | 15456 | ~18 min |
| HYP | 2392 | 150 | 8400 | ~10 min |

All five classes can be submitted at once; they write distinct filenames into a
shared directory. Throughput is ~14.6 it/s on an H100 MIG 20GB slice.

## Four changes that were needed, and why

Each was isolated by a separate experiment; the first three are defaults now,
the fourth is a constraint on how you run it.

**1. Anchor the generator's output scale** (`ScaleNormalizedGenerator`). The
generator has no bounded output activation and no final LayerNorm, so nothing
holds its output at the data scale. The discriminator's head ends in
`LayerNorm -> Linear`, which is scale-blind: measured on the PTB-XL geometry,
its output moves by 0.004 as the input std goes from 1000 to 10000. Once the
generator drifts out of range, the discriminator saturates, returns a constant
for every fake, and the generator's gradient dies. Unanchored, a 100k run
produced samples with std 22147 against real data at std 1.

**2. Discriminator patch size 25 -> 100.** The head mean-pools its patch tokens
before that LayerNorm. Averaging 40 tokens (patch 25) collapses all fake
embeddings onto nearly the same vector — measured spread 0.102, versus 0.187 at
patch 100 — so the discriminator cannot tell fakes apart. Patch 100 gives 10
tokens plus cls, the same count as UniMiB's 150/15. The cost is coarser time
resolution in the discriminator (1 s per token instead of 0.25 s).

**3. Generator `embed_dim` 10 -> 40.** The blocks hardcode 5 attention heads, so
the upstream default leaves each head 2 dimensions to model a 1000-step 12-lead
record. At 40 (8 dims per head, ~4.1M generator parameters) the run survives
several times longer before degrading.

**4. Cap training at ~56 epochs.** The stability window is measured in epochs,
not iterations. At a fixed 30k iterations NORM (56 epochs) scored 0.941 while
MI (98 epochs) and STTC (102 epochs) degraded to 1.389 and 1.506; equalizing to
56 epochs brought all five to 1.035-1.085. Every 187-epoch run collapsed.

**The failure signature is always identical and easy to spot:** both LSGAN
losses freeze at exactly 0.25 — solving the objectives gives D(real)=1.0 and
D(fake)=0.5, i.e. the discriminator has stopped discriminating — and the output
becomes pointwise white noise. `grep "D loss" logs/job-<id>.out` and look for
dead-flat 0.250/0.250; a healthy run keeps both losses fluctuating in roughly
0.24-0.40.

Linear learning-rate decay (`TTS_GAN_LR_DECAY=1`) helps but does not lift the
ceiling: at 187 epochs it improved the ratio from 1.470 to 1.064, still worse
than a plain 56-epoch run.

## Results

Per-record z-normalized, lead II. `diff/signal` is the std of the first
difference over the std of the signal — pointwise white noise scores ~1.41.
Real and SSSD-ECG are measured on 400 records per class (HYP: 245 for
SSSD-ECG); TTS-GAN on its full 1000 generated samples per class.

| Class | Source | diff/signal | HR-band energy | Autocorr peak |
|---|---|---|---|---|
| NORM | Real | 0.663 | 15.2% | 0.176 |
| | SSSD-ECG | 0.664 | 16.1% | 0.271 |
| | TTS-GAN | 0.939 | 13.7% | 0.089 |
| MI | Real | 0.568 | 26.4% | 0.167 |
| | SSSD-ECG | 0.589 | 27.7% | 0.160 |
| | TTS-GAN | 1.060 | 18.3% | 0.050 |
| STTC | Real | 0.619 | 20.1% | 0.142 |
| | SSSD-ECG | 0.637 | 19.8% | 0.152 |
| | TTS-GAN | 1.085 | 13.5% | 0.063 |
| CD | Real | 0.538 | 28.4% | 0.177 |
| | SSSD-ECG | 0.527 | 29.5% | 0.186 |
| | TTS-GAN | 1.035 | 18.9% | 0.050 |
| HYP | Real | 0.630 | 21.7% | 0.127 |
| | SSSD-ECG | 0.638 | 21.5% | 0.120 |
| | TTS-GAN | 1.057 | 18.7% | 0.053 |

SSSD-ECG tracks the real data to within 0.021 on every class. TTS-GAN is
smoother than white noise and shows isolated spikes, but has not learned
recognizable QRS morphology or a stable rhythm — its autocorrelation peak is
30-40% of the real value.

**Lead consistency** (`|II - (0.5*I + aVF)| / std`, on unnormalized data): real
PTB-XL 0.000000, SSSD-ECG 0.000000, TTS-GAN 0.672. A 12-lead ECG has only 8
independent leads; SSSD-ECG generates 8 and derives the rest by the standard
linear formula, so it satisfies the relation by construction. TTS-GAN generates
all 12 independently and has no such constraint. Note this metric is not
strictly comparable for TTS-GAN, whose output is per-lead normalized by design.

## Known issues

- **Do not compute lead consistency after per-record z-normalization.**
  Normalizing each lead by its own std destroys the linear relation and makes
  real data score ~0.26 instead of 0. Use unnormalized signals for that metric.
- **`diff/signal` is a health check, not a quality score.** Low-pass filtering
  the collapsed output with a width-5 moving average drops it to 0.401 — better
  than real data at 0.657 — without producing anything resembling an ECG. For
  the same reason, do not "fix" the metric by giving the generator's output
  convolution a temporal kernel (it is currently 1x1, i.e. pointwise in time):
  that would buy the number and not the morphology.
- **Absolute amplitude is not preserved.** `per_sample` normalization is
  required for stable training, so compare all sources on per-record
  z-normalized signals.
- **The two models are not solving the same task.** SSSD-ECG is one conditional
  model over the full 71-dim label vector; TTS-GAN is five unconditional models,
  one per superclass, and so uses roughly 5x the training compute in total. Its
  outputs also carry only a one-hot superclass label. State this when comparing.
- **`label_mode=any` means classes overlap.** A record with several superclasses
  appears in several training sets. `exclusive` is cleaner but leaves HYP with
  only 480 records, too few to train on.
- **`--diff_aug` in `generate_ttsgan.sh` is dead.** `cfg.py` defines it but no
  code reads it; it is inherited from the TransGAN codebase.
- **Checkpoints are tied to their `embed_dim`.** Loading one with a different
  value fails loudly, which is intended — but it means old checkpoints cannot be
  reused after changing that setting.

## Recommended next steps

1. **Shorter windows — the highest-value experiment.** Cut each 10 s record into
   four 2.5 s windows (250 steps). That puts the sequence length near the 150
   steps the architecture was tuned for, where its output is healthy, and it
   also multiplies the training set by four, which most helps the smallest class
   (HYP 2392 -> 9568). The cost is a protocol change: real data and SSSD-ECG
   output must be cropped to the same window for comparison. This is standard
   practice in the ECG generation literature and is worth doing before spending
   more on hyperparameters. Implementation is a windowing option in
   `ptbxl_dataLoader.py` plus `SEQ_LEN` and `patch_size`.
2. **Build `evaluation/`.** `diff/signal` was built to detect training failure
   quickly and should not be the reported quality metric. A benchmark needs
   several axes: morphology, power spectra, distributional distance, and a
   train-on-synthetic / test-on-real classifier score.
3. **Only then consider more capacity** (`embed_dim` 40 -> 80, `depth` 3 -> 5).
   Expected value is lower than the windowing change, and larger generators may
   shorten the stability window further.
