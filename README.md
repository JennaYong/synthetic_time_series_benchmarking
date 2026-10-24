# Synthetic Time-Series Data Benchmarking
This repo implements the unified synthetic time-series data benchmarking framework, covering dataset downloading, generation model installation, and synthesis evaluation. The current progress is denoted as below. 
### Available Datasets
✅ PTB-XL \
✅ MIMIC-IV — ICU stays on a 72-hour hourly grid, built by `preprocess/mimic_pyhealth.py` \
More datasets to come...
### Available Models
✅ SSSD-ECG \
✅ TTS-GAN — works out of the box on UniMiB; PTB-XL needed four adaptations and
has a training-length limit, see [TTS-GAN on PTB-XL](#tts-gan-on-ptb-xl);
MIMIC-IV runs through its own adapter, see [TTS-GAN on MIMIC-IV](#tts-gan-on-mimic-iv) \
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
|   |--mimic_pyhealth.py    # MIMIC-IV -> (n_stays, 16, 72) tensors, shared by all models
|   |--ttsgan/              # Dataset adapters for TTS-GAN. Upstream model code is
|      |--ptbxl_dataLoader.py   #   never edited; the drivers rebind names at runtime
|      |--train_ptbxl_GAN.py    #   so the UniMiB path stays unchanged
|      |--mimic_dataLoader.py   # MIMIC-IV: signal selection, scaling, class filter
|      |--train_mimic_GAN.py    # MIMIC-IV: training driver and output anchors
|      |--generate_mimic.py     # MIMIC-IV: sampling back to clinical units + checks
|--generate_scripts/        # Model-specific training and generation scripts
|   |--generate_sssdecg.sh
|   |--generate_ttsgan.sh
|--relocate_scripts/        # Model-specific dataset relocation scripts
|   |--relocate_sssdecg.sh
|   |--relocate_ttsgan.sh         # UniMiB motion data
|   |--relocate_ttsgan_ptbxl.sh   # PTB-XL ECG data
|   |--relocate_ttsgan_mimic.sh   # MIMIC-IV ICU data
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
For TTS-GAN there are three relocate scripts, one per dataset:
- `relocate_ttsgan.sh` — UniMiB motion data (copies `UniMiB-SHAR.zip` into the model dir)
- `relocate_ttsgan_ptbxl.sh` — PTB-XL ECG data (copies the six `ptbxl_*.npy` files into `model/tts-gan/ptbxl/` and the adapter code from `preprocess/ttsgan/` into the model dir; re-run it after editing the adapters)
- `relocate_ttsgan_mimic.sh` — MIMIC-IV ICU data (copies the output of `preprocess/mimic_pyhealth.py` into `model/tts-gan/mimic/` and the MIMIC adapter code into the model dir; re-run it after editing the adapters or regenerating the data). It reads `processed_mimic_pyhealth/` at the project root by default, or the directory named by `MIMIC_DATA_DIR`. That directory is gitignored: MIMIC-IV is credentialed data and must never be committed.

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
| tts-gan  | mimic   | 02:00:00 | measured 1:16:05 at `TTS_GAN_MAX_ITER=110000` with 21,860 samples; set `TTS_GAN_MAX_ITER` explicitly |

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

# MIMIC-IV ICU stays. Classes: all (default, unconditional), survived, died
TTS_GAN_MAX_ITER=110000 TTS_GAN_NUM_SAMPLES=21860 ./job.sh tts-gan mimic
```
The MIMIC-IV options and outputs are described in
[TTS-GAN on MIMIC-IV](#tts-gan-on-mimic-iv).

The rest of this step is about PTB-XL.
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
| `TTS_GAN_PTBXL_PATCH_SIZE` | window/10 | Discriminator patch size; must divide the window. The default keeps 10 tokens at any window length. |
| `TTS_GAN_PTBXL_EMBED_DIM` | 40 | Generator width per timestep; must be a multiple of 5. |
| `TTS_GAN_PTBXL_WINDOW` | 1000 | Timesteps per training item; must divide 1000. `250` splits each record into four 2.5 s windows. |

Outputs land in `synthesis/TTS-GAN/<date>_i<max_iter>[_w<window>_e<embed>_p<patch>][_lrdecay]/`.
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

## Two working configurations

The window and capacity ablation produced a second, better configuration. Both
are kept because they are not interchangeable — they generate different signal
lengths.

| | Full-length | Short-window |
|---|---|---|
| `TTS_GAN_PTBXL_WINDOW` | 1000 (default) | 250 |
| `TTS_GAN_PTBXL_EMBED_DIM` | 40 (default) | 80 |
| `TTS_GAN_PTBXL_PATCH_SIZE` | 100 (auto) | 25 (auto) |
| Training length | 56 epochs per class | ~8000 gradient steps per class |
| Output shape | `(N, 12, 1, 1000)` | `(N, 12, 1, 250)` |
| Use it for | direct comparison against SSSD-ECG, which emits 10 s records | best available TTS-GAN morphology, if the comparison set is cropped to 2.5 s |

`diff/signal` gap against a real-data baseline computed at the same window
length (a 2.5 s crop of real PTB-XL scores 0.571-0.700, not 0.66):

| Class | Full-length gap | Short-window gap |
|---|---|---|
| NORM | +0.276 | **+0.209** |
| MI | +0.492 | **+0.362** |
| STTC | +0.466 | **+0.299** |
| CD | +0.497 | **+0.449** |
| HYP | +0.427 | **+0.312** |

The short-window configuration closes roughly a quarter of the gap and is the
first setting whose output shows sharp isolated spikes on a comparatively flat
baseline. It is still clearly distinguishable from real ECG.

## What the ablation showed about stability

Collapse point by configuration, measured from the loss trace (both LSGAN
losses frozen at 0.25):

| Configuration | Collapse at |
|---|---|
| window 1000, embed 40 | > 30,000 steps (56 epochs) |
| window 1000, embed 80 | ~14,500 steps (~27 epochs) |
| window 250, embed 80 | ~8,500 steps (~4 epochs) |

Both shortening the window and widening the generator **shorten** the stability
window in gradient steps, so each configuration has to be stopped at its own
point rather than trained to a common budget. Which quantity drives the
collapse also changes with the window: at 1000 steps per record, equalizing
epochs across classes fixed it, while at 250 steps HYP ran 13.4 epochs without
collapsing and matching gradient steps was the right protocol instead.

A training-length sweep at window 250 / embed 80 (4k, 6k, 8k, 12k, 16k steps)
put the quality peak at 8000, just before the collapse — and `diff/signal`
ranked 6000 higher, another reason not to trust it for ranking.

## Recommended next steps

1. **Build `evaluation/`.** `diff/signal` was built to detect training failure
   quickly and repeatedly proved unreliable as a quality score: it ranked a
   low-pass-filtered collapse above real data, and it ranked a 6000-step run
   above the 8000-step run that is visibly better. Every quality judgement in
   this work ended up being made from waveform plots. A benchmark needs
   morphology, power spectra, distributional distance, and a train-on-synthetic
   / test-on-real classifier score.
2. **Extend the stability window before spending capacity on it.** Every failure
   so far is the discriminator winning outright, and the two cheapest untried
   remedies target that directly: the discriminator learns 3x faster than the
   generator by default (`TTS_GAN_D_LR` 3e-4 against `TTS_GAN_G_LR` 1e-4), and
   `TTS_GAN_LOSS=wgangp` selects an objective that upstream already implements
   and that is far less prone to this particular vanishing-gradient collapse.
   Note that `--d_spectral_norm`, the other standard stabiliser, is defined in
   `cfg.py` but read nowhere in the training code -- using it means writing it.
3. **Then raise model capacity.** `latent_dim` (100) and the generator's `depth`
   (3) are both untried, and neither needs a change to the model code -- only
   the same runtime rebinding `embed_dim` already uses. Do this after step 2,
   not before: the one capacity change already measured, `embed_dim` 40 -> 80,
   improved quality slightly but cut the stability window from >30k gradient
   steps to ~14,500, so extra capacity is currently paid for in training length.
   `latent_dim` additionally appears hardcoded as 100 in the generation step of
   `generate_ttsgan.sh` and in upstream's `gen_plot`, so both need updating with
   it or generation will fail on a shape mismatch.
4. **Reduce the difficulty of the task itself.** Windows are currently cut at
   arbitrary offsets, so the generator has to learn where beats fall as well as
   what they look like; centring each window on a detected beat removes half of
   that. Generating 8 leads and deriving the other 4 by the standard formula --
   as SSSD-ECG does -- would also stop the model spending capacity on 4 leads
   that are not independent, and would make its output physiologically
   consistent by construction.

# TTS-GAN on MIMIC-IV

MIMIC-IV is the benchmark's EHR dataset: ICU stays on a 72-hour hourly grid,
written by `preprocess/mimic_pyhealth.py` as `(n_stays, 16, 72)` tensors in raw
clinical units (a value and a mask channel for each of eight signals). Its geometry (72 steps, 5-8 signals) is much closer to the
UniMiB data TTS-GAN was published on (150 steps, 3 axes) than PTB-XL is, but
it brings a constraint PTB-XL did not have: **the absolute level of a signal is
the clinical information**, so the per-record normalization that made PTB-XL
trainable cannot be used.

The adapter lives in `preprocess/ttsgan/` (`mimic_dataLoader.py`,
`train_mimic_GAN.py`, `generate_mimic.py`) and follows the PTB-XL pattern:
upstream code is never edited, and the UniMiB and PTB-XL paths are unchanged.

**Benchmark rules this adapter follows** (agreed 2026-10-13): every model
trains on the same complete data matrix, imputed slots included; the mask is
used only at evaluation; and each model keeps its original training objective.
Unconditional models, TTS-GAN among them, train on the whole training set;
only conditional models get a conditional input (decided 2026-10-16). So the
adapter only applies invertible transforms to the shared tensor by default
(see *What the model is trained on*), the upstream LSGAN loss is unchanged,
and the benchmark run is `TTS_GAN_CLASS=all`.

**Status:** the current release is V3 (`processed_mimic_pyhealth_V3`,
2026-10-16). With all eight signals the recommended setting is the defaults
with `TTS_GAN_MAX_ITER=110000`. The model reproduces the vitals reasonably and
the three labs poorly; *Results with all eight signals* has the measurements.
The output anchor and the use of the averaged (EMA) weights were settled
earlier, on the first data release (`processed_mimic_pyhealth` of 2026-10-11)
with the five vitals only (*Anchor comparison*, *Training length and
stability*). The comparisons were run locally; the recommended setting has
also run on nibi (see *How it was verified*).

## Running it

```
MIMIC_DATA_DIR=<preprocessed dir> ./relocate_scripts/relocate_ttsgan_mimic.sh   # locally, then sync.sh
TTS_GAN_MAX_ITER=110000 TTS_GAN_NUM_SAMPLES=21860 ./job.sh tts-gan mimic   # on the cluster
TTS_GAN_CLASS=died TTS_GAN_MAX_ITER=20000 ./job.sh tts-gan mimic   # per-class: not tested at length
```

| Variable | Default | Meaning |
|---|---|---|
| `TTS_GAN_CLASS` | `all` | `all` trains one unconditional model; `survived` / `died` train on one outcome only. |
| `TTS_GAN_MAX_ITER` | 500000 | Training iterations. The default is the UniMiB one; always set it. One epoch is 1367 iterations at batch 16 for `all` on the current release (`survived` 1122, `died` 245), and upstream rounds up to whole epochs. 110000 (81 epochs of `all`) is the recommended setting. |
| `TTS_GAN_MIMIC_SIGNALS` | `all` | `all` (heart_rate, sbp, dbp, resp_rate, spo2, glucose, creatinine, sodium: the whole data matrix), `vitals` (the first five) or a comma-separated list of at least three signals. |
| `TTS_GAN_MIMIC_ANCHOR` | `none` | How the generator output is held at the data scale: `none`, `batch` or `tanh`. See below. |
| `TTS_GAN_MIMIC_WEIGHTS` | `ema` | Which generator weights to sample from: `ema` (the moving average upstream keeps during training) or `raw` (the last training step). Affects generation only. |
| `TTS_GAN_MIMIC_PATCH_SIZE` | 8 | Discriminator patch size; must divide 72. 8 gives 9 tokens, close to the 10 that PTB-XL needed. |
| `TTS_GAN_MIMIC_EMBED_DIM` | 10 | Generator width per timestep (upstream's UniMiB value); a multiple of 5. |
| `TTS_GAN_MIMIC_HEAD_FILL` | `none` | `none` feeds the shared tensor as it is. `backfill` is an experiment-only option that alters the data, see below. |

`TTS_GAN_BATCH_SIZE`, `TTS_GAN_D_LR`, `TTS_GAN_G_LR`, `TTS_GAN_LOSS`,
`TTS_GAN_LR_DECAY` and `TTS_GAN_NUM_SAMPLES` work as for PTB-XL.

Outputs land in
`synthesis/TTS-GAN/<date>_i<max_iter>_mimic_s<signals>_e<embed>_p<patch>_<anchor>/`
(with `_rawweights` appended for `TTS_GAN_MIMIC_WEIGHTS=raw`):

- `ttsgan_mimic_<class>_samples.npy` — float32 `(N, C, 72)` **in clinical
  units**, axes `(stay, signal, hour)` like the real tensor. With all eight
  signals this is the benchmark's delivery format, `(N, 8, 72)`: the value
  channels of the shared tensor, in its order, without masks. `N` is
  `TTS_GAN_NUM_SAMPLES` (default 1000; 21860, the number of training stays,
  for the benchmark run).
- `ttsgan_mimic_<class>_labels.npy` — `(N,)`, `mortality_in_hospital`; only for
  `survived` / `died`.
- `ttsgan_mimic_<class>_config.json` — the signal list, the scaling that was
  inverted and the health check below.
- the checkpoint.

## What the model is trained on

By default the loader changes the shared tensor only in invertible ways. All
statistics come from the train split.

1. **Signal selection.** All eight value channels; the masks are not fed to
   the model.
2. **Log transform** of glucose and creatinine, which are right-skewed.
3. **Global per-channel z-score**, never per record.

Generated samples are mapped back through 3 and 2 into clinical units and
clipped to the physiological bounds that the shared preprocessing clips the
real data to (`clip_bounds` in `channels.json`: heart rate 20 - 300, SBP
40 - 300, DBP 20 - 200, respiratory rate 4 - 120, SpO2 50 - 100, glucose
10 - 1000, creatinine 0.1 - 15, sodium 110 - 165); that is the only constraint
added. Releases before V3 have no bounds, and there the output is only floored
at 0. The model is trained on the imputed values too (forward fill, and the
train median before a channel's first reading), so it reproduces them; its
output has no mask. Implausible readings (heart rate 1, glucose 5840) are
clipped by the shared preprocessing from V3 on; the loader adds no clipping of
its own.

Two options alter the data and are kept for experiments only:
`TTS_GAN_MIMIC_HEAD_FILL=backfill` replaces the slots before a channel's first
reading with the stay's own first reading, and the `tanh` anchor's min-max
scaling clips each channel to the 0.5 / 99.5 percentiles of the measured train
values. Both were defaults on the first data release, when the slots before
the first reading held 0.

**Class labels** come from `mortality_in_hospital` in `annotations.json`, i.e.
death during the stay's own admission (3919 of 21860 train stays).

**Mask convention.** From V3 on, a mask channel is 1 where the value was
*measured* and 0 where it was imputed, the convention the evaluation uses
(`values[mask == 1]`). The two earlier releases stored the opposite. The loader
reads the convention off the data (heart rate is measured in ~96% of hours, so
whichever value dominates its mask means "measured"), loads every release
correctly, and records the convention in the config file of each run.

## Results with all eight signals

The settings were compared on the release of 2026-10-13. V3 differs from it
only in the mask convention, which the model never sees, and in the clipping
of under 0.1% of the slots, so the recommended setting was then trained twice
more on V3, locally and on nibi (the last table). Class `all`, all eight
signals. Unless a column says otherwise: the defaults above (anchor `none`,
patch 8, `embed_dim` 10), LSGAN, batch 16, learning rates G 1e-4 / D 3e-4,
samples from the averaged weights. Each setting was trained once, locally (see
*How it was verified*); the later epochs of a setting continue the same run.
Samples are compared with the real train split, and the real test split is
listed as the score of a perfect generator.

| | Test split | `embed_dim` 10, 40 epochs | **10, 80 epochs** | 10, 120 epochs | 20, 40 epochs | 40, 80 epochs | 10, D lr 1e-4, 40 epochs |
|---|---|---|---|---|---|---|---|
| Hourly values, W1 / real std: vitals | 0.009 | 0.137 | 0.139 | 0.148 | 0.144 | 0.204 | 0.226 |
| ... labs | 0.019 | 0.296 | 0.257 | 0.230 | 0.216 | 0.280 | 0.209 |
| Autocorrelation error: vitals | 0.002 | 0.075 | 0.070 | 0.068 | 0.115 | 0.079 | 0.066 |
| ... labs | 0.011 | 0.286 | 0.265 | 0.239 | 0.292 | 0.168 | 0.317 |
| Cross-signal correlation error | 0.007 | 0.040 | 0.033 | 0.034 | 0.047 | 0.070 | 0.063 |
| Level spread, synthetic / real: vitals | 0.98 - 1.00 | 0.87 - 0.96 | 0.87 - 1.05 | 0.73 - 0.96 | 0.79 - 0.99 | 0.32 - 0.77 | 0.44 - 0.78 |
| ... labs | 1.00 - 1.01 | 0.36 - 0.53 | 0.47 - 0.52 | 0.51 - 0.58 | 0.58 - 0.72 | 0.46 - 0.56 | 0.47 - 0.77 |
| Roughness, synthetic / real: labs | 1.0 | 2.2 - 5.3 | 2.1 - 4.5 | 2.0 - 3.6 | 2.1 - 2.9 | 1.4 - 2.1 | 2.1 - 4.4 |
| 99th percentile, synthetic / real: glucose, creatinine | 0.99, 1.01 | 0.62, 0.27 | 0.64, 0.37 | 0.63, 0.42 | 0.66, 0.41 | 0.63, 0.29 | 0.71, 0.29 |

4000 samples per column. W1 is the Wasserstein distance between the
distributions of hourly values, averaged over the five vitals or the three
labs. Autocorrelation error is the mean absolute error of the within-stay
autocorrelation at lags of 1, 2, 4, 8 and 12 hours; correlation error is
defined under *Training length and stability*. Level spread and roughness are
defined under *Health check*, measured here on the scale the model works on
(log for glucose and creatinine) so that the missing tails do not dominate
them; ranges are over the signals in the group.

Three of the settings were evaluated further (14,000 samples; GRU classifiers
and forecasters trained on 10,000 real and 10,000 synthetic stays, tested on
held-out stays):

| | Real test split | `embed_dim` 10, 40 epochs | **10, 80 epochs** | 20, 40 epochs |
|---|---|---|---|---|
| Real-vs-synthetic accuracy (0.5 = indistinguishable) | 0.50 | 0.991 | 0.986 | 0.984 |
| ... on 6-hour means only | | 0.958 | 0.958 | 0.967 |
| Next-hour forecast error, trained on synthetic (TSTR) | 0.276 (trained on real) | 0.329 | 0.319 | 0.340 |
| Autocorrelation error, lags 1 - 48 h | 0.005 | 0.124 | 0.116 | 0.144 |

Forecast error is the MAE on real test stays in units of the real std;
predicting that the next hour equals the current one scores 0.301.

What these show:

- **With the labs added the model takes longer to train, but the vitals end
  up about as good.** A vitals-only control on the same release reaches its
  plateau by about epoch 8. With all eight signals the cross-signal
  correlation error is still 0.14 at epoch 10 and reaches 0.035 at epoch 20.
  At epoch 40 the vitals of the eight-signal model and those of the
  vitals-only model score alike: real-vs-synthetic accuracy 0.98 vs 0.97,
  TSTR 0.432 vs 0.426 (trained on real: 0.412; persistence: 0.462),
  autocorrelation error over lags 1 - 48 h 0.061 vs 0.100. Only the marginals
  are matched better by the vitals-only model (W1 0.04 - 0.15 vs 0.10 - 0.23).
- **The labs are the weak part under every setting tried.** After forward
  fill a real lab is a step function: 92 - 94% of its hours repeat the
  previous value, and most of its variation is between stays. The synthetic
  labs never repeat a value, change 2 - 5 times as much from hour to hour,
  spread about half as much between stays, and lack the upper tail (at 80
  epochs the 99th percentile of creatinine is 3 against 8 mg/dL, of glucose
  213 against 335 mg/dL).
- **This is why TSTR loses to persistence.** On the vitals alone, a forecaster
  trained on synthetic data beats "next hour = this hour"; on all eight
  signals it does not, because the synthetic labs teach it that labs move
  every hour.
- **A wider generator trades the vitals for the labs.** `embed_dim` 40 gives
  the smoothest labs but shrinks the between-stay spread of the vitals to
  0.32 - 0.77 of the real one; `embed_dim` 20 gives the widest lab spread but
  a between-stay SBP-DBP correlation of 0.85 against a real 0.43, and the
  worst TSTR. A lower discriminator learning rate (1e-4) is no better overall
  than the default.
- **At the default width the labs keep improving slowly with training.** From
  40 to 80 epochs the lab measures, TSTR and the real-vs-synthetic accuracy
  improve in both runs of this setting. The vitals hold in the first run until
  epoch 100, after which their between-stay spread shrinks (0.73 - 0.96 at
  120); in the second run the lowest, SBP and SpO2, is already down to 0.68
  by epoch 80. No run collapsed: after epoch 5 the epoch-mean losses stay at
  D 0.26 - 0.39 / G 0.19 - 0.25 with all eight signals at the default learning
  rates, and at D 0.55 / G 0.11 for the vitals only, as on the first release.
- **Runs of the same setting differ about as much as 80 and 120 epochs do**
  (next table). With one run per setting, only the larger differences in the
  first table stand out from that: what a wider generator costs the vitals,
  and the lab improvement from 40 to 80 epochs.

The recommended setting was trained three times: the run in the tables above,
a second local run on V3, and the benchmark run on nibi (V3, job 24135376,
21,860 samples). Same measures as above, at epoch 80 (81 on nibi), computed
from the saved samples, so they can differ from the tables above in the last
digit:

| | First run, local | Second run, V3, local | **Third run, V3, nibi** |
|---|---|---|---|
| Real-vs-synthetic accuracy | 0.986 | 0.986 | 0.989 |
| ... on 6-hour means only | 0.958 | 0.961 | 0.955 |
| TSTR (persistence 0.301 - 0.303, trained on real 0.276 - 0.277) | 0.319 | 0.326 | 0.325 |
| Autocorrelation error, lags 1 - 48 h | 0.116 | 0.124 | 0.123 |
| Cross-signal correlation error | 0.031 | 0.042 | 0.035 |
| Hourly values, W1 / real std: vitals, labs | 0.14, 0.26 | 0.14, 0.30 | 0.15, 0.28 |
| Lowest between-stay spread of a vital, synthetic / real | 0.86 | 0.68 | 0.71 |

The nibi run, the first through the real CUDA path, lands within the spread
of the two local ones. Since V3 the samples are clipped to `clip_bounds`, so
SpO2 now piles up at 100 like the real data (9.6% of synthetic values, 21% of
real ones) instead of running past it.

**Recommended: the defaults with `TTS_GAN_MAX_ITER=110000`** (81 epochs).
`embed_dim` 10 is upstream's own value and no other width was better overall.
From 40 to 80 epochs both runs gain on the labs, TSTR and the
real-vs-synthetic accuracy. On nibi the whole job, 110000 iterations plus
21,860 samples, takes 1 h 16 min.

Gaps that remain at that setting, besides the labs: the autocorrelation is
gone after about 8 hours (real data: 13 - 20 hours); the within-stay
variability is too uniform across stays (heart rate: 10th - 90th percentile of
the per-stay std 9 - 15 bpm against a real 6 - 16); the mean at each hour
deviates from the real one by up to 0.3 real std, in a fixed pattern over the
72 hours (test split: under 0.04); and the values are continuous, whereas the
real vitals are 91 - 97% integers, and 43 - 46% of the stays start with the
train-median fill of the vitals (86% for the labs) before their first reading.

## Output anchor

The generator ends in a 1x1 convolution with no activation and the
discriminator head is blind to input scale, which is why every unanchored
PTB-XL run diverged (see *Four changes that were needed* above). PTB-XL's fix
z-normalizes each generated record over time; that erases the record's level
and is not usable here. Whether MIMIC needs an anchor at all was measured
rather than assumed. Three settings are implemented:

| `TTS_GAN_MIMIC_ANCHOR` | Data scaling | Generator output | Note |
|---|---|---|---|
| `none` (default) | z-score | raw | Upstream TTS-GAN with nothing added. |
| `batch` | z-score with the training subset's mean/std | standardized per channel over (batch, time) | Pins the scale but lets records in a batch differ in level. It also forces the synthetic mean and std of every channel to match the real ones exactly, so those two statistics say nothing about model quality under this anchor. |
| `tanh` | clip range mapped to [-0.8, 0.8] | `tanh` | Bounded by construction. The margin keeps common values such as SpO2 = 100 inside tanh's reachable range. |

None of them has parameters, so a checkpoint can be loaded under any anchor;
what must match the checkpoint is the signal list and `embed_dim`.

## Anchor comparison

Five vitals, class `all`, `embed_dim` 10, patch 8, LSGAN with the default
learning rates, batch 16, 20 epochs (27,260 iterations), 1000 samples from the
final checkpoint's last-step (`raw`) weights. "Real" is the train split as the
model saw it; the test split is listed as the score a perfect generator would
get.

| | Real | Test split | `none` | `batch` | `tanh` |
|---|---|---|---|---|---|
| Lag-1 autocorrelation within a stay: HR | 0.72 | 0.72 | 0.74 | 0.80 | 0.50 |
| ... SBP | 0.49 | 0.50 | 0.55 | 0.63 | 0.12 |
| ... DBP | 0.40 | 0.40 | 0.48 | 0.51 | 0.15 |
| ... resp. rate | 0.40 | 0.40 | 0.54 | 0.63 | 0.14 |
| ... SpO2 | 0.46 | 0.46 | 0.52 | 0.62 | 0.15 |
| SBP-DBP correlation | 0.49 | 0.48 | 0.39 | 0.43 | 0.19 |
| Cross-signal correlation, mean abs. error | 0 | 0.006 | 0.084 | 0.138 | 0.143 |
| Heart rate, 5 / 50 / 95 percentile | 58 / 84 / 119 | 59 / 85 / 119 | 60 / 86 / 121 | 58 / 85 / 119 | 61 / 93 / 126 |
| Std, synthetic / real, range over signals | 1 | | 0.72 - 0.99 | 0.97 - 1.00 | 0.93 - 1.08 |
| Level spread, synthetic / real, range | 1 | | 0.82 - 1.09 | 1.00 - 1.19 | 0.83 - 1.12 |

- **No run collapsed.** In all three the epoch-mean losses settle near
  D 0.55 / G 0.11 within the first dozen epochs and stay there; the frozen
  0.25 / 0.25 signature that ended every long PTB-XL run did not appear.
- **`none` did not drift.** The unanchored generator that ran away to std
  ~22000 on PTB-XL stays within about 30% of the real amplitude here at every
  checkpoint examined, and gives the closest temporal and cross-signal
  structure. It is the default because it is also the least modified model.
  Its amplitude is the least well calibrated of the three (SpO2 std at 0.72
  of the real one).
- **`batch` is comparable and too smooth** (autocorrelation above the real
  value on every signal). Its amplitude match is by construction. It is the
  fallback if `none` drifts in longer training.
- **`tanh` is clearly worse**: hour-to-hour autocorrelation far below the real
  one, i.e. visibly noisy output, and a shifted heart-rate distribution.

These runs used the local CPU shim described under *How it was verified*, with
one seed per anchor and no repeat, so the small differences between `none` and
`batch` should not be over-read. The gap to `tanh` is large on every temporal
measure, and it persists with the averaged weights and through 80 epochs.

## Training length and stability

Each anchor was then trained for 80 epochs (109,040 iterations) with a snapshot
after every epoch.

- **Nothing collapsed.** For `none` and `batch` the epoch-mean losses stay at
  D 0.54-0.56 / G 0.11-0.12 from epoch 6 to epoch 80; `tanh` reaches the same
  level by about epoch 12. PTB-XL's stability window (56 epochs at best) has no
  counterpart here within the range tried.
- **`none` never drifted.** Its amplitude stays between 0.71 and 1.12 of the
  real one at every snapshot examined.
- **Quality is reached by about epoch 10 and does not improve afterwards.**
- **The last-step weights wander; the averaged weights do not.** Upstream keeps
  an exponential moving average of the generator and saves it in every
  checkpoint (`avg_gen_state_dict`). Sampling from it is the default
  (`TTS_GAN_MIMIC_WEIGHTS=ema`):

| Epoch (`none`) | `raw`: correlation error | `raw`: SBP-DBP corr. | `ema`: correlation error | `ema`: SBP-DBP corr. | `ema`: std, synthetic / real |
|---|---|---|---|---|---|
| 5 | 0.051 | 0.43 | 0.022 | 0.51 | 0.81 - 0.96 |
| 10 | 0.092 | 0.54 | 0.043 | 0.50 | 0.87 - 1.00 |
| 20 | 0.089 | 0.40 | 0.028 | 0.47 | 0.87 - 0.97 |
| 40 | 0.058 | 0.54 | 0.031 | 0.51 | 0.85 - 0.97 |
| 60 | 0.110 | 0.45 | 0.047 | 0.56 | 0.82 - 0.96 |
| 80 | 0.138 | 0.53 | 0.041 | 0.51 | 0.78 - 0.93 |

  Correlation error is the mean absolute difference between the synthetic and
  the real cross-signal correlation matrix (the real test split scores 0.006);
  the real SBP-DBP correlation is 0.49. With `ema` the same holds for the other
  anchors: `batch` scores 0.014-0.032, `tanh` 0.06-0.19.

**On that release, `TTS_GAN_MAX_ITER=20000`** (15 epochs) was the recommended
setting for the vitals. It sits
inside the plateau; training longer costs time and, with the averaged weights,
slowly shrinks the amplitude (down to 0.78 of the real std by epoch 80). With
all eight signals the plateau comes much later; see *Results with all eight
signals*.

What remains off at that setting: the synthetic signals are a little too
smooth except for heart rate (lag-1 autocorrelation 0.43-0.59 against a real
0.40-0.49), and their amplitude is up to 15% low.

## Health check

The generation step prints, per signal and in clinical units, the real and
synthetic mean, standard deviation, **level spread** (std across stays of each
stay's mean — it collapses if every sample sits at the same level) and
**roughness** (std of the hour-to-hour difference over the std; pointwise white
noise gives 1.41, real vitals 0.45-0.82). It warns on NaN, on an amplitude far
from the real one, on a collapsed level spread, on noise-like roughness and on
a saturated tanh. As on PTB-XL, this detects failed runs; it does not rank
quality, which still has to be judged from waveforms until `evaluation/`
exists. With all eight signals it flags the labs as too rough on every run,
and sometimes creatinine's level spread as collapsed (0.19 - 0.29 of the real
one in clinical units across three runs, where the missing upper tail
dominates, against 0.34 - 0.46 on the model's log scale): that is the known lab gap described
above, not a failed run.

## How it was verified

- On V3 the loader was checked against the raw tensor: with the defaults the
  inverse transform reproduces the shared data to within 3e-7 relative error
  (nothing clipped or filled by the loader), the class filter returns
  17941 / 3919 train stays, and the mask is read as 1 = measured. The two
  earlier releases still load, with their masks read as 1 = imputed, and the
  `backfill` option finds the same first readings under both conventions.
  A mask that is neither mostly 0 nor mostly 1 on heart rate is refused.
- V3 was compared with the 2026-10-13 release slot by slot: its masks are
  exactly one minus the earlier ones, its values equal the earlier values
  clipped to `clip_bounds` (0.001 - 0.08% of the slots of a signal changed;
  for the labs most of them are imputed slots that carried an implausible
  reading forward), and the stays, subjects and annotations are identical.
- `generate_ttsgan.sh` gives identical experiment names, entry points and
  output directories for UniMiB and PTB-XL before and after the change.
- One epoch was trained locally through the real `train_GAN.main()` for each
  anchor, class and signal set, followed by the generation step; the runs in
  *Results with all eight signals* went through the same path (about 40 s per
  epoch on a local CPU). Upstream hardcodes CUDA, so this used a test-only
  shim that maps those calls onto the CPU; the shim is not part of the repo.
- On nibi the recommended setting ran through the real CUDA path (job
  24135376, V3, `TTS_GAN_MAX_ITER=110000`, 21,860 samples) and completed in
  1:16:05, which sets the job's 2-hour limit in `job.sh`.
- Upstream's resume option (`--load_path`) fails on the cluster path:
  checkpoints are saved from `.module`, but resuming loads them into the
  `DataParallel` wrapper, whose keys carry a `module.` prefix. A job stopped
  by its time limit therefore cannot be continued, so give it enough wall
  time. The local runs past 40 and 80 epochs were continued through the shim,
  which loads into the inner module.

## Open points

- **The labs.** No setting tried reproduces their step shape, their spread
  between stays or their upper tail (*Results with all eight signals*). Within
  the benchmark rules (the whole data matrix, the original loss) what remains
  is model capacity and training length, which trade the vitals for the labs.
  Other models face the same forward-filled lab channels, so this is worth
  comparing across models before tuning TTS-GAN further.
- **Conditioning** (decided 2026-10-16): unconditional models train on the
  whole training set, and only conditional models get a conditional input.
  TTS-GAN has no conditional input, so its benchmark run is `TTS_GAN_CLASS=all`;
  the `survived` / `died` options remain for experiments.
- Generating masks is not implemented; under the current rules masks are
  only used at evaluation, so none is needed.
