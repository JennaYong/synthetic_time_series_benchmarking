# Synthetic Time-Series Data Benchmarking
This repo implements the unified synthetic time-series data benchmarking framework, covering dataset downloading, generation model installation, and synthesis evaluation. The current progress is denoted as below. 
### Available Datasets
✅ PTB-XL \
More datasets to come...
### Available Models
✅ SSSD-ECG \
✅ TTS-GAN \
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
|   |--ttsgan/              # PTB-XL adapter for TTS-GAN (label mapping + dataset + driver)
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
faster than the 1000-step sequence length suggests because the generator's
`embed_dim` is only 10. The 3h budget assumes `TTS_GAN_MAX_ITER=100000` — a
much larger value (500000 is roughly 10h) needs a manual override.

To override the wall-time manually, submit explicitly: `sbatch --time=<HH:MM:SS> job.sh <model>`.

`job.sh` handles job details and computing resource allocation, so double-check before submitting. For SSSD-ECG it runs `./generate_scripts/generate_sssdecg.sh`; for TTS-GAN, `./generate_scripts/generate_ttsgan.sh`. The generated synthesis is stored in `synthesis/<MODEL>/{date}` where `date` is the execution timestamp.

TTS-GAN trains one unconditional model per class; select dataset and class via arguments/environment variables:
```
# UniMiB motion data (default). Classes: Running (default), Jumping, ...
TTS_GAN_CLASS=Jumping ./job.sh tts-gan

# PTB-XL ECG. Classes are the 5 diagnostic superclasses: NORM (default), MI, STTC, CD, HYP
TTS_GAN_CLASS=MI ./job.sh tts-gan ptbxl
```
PTB-XL specifics (see `preprocess/ttsgan/` for details): records are filtered by
diagnostic superclass derived from the 71-dim multi-hot SCP-statement labels
(alphabetical column order, verified against `ptbxl_database.csv`). Optional env vars:
- `TTS_GAN_PTBXL_LABEL_MODE`: `any` (default; record contains the class) or `exclusive` (record has exactly that one superclass)
- `TTS_GAN_PTBXL_NORMALIZE`: `per_sample` (default; UniMiB-style per-record z-norm) or `none` (keeps the SSSD-ECG global standardization). **Keep the default.** The generator has no bounded output activation and no final LayerNorm, so nothing anchors its output scale; it was tuned for data with std ~1, and its own output at initialization has std ~0.47. Feeding the globally standardized PTB-XL (std 0.133) instead starts the generator 3.5x above the data scale, and training runs away — a 100k-iteration run produced samples with std 8078 against real data at std 0.133, with no recognizable QRS morphology. Because absolute scale is not preserved, compare models on per-record z-normalized signals at evaluation time.
- `TTS_GAN_PTBXL_PATCH_SIZE`: discriminator patch size, must divide 1000 (default 100 — 10 tokens + cls, matching UniMiB's token count; 25 made the discriminator's mean-pooled head go blind to fakes and training collapsed by epoch ~9 with both losses frozen at 0.25)
- `TTS_GAN_PTBXL_EMBED_DIM`: generator embedding width per timestep (default 40; must be a multiple of 5 — the generator blocks hardcode 5 attention heads). The upstream default of 10 gives each attention head just 2 dimensions to model a 1000-step 12-lead record; all three full runs with it eventually fell back into the frozen-0.25 collapse even after the scale and patch fixes. Checkpoints only load with the embed_dim they were trained with.

**Known limit on PTB-XL:** training is stable for roughly the first 60 epochs
and then collapses. Runs at 5k, 10k and 30k iterations all stay healthy; every
187-epoch (100k iteration) run so far has ended in the frozen-0.25 state with
white-noise output, including after the scale, patch-size and embed-dim fixes.
`TTS_GAN_MAX_ITER=30000` is the largest verified-good setting. Set
`TTS_GAN_LR_DECAY=1` to decay both learning rates linearly to zero over the run
(upstream's `--lr_decay`, off by default) when trying to push past that window.

Shared TTS-GAN knobs (both datasets): `TTS_GAN_MAX_ITER` (default 500000),
`TTS_GAN_NUM_SAMPLES` (default 1000), `TTS_GAN_BATCH_SIZE` (default 16 — lower
it if a PTB-XL job runs out of GPU memory; the generator's attention is
seq_len x seq_len, so 1000-step ECG costs far more per sample than 150-step motion).

Outputs per run: `ttsgan_ptbxl_<class>_samples.npy` with shape `(N, 12, 1, 1000)`,
`ttsgan_ptbxl_<class>_labels.npy` with one-hot superclass rows `(N, 5)` in the
order `NORM, MI, STTC, CD, HYP`, plus the checkpoint and a config txt.

**Important: cd to the root directory (where `job.sh` is located) before submission so relative paths resolve correctly.**

### Step 6: Acquire Generated Synthesis from Remote Server
To acquire the generated synthetic data from remote server to local, run 
```
./sync.sh remote-to-local <path_to_synthesis_dir>
```
`<path_to_synthesis_dir>` is the full path (`userid@remote-server:path_to_synthesis_dir`) to the `synthesis/` directory
## Synthesis Evaluation - Developing...
