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
|--preprocess/              # Data preprocessing methods
|--evaluation/              # Evaluation methods
|--synthesis/               # Generated data stored in .npy format. Not tracked by git
|--results/                 # Evaluation results. Not tracked by git. Not tracked by git
|
|--generate_scripts/        # Model-specific training and generation scripts
|   |--generate_sssdecg.sh
|--relocate_scripts/        # Model-specific dataset relocation scripts
|   |--relocate_sssdecg.sh
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

### Step 4: Sync Local Setup with Remote Server
To sync local setup with remote server, run
```
./sync.sh local-to-remote <destination>
```
`<destination>` is the full path (`userid@remote-server:path_to_dest`) to the target location on the remote server. 

### Step 5: Synthesis Generation on Remote Server
To train the model and generate synthetic data with it, SSH to the remote server and run
```
./job.sh <model>
```
Run `job.sh` **directly** — do NOT use `sbatch ./job.sh <model>`. SBATCH `--time` headers are static (parsed before the script runs), so `job.sh` self-submits: it picks a per-model wall-time and submits itself via `sbatch`. Running it under `sbatch` yourself bypasses this and falls back to the static 24h header time.

Per-model wall-time:

| Model    | `--time` |
|----------|----------|
| sssd-ecg | 24:00:00 |
| tts-gan  | 10:00:00 |

To override the wall-time manually, submit explicitly: `sbatch --time=<HH:MM:SS> job.sh <model>`.

`job.sh` handles job details and computing resource allocation, so double-check before submitting. For SSSD-ECG it runs `./generate_scripts/generate_sssdecg.sh`; for TTS-GAN, `./generate_scripts/generate_ttsgan.sh`. The generated synthesis is stored in `synthesis/<MODEL>/{date}` where `date` is the execution timestamp.

TTS-GAN trains one model per activity class; select it via an environment variable (default `Running`):
```
TTS_GAN_CLASS=Jumping ./job.sh tts-gan
```

**Important: cd to the root directory (where `job.sh` is located) before submission so relative paths resolve correctly.**

### Step 6: Acquire Generated Synthesis from Remote Server
To acquire the generated synthetic data from remote server to local, run 
```
./sync.sh remote-to-local <path_to_synthesis_dir>
```
`<path_to_synthesis_dir>` is the full path (`userid@remote-server:path_to_synthesis_dir`) to the `synthesis/` directory
## Synthesis Evaluation - Developing...
