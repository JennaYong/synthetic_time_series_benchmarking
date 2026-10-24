# -*- coding: utf-8 -*-
"""Train DoppelGANger on the shared MIMIC-IV tensors and sample from it.

DoppelGANger (Lin et al., IMC 2020) runs through Gretel's PyTorch
implementation, the gretel_synthetics.timeseries_dgan package, which
load_model.sh clones into model/doppelganger/. The authors' own release is
TensorFlow 1.4 code that no longer installs on current clusters. Only that one
package is imported, from source, so neither the rest of gretel-synthetics nor
its pinned dependencies are needed: it imports torch, numpy, pandas, scipy and
category_encoders. Upstream code is not edited.

Run from model/doppelganger/ with src/ on PYTHONPATH
(generate_scripts/generate_doppelganger.sh does this):

    python mimic_dgan.py <output_dir> <prefix>

Benchmark rules (agreed 2026-10-13 and 2026-10-16): the model trains on the
whole data matrix -- the eight value channels of the shared tensor, imputed
slots included, masks dropped -- with its original objective (Wasserstein loss
with gradient penalty, plus the auxiliary discriminator on attributes). It is
an unconditional model, so no static attributes are passed in; DoppelGANger's
auto-normalization still adds each series' own min and max as attributes it
generates first.

The only change to the data is optional and invertible: log1p on glucose and
creatinine (DGAN_MIMIC_LOG, on by default as in the TTS-GAN adapter). Samples
are mapped back to clinical units and clipped to the physiological bounds of
the shared preprocessing (channels.json clip_bounds, release V3 on).

Writes into <output_dir>:

    <prefix>_samples.npy   float32 (N, 8, 72) in clinical units, axes
                           (stay, signal, hour) like the value channels of
                           train.npy
    <prefix>_config.json   run configuration, the transform that was inverted
                           and the health check printed below
    <prefix>_model.pt      the trained model (DGAN.save; load it with
                           DGAN.load(path, weights_only=False))

Options (environment variables):

    DGAN_MIMIC_DATA        directory with train.npy and channels.json
                           (default ./mimic/)
    DGAN_EPOCHS            training epochs (default 400, the original paper's
                           and Gretel's default)
    DGAN_BATCH_SIZE        batch size (default 100, the original paper's)
    DGAN_SAMPLE_LEN        hours generated per LSTM step, must divide 72
                           (default 6)
    DGAN_MIMIC_LOG         1 (default) to model glucose and creatinine on a
                           log scale, 0 for raw units
    DGAN_LR                learning rate of the generator and both
                           discriminators (default 0.0001; the original
                           paper's 0.001 never settled on MIMIC-IV, see the
                           README section *DoppelGANger on MIMIC-IV*)
    DGAN_EXAMPLE_SCALING   1 (default) for DoppelGANger's auto-normalization,
                           which generates each series' min and max as
                           attributes; 0 to scale every series globally
    DGAN_NUM_SAMPLES       samples to generate (default: the number of training
                           stays)
    DGAN_SEED              random seed (default 0)
    DGAN_SNAPSHOT_EPOCHS   comma-separated epochs after which to also save the
                           model as <prefix>_model_epoch<k>.pt (default none)
"""

import json
import logging
import os
import subprocess
import sys
import time
from enum import Enum
from pathlib import Path

import numpy as np
import torch

from gretel_synthetics.timeseries_dgan.config import DGANConfig, OutputType
from gretel_synthetics.timeseries_dgan.dgan import DGAN

N_STEPS = 72
SIGNALS = ['heart_rate', 'sbp', 'dbp', 'resp_rate', 'spo2', 'glucose', 'creatinine', 'sodium']
LOG_SIGNALS = ('glucose', 'creatinine')


def config_from_env(env=None):
    env = os.environ if env is None else env
    sample_len = int(env.get('DGAN_SAMPLE_LEN', '6'))
    if sample_len <= 0 or N_STEPS % sample_len != 0:
        raise SystemExit(f'DGAN_SAMPLE_LEN must divide {N_STEPS}, got {sample_len}')
    log = env.get('DGAN_MIMIC_LOG', '1')
    if log not in ('0', '1'):
        raise SystemExit(f'DGAN_MIMIC_LOG must be 0 or 1, got {log!r}')
    example_scaling = env.get('DGAN_EXAMPLE_SCALING', '1')
    if example_scaling not in ('0', '1'):
        raise SystemExit(f'DGAN_EXAMPLE_SCALING must be 0 or 1, got {example_scaling!r}')
    snapshots = [int(e) for e in env.get('DGAN_SNAPSHOT_EPOCHS', '').split(',') if e.strip()]
    return {
        'data_path': env.get('DGAN_MIMIC_DATA', './mimic/'),
        'epochs': int(env.get('DGAN_EPOCHS', '400')),
        'batch_size': int(env.get('DGAN_BATCH_SIZE', '100')),
        'sample_len': sample_len,
        'log_labs': log == '1',
        'learning_rate': float(env.get('DGAN_LR', '0.0001')),
        'example_scaling': example_scaling == '1',
        'num_samples': int(env['DGAN_NUM_SAMPLES']) if env.get('DGAN_NUM_SAMPLES') else None,
        'seed': int(env.get('DGAN_SEED', '0')),
        'snapshot_epochs': snapshots,
    }


def load_values(data_path):
    """The eight value channels of train.npy, (n, 8, 72) float32, and channels.json."""
    with open(os.path.join(data_path, 'channels.json')) as f:
        meta = json.load(f)
    names = meta['channels']
    missing = [s for s in SIGNALS if s not in names]
    if missing:
        raise SystemExit(f'channels.json has no channel(s) {missing}; found {names}')
    tensor = np.load(os.path.join(data_path, 'train.npy'))
    if tensor.ndim != 3 or tensor.shape[2] != N_STEPS:
        raise SystemExit(f'expected (n_stays, channels, {N_STEPS}), got {tensor.shape}')
    values = tensor[:, [names.index(s) for s in SIGNALS], :].astype(np.float32)
    if not np.isfinite(values).all():
        raise SystemExit('value channels contain NaN/inf; expected the imputed tensor')
    return values, meta


def output_bounds(meta):
    """Per-signal (low, high) from clip_bounds, or a floor at 0 for older releases."""
    bounds = meta.get('clip_bounds')
    if not bounds:
        return np.zeros(len(SIGNALS)), np.full(len(SIGNALS), np.inf)
    return (np.array([bounds[s]['low'] for s in SIGNALS], dtype=np.float64),
            np.array([bounds[s]['high'] for s in SIGNALS], dtype=np.float64))


def signal_stats(x):
    """Per-signal summary of (n, C, T) data in clinical units (as in the TTS-GAN check)."""
    pooled_std = x.std(axis=(0, 2))
    return {
        'mean': x.mean(axis=(0, 2)),
        'std': pooled_std,
        # spread of the per-stay level: collapses to 0 if every sample sits at
        # the same level, however realistic each one looks on its own
        'level_spread': x.mean(axis=2).std(axis=0),
        # typical variation within a stay: collapses to 0 if every sample is a
        # flat line, however well the levels are spread
        'within_stay_std': np.median(x.std(axis=2), axis=0),
        # hour-to-hour roughness: std of the first difference over the std;
        # pointwise white noise gives sqrt(2) ~ 1.41
        'roughness': np.diff(x, axis=2).std(axis=(0, 2)) / (pooled_std + 1e-12),
    }


def health_check(real, synthetic):
    r, s = signal_stats(real), signal_stats(synthetic)
    print(f'\nHealth check, clinical units (real = the {len(real)} train stays, imputed slots included):')
    print(f"{'signal':12s} {'mean real/syn':>19s} {'std real/syn':>17s} "
          f"{'level spread r/s':>19s} {'roughness r/s':>17s}")
    warnings = []
    for c, name in enumerate(SIGNALS):
        print(f"{name:12s} {r['mean'][c]:9.2f}/{s['mean'][c]:<9.2f} {r['std'][c]:8.2f}/{s['std'][c]:<8.2f} "
              f"{r['level_spread'][c]:9.2f}/{s['level_spread'][c]:<9.2f} "
              f"{r['roughness'][c]:8.3f}/{s['roughness'][c]:<8.3f}")
        std_ratio = s['std'][c] / (r['std'][c] + 1e-12)
        level_ratio = s['level_spread'][c] / (r['level_spread'][c] + 1e-12)
        rough_ratio = s['roughness'][c] / (r['roughness'][c] + 1e-12)
        within_ratio = s['within_stay_std'][c] / (r['within_stay_std'][c] + 1e-12)
        if not 0.2 < std_ratio < 5.0:
            warnings.append(f'{name}: amplitude is {std_ratio:.2f}x the real one')
        if level_ratio < 0.2:
            warnings.append(f'{name}: samples share one level ({level_ratio:.2f}x the real spread) '
                            f'-- mode collapse')
        if within_ratio < 0.2:
            warnings.append(f'{name}: samples are flat within a stay ({within_ratio:.2f}x the real '
                            f'median within-stay std)')
        if rough_ratio > 2.0:
            warnings.append(f'{name}: {rough_ratio:.1f}x rougher than real data -- output is close '
                            f'to pointwise noise')
    for w in warnings:
        print(f'WARNING: {w}')
    if not warnings:
        print('No warnings. This checks for failed runs only; it does not rank quality.')
    return {'real': {k: [float(v) for v in vals] for k, vals in r.items()},
            'synthetic': {k: [float(v) for v in vals] for k, vals in s.items()},
            'warnings': warnings}


def upstream_commit():
    """The gretel-synthetics commit, recorded by the relocate script.

    sync.sh skips .git, so on the cluster `git rev-parse` would report the
    enclosing benchmark repo instead; only ask git when the model dir is a
    checkout of its own.
    """
    recorded = Path('UPSTREAM_COMMIT')
    if recorded.is_file():
        return recorded.read_text().strip() or None
    if not Path('.git').exists():
        return None
    try:
        return subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def full_precision_on_gpu():
    """Keep Gretel's feature-discriminator step in float32 on the GPU.

    DGAN._train runs that step under torch.cuda.amp.autocast(enabled=True)
    whatever mixed_precision_training says, while its GradScaler follows the
    setting and is off by default. On a GPU the step therefore runs in float16
    without loss scaling; on a CPU autocast does nothing. Turning autocast off
    while mixed precision is off makes both devices train the model the same
    way, in float32. Upstream code is not edited.
    """
    autocast = torch.cuda.amp.autocast
    torch.cuda.amp.autocast = lambda *args, **kwargs: autocast(*args, **{**kwargs, 'enabled': False})


def main(argv):
    if len(argv) != 2:
        raise SystemExit(__doc__)
    out_dir, prefix = Path(argv[0]), argv[1]
    cfg = config_from_env()
    logging.basicConfig(level=logging.WARNING)
    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])

    values, meta = load_values(cfg['data_path'])                  # (n, 8, 72)
    log_mask = np.array([cfg['log_labs'] and s in LOG_SIGNALS for s in SIGNALS])
    model_space = values.copy()
    model_space[:, log_mask, :] = np.log1p(np.maximum(model_space[:, log_mask, :], 0.0))
    features = np.ascontiguousarray(model_space.transpose(0, 2, 1))  # (n, 72, 8), as DGAN expects

    dgan_config = DGANConfig(
        max_sequence_len=N_STEPS,
        sample_len=cfg['sample_len'],
        batch_size=cfg['batch_size'],
        epochs=cfg['epochs'],
        generator_learning_rate=cfg['learning_rate'],
        discriminator_learning_rate=cfg['learning_rate'],
        attribute_discriminator_learning_rate=cfg['learning_rate'],
        apply_example_scaling=cfg['example_scaling'],
    )
    print(f'DoppelGANger on MIMIC-IV: {len(features)} train stays, signals {SIGNALS}, '
          f'log scale for {[s for s, m in zip(SIGNALS, log_mask) if m]}, epochs {cfg["epochs"]}, '
          f'batch {cfg["batch_size"]}, sample_len {cfg["sample_len"]}, lr {cfg["learning_rate"]}, '
          f'example scaling {cfg["example_scaling"]}, '
          f'device {"cuda" if torch.cuda.is_available() else "cpu"}', flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    if not dgan_config.mixed_precision_training:
        full_precision_on_gpu()
    model = DGAN(dgan_config)
    start = time.time()

    def on_progress(info):
        if info.batch != info.total_batches - 1:
            return
        epoch = info.epoch + 1
        if epoch == 1 or epoch % 10 == 0 or epoch == info.total_epochs:
            print(f'epoch {epoch}/{info.total_epochs} done, {time.time() - start:.0f}s', flush=True)
        if epoch in cfg['snapshot_epochs']:
            model.save(str(out_dir / f'{prefix}_model_epoch{epoch:03d}.pt'))

    model.train_numpy(features, feature_types=[OutputType.CONTINUOUS] * len(SIGNALS),
                      progress_callback=on_progress)
    train_seconds = time.time() - start
    model_path = out_dir / f'{prefix}_model.pt'
    model.save(str(model_path))

    num_samples = cfg['num_samples'] or len(features)
    _, generated = model.generate_numpy(num_samples)
    synthetic = np.stack(generated).transpose(0, 2, 1).astype(np.float64)   # (N, 8, 72)
    if not np.isfinite(synthetic).all():
        raise SystemExit('ERROR: generator produced NaN/inf -- training diverged.')
    synthetic[:, log_mask, :] = np.expm1(synthetic[:, log_mask, :])
    low, high = output_bounds(meta)
    synthetic = np.clip(synthetic, low[None, :, None], high[None, :, None]).astype(np.float32)

    samples_path = out_dir / f'{prefix}_samples.npy'
    np.save(samples_path, synthetic)
    health = health_check(values, synthetic)

    config = {
        'dataset': 'mimic',
        'model': 'DoppelGANger (gretel_synthetics.timeseries_dgan)',
        'upstream_commit': upstream_commit(),
        'num_train_stays': int(len(features)),
        'num_samples': int(num_samples),
        'samples_shape': list(synthetic.shape),
        'samples_units': 'clinical units, axes (stay, signal, hour)',
        'signals': SIGNALS,
        'log_transformed': [s for s, m in zip(SIGNALS, log_mask) if m],
        'output_bounds': {s: [float(lo), float(hi)] for s, lo, hi in zip(SIGNALS, low, high)},
        'seed': cfg['seed'],
        'train_seconds': round(train_seconds, 1),
        'dgan_config': {k: (v.name if isinstance(v, Enum) else v)
                        for k, v in dgan_config.to_dict().items()},
        'health': health,
    }
    config_path = out_dir / f'{prefix}_config.json'
    config_path.write_text(json.dumps(config, indent=2) + '\n')
    print(f'Saved {synthetic.shape} synthetic samples to {samples_path}')
    print(f'Saved the model to {model_path}; training took {train_seconds:.0f}s')


if __name__ == '__main__':
    main(sys.argv[1:])
