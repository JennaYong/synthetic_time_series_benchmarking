# -*- coding: utf-8 -*-
"""Generate MIMIC-IV samples from a TTS-GAN checkpoint, in clinical units.

Called by generate_scripts/generate_ttsgan.sh after training, from the tts-gan
repo directory and with the same TTS_GAN_MIMIC_* environment as the training
run, so the generator and the scaling are rebuilt exactly as they were trained:

    python generate_mimic.py <checkpoint> <synthesis_dir> <class> <n> <prefix>

Writes into <synthesis_dir>:

    <prefix>_samples.npy   float32 (n, C, 72) in clinical units; axis order
                           (stay, signal, hour) as in the real train.npy, with
                           the signals listed in the config file
    <prefix>_labels.npy    int64 (n,), mortality_in_hospital of the class the
                           model was trained on; only for survived / died
    <prefix>_config.json   run configuration, the scaling that was inverted,
                           and the health statistics printed below

The job log gets a per-signal comparison against the training data the model
saw, so a failed run is visible without copying anything back.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

from mimic_dataLoader import CLASSES, config_from_env
from train_mimic_GAN import apply_anchor, build_dataset, build_generator

LATENT_DIM = 100
CHUNK = 500
# Upstream keeps an exponential moving average of the generator and saves it
# next to the last-step weights. Sampling from the average is the default: over
# an 80-epoch run it cut the cross-signal correlation error by 2-4x and removed
# the checkpoint-to-checkpoint swings the last-step weights show.
STATE_KEY = {'ema': 'avg_gen_state_dict', 'raw': 'gen_state_dict'}


def signal_stats(x):
    """Per-signal summary of (n, C, T) data in clinical units."""
    pooled_std = x.std(axis=(0, 2))
    return {
        'mean': x.mean(axis=(0, 2)),
        'std': pooled_std,
        # spread of the per-stay level: collapses to 0 if every sample sits at
        # the same level, however realistic each one looks on its own
        'level_spread': x.mean(axis=2).std(axis=0),
        # hour-to-hour roughness: std of the first difference over the std;
        # pointwise white noise gives sqrt(2) ~ 1.41
        'roughness': np.diff(x, axis=2).std(axis=(0, 2)) / (pooled_std + 1e-12),
    }


def main(argv):
    if len(argv) != 5:
        raise SystemExit(__doc__)
    ckpt_path, synthesis_dir = Path(argv[0]), Path(argv[1])
    class_name, num_samples, output_prefix = argv[2], int(argv[3]), argv[4]
    if class_name not in CLASSES:
        raise SystemExit(f'class must be one of {CLASSES}, got {class_name!r}')

    cfg = config_from_env()
    # The train subset defines the scaling to invert and is the reference the
    # samples are checked against.
    dataset = build_dataset(cfg, data_mode='Train', class_name=class_name)

    gen_net = build_generator(cfg)
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    gen_net.load_state_dict(checkpoint[STATE_KEY[cfg['weights']]])
    gen_net.eval()

    z = torch.FloatTensor(np.random.normal(0, 1, (num_samples, LATENT_DIM)))
    with torch.no_grad():
        raw = torch.cat([gen_net.forward_raw(z[i:i + CHUNK])
                         for i in range(0, num_samples, CHUNK)])
        # Applied once over the whole set: the 'batch' anchor standardizes over
        # the samples it is given, and all of them is the better estimate.
        scaled = apply_anchor(raw, cfg['anchor']).numpy()
    if not np.isfinite(scaled).all():
        raise SystemExit('ERROR: generator produced NaN/inf -- training diverged.')

    synthetic = dataset.inverse_transform(scaled)             # (n, C, 72)
    synthesis_dir.mkdir(parents=True, exist_ok=True)
    samples_path = synthesis_dir / f'{output_prefix}_samples.npy'
    np.save(samples_path, synthetic)

    if class_name != 'all':
        labels = np.full(num_samples, int(class_name == 'died'), dtype=np.int64)
        labels_path = synthesis_dir / f'{output_prefix}_labels.npy'
        np.save(labels_path, labels)
        print(f'Saved {labels.shape} mortality_in_hospital labels to {labels_path}')

    real = signal_stats(dataset.inverse_transform(dataset.data))
    fake = signal_stats(synthetic)
    saturated = float((np.abs(scaled) > 0.98).mean()) if cfg['anchor'] == 'tanh' else None

    print(f'\nHealth check, clinical units (real = the {dataset.n_train} train stays as the '
          f'model saw them, imputed slots included):')
    print(f"{'signal':12s} {'mean real/syn':>19s} {'std real/syn':>17s} "
          f"{'level spread r/s':>19s} {'roughness r/s':>17s}")
    warnings = []
    for c, name in enumerate(cfg['signals']):
        print(f"{name:12s} {real['mean'][c]:9.2f}/{fake['mean'][c]:<9.2f} "
              f"{real['std'][c]:8.2f}/{fake['std'][c]:<8.2f} "
              f"{real['level_spread'][c]:9.2f}/{fake['level_spread'][c]:<9.2f} "
              f"{real['roughness'][c]:8.3f}/{fake['roughness'][c]:<8.3f}")
        std_ratio = fake['std'][c] / (real['std'][c] + 1e-12)
        level_ratio = fake['level_spread'][c] / (real['level_spread'][c] + 1e-12)
        rough_ratio = fake['roughness'][c] / (real['roughness'][c] + 1e-12)
        if not 0.2 < std_ratio < 5.0:
            warnings.append(f'{name}: amplitude is {std_ratio:.2f}x the real one')
        if level_ratio < 0.2:
            warnings.append(f'{name}: samples share one level ({level_ratio:.2f}x the real '
                            f'spread) -- mode collapse')
        if rough_ratio > 2.0:
            warnings.append(f'{name}: {rough_ratio:.1f}x rougher than real data -- output is '
                            f'close to pointwise noise')
    if saturated is not None:
        print(f'tanh saturation (|output| > 0.98): {saturated:.1%} of values')
        if saturated > 0.2:
            warnings.append(f'{saturated:.0%} of the tanh output is saturated')
    for w in warnings:
        print(f'WARNING: {w}')
    if not warnings:
        print('No warnings. This checks for failed runs only; it does not rank quality.')

    config = {
        'dataset': 'mimic',
        'class': class_name,
        'checkpoint': str(ckpt_path),
        'epoch': int(checkpoint['epoch']),
        'num_samples': num_samples,
        'samples_shape': list(synthetic.shape),
        'samples_units': 'clinical units, axes (stay, signal, hour)',
        'anchor': cfg['anchor'],
        'weights': cfg['weights'],
        'embed_dim': cfg['embed_dim'],
        'patch_size': cfg['patch_size'],
        'transform': dataset.describe(),
        'health': {
            'real': {k: [float(v) for v in vals] for k, vals in real.items()},
            'synthetic': {k: [float(v) for v in vals] for k, vals in fake.items()},
            'tanh_saturation': saturated,
            'warnings': warnings,
        },
    }
    config_path = synthesis_dir / f'{output_prefix}_config.json'
    config_path.write_text(json.dumps(config, indent=2) + '\n')

    print(f'Saved {synthetic.shape} synthetic samples to {samples_path}')


if __name__ == '__main__':
    main(sys.argv[1:])
