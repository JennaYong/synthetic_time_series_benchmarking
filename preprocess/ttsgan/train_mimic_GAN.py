# -*- coding: utf-8 -*-
"""Train TTS-GAN on MIMIC-IV without modifying the upstream code.

Same mechanism as train_ptbxl_GAN.py: train_GAN.py hardcodes the dataset
(unimib_load_dataset) and the model dimensions (Generator() / Discriminator()
defaults), so this driver rebinds those names inside the train_GAN module
namespace and then calls the untouched train_GAN.main(). The UniMiB and PTB-XL
paths are unaffected.

Run from the tts-gan repo directory (this file, mimic_dataLoader.py and
generate_mimic.py are copied there by relocate_scripts/relocate_ttsgan_mimic.sh):

    python train_mimic_GAN.py --class_name all --exp_name mimic_all ...

CLI arguments are the ordinary cfg.py arguments of train_GAN.py.
--class_name must be one of all / survived / died. The TTS_GAN_MIMIC_*
environment variables are documented in mimic_dataLoader.config_from_env.

Output anchor
-------------
The generator ends in a 1x1 convolution with no activation, so nothing bounds
its output, and the discriminator head (mean-pool -> LayerNorm -> Linear) is
blind to input scale. On PTB-XL that combination killed every unanchored run:
the generator drifted to std ~22000, the discriminator returned a constant and
the gradient died. The PTB-XL fix -- z-normalizing each generated record over
time -- cannot be reused here, because it removes each record's level and the
level is what distinguishes a heart rate of 60 from one of 130. Whether MIMIC
needs an anchor at all was therefore measured rather than assumed;
TTS_GAN_MIMIC_ANCHOR selects among:

    none   (default) z-scored data, raw generator output: upstream TTS-GAN
           with nothing added. Over 20 epochs on the five vitals it did not
           drift, and it reproduced the data's temporal and cross-signal
           correlation best.
    batch  Data are z-scored with the training subset's own mean/std, and the
           generator output is standardized per channel over (batch, time).
           The scale is pinned to the data's, while records in a batch keep
           their relative levels. The discriminator scores records one at a
           time, so it cannot see the batch-level constraint. Comparable
           quality to none, somewhat too smooth; the fallback if none drifts.
    tanh   Data are min-max scaled to [-0.8, 0.8] and the generator output
           passes through tanh, so it is bounded by construction. Clearly the
           worst of the three in the same comparison: output too noisy.

The measurements are in the README section "TTS-GAN on MIMIC-IV". No anchor
has parameters, so a checkpoint loads under any of them; what must match the
checkpoint is the signal list and embed_dim.
"""

import torch

import train_GAN
from GANModels import Generator, Discriminator
from mimic_dataLoader import CLASSES, N_STEPS, config_from_env, mimic_load_dataset

CFG = config_from_env()
SEQ_LEN = N_STEPS
CHANNELS = len(CFG['signals'])

# Upstream gen_plot() draws channels 0, 1 and 2 of every sample it previews.
if CHANNELS < 3:
    raise ValueError(
        f"TTS-GAN's training preview plots three channels, so at least three signals "
        f"are required; got {CFG['signals']}")


def apply_anchor(out, anchor):
    """Hold a raw generator output (B, C, 1, T) at the data scale."""
    if anchor == 'tanh':
        return torch.tanh(out)
    if anchor == 'batch':
        mean = out.mean(dim=(0, 2, 3), keepdim=True)
        std = out.std(dim=(0, 2, 3), keepdim=True)
        return (out - mean) / (std + 1e-8)
    return out


class AnchoredGenerator(Generator):
    """Upstream Generator followed by a parameter-free output anchor."""

    def __init__(self, *args, anchor='none', **kwargs):
        super().__init__(*args, **kwargs)
        self.anchor = anchor

    def forward_raw(self, z):
        return super().forward(z)

    def forward(self, z):
        return apply_anchor(super().forward(z), self.anchor)


def build_generator(cfg=None):
    cfg = CFG if cfg is None else cfg
    return AnchoredGenerator(seq_len=SEQ_LEN, channels=len(cfg['signals']),
                             embed_dim=cfg['embed_dim'], anchor=cfg['anchor'])


def build_discriminator(cfg=None):
    cfg = CFG if cfg is None else cfg
    return Discriminator(in_channels=len(cfg['signals']), patch_size=cfg['patch_size'],
                         seq_length=SEQ_LEN)


def build_dataset(cfg=None, data_mode='Train', class_name='all', verbose=True):
    cfg = CFG if cfg is None else cfg
    return mimic_load_dataset(
        data_path=cfg['data_path'],
        data_mode=data_mode,
        class_name=class_name,
        signals=cfg['signals'],
        head_fill=cfg['head_fill'],
        scaling=cfg['scaling'],
        verbose=verbose,
    )


def _make_mimic_dataset(incl_xyz_accel=None, incl_rms_accel=None, incl_val_group=None,
                        is_normalize=None, one_hot_encode=None, data_mode='Train',
                        single_class=True, class_name='all', augment_times=None,
                        **_unused):
    """Adapter with unimib_load_dataset's call signature.

    The accelerometer/one-hot flags have no MIMIC counterpart, and
    is_normalize (hardcoded True by train_GAN.py, meaning per-record
    z-normalization) is deliberately ignored: scaling is global and follows
    the anchor, see the module docstring.
    """
    if augment_times:
        print(f'Warning: augment_times={augment_times} is ignored for MIMIC-IV')
    if class_name not in CLASSES:
        raise ValueError(f'--class_name must be one of {CLASSES}, got {class_name!r}')
    return build_dataset(data_mode=data_mode, class_name=class_name)


train_GAN.unimib_load_dataset = _make_mimic_dataset
train_GAN.Generator = build_generator
train_GAN.Discriminator = build_discriminator

if __name__ == '__main__':
    print(f'TTS-GAN on MIMIC-IV: signals={CFG["signals"]}, anchor={CFG["anchor"]}, '
          f'scaling={CFG["scaling"]}, embed_dim={CFG["embed_dim"]}, '
          f'patch_size={CFG["patch_size"]} ({SEQ_LEN // CFG["patch_size"]} tokens), '
          f'head_fill={CFG["head_fill"]}')
    train_GAN.main()
