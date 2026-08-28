# -*- coding: utf-8 -*-
"""Train TTS-GAN on PTB-XL without modifying the upstream code.

train_GAN.py hardcodes both the dataset (unimib_load_dataset) and the model
dimensions (Generator() / Discriminator() defaults: 3 channels x 150 steps).
Instead of patching that file, this driver rebinds those names inside the
train_GAN module namespace (train_GAN does `from dataLoader import *` and
`from GANModels import *`, so the classes are plain module-level names) and
then calls the untouched train_GAN.main(). The UniMiB path keeps working
exactly as before via plain `python train_GAN.py`.

Run from the tts-gan repo directory (this file and ptbxl_dataLoader.py are
copied there by relocate_scripts/relocate_ttsgan_ptbxl.sh):

    python train_ptbxl_GAN.py --class_name NORM --exp_name ptbxl_NORM ...

CLI arguments are the ordinary cfg.py arguments of train_GAN.py.
--class_name must be one of NORM / MI / STTC / CD / HYP.

Environment overrides:
    TTS_GAN_PTBXL_DATA        directory with the ptbxl_*.npy files
                              (default ./ptbxl/)
    TTS_GAN_PTBXL_LABEL_MODE  'any' (default) or 'exclusive', see
                              ptbxl_dataLoader.ptbxl_load_dataset
    TTS_GAN_PTBXL_NORMALIZE   'per_sample' (default) applies UniMiB-style
                              per-record z-normalization; 'none' keeps the
                              SSSD-ECG global standardization.
                              Keep the default: the generator has no output
                              activation and no final LayerNorm, so nothing
                              bounds its output scale. It was tuned for data
                              with std ~1 (its own output at init has std
                              ~0.47). Feeding the globally standardized PTB-XL
                              (std 0.133) instead starts the generator 3.5x
                              above the data scale and training runs away: a
                              100k-iteration run produced samples with
                              std 8078 vs 0.133 for real data.
    TTS_GAN_PTBXL_PATCH_SIZE  discriminator patch size, must divide 1000
                              (default 100 -> 10 patches + cls, the same token
                              count as UniMiB's 150/15. With 25 (40 patches)
                              training collapsed twice: the discriminator head
                              mean-pools the tokens, and averaging 4x more
                              tokens washes out the differences between fakes
                              until D returns a constant 0.5 for all of them
                              and G's gradient dies -- both losses freeze at
                              exactly 0.25 by epoch ~9. At 100, the losses stay
                              live through the same window.)
"""

import functools
import os

import train_GAN
from GANModels import Generator, Discriminator
from ptbxl_dataLoader import ptbxl_load_dataset, SUPERCLASSES

SEQ_LEN = 1000
CHANNELS = 12

DATA_PATH = os.environ.get('TTS_GAN_PTBXL_DATA', './ptbxl/')
LABEL_MODE = os.environ.get('TTS_GAN_PTBXL_LABEL_MODE', 'any')
NORMALIZE = os.environ.get('TTS_GAN_PTBXL_NORMALIZE', 'per_sample')
PATCH_SIZE = int(os.environ.get('TTS_GAN_PTBXL_PATCH_SIZE', '100'))

if NORMALIZE not in ('none', 'per_sample'):
    raise ValueError(f"TTS_GAN_PTBXL_NORMALIZE must be 'none' or 'per_sample', got {NORMALIZE!r}")
if SEQ_LEN % PATCH_SIZE != 0:
    raise ValueError(f"TTS_GAN_PTBXL_PATCH_SIZE must divide {SEQ_LEN}, got {PATCH_SIZE}")


def _make_ptbxl_dataset(incl_xyz_accel=None, incl_rms_accel=None, incl_val_group=None,
                        is_normalize=None, one_hot_encode=None, data_mode='Train',
                        single_class=True, class_name='NORM', augment_times=None,
                        **_unused):
    """Adapter with unimib_load_dataset's call signature.

    The accelerometer/one-hot flags have no PTB-XL counterpart and
    augment_times (tsaug motion augmentation) is not applied to ECG.
    is_normalize as passed by train_GAN.py is hardcoded True for UniMiB;
    for PTB-XL the TTS_GAN_PTBXL_NORMALIZE env var decides instead, and it
    defaults to the same per-record z-normalization for the stability reason
    documented at the top of this file. Absolute scale is therefore not
    preserved -- compare models on per-record z-normalized signals.
    """
    if augment_times:
        print(f'Warning: augment_times={augment_times} is ignored for PTB-XL')
    if class_name not in SUPERCLASSES:
        raise ValueError(
            f'--class_name must be a PTB-XL superclass {SUPERCLASSES}, got {class_name!r}')
    return ptbxl_load_dataset(
        data_path=DATA_PATH,
        data_mode=data_mode,
        class_name=class_name,
        label_mode=LABEL_MODE,
        is_normalize=(NORMALIZE == 'per_sample'),
    )


class ScaleNormalizedGenerator(Generator):
    """Generator whose output is z-normalized per (sample, lead) over time.

    Without this, PTB-XL training dies in a specific way: the generator's
    output scale escapes the data scale early on, the discriminator's
    classification head (Reduce -> LayerNorm -> Linear) then saturates and
    returns the same value for every fake, so G's gradient vanishes. Both
    losses freeze at exactly 0.25 -- D(real)=1.0, D(fake)=0.5 -- within ~5
    epochs and never move again, while G drifts until its output is white
    noise at std 22000 (real data: std 1 after normalization).

    Normalizing here puts every generated record on the same scale as the
    per-record z-normalized training data, so the discriminator never sees
    an out-of-scale input and keeps producing a usable gradient. The
    operation has no parameters, so checkpoints stay compatible with the
    plain Generator -- the generation step in generate_scripts/generate_ttsgan.sh
    applies the identical transform to the numpy output instead.
    """

    def forward(self, z):
        out = super().forward(z)                       # (B, C, 1, T)
        mean = out.mean(dim=3, keepdim=True)
        std = out.std(dim=3, keepdim=True)
        return (out - mean) / (std + 1e-8)


train_GAN.unimib_load_dataset = _make_ptbxl_dataset
train_GAN.Generator = functools.partial(
    ScaleNormalizedGenerator, seq_len=SEQ_LEN, channels=CHANNELS)
train_GAN.Discriminator = functools.partial(
    Discriminator, in_channels=CHANNELS, patch_size=PATCH_SIZE, seq_length=SEQ_LEN)

if __name__ == '__main__':
    train_GAN.main()
