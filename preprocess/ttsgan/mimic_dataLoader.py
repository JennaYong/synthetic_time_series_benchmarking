# -*- coding: utf-8 -*-
"""MIMIC-IV dataset loader for TTS-GAN.

Reads the tensors written by `preprocess/mimic_pyhealth.py` (`train.npy` /
`test.npy`, shape `(n_stays, 16, 72)`, raw clinical units, one value channel
and one mask channel per signal; channels are looked up by name in
`channels.json`) and exposes them through the Dataset interface TTS-GAN's
`unimib_load_dataset` provides, so upstream `train_GAN.py` can consume them
unmodified (see `train_mimic_GAN.py`).

The benchmark's rule is that every model trains on the same, complete data
matrix: imputed slots included, masks used only at evaluation. So by default
this loader changes the shared tensor only in invertible ways:

1. **Selects signals.** All eight (five vitals, three labs) by default.
2. **Log-transforms the right-skewed labs** (glucose, creatinine).
3. **Scales** per channel, globally, with the mean/std of the training subset.
   Unlike the PTB-XL adapter this is never per record: in ICU data the
   absolute level (heart rate 60 vs 130) is the clinical information, and
   per-record normalization would erase it.

`inverse_transform` undoes 3 and 2, so generated samples are written in
clinical units, and clips them to the physiological bounds the shared
preprocessing clips the real data to (`clip_bounds` in `channels.json`, from
release V3 on); for older releases, which have no bounds, it only floors at 0,
since all eight signals are non-negative quantities. The masks are not fed to
the model.

From V3 on, a mask channel is 1 where the slot was measured and 0 where it was
imputed; earlier releases stored the opposite. Here the masks matter only for
the two experiment-only options below, and the loader reads the convention off
the data (heart rate is measured in ~96% of hours), so both kinds of release
load correctly.

Two non-default options alter the data and are kept for experiments only:
`head_fill='backfill'` replaces the slots before a channel's first reading
(which the shared preprocessing fills with the train median) by the stay's own
first reading, and the `minmax` scaling used by the tanh anchor clips each
channel to the 0.5 / 99.5 percentiles of the measured train values, because a
bounded output needs a finite range.

Class filtering uses `mortality_in_hospital` from `annotations.json` (death
during the stay's own admission).

Everything is fit on the train split only; the test split is transformed with
the train statistics and never influences them.
"""

import json
import os

import numpy as np
from torch.utils.data import Dataset

N_STEPS = 72
VITALS = ['heart_rate', 'sbp', 'dbp', 'resp_rate', 'spo2']
LABS = ['glucose', 'creatinine', 'sodium']
SIGNALS = VITALS + LABS
SIGNAL_PRESETS = {'vitals': VITALS, 'all': SIGNALS}
LOG_SIGNALS = ('glucose', 'creatinine')

CLASSES = ('all', 'survived', 'died')
HEAD_FILLS = ('none', 'backfill')
SCALINGS = ('zscore', 'minmax')
ANCHORS = ('tanh', 'batch', 'none')
WEIGHTS = ('ema', 'raw')
# Each output anchor needs its matching input scale (see train_mimic_GAN.py).
SCALING_FOR_ANCHOR = {'tanh': 'minmax', 'batch': 'zscore', 'none': 'zscore'}

CLIP_PERCENTILES = (0.5, 99.5)
# The clip range maps to +-TANH_RANGE rather than +-1: SpO2 = 100 is both the
# clip bound and one of the most common values, and tanh only reaches 1
# asymptotically, so a target of exactly 1 would push the generator into
# saturation to hit an everyday value.
TANH_RANGE = 0.8


def parse_signals(spec):
    """'vitals' | 'all' | comma-separated signal names -> ordered name list."""
    if spec in SIGNAL_PRESETS:
        return list(SIGNAL_PRESETS[spec])
    names = [s.strip() for s in spec.split(',') if s.strip()]
    unknown = [s for s in names if s not in SIGNALS]
    if not names or unknown:
        raise ValueError(
            f"TTS_GAN_MIMIC_SIGNALS must be 'vitals', 'all' or a comma-separated subset "
            f"of {SIGNALS}; got {spec!r}")
    if len(set(names)) != len(names):
        raise ValueError(f"TTS_GAN_MIMIC_SIGNALS repeats a signal: {spec!r}")
    return names


def config_from_env(env=None):
    """The MIMIC adapter configuration, read from TTS_GAN_MIMIC_* variables.

    Training (`train_mimic_GAN.py`) and generation (`generate_mimic.py`) both
    call this, so the two cannot disagree about dimensions or scaling.

        TTS_GAN_MIMIC_DATA        directory with train.npy, test.npy,
                                  stays_*.npy, channels.json, annotations.json
                                  (default ./mimic/)
        TTS_GAN_MIMIC_SIGNALS     'all' (default: five vitals + three labs),
                                  'vitals' or a comma-separated list of signals
        TTS_GAN_MIMIC_HEAD_FILL   'none' (default: use the shared tensor as it
                                  is) or 'backfill'
        TTS_GAN_MIMIC_ANCHOR      how the generator output is held at the data
                                  scale: 'none' (default), 'batch' or 'tanh'
        TTS_GAN_MIMIC_WEIGHTS     generator weights used for sampling: 'ema'
                                  (default; the moving average upstream keeps
                                  and saves as avg_gen_state_dict) or 'raw'
                                  (the last training step). Generation only.
        TTS_GAN_MIMIC_PATCH_SIZE  discriminator patch size, must divide 72
                                  (default 8 -> 9 tokens + cls)
        TTS_GAN_MIMIC_EMBED_DIM   generator width per timestep, a multiple of 5
                                  (default 10, the upstream UniMiB value)
    """
    env = os.environ if env is None else env
    anchor = env.get('TTS_GAN_MIMIC_ANCHOR', 'none')
    if anchor not in ANCHORS:
        raise ValueError(f"TTS_GAN_MIMIC_ANCHOR must be one of {ANCHORS}, got {anchor!r}")
    head_fill = env.get('TTS_GAN_MIMIC_HEAD_FILL', 'none')
    if head_fill not in HEAD_FILLS:
        raise ValueError(f"TTS_GAN_MIMIC_HEAD_FILL must be one of {HEAD_FILLS}, got {head_fill!r}")
    weights = env.get('TTS_GAN_MIMIC_WEIGHTS', 'ema')
    if weights not in WEIGHTS:
        raise ValueError(f"TTS_GAN_MIMIC_WEIGHTS must be one of {WEIGHTS}, got {weights!r}")
    patch_size = int(env.get('TTS_GAN_MIMIC_PATCH_SIZE', '8'))
    if patch_size <= 0 or N_STEPS % patch_size != 0:
        raise ValueError(f"TTS_GAN_MIMIC_PATCH_SIZE must divide {N_STEPS}, got {patch_size}")
    embed_dim = int(env.get('TTS_GAN_MIMIC_EMBED_DIM', '10'))
    if embed_dim <= 0 or embed_dim % 5 != 0:
        raise ValueError(
            f"TTS_GAN_MIMIC_EMBED_DIM must be a positive multiple of 5 (the generator "
            f"blocks hardcode 5 attention heads), got {embed_dim}")
    return {
        'data_path': env.get('TTS_GAN_MIMIC_DATA', './mimic/'),
        'signals': parse_signals(env.get('TTS_GAN_MIMIC_SIGNALS', 'all')),
        'head_fill': head_fill,
        'anchor': anchor,
        'scaling': SCALING_FOR_ANCHOR[anchor],
        'weights': weights,
        'patch_size': patch_size,
        'embed_dim': embed_dim,
    }


def fill_heads(values, imputed, never_fill):
    """Back-fill the slots before each channel's first measurement.

    values (n, C, T) and imputed (n, C, T, bool) come from the shared tensor,
    where slots after the first measurement already hold a carried-forward
    value. never_fill (C,) is used for channels with no measurement at all.
    """
    measured = ~imputed
    has_any = measured.any(axis=2)
    first = measured.argmax(axis=2)
    first_val = np.take_along_axis(values, first[..., None], axis=2)
    head = np.arange(values.shape[2])[None, None, :] < first[..., None]
    out = np.where(head, first_val, values)
    return np.where(has_any[..., None], out, never_fill[None, :, None]).astype(np.float32)


class mimic_load_dataset(Dataset):
    """MIMIC-IV ICU stays as (C, 1, 72) float32 items for TTS-GAN.

    Args:
        data_path: directory with the files written by mimic_pyhealth.py.
        data_mode: 'Train' or 'Test'. Statistics always come from train.
        class_name: 'all' (no filtering), 'survived' or 'died', by
            `mortality_in_hospital` of the stay's own admission.
        signals: ordered signal names to use as channels (default: all eight).
        head_fill: 'none' or 'backfill', see the module docstring.
        scaling: 'zscore' or 'minmax', see the module docstring.
    """

    def __init__(self,
                 data_path='./mimic/',
                 data_mode='Train',
                 class_name='all',
                 signals=None,
                 head_fill='none',
                 scaling='zscore',
                 verbose=True):
        signals = list(SIGNALS if signals is None else signals)
        if data_mode not in ('Train', 'Test'):
            raise ValueError(f"data_mode must be 'Train' or 'Test', got {data_mode!r}")
        if class_name not in CLASSES:
            raise ValueError(f"class_name must be one of {CLASSES}, got {class_name!r}")
        if head_fill not in HEAD_FILLS:
            raise ValueError(f"head_fill must be one of {HEAD_FILLS}, got {head_fill!r}")
        if scaling not in SCALINGS:
            raise ValueError(f"scaling must be one of {SCALINGS}, got {scaling!r}")

        self.data_path = data_path
        self.data_mode = data_mode
        self.class_name = class_name
        self.signals = signals
        self.head_fill = head_fill
        self.scaling = scaling
        self._log = np.array([s in LOG_SIGNALS for s in signals])

        with open(self._file('channels.json')) as f:
            meta = json.load(f)
        channel_names = meta['channels']
        missing = [n for s in signals for n in (s, f'{s}_mask') if n not in channel_names]
        if missing:
            raise ValueError(f"channels.json has no channel(s) {missing}; found {channel_names}")
        self._value_idx = [channel_names.index(s) for s in signals]
        self._mask_idx = [channel_names.index(f'{s}_mask') for s in signals]
        # Physiological bounds of the shared preprocessing (V3 on); generated
        # samples are clipped to them so that they obey the same range as the
        # real data.
        clip_bounds = meta.get('clip_bounds')
        self.out_low = self.out_high = None
        if clip_bounds:
            unbounded = [s for s in signals if s not in clip_bounds]
            if unbounded:
                raise ValueError(f"channels.json clip_bounds has no entry for {unbounded}")
            self.out_low = np.array([clip_bounds[s]['low'] for s in signals], dtype=np.float64)
            self.out_high = np.array([clip_bounds[s]['high'] for s in signals], dtype=np.float64)

        train_tensor = np.load(self._file('train.npy'))
        self.mask_measured = self._measured_mask_value(train_tensor, channel_names)
        train_values, train_imputed = self._select(train_tensor)
        del train_tensor
        # Statistics from the measured train values only: imputed slots would
        # weight each reading by how long it was carried forward. They are used
        # only by the two non-default options (backfill, minmax).
        measured = [train_values[:, c][~train_imputed[:, c]] for c in range(len(signals))]
        if any(m.size == 0 for m in measured):
            empty = [s for s, m in zip(signals, measured) if m.size == 0]
            raise ValueError(f"no measured train values for {empty}")
        self.median = np.array([np.median(m) for m in measured], dtype=np.float32)
        bounds = np.array([np.percentile(self._forward_log(m, c), CLIP_PERCENTILES)
                           for c, m in enumerate(measured)], dtype=np.float32)
        self.lo, self.hi = bounds[:, 0], bounds[:, 1]
        if np.any(self.hi <= self.lo):
            flat = [s for s, lo, hi in zip(signals, self.lo, self.hi) if hi <= lo]
            raise ValueError(f"clip range is empty for {flat}")

        train = self._prepare(train_values, train_imputed)
        train = train[self._class_mask('train', len(train))]
        if len(train) == 0:
            raise ValueError(f"no train stays for class {class_name!r}")
        # z-score statistics describe the tensor exactly as the discriminator
        # sees it (this class, after any transform), which is also what the
        # 'batch' anchor reproduces on the generator side.
        self.mean = train.mean(axis=(0, 2)).astype(np.float32)
        self.std = train.std(axis=(0, 2)).astype(np.float32)
        self.n_train = len(train)

        if data_mode == 'Train':
            prepared = train
        else:
            test_values, test_imputed = self._select(np.load(self._file('test.npy')))
            prepared = self._prepare(test_values, test_imputed)
            prepared = prepared[self._class_mask('test', len(prepared))]

        scaled = self._scale(prepared)
        # (n, C, T) -> (n, C, 1, T): the (C, 1, T) layout TTS-GAN uses.
        self.data = scaled[:, :, None, :].astype(np.float32)
        self.labels = np.full(len(self.data), CLASSES.index(class_name), dtype=np.int64)

        if verbose:
            print(f'MIMIC-IV {data_mode}: {len(self.data)} stays, class={class_name}, '
                  f'signals={signals}, head_fill={head_fill}, scaling={scaling}, '
                  f'mask 1 = {"measured" if self.mask_measured == 1 else "imputed"}, '
                  f'output {"clipped to clip_bounds" if self.out_low is not None else "floored at 0"}')
            print(f'data shape is {self.data.shape}, labels shape is {self.labels.shape}')

    def _file(self, name):
        path = os.path.join(self.data_path, name)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Missing {path}. Run relocate_scripts/relocate_ttsgan_mimic.sh or point "
                f"TTS_GAN_MIMIC_DATA at the directory written by mimic_pyhealth.py.")
        return path

    @staticmethod
    def _measured_mask_value(tensor, channel_names):
        """The mask value that marks a measured slot: 1 from release V3 on, 0 before.

        Read off the data rather than from a label: heart rate is charted in
        ~96% of an ICU stay's hours, so whichever value dominates its mask
        channel means 'measured'.
        """
        if 'heart_rate_mask' not in channel_names:
            raise ValueError("channels.json has no heart_rate_mask to read the mask convention from")
        share_one = float(tensor[:, channel_names.index('heart_rate_mask'), :].mean())
        if share_one > 0.8:
            return 1.0
        if share_one < 0.2:
            return 0.0
        raise ValueError(
            f"heart_rate_mask is 1 in {share_one:.0%} of slots, so it is unclear whether 1 "
            f"means measured or imputed (heart rate is measured in ~96% of hours).")

    def _select(self, tensor):
        """Shared tensor -> (values, imputed) for the selected signals, (n, C, T) each."""
        if tensor.ndim != 3 or tensor.shape[2] != N_STEPS:
            raise ValueError(f"expected (n_stays, channels, {N_STEPS}), got {tensor.shape}")
        values = tensor[:, self._value_idx, :].astype(np.float32)
        mask = tensor[:, self._mask_idx, :]
        if not np.isfinite(values).all():
            raise ValueError("value channels contain NaN/inf; this loader expects the imputed tensor")
        if not np.isin(mask, (0.0, 1.0)).all():
            raise ValueError("mask channels must hold only 0 and 1")
        return values, mask != self.mask_measured

    def _forward_log(self, x, channel):
        return np.log1p(x) if self._log[channel] else x

    def _prepare(self, values, imputed):
        """Shared tensor slice -> the (n, C, T) values the model is trained on."""
        if self.head_fill == 'backfill':
            values = fill_heads(values, imputed, self.median)
        out = values.astype(np.float32, copy=True)
        if self._log.any():
            out[:, self._log, :] = np.log1p(np.maximum(out[:, self._log, :], 0.0))
        if self.scaling == 'minmax':
            out = np.clip(out, self.lo[None, :, None], self.hi[None, :, None])
        return out

    def _class_mask(self, split, n_rows):
        if self.class_name == 'all':
            return np.ones(n_rows, dtype=bool)
        stay_ids = np.load(self._file(f'stays_{split}.npy'))
        if len(stay_ids) != n_rows:
            raise ValueError(f"stays_{split}.npy has {len(stay_ids)} rows, tensor has {n_rows}")
        with open(self._file('annotations.json')) as f:
            stays = json.load(f)['stays']
        died = np.array([float(stays[str(int(s))]['mortality_in_hospital']) for s in stay_ids])
        return died == (1.0 if self.class_name == 'died' else 0.0)

    def _scale(self, x):
        if self.scaling == 'minmax':
            unit = (x - self.lo[None, :, None]) / (self.hi - self.lo)[None, :, None]
            return TANH_RANGE * (2.0 * unit - 1.0)
        return (x - self.mean[None, :, None]) / self.std[None, :, None]

    def inverse_transform(self, scaled):
        """Scaled (n, C, T) or (n, C, 1, T) -> clinical units, shape (n, C, T).

        The only constraint added is the shared physiological bounds, or a
        floor at 0 for releases without them (and, for minmax, the clip range
        the model was trained on).
        """
        x = np.asarray(scaled, dtype=np.float64)
        if x.ndim == 4:
            x = x[:, :, 0, :]
        if self.scaling == 'minmax':
            x = (x / TANH_RANGE + 1.0) / 2.0 * (self.hi - self.lo)[None, :, None] + self.lo[None, :, None]
            x = np.clip(x, self.lo[None, :, None], self.hi[None, :, None])
        else:
            x = x * self.std[None, :, None] + self.mean[None, :, None]
        if self._log.any():
            x[:, self._log, :] = np.expm1(x[:, self._log, :])
        if self.out_low is not None:
            x = np.clip(x, self.out_low[None, :, None], self.out_high[None, :, None])
        else:
            x = np.maximum(x, 0.0)
        return x.astype(np.float32)

    def describe(self):
        """Everything needed to reproduce or invert the transform, JSON-ready."""
        info = {
            'signals': self.signals,
            'class': self.class_name,
            'n_train_stays': int(self.n_train),
            'head_fill': self.head_fill,
            'scaling': self.scaling,
            'log_transformed': [s for s in self.signals if s in LOG_SIGNALS],
            'zscore_mean': [float(v) for v in self.mean],
            'zscore_std': [float(v) for v in self.std],
            'mask_convention': '1 = measured' if self.mask_measured == 1 else '1 = imputed',
        }
        if self.out_low is not None:
            info['output_bounds'] = {s: [float(lo), float(hi)]
                                     for s, lo, hi in zip(self.signals, self.out_low, self.out_high)}
        else:
            info['output_floor'] = 0.0
        if self.scaling == 'minmax':
            info['clip_percentiles'] = list(CLIP_PERCENTILES)
            info['clip_low'] = [float(np.expm1(v)) if lg else float(v) for v, lg in zip(self.lo, self._log)]
            info['clip_high'] = [float(np.expm1(v)) if lg else float(v) for v, lg in zip(self.hi, self._log)]
            info['tanh_range'] = TANH_RANGE
        if self.head_fill == 'backfill':
            info['never_measured_fill'] = [float(v) for v in self.median]
        return info

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
