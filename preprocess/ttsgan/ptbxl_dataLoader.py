# -*- coding: utf-8 -*-
"""PTB-XL dataset loader for TTS-GAN.

Loads the SSSD-ECG preprocessed PTB-XL npy files (figshare release, 100 Hz,
10 s, 12 leads) and exposes them through the same Dataset interface that
TTS-GAN's unimib_load_dataset provides, so upstream train_GAN.py can consume
PTB-XL without modification (see train_ptbxl_GAN.py).

Input files (under data_path):
    ptbxl_train_data.npy       (17441, 12, 1000) float32
    ptbxl_validation_data.npy  ( 2193, 12, 1000) float32
    ptbxl_test_data.npy        ( 2203, 12, 1000) float32
    ptbxl_train_labels.npy     (17441, 71) float32 multi-hot
    ptbxl_validation_labels.npy( 2193, 71)
    ptbxl_test_labels.npy      ( 2203, 71)

Label handling: the 71 columns are the 71 PTB-XL SCP statements in
alphabetical order (clinical_ts's map_and_filter_labels builds lbl_itos with
sorted(); verified column-by-column against ptbxl_database.csv counts).
44 of them are diagnostic codes that aggregate into the 5 diagnostic
superclasses NORM / MI / STTC / CD / HYP via scp_statements.csv
(diagnostic_class). TTS-GAN trains one unconditional GAN per class, so this
loader filters records by one chosen superclass.

Items are returned as (channels=12, 1, timesteps=window) float32 — the
(C, 1, T) layout train_GAN.py / functions.py expect. `window` defaults to the
full 1000-step record; a smaller divisor of 1000 splits every record into
consecutive non-overlapping windows, which both shortens the sequence the
generator has to model and multiplies the number of training samples.
"""

import os

import numpy as np
from torch.utils.data import Dataset

PTBXL_SCP_CODES = [  # alphabetical; column order of the 71-dim SSSD-ECG label npys
    '1AVB', '2AVB', '3AVB', 'ABQRS', 'AFIB', 'AFLT', 'ALMI', 'AMI',
    'ANEUR', 'ASMI', 'BIGU', 'CLBBB', 'CRBBB', 'DIG', 'EL', 'HVOLT',
    'ILBBB', 'ILMI', 'IMI', 'INJAL', 'INJAS', 'INJIL', 'INJIN', 'INJLA',
    'INVT', 'IPLMI', 'IPMI', 'IRBBB', 'ISCAL', 'ISCAN', 'ISCAS', 'ISCIL',
    'ISCIN', 'ISCLA', 'ISC_', 'IVCD', 'LAFB', 'LAO/LAE', 'LMI', 'LNGQT',
    'LOWT', 'LPFB', 'LPR', 'LVH', 'LVOLT', 'NDT', 'NORM', 'NST_',
    'NT_', 'PAC', 'PACE', 'PMI', 'PRC(S)', 'PSVT', 'PVC', 'QWAVE',
    'RAO/RAE', 'RVH', 'SARRH', 'SBRAD', 'SEHYP', 'SR', 'STACH', 'STD_',
    'STE_', 'SVARR', 'SVTAC', 'TAB_', 'TRIGU', 'VCLVH', 'WPW',
]

SUPERCLASS_OF = {  # diagnostic code -> superclass (form/rhythm codes have none)
    '1AVB': 'CD', '2AVB': 'CD', '3AVB': 'CD', 'ALMI': 'MI', 'AMI': 'MI',
    'ANEUR': 'STTC', 'ASMI': 'MI', 'CLBBB': 'CD', 'CRBBB': 'CD',
    'DIG': 'STTC', 'EL': 'STTC', 'ILBBB': 'CD', 'ILMI': 'MI', 'IMI': 'MI',
    'INJAL': 'MI', 'INJAS': 'MI', 'INJIL': 'MI', 'INJIN': 'MI',
    'INJLA': 'MI', 'IPLMI': 'MI', 'IPMI': 'MI', 'IRBBB': 'CD',
    'ISCAL': 'STTC', 'ISCAN': 'STTC', 'ISCAS': 'STTC', 'ISCIL': 'STTC',
    'ISCIN': 'STTC', 'ISCLA': 'STTC', 'ISC_': 'STTC', 'IVCD': 'CD',
    'LAFB': 'CD', 'LAO/LAE': 'HYP', 'LMI': 'MI', 'LNGQT': 'STTC',
    'LPFB': 'CD', 'LVH': 'HYP', 'NDT': 'STTC', 'NORM': 'NORM',
    'NST_': 'STTC', 'PMI': 'MI', 'RAO/RAE': 'HYP', 'RVH': 'HYP',
    'SEHYP': 'HYP', 'WPW': 'CD',
}

SUPERCLASSES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']

# (71, 5) 0/1 matrix: multi-hot @ this = per-superclass membership counts
_CODE_TO_SUPER = np.zeros((len(PTBXL_SCP_CODES), len(SUPERCLASSES)), dtype=np.float32)
for _i, _code in enumerate(PTBXL_SCP_CODES):
    _super = SUPERCLASS_OF.get(_code)
    if _super is not None:
        _CODE_TO_SUPER[_i, SUPERCLASSES.index(_super)] = 1.0


def superclass_multihot(labels_71):
    """(N, 71) multi-hot -> (N, 5) superclass multi-hot (NORM,MI,STTC,CD,HYP)."""
    return (np.asarray(labels_71, dtype=np.float32) @ _CODE_TO_SUPER > 0).astype(np.float32)


class ptbxl_load_dataset(Dataset):
    """PTB-XL superclass-filtered dataset with the unimib_load_dataset interface.

    Args:
        data_path: directory containing the six ptbxl_*.npy files.
        data_mode: 'Train' (train + validation splits combined, mirroring
            unimib's incl_val_group=False) or 'Test'.
        class_name: one of NORM / MI / STTC / CD / HYP.
        label_mode:
            'any'       - keep records whose label set contains class_name
                          (records may also carry other superclasses);
            'exclusive' - keep records annotated with exactly this one
                          superclass (cleaner class signal, fewer records).
        is_normalize: per-record, per-lead z-normalization (the UniMiB
            pipeline's normalization). train_ptbxl_GAN.py turns this on by
            default; without it TTS-GAN training diverges (see the
            TTS_GAN_PTBXL_NORMALIZE notes there).
        window: timesteps per training item; must divide 1000. The default
            1000 keeps whole records. A smaller value (e.g. 250 = 2.5 s) cuts
            each record into 1000//window consecutive windows, so the sample
            count grows by the same factor. Normalization is applied per
            window, after splitting, so each item still has unit std.
    """

    def __init__(self,
                 data_path='./ptbxl/',
                 data_mode='Train',
                 class_name='NORM',
                 label_mode='any',
                 is_normalize=False,
                 window=1000,
                 verbose=True):
        if class_name not in SUPERCLASSES:
            raise ValueError(f"class_name must be one of {SUPERCLASSES}, got {class_name!r}")
        if label_mode not in ('any', 'exclusive'):
            raise ValueError(f"label_mode must be 'any' or 'exclusive', got {label_mode!r}")
        if data_mode not in ('Train', 'Test'):
            raise ValueError(f"data_mode must be 'Train' or 'Test', got {data_mode!r}")
        if window <= 0 or 1000 % window != 0:
            raise ValueError(f"window must be a positive divisor of 1000, got {window}")

        self.data_mode = data_mode
        self.class_name = class_name

        if data_mode == 'Train':
            splits = ['train', 'validation']
        else:
            splits = ['test']

        data_parts, label_parts = [], []
        for split in splits:
            data_file = os.path.join(data_path, f'ptbxl_{split}_data.npy')
            label_file = os.path.join(data_path, f'ptbxl_{split}_labels.npy')
            for f in (data_file, label_file):
                if not os.path.isfile(f):
                    raise FileNotFoundError(
                        f"Missing {f}. Run relocate_scripts/relocate_ttsgan_ptbxl.sh "
                        f"or point data_path at the PTB-XL npy directory.")
            data_parts.append(np.load(data_file))
            label_parts.append(np.load(label_file))

        data = np.concatenate(data_parts, axis=0)          # (N, 12, 1000)
        labels_71 = np.concatenate(label_parts, axis=0)    # (N, 71)
        super_labels = superclass_multihot(labels_71)      # (N, 5)

        class_idx = SUPERCLASSES.index(class_name)
        has_class = super_labels[:, class_idx] > 0
        if label_mode == 'exclusive':
            keep = has_class & (super_labels.sum(axis=1) == 1)
        else:
            keep = has_class

        data = data[keep].astype(np.float32)
        n_records = data.shape[0]
        # (N, 12, 1000) -> (N * 1000//window, 12, 1, window): the (C, 1, T)
        # layout TTS-GAN uses, with each record split into consecutive windows.
        # Splitting before normalizing means every window has unit std on its
        # own, which is what the discriminator then sees.
        per_record = data.shape[2] // window
        data = data.reshape(n_records, data.shape[1], per_record, window)
        data = data.transpose(0, 2, 1, 3).reshape(-1, data.shape[1], 1, window)
        self.data = data
        self.labels = np.full(self.data.shape[0], class_idx, dtype=np.int64)

        if is_normalize:
            self.data = self._normalize_per_record(self.data)

        if verbose:
            windowing = '' if per_record == 1 else f' -> {per_record} windows of {window} each'
            print(f'PTB-XL {data_mode} split(s) {splits}: {keep.sum()} of {len(keep)} '
                  f'records kept for superclass {class_name} (label_mode={label_mode})'
                  f'{windowing}')
            print(f'data shape is {self.data.shape}, labels shape is {self.labels.shape}')

    @staticmethod
    def _normalize_per_record(x):
        eps = 1e-10
        mean = x.mean(axis=3, keepdims=True)
        std = x.std(axis=3, keepdims=True)
        return (x - mean) / (std + eps)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
