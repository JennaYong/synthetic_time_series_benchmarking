# -*- coding: utf-8 -*-
"""MIMIC-IV ICU time-series extraction through PyHealth, with LOCF imputation.

Standalone; it does not import `mimic.py`.  Same signals and value/mask channel
pairs as `mimic.py`, loaded through `pyhealth.datasets.MIMIC4EHRDataset` and a
`BaseTask`.  Deliberate departures from it:

* Every channel of a stay shares one clock.  Hour h is
  ``[intime + h, intime + h + 1)`` with ICU ``intime`` floored to the hour, and
  each slot is the mean of the readings that fall in it.  `mimic.py` instead
  starts each signal at its own first reading (labs at the subject's first lab
  ever) and keeps only readings exactly on the hour, so its slot t means a
  different absolute time in every channel.
* Value channels are imputed by PyHealth's `TimeseriesProcessor` instead of
  holding NaN.
* Train and test share no subject, for membership/privacy evaluation.
* The tensors hold only the time series.  Age and mortality, which `mimic.py`
  repeats across all 72 slots as channels 18 and 19, are written to
  `annotations.json` instead, and mortality is per admission rather than per
  subject.  The ICD and antibiotic indicator channels are disabled for now.

Pipeline:

1. `TimeDiffMIMIC4Dataset` loads the tables declared in `mimic_pyhealth.yaml`.
   Its `preprocess_<table>` hooks keep only the vital/lab itemids (and the
   antibiotic prescriptions, when that table is loaded) before PyHealth caches
   the global event table.
2. (Disabled) The top-72 ICD codes and antibiotics are ranked over the whole
   event table, as `mimic.py` ranks them over the whole CSV.
3. `TimeDiffHourlyMIMIC4` runs per patient: it rebuilds the vitalsign and
   chemistry rows, keeps stays with >= 70 values of every vital whose ICU stay
   lasts the full 72 hours, and averages each signal into the stay's 72 hourly
   slots.  The missing mask is taken there, before imputation.
4. `set_task` hands the (72, 8) value grid to `TimeseriesProcessor`, which
   forward-fills each signal from its last observation (LOCF) and zero-fills
   slots before the first one, including signals never observed in the window.
5. The samples are stacked into a `(16, 72)` layout and split ~75/25 by
   subject, with subjects stratified on mortality.
6. With `forward_fill`, the slots before each channel's first reading are reset
   from 0 to that channel's median over the measured train slots.
7. Every value channel is clipped to its physiological bounds (`CLIP_BOUNDS`),
   and the result is saved.

Outputs
-------

Eight files are written to `--out-dir` (default `~/Desktop/processed_mimic_pyhealth/`).
One row is one ICU stay.  Rows are sorted by stay_id within each split, and row
i of `train.npy`, `stays_train.npy` and `subjects_train.npy` is the same stay
(likewise for test).

`train.npy`, `test.npy` -- float32, shape `(n_stays, 16, 72)`, no NaN.
  Axis 0 is the stay, axis 1 the channel, axis 2 the hour 0..71 after ICU
  admission (hour h covers ``[intime + h, intime + h + 1)``, intime floored to
  the hour; the same clock for every channel).  Values are in raw clinical
  units, not normalised.  Channels:

      idx  name                  content
      ---  --------------------  ------------------------------------------------
       0   heart_rate            beats/min
       1   heart_rate_mask       1 = measured, 0 = imputed
       2   sbp                   systolic blood pressure, mmHg (invasive or cuff)
       3   sbp_mask
       4   dbp                   diastolic blood pressure, mmHg
       5   dbp_mask
       6   resp_rate             breaths/min
       7   resp_rate_mask
       8   spo2                  peripheral O2 saturation, %
       9   spo2_mask
      10   glucose               serum glucose, mg/dL
      11   glucose_mask
      12   creatinine            serum creatinine, mg/dL
      13   creatinine_mask
      14   sodium                serum sodium, mEq/L
      15   sodium_mask

  Value channels (even indices 0..14): a measured slot holds the mean of the
  readings in that hour.  A missing slot holds the last measured value of the
  same channel (LOCF); slots before the channel's first reading, and every slot
  of a channel never measured in the window, hold the channel's median over the
  measured train slots (`channels.json["leading_fill"]`; computed on train only
  and applied to both splits).  With `--impute zero` every missing slot holds 0
  instead.  Every value channel is then clipped to its physiological bounds
  (`CLIP_BOUNDS`, also in `channels.json["clip_bounds"]`); with `--impute zero`
  only measured slots are clipped.  The mask channel right after each value
  channel (odd indices 1..15) is 1 where the slot holds an actual measurement
  and 0 where it was imputed, so evaluate on ``values[mask == 1]``.  Labs are
  drawn a few times a day, so their masks are mostly 0; vitals are mostly 1.

`stays_train.npy`, `stays_test.npy` -- int64, shape `(n_stays,)`: the MIMIC-IV
  `stay_id` of each row.

`subjects_train.npy`, `subjects_test.npy` -- int64, shape `(n_stays,)`: the
  MIMIC-IV `subject_id` of each row.  A subject with several stays repeats, but
  never appears in both splits.  About 75% of subjects are in train, drawn
  separately within mortality = 0 and mortality = 1 (a subject counts as 1 when
  any of their saved stays has mortality = 1).

`channels.json` -- run metadata: `channels` (the 16 names above, in order),
  `n_train` / `n_test` (stays), `n_subjects_train` / `n_subjects_test`,
  `n_steps`, `granularity`, `time_axis`, `split`, `imputation`, `mask`,
  `leading_fill` (the per-signal value of the slots before the first reading,
  null with `--impute zero`), `clip_bounds`, `clipping`,
  `min_measurements_per_vital` and `dev`.

`annotations.json` -- per-stay labels that are NOT in the tensors, for
  conditioning or evaluation.  `stays` maps each stay_id (as a string) to a
  record holding: split, subject_id, hadm_id, age (patients.anchor_age),
  mortality (1 when the subject died in the hospital admission this ICU stay
  belongs to; the label the split is stratified on), demographics (gender,
  race, insurance, marital_status), admission details (admission_type /
  admission_location, via_ed, first/last care unit, icu_stay_number), length of
  stay (los_icu_days, los_hospital_days), outcomes (mortality_in_hospital for
  this admission, mortality_in_icu, mortality_30day, mortality_1year,
  days_discharge_to_death, readmission_30day, discharge_location), diagnoses
  (primary_icd_code / version / title, icd_codes, icd_version), APR-DRG
  description / severity / mortality, and hospital services.  `fields`
  describes every record key, and `label_rates` gives the positive count and
  rate of each binary outcome.

Loading:

    import json, numpy as np
    out = "/Users/yenping/Desktop/processed_mimic_pyhealth"
    train = np.load(f"{out}/train.npy")                   # (n, 16, 72)
    stays = np.load(f"{out}/stays_train.npy")
    meta = json.load(open(f"{out}/channels.json"))
    ann = json.load(open(f"{out}/annotations.json"))["stays"]
    hr, hr_mask = train[:, 0], train[:, 1]                # (n, 72) each
    measured_hr = hr[hr_mask == 1]
    mortality = np.array([ann[str(s)]["mortality"] for s in stays])

Each run converts every listed CSV to Parquet under a PyHealth cache
(`--cache-dir`, default `~/Library/Caches/pyhealth`), which for the 42 GB
chartevents.csv takes a while and several GB of disk.  The cache is deleted
when the run ends, whether it succeeded or not, so every run starts from the
CSVs.

Usage:
    python mimic_pyhealth.py          # defaults: ~/Desktop/physionet.org/files/mimiciv/3.1 -> ~/Desktop/processed_mimic_pyhealth
    python mimic_pyhealth.py --dev    # 1000-patient smoke run
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Sequence

import narwhals as nw
import numpy as np
import pandas as pd
import polars as pl
from pyhealth.datasets import MIMIC4EHRDataset
from pyhealth.tasks import BaseTask

LOG = logging.getLogger("mimic_pyhealth")

MIMIC_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"

# (itemids, low, high, high_is_inclusive) per concepts/measurement/vitalsign.sql
VITAL_ITEMS: dict[str, tuple[tuple[int, ...], float, float, bool]] = {
    "heart_rate": ((220045,), 0, 300, False),
    "sbp": ((220179, 220050, 225309), 0, 400, False),
    "dbp": ((220180, 220051, 225310), 0, 300, False),
    "resp_rate": ((220210, 224690), 0, 70, False),
    "spo2": ((220277,), 0, 100, True),
}
VITALS: tuple[str, ...] = tuple(VITAL_ITEMS)

# (itemid, high) per concepts/measurement/chemistry.sql; all also require > 0
LAB_ITEMS: dict[str, tuple[int, float]] = {
    "glucose": (50931, 10000),
    "creatinine": (50912, 150),
    "sodium": (50983, 200),
}
LABS: tuple[str, ...] = tuple(LAB_ITEMS)

# channel order of the saved tensor: value/mask per signal
SIGNALS: tuple[str, ...] = VITALS + LABS

# (low, high) physiological bounds the value channels are clipped to after
# imputation.  Upper bounds and the spo2 / sodium lower bounds are ricu's
# concept-dict.json min/max; where ricu's lower bound is 0, the adult
# implausible-value screening bound is used instead.
CLIP_BOUNDS: dict[str, tuple[float, float]] = {
    "heart_rate": (20.0, 300.0),   # beats/min
    "sbp": (40.0, 300.0),          # mmHg
    "dbp": (20.0, 200.0),          # mmHg
    "resp_rate": (4.0, 120.0),     # breaths/min
    "spo2": (50.0, 100.0),         # %
    "glucose": (10.0, 1000.0),     # mg/dL
    "creatinine": (0.1, 15.0),     # mg/dL
    "sodium": (110.0, 165.0),      # mEq/L
}
assert set(CLIP_BOUNDS) == set(SIGNALS)

N_STEPS = 72  # hours kept per stay; also the width of the ICD/antibiotic vectors

# drug-name and route rules of concepts/medication/antibiotic.sql
_ANTIBIOTIC_SUBSTRINGS = (
    "adoxa", "ala-tet", "alodox", "amikacin", "amikin", "amoxicill", "amphotericin",
    "anidulafungin", "ancef", "clavulanate", "ampicillin", "augmentin", "avelox",
    "avidoxy", "azactam", "azithromycin", "aztreonam", "axetil", "bactocill",
    "bactrim", "bactroban", "bethkis", "biaxin", "bicillin l-a", "cayston",
    "cefazolin", "cedax", "cefoxitin", "ceftazidime", "cefaclor", "cefadroxil",
    "cefdinir", "cefditoren", "cefepime", "cefotan", "cefotetan", "cefotaxime",
    "ceftaroline", "cefpodoxime", "cefpirome", "cefprozil", "ceftibuten", "ceftin",
    "ceftriaxone", "cefuroxime", "cephalexin", "cephalothin", "cephapririn",
    "chloramphenicol", "cipro", "ciprofloxacin", "claforan", "clarithromycin",
    "cleocin", "clindamycin", "cubicin", "dicloxacillin", "dirithromycin", "doryx",
    "doxycy", "duricef", "dynacin", "ery-tab", "eryped", "eryc", "erythrocin",
    "erythromycin", "factive", "flagyl", "fortaz", "furadantin", "garamycin",
    "gentamicin", "kanamycin", "keflex", "kefzol", "ketek", "levaquin",
    "levofloxacin", "lincocin", "linezolid", "macrobid", "macrodantin", "maxipime",
    "mefoxin", "metronidazole", "meropenem", "methicillin", "minocin",
    "minocycline", "monodox", "monurol", "morgidox", "moxatag", "moxifloxacin",
    "mupirocin", "myrac", "nafcillin", "neomycin", "nicazel doxy 30",
    "nitrofurantoin", "norfloxacin", "noroxin", "ocudox", "ofloxacin", "omnicef",
    "oracea", "oraxyl", "oxacillin", "pc pen vk", "pce dispertab", "panixine",
    "pediazole", "penicillin", "periostat", "pfizerpen", "piperacillin",
    "tazobactam", "primsol", "proquin", "raniclor", "rifadin", "rifampin",
    "rocephin", "smz-tmp", "septra", "septra ds", "solodyn", "spectracef",
    "streptomycin", "sulfadiazine", "sulfamethoxazole", "trimethoprim", "sulfatrim",
    "sulfisoxazole", "suprax", "synercid", "tazicef", "tetracycline", "timentin",
    "tobramycin", "unasyn", "vancocin", "vancomycin", "vantin", "vibativ",
    "vibra-tabs", "vibramycin", "zinacef", "zithromax", "zosyn", "zyvox",
)
_ANTIBIOTIC_PATTERN = "|".join(re.escape(s) for s in _ANTIBIOTIC_SUBSTRINGS)
_EXCLUDED_ROUTES = frozenset({"OU", "OS", "OD", "AU", "AS", "AD", "TP"})

DEFAULT_EHR_ROOT = Path("/path/to/physionet.org/files/mimiciv/3.1")
DEFAULT_OUT_DIR = Path("/path/to/processed_mimic_pyhealth")

CONFIG_PATH = Path(__file__).with_name("mimic_pyhealth.yaml")
TABLES = [
    # "diagnoses_icd",  # icd channel
    # "prescriptions",  # antibiotic channel
    "labevents",
    "chartevents",
]

_VITAL_ITEMIDS = sorted({str(i) for ids, *_ in VITAL_ITEMS.values() for i in ids})
_LAB_ITEMIDS = sorted(str(itemid) for itemid, _ in LAB_ITEMS.values())
_HOUR_MS = 3_600_000


class TimeDiffMIMIC4Dataset(MIMIC4EHRDataset):
    """`MIMIC4EHRDataset` that drops rows `mimic.py` never reads before caching.

    `MIMIC4Dataset` builds a plain `MIMIC4EHRDataset` internally, so the
    filtering hooks have to live on this subclass.  Every CSV column arrives as a
    string, hence the string itemids.
    """

    def preprocess_chartevents(self, df: nw.LazyFrame) -> nw.LazyFrame:
        return df.filter(
            ~nw.col("stay_id").is_null() & nw.col("itemid").is_in(_VITAL_ITEMIDS)
        )

    def preprocess_labevents(self, df: nw.LazyFrame) -> nw.LazyFrame:
        return df.filter(
            ~nw.col("valuenum").is_null() & nw.col("itemid").is_in(_LAB_ITEMIDS)
        )

    def preprocess_prescriptions(self, df: nw.LazyFrame) -> nw.LazyFrame:
        """Drug-name and route rules of concepts/medication/antibiotic.sql."""
        df = df.filter(
            ~nw.col("drug").is_null()
            & ~nw.col("route").is_null()
            & ~nw.col("drug_type").is_null()
        )
        drug = nw.col("drug").str.to_lowercase()
        route = nw.col("route").str.to_lowercase()
        return df.filter(
            (nw.col("drug_type") != "BASE")
            & ~nw.col("route").is_in(sorted(_EXCLUDED_ROUTES))
            & ~route.str.contains("ear", literal=True)
            & ~route.str.contains("eye", literal=True)
            & ~drug.str.contains("cream", literal=True)
            & ~drug.str.contains("desensitization", literal=True)
            & ~drug.str.contains("ophth oint", literal=True)
            & ~drug.str.contains("gel", literal=True)
            & drug.str.contains(_ANTIBIOTIC_PATTERN)
        )


def rank_top_codes(events: pl.LazyFrame, n_codes: int = N_STEPS) -> tuple[list[str], list[str]]:
    """Top ICD codes by row count and top antibiotics by subject count.

    Ties are broken by code so the ranking is deterministic; `mimic.py` leaves
    them in pandas' `value_counts` order.
    """
    code = "diagnoses_icd/icd_code"
    icd = (
        events.filter((pl.col("event_type") == "diagnoses_icd") & pl.col(code).is_not_null())
        .group_by(code)
        .agg(pl.len().alias("n"))
        .sort(["n", code], descending=[True, False])
        .head(n_codes)
        .collect(engine="streaming")[code]
        .to_list()
    )
    drug = "prescriptions/drug"
    abx = (
        events.filter(pl.col("event_type") == "prescriptions")
        .select("patient_id", drug)
        .unique()
        .group_by(drug)
        .agg(pl.len().alias("n"))
        .sort(["n", drug], descending=[True, False])
        .head(n_codes)
        .collect(engine="streaming")[drug]
        .to_list()
    )
    LOG.info("ranked top %s ICD codes and top %s antibiotics", len(icd), len(abx))
    return icd, abx


def _vitalsign(charts: pl.DataFrame) -> pl.DataFrame:
    """One row per (stay_id, charttime), as `mimic.load_vitalsign` builds it."""
    df = charts.select(
        pl.col("chartevents/stay_id").cast(pl.Int64).alias("stay_id"),
        pl.col("timestamp").alias("charttime"),
        pl.col("chartevents/itemid").cast(pl.Int32).alias("itemid"),
        pl.col("chartevents/valuenum").cast(pl.Float32, strict=False).alias("valuenum"),
    )
    value = pl.col("valuenum")
    aggs = []
    for name, (ids, low, high, inclusive) in VITAL_ITEMS.items():
        upper = value <= high if inclusive else value < high
        keep = pl.col("itemid").is_in(ids) & (value > low) & upper
        aggs.append(value.filter(keep).cast(pl.Float64).mean().cast(pl.Float32).alias(name))
    return df.group_by(["stay_id", "charttime"]).agg(aggs).sort(["stay_id", "charttime"])


def _chemistry(labs: pl.DataFrame) -> pl.DataFrame:
    """One row per specimen, as `mimic.load_chemistry` builds it."""
    df = labs.select(
        pl.col("labevents/specimen_id").cast(pl.Int64).alias("specimen_id"),
        pl.col("timestamp").alias("charttime"),
        pl.col("labevents/itemid").cast(pl.Int32).alias("itemid"),
        pl.col("labevents/valuenum").cast(pl.Float32, strict=False).alias("valuenum"),
    ).filter(pl.col("valuenum") > 0)
    value = pl.col("valuenum")
    aggs = [
        value.filter((pl.col("itemid") == itemid) & (value <= high)).max().alias(name)
        for name, (itemid, high) in LAB_ITEMS.items()
    ]
    return (
        df.group_by("specimen_id")
        .agg(pl.col("charttime").max(), *aggs)
        .sort(["charttime", "specimen_id"])
    )


def _hourly_mean(frame: pl.DataFrame, name: str, anchor_ms: int) -> np.ndarray:
    """Mean of ``name`` over each hour ``[anchor + h, anchor + h + 1)``, NaN where empty."""
    observed = frame.filter(pl.col(name).is_not_null())
    hours = (observed["charttime"].dt.epoch("ms").to_numpy() - anchor_ms) // _HOUR_MS
    keep = (hours >= 0) & (hours < N_STEPS)
    hours = hours[keep]
    values = observed[name].to_numpy().astype(np.float64)[keep]
    sums = np.bincount(hours, weights=values, minlength=N_STEPS)
    counts = np.bincount(hours, minlength=N_STEPS)
    out = np.full(N_STEPS, np.nan)
    np.divide(sums, counts, out=out, where=counts > 0)
    return out


def _top_indicator(codes: set[str], top: Sequence[str]) -> np.ndarray:
    out = np.zeros(N_STEPS, dtype=np.float32)
    for i, code in enumerate(top):
        if code in codes:
            out[i] = 1.0
    return out


def channel_names() -> list[str]:
    return [f"{s}{suffix}" for s in SIGNALS for suffix in ("", "_mask")]


class TimeDiffHourlyMIMIC4(BaseTask):
    """One sample per ICU stay: the 72-hour signal grid, its mask, age and mortality.

    ``signals`` goes through PyHealth's ``TimeseriesProcessor`` and comes out
    imputed; ``mask`` is 1 where that slot was measured and 0 where it was
    imputed.  ``age`` is
    ``patients.anchor_age`` (NaN when unknown) and ``mortality`` the
    ``hospital_expire_flag`` of the admission the stay belongs to.
    """

    task_name: str = "TimeDiffHourlyMIMIC4"

    def __init__(
        self,
        # top_icd: Sequence[str],
        # top_antibiotics: Sequence[str],
        min_measurements: int = 70,
        impute_strategy: str = "forward_fill",
    ):
        # self.top_icd = list(top_icd)
        # self.top_antibiotics = list(top_antibiotics)
        self.min_measurements = min_measurements
        self.input_schema: Dict[str, Any] = {
            "signals": (
                "timeseries",
                {"sampling_rate": timedelta(hours=1), "impute_strategy": impute_strategy},
            ),
            "mask": "tensor",
            # "icd": "tensor",
            # "antibiotic": "tensor",
            "age": "tensor",
        }
        self.output_schema: Dict[str, Any] = {"mortality": "tensor"}
        super().__init__()

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        charts = patient.get_events(event_type="chartevents", return_df=True)
        if charts.height == 0:
            return []
        vitals = _vitalsign(charts)
        counts = vitals.group_by("stay_id").agg(pl.col(v).count() for v in VITALS)
        icustays = patient.get_events(event_type="icustays", return_df=True).select(
            pl.col("icustays/stay_id").cast(pl.Int64).alias("stay_id"),
            pl.col("icustays/hadm_id").cast(pl.Int64, strict=False).alias("hadm_id"),
            pl.col("timestamp").dt.truncate("1h").alias("anchor"),
            pl.col("icustays/outtime")
            .str.strptime(pl.Datetime("ms"), "%Y-%m-%d %H:%M:%S", strict=False)
            .alias("outtime"),
        )
        # the stay has to last through the whole window
        eligible = (
            counts.filter(pl.all_horizontal(pl.col(v) >= self.min_measurements for v in VITALS))
            .join(icustays, on="stay_id", how="inner")
            .filter(pl.col("outtime") >= pl.col("anchor") + pl.duration(hours=N_STEPS))
            .with_columns(pl.col("anchor").dt.epoch("ms").alias("anchor_ms"))
            .sort("stay_id")
        )
        if eligible.is_empty():
            return []

        chemistry = _chemistry(patient.get_events(event_type="labevents", return_df=True))
        # diagnoses = patient.get_events(event_type="diagnoses_icd", return_df=True)
        # icd = _top_indicator(set(diagnoses["diagnoses_icd/icd_code"].drop_nulls()), self.top_icd)
        # drugs = patient.get_events(event_type="prescriptions", return_df=True)
        # abx = _top_indicator(set(drugs["prescriptions/drug"].drop_nulls()), self.top_antibiotics)

        demographics = patient.get_events(event_type="patients")
        age = demographics[0].attr_dict.get("anchor_age") if demographics else None
        age = float(age) if age is not None else float("nan")
        admissions = patient.get_events(event_type="admissions", return_df=True)
        died_in = set(
            admissions.filter(
                pl.col("admissions/hospital_expire_flag").cast(pl.Float64, strict=False) == 1
            )["admissions/hadm_id"]
            .cast(pl.Int64, strict=False)
            .drop_nulls()
            .to_list()
        )

        samples = []
        for row in eligible.iter_rows(named=True):
            stay = vitals.filter(pl.col("stay_id") == row["stay_id"])
            grid = np.stack(
                [
                    _hourly_mean(chemistry if name in LAB_ITEMS else stay, name, row["anchor_ms"])
                    for name in SIGNALS
                ],
                axis=1,
            )
            hours = [row["anchor"] + timedelta(hours=h) for h in range(N_STEPS)]

            samples.append(
                {
                    "patient_id": patient.patient_id,
                    "stay_id": int(row["stay_id"]),
                    "signals": (hours, grid),
                    "mask": (~np.isnan(grid)).astype(np.float32),
                    # "icd": icd,
                    # "antibiotic": abx,
                    "age": age,
                    "mortality": float(row["hadm_id"] in died_in),
                }
            )
        return samples


def stack_samples(samples) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """`(n_stays, 16, 72)` tensor in `channel_names()` order, rows sorted by stay_id,
    with the stay_id, subject_id, age and mortality of each row."""
    rows, stays, subjects, ages, mortality = [], [], [], [], []
    for i in range(len(samples)):
        s = samples[i]
        values = s["signals"].numpy().T  # (8, 72)
        mask = s["mask"].numpy().T
        channels = [c for k in range(len(SIGNALS)) for c in (values[k], mask[k])]
        # channels += [s["icd"].numpy(), s["antibiotic"].numpy()]
        rows.append(np.stack(channels).astype(np.float32))
        stays.append(int(s["stay_id"]))
        subjects.append(int(s["patient_id"]))
        ages.append(float(s["age"]))
        mortality.append(float(s["mortality"]))
    order = np.argsort(stays, kind="stable")
    return (
        np.stack(rows)[order],
        np.asarray(stays, dtype=np.int64)[order],
        np.asarray(subjects, dtype=np.int64)[order],
        np.asarray(ages, dtype=np.float64)[order],
        np.asarray(mortality, dtype=np.int64)[order],
    )


def split_by_subject(
    tensor: np.ndarray,
    stay_ids: np.ndarray,
    subject_ids: np.ndarray,
    mortality: np.ndarray,
    seed: int = 2023,
    train_frac: float = 0.75,
) -> dict[str, np.ndarray]:
    """75/25 split with all of a subject's stays on the same side, stratified on mortality.

    Mortality is per stay, so a subject is labelled 1 when any of their stays
    has mortality 1.  The stay ratio lands near 75/25 rather than exactly on it.
    """
    rng = np.random.default_rng(seed)
    subjects, inverse = np.unique(subject_ids, return_inverse=True)
    labels = np.zeros(len(subjects), dtype=mortality.dtype)
    np.maximum.at(labels, inverse, mortality)
    test_subjects = []
    for label in np.unique(labels):
        members = rng.permutation(subjects[labels == label])
        test_subjects.append(members[: len(members) - int(len(members) * train_frac)])
    test = np.isin(subject_ids, np.concatenate(test_subjects))

    overlap = np.intersect1d(subject_ids[test], subject_ids[~test])
    assert overlap.size == 0, f"{overlap.size} subjects on both sides of the split"
    LOG.info(
        "split by subject: train %s stays / %s subjects, test %s stays / %s subjects",
        f"{(~test).sum():,}", f"{np.unique(subject_ids[~test]).size:,}",
        f"{test.sum():,}", f"{np.unique(subject_ids[test]).size:,}",
    )
    return {
        "train": tensor[~test],
        "test": tensor[test],
        "stays_train": stay_ids[~test],
        "stays_test": stay_ids[test],
        "subjects_train": subject_ids[~test],
        "subjects_test": subject_ids[test],
    }


def fill_leading_with_median(parts: dict[str, np.ndarray]) -> dict[str, float]:
    """Set the slots before each channel's first reading to the channel's train median.

    LOCF has nothing to carry into those slots, so `TimeseriesProcessor` leaves
    them at 0.  The median is taken over measured train slots only, so the test
    split never informs it.  Edits `parts["train"]` and `parts["test"]` in place.
    """
    medians: dict[str, float] = {}
    for k, name in enumerate(SIGNALS):
        value, mask = 2 * k, 2 * k + 1
        measured = parts["train"][:, value][parts["train"][:, mask] == 1]
        medians[name] = float(np.median(measured))
        for split in ("train", "test"):
            x = parts[split]
            leading = np.cumsum(x[:, mask] == 1, axis=1) == 0
            x[:, value][leading] = medians[name]
        LOG.info("leading fill %-12s median %.2f", name, medians[name])
    return medians


def clip_physiological(parts: dict[str, np.ndarray], impute_strategy: str) -> None:
    """Clip each value channel to `CLIP_BOUNDS`, measured and imputed slots alike.

    With ``zero`` imputation the imputed slots hold 0 as a placeholder rather
    than a value, so only measured slots are clipped.  Edits `parts["train"]`
    and `parts["test"]` in place.
    """
    for k, name in enumerate(SIGNALS):
        low, high = CLIP_BOUNDS[name]
        value, mask = 2 * k, 2 * k + 1
        for split in ("train", "test"):
            x = parts[split]
            target = (
                np.ones_like(x[:, mask], dtype=bool)
                if impute_strategy == "forward_fill"
                else x[:, mask] == 1
            )
            v = x[:, value]
            out = target & ((v < low) | (v > high))
            measured_out = out & (x[:, mask] == 1)
            v[target] = np.clip(v[target], low, high)
            LOG.info(
                "clip %-5s %-12s [%g, %g]: %s slots (%s measured)",
                split, name, low, high, f"{out.sum():,}", f"{measured_out.sum():,}",
            )


# every annotation emitted per stay, with the description written into the json
ANNOTATION_FIELDS: dict[str, str] = {
    "split": "train or test, matching which tensor the stay was saved in",
    "subject_id": "MIMIC-IV patient identifier",
    "hadm_id": "hospital admission the ICU stay belongs to",
    "age": "patients.anchor_age, the age at the anchor year",
    "mortality": (
        "admissions.hospital_expire_flag of the admission this ICU stay belongs to,"
        " i.e. whether the subject died in this admission; the split is stratified on it"
    ),
    "gender": "patients.gender",
    "admission_type": "admissions.admission_type, e.g. EW EMER. / ELECTIVE / OBSERVATION",
    "admission_location": "admissions.admission_location",
    "insurance": "admissions.insurance",
    "marital_status": "admissions.marital_status",
    "race": "admissions.race",
    "via_ed": "1 when admissions.edregtime is set, i.e. the patient came through the ED",
    "first_careunit": "icustays.first_careunit",
    "last_careunit": "icustays.last_careunit",
    "icu_stay_number": "1-based rank of this ICU stay among the subject's stays, by intime",
    "los_icu_days": "icustays.los",
    "los_hospital_days": "dischtime - admittime, in days",
    "discharge_location": "admissions.discharge_location; DIED and HOSPICE are outcomes too",
    "mortality_in_hospital": "same label as mortality, read from admissions.csv",
    "mortality_in_icu": "1 when admissions.deathtime falls inside the ICU intime..outtime window",
    "days_discharge_to_death": (
        "patients.dod minus dischtime, in days; null when no date of death is known."
        " Zero or negative for an in-hospital death, since dod is only a date"
    ),
    "mortality_30day": "1 when dod is known and falls no later than 30 days after discharge",
    "mortality_1year": "1 when dod is known and falls no later than 365 days after discharge",
    "readmission_30day": (
        "1 when the subject is admitted again within 30 days of this discharge;"
        " 0 for patients who died in hospital, as they cannot be readmitted"
    ),
    "primary_icd_code": "icd_code at the lowest seq_num of the admission",
    "primary_icd_version": "9 or 10",
    "primary_icd_title": "d_icd_diagnoses.long_title of the primary code",
    "icd_codes": "every icd_code billed for the admission, ordered by seq_num",
    "icd_version": "the admission's coding system, or null for the rare mixed admission",
    "drg_description": "APR-DRG description from drgcodes.csv",
    "drg_severity": "APR-DRG severity of illness, 1-4",
    "drg_mortality": "APR-DRG risk of mortality, 1-4",
    "services": "curr_service values of the admission, ordered by transfertime",
    "first_service": "the admission's first curr_service, e.g. MED / SURG / CMED",
}

_BINARY_LABELS = (
    "mortality",
    "mortality_in_hospital",
    "mortality_in_icu",
    "mortality_30day",
    "mortality_1year",
    "readmission_30day",
    "via_ed",
)


def _to_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, format=MIMIC_TIME_FORMAT, errors="coerce")


def _json_safe(value):
    """numpy scalars, pandas NA and NaN are not json-serialisable as they are."""
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (bool, np.bool_)):  # before int, bool subclasses it
        return int(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return None if np.isnan(value) else float(value)
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def load_annotations(
    csv_dir: Path,
    splits: dict[str, np.ndarray],
    statics: dict[int, tuple[float, int]],
) -> dict:
    """Per-stay status labels for the saved cohort, keyed by ``stay_id``.

    None of this is in the tensors.  ``statics`` maps each stay_id to the
    (age, mortality) the task computed; the rest is the outcome and diagnosis
    metadata the CSV release carries.
    """
    stay_ids = np.concatenate([splits["stays_train"], splits["stays_test"]])
    split_of = {int(s): "train" for s in splits["stays_train"]}
    split_of.update({int(s): "test" for s in splits["stays_test"]})
    LOG.info("annotations: collecting for %s stays", f"{len(stay_ids):,}")

    icu = pd.read_csv(
        csv_dir / "icu" / "icustays.csv",
        usecols=["subject_id", "hadm_id", "stay_id", "first_careunit", "last_careunit", "intime", "outtime", "los"],
    )
    icu["intime"] = _to_datetime(icu["intime"])
    icu["outtime"] = _to_datetime(icu["outtime"])
    icu["icu_stay_number"] = (
        icu.sort_values("intime").groupby("subject_id").cumcount().add(1).reindex(icu.index)
    )
    icu = icu[icu["stay_id"].isin(set(int(s) for s in stay_ids))].set_index("stay_id")

    adm = pd.read_csv(
        csv_dir / "hosp" / "admissions.csv",
        usecols=[
            "subject_id", "hadm_id", "admittime", "dischtime", "deathtime", "admission_type",
            "admission_location", "discharge_location", "insurance", "marital_status", "race",
            "edregtime", "hospital_expire_flag",
        ],
    )
    for col in ("admittime", "dischtime", "deathtime"):
        adm[col] = _to_datetime(adm[col])
    # a readmission is the subject's next admission, so this needs every row
    ordered = adm.sort_values(["subject_id", "admittime"])
    gap = (
        ordered.groupby("subject_id")["admittime"].shift(-1) - ordered["dischtime"]
    ).dt.total_seconds() / 86400.0
    adm["readmit_gap_days"] = gap.reindex(adm.index)
    adm = adm.set_index("hadm_id")

    patients = pd.read_csv(csv_dir / "hosp" / "patients.csv", usecols=["subject_id", "gender", "dod"])
    patients["dod"] = pd.to_datetime(patients["dod"], format="%Y-%m-%d", errors="coerce")
    patients = patients.drop_duplicates("subject_id").set_index("subject_id")

    hadm_ids = set(icu["hadm_id"].dropna().astype("int64"))
    dx = pd.read_csv(
        csv_dir / "hosp" / "diagnoses_icd.csv",
        usecols=["hadm_id", "seq_num", "icd_code", "icd_version"],
    )
    dx = dx[dx["hadm_id"].isin(hadm_ids)].sort_values(["hadm_id", "seq_num"], kind="stable")
    titles = pd.read_csv(csv_dir / "hosp" / "d_icd_diagnoses.csv", usecols=["icd_code", "icd_version", "long_title"])
    title_of = {
        (code, version): title
        for code, version, title in titles.itertuples(index=False, name=None)
    }
    codes_of = dx.groupby("hadm_id")["icd_code"].apply(list).to_dict()
    # seq_num 1 is not unique for a handful of admissions, so take the first row
    primary = dx.drop_duplicates("hadm_id").set_index("hadm_id")
    versions = dx.groupby("hadm_id")["icd_version"].nunique()

    drg = pd.read_csv(
        csv_dir / "hosp" / "drgcodes.csv",
        usecols=["hadm_id", "drg_type", "description", "drg_severity", "drg_mortality"],
    )
    drg = drg[(drg["drg_type"] == "APR") & drg["hadm_id"].isin(hadm_ids)]
    drg = drg.drop_duplicates("hadm_id").set_index("hadm_id")

    svc = pd.read_csv(csv_dir / "hosp" / "services.csv", usecols=["hadm_id", "transfertime", "curr_service"])
    svc = svc[svc["hadm_id"].isin(hadm_ids)]
    svc["transfertime"] = _to_datetime(svc["transfertime"])
    services_of = (
        svc.sort_values("transfertime", kind="stable").groupby("hadm_id")["curr_service"].apply(list).to_dict()
    )

    records: dict[str, dict] = {}
    for stay_id in stay_ids:
        stay_id = int(stay_id)
        if stay_id not in icu.index:
            continue
        stay = icu.loc[stay_id]
        hadm_id = stay["hadm_id"]
        subject_id = int(stay["subject_id"])
        admission = adm.loc[hadm_id] if hadm_id in adm.index else None
        person = patients.loc[subject_id] if subject_id in patients.index else None
        age, mortality = statics[stay_id]

        died_in_hospital = int(admission["hospital_expire_flag"]) if admission is not None else 0
        deathtime = admission["deathtime"] if admission is not None else pd.NaT
        dischtime = admission["dischtime"] if admission is not None else pd.NaT
        dod = person["dod"] if person is not None else pd.NaT

        to_death = (
            (dod - dischtime).total_seconds() / 86400.0
            if not pd.isna(dod) and not pd.isna(dischtime)
            else None
        )
        gap_days = admission["readmit_gap_days"] if admission is not None else np.nan
        diagnoses = codes_of.get(hadm_id, [])
        row = primary.loc[hadm_id] if hadm_id in primary.index else None
        drg_row = drg.loc[hadm_id] if hadm_id in drg.index else None

        record = {
            "split": split_of[stay_id],
            "subject_id": subject_id,
            "hadm_id": int(hadm_id) if not pd.isna(hadm_id) else None,
            "age": age,
            "mortality": mortality,
            "gender": person["gender"] if person is not None else None,
            "admission_type": admission["admission_type"] if admission is not None else None,
            "admission_location": admission["admission_location"] if admission is not None else None,
            "insurance": admission["insurance"] if admission is not None else None,
            "marital_status": admission["marital_status"] if admission is not None else None,
            "race": admission["race"] if admission is not None else None,
            "via_ed": int(admission is not None and not pd.isna(admission["edregtime"])),
            "first_careunit": stay["first_careunit"],
            "last_careunit": stay["last_careunit"],
            "icu_stay_number": stay["icu_stay_number"],
            "los_icu_days": stay["los"],
            "los_hospital_days": (
                (dischtime - admission["admittime"]).total_seconds() / 86400.0
                if admission is not None and not pd.isna(dischtime) and not pd.isna(admission["admittime"])
                else None
            ),
            "discharge_location": admission["discharge_location"] if admission is not None else None,
            "mortality_in_hospital": died_in_hospital,
            "mortality_in_icu": int(
                not pd.isna(deathtime)
                and not pd.isna(stay["intime"])
                and stay["intime"] <= deathtime <= stay["outtime"]
            ),
            "days_discharge_to_death": to_death,
            "mortality_30day": int(to_death is not None and to_death <= 30),
            "mortality_1year": int(to_death is not None and to_death <= 365),
            "readmission_30day": int(
                not died_in_hospital and not pd.isna(gap_days) and 0 <= gap_days <= 30
            ),
            "primary_icd_code": row["icd_code"] if row is not None else None,
            "primary_icd_version": row["icd_version"] if row is not None else None,
            "primary_icd_title": (
                title_of.get((row["icd_code"], row["icd_version"])) if row is not None else None
            ),
            "icd_codes": diagnoses,
            "icd_version": (
                row["icd_version"] if row is not None and versions.get(hadm_id, 0) == 1 else None
            ),
            "drg_description": drg_row["description"] if drg_row is not None else None,
            "drg_severity": drg_row["drg_severity"] if drg_row is not None else None,
            "drg_mortality": drg_row["drg_mortality"] if drg_row is not None else None,
            "services": services_of.get(hadm_id, []),
            "first_service": (services_of.get(hadm_id) or [None])[0],
        }
        records[str(stay_id)] = {k: _json_safe(v) for k, v in record.items()}

    summary = {}
    for label in _BINARY_LABELS:
        values = [r[label] for r in records.values() if r[label] is not None]
        if values:
            summary[label] = {
                "positive": int(sum(values)),
                "rate": round(sum(values) / len(values), 4),
            }
    missing = len(stay_ids) - len(records)
    if missing:
        LOG.warning("annotations: %s stays absent from icustays.csv", missing)
    for label, stats in summary.items():
        LOG.info("  %-22s %6d  %.3f", label, stats["positive"], stats["rate"])

    return {
        "description": (
            "Per-ICU-stay status annotations for the cohort saved in train.npy / test.npy,"
            " keyed by stay_id. Independent of the tensors: nothing here is a model input"
            " unless you make it one."
        ),
        "source": "MIMIC-IV 3.1 hosp/ and icu/ CSVs",
        "n_stays": len(records),
        "label_rates": summary,
        "fields": ANNOTATION_FIELDS,
        "stays": records,
    }


def preprocess(
    ehr_root: Path,
    out_dir: Path,
    cache_dir: Path | None = None,
    min_measurements: int = 70,
    impute_strategy: str = "forward_fill",
    num_workers: int = 1,
    dev: bool = False,
    seed: int = 2023,
) -> dict[str, Path]:
    dataset = TimeDiffMIMIC4Dataset(
        root=str(ehr_root),
        tables=TABLES,
        dataset_name="mimic4_timediff",
        config_path=str(CONFIG_PATH),
        cache_dir=str(cache_dir) if cache_dir else None,
        dev=dev,
        num_workers=num_workers,
    )
    samples = None
    try:
        # top_icd, top_abx = rank_top_codes(dataset.global_event_df)
        task = TimeDiffHourlyMIMIC4(
            min_measurements=min_measurements, impute_strategy=impute_strategy
        )
        samples = dataset.set_task(task, num_workers=num_workers)
        if len(samples) == 0:
            raise RuntimeError("no ICU stay met the vital-sign count and 72-hour length criteria")
        tensor, stays, subjects, ages, mortality = stack_samples(samples)
    finally:
        if samples is not None:
            samples.close()
        shutil.rmtree(dataset.cache_dir, ignore_errors=True)
        LOG.info("removed PyHealth cache %s", dataset.cache_dir)

    LOG.info("tensor: %s", tensor.shape)
    LOG.info("mortality rate %.3f", mortality.mean())

    parts = split_by_subject(tensor, stays, subjects, mortality, seed=seed)
    leading_fill = (
        fill_leading_with_median(parts) if impute_strategy == "forward_fill" else None
    )
    clip_physiological(parts, impute_strategy)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for name, array in parts.items():
        path = out_dir / f"{name}.npy"
        np.save(path, array)
        written[name] = path
        LOG.info("wrote %s %s", path, array.shape)

    meta = out_dir / "channels.json"
    meta.write_text(
        json.dumps(
            {
                "source": "MIMIC-IV CSV via PyHealth, adapted from mimic.py",
                "n_steps": N_STEPS,
                "granularity": "1 hour",
                "channels": channel_names(),
                "n_train": int(len(parts["train"])),
                "n_test": int(len(parts["test"])),
                "n_subjects_train": int(np.unique(parts["subjects_train"]).size),
                "n_subjects_test": int(np.unique(parts["subjects_test"]).size),
                "split": "by subject, no subject in both; subjects stratified on mortality",
                "time_axis": (
                    "hour h covers [intime + h, intime + h + 1) with intime floored to the hour,"
                    " shared by every channel; readings within an hour are averaged"
                ),
                "min_measurements_per_vital": min_measurements,
                "imputation": (
                    f"pyhealth TimeseriesProcessor impute_strategy={impute_strategy!r};"
                    " slots before a channel's first reading hold leading_fill[channel]"
                    " (forward_fill only)"
                ),
                "mask": (
                    "each <signal>_mask channel is 1 where the slot holds an actual"
                    " measurement and 0 where it was imputed; evaluate on values[mask == 1]"
                ),
                "leading_fill": leading_fill,
                "clip_bounds": {
                    name: {"low": low, "high": high} for name, (low, high) in CLIP_BOUNDS.items()
                },
                "clipping": (
                    "value channels clipped to clip_bounds after imputation, measured and"
                    " imputed slots alike (measured slots only with impute_strategy='zero')"
                ),
                "dev": dev,
                # "icd_codes": top_icd,
                # "antibiotics": top_abx,
            },
            indent=2,
        )
    )
    written["channels"] = meta

    annotations = out_dir / "annotations.json"
    statics = {
        int(s): (float(a), int(m)) for s, a, m in zip(stays, ages, mortality)
    }
    annotations.write_text(
        json.dumps(load_annotations(ehr_root, parts, statics), indent=2)
    )
    written["annotations"] = annotations
    LOG.info("wrote %s", annotations)
    return written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MIMIC-IV CSV -> hourly ICU tensors via PyHealth, with LOCF imputation."
    )
    parser.add_argument(
        "--ehr-root",
        type=Path,
        default=DEFAULT_EHR_ROOT,
        help="directory holding the MIMIC-IV hosp/ and icu/ CSVs",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="where PyHealth builds its working cache; deleted when the run ends",
    )
    parser.add_argument("--min-measurements", type=int, default=70)
    parser.add_argument(
        "--impute",
        choices=["forward_fill", "zero"],
        default="forward_fill",
        help="TimeseriesProcessor impute_strategy",
    )
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--dev", action="store_true", help="limit PyHealth to 1000 patients")
    parser.add_argument("--seed", type=int, default=2023)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    preprocess(
        ehr_root=args.ehr_root,
        out_dir=args.out_dir,
        cache_dir=args.cache_dir,
        min_measurements=args.min_measurements,
        impute_strategy=args.impute,
        num_workers=args.num_workers,
        dev=args.dev,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
