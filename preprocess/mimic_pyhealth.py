# -*- coding: utf-8 -*-
"""MIMIC-IV ICU time-series extraction through PyHealth, with LOCF imputation.

Same signals, statics and channel layout as `mimic.py`, loaded through
`pyhealth.datasets.MIMIC4EHRDataset` and a `BaseTask`.  Three deliberate
departures from it:

* Every channel of a stay shares one clock.  Hour h is
  ``[intime + h, intime + h + 1)`` with ICU ``intime`` floored to the hour, and
  each slot is the mean of the readings that fall in it.  `mimic.py` instead
  starts each signal at its own first reading (labs at the subject's first lab
  ever) and keeps only readings exactly on the hour, so its slot t means a
  different absolute time in every channel.
* Value channels are imputed by PyHealth's `TimeseriesProcessor` instead of
  holding NaN.
* Train and test share no subject, for membership/privacy evaluation.

Pipeline:

1. `TimeDiffMIMIC4Dataset` loads the tables declared in `mimic_pyhealth.yaml`.
   Its `preprocess_<table>` hooks keep only the vital/lab itemids and the
   antibiotic prescriptions before PyHealth caches the global event table.
2. The top-72 ICD codes and antibiotics are ranked over the whole event table,
   as `mimic.py` ranks them over the whole CSV.
3. `TimeDiffHourlyMIMIC4` runs per patient: it rebuilds the vitalsign and
   chemistry rows, keeps stays with >= 70 values of every vital whose ICU stay
   lasts the full 72 hours, and averages each signal into the stay's 72 hourly
   slots.  The missing mask is taken there, before imputation.
4. `set_task` hands the (72, 8) value grid to `TimeseriesProcessor`, which
   forward-fills each signal from its last observation (LOCF) and zero-fills
   slots before the first one, including signals never observed in the window.
5. The samples are stacked into the `mimic.py` layout, split ~75/25 by subject
   with subjects stratified on mortality, and saved.

Outputs
-------

Eight files are written to `--out-dir` (default `~/Desktop/processed_mimic_pyhealth/`).
One row is one ICU stay.  Rows are sorted by stay_id within each split, and row
i of `train.npy`, `stays_train.npy` and `subjects_train.npy` is the same stay
(likewise for test).

`train.npy`, `test.npy` -- float32, shape `(n_stays, 20, 72)`, no NaN.
  Axis 0 is the stay, axis 1 the channel, axis 2 the hour 0..71 after ICU
  admission (hour h covers ``[intime + h, intime + h + 1)``, intime floored to
  the hour; the same clock for every channel).  Values are in raw clinical
  units, not normalised.  Channels:

      idx  name                  content
      ---  --------------------  ------------------------------------------------
       0   heart_rate            beats/min
       1   heart_rate_mask       1 = imputed, 0 = measured
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
      16   icd                   NOT a time series: slot j is 1 when the subject has
                                 ever been billed `channels.json["icd_codes"][j]`,
                                 the j-th most frequent ICD code in MIMIC-IV
      17   antibiotic            NOT a time series: slot j is 1 when the subject has
                                 ever been prescribed `channels.json["antibiotics"][j]`,
                                 the j-th antibiotic by number of subjects
      18   age                   patients.anchor_age, repeated in all 72 slots
      19   mortality             1 when the subject died in ANY hospital admission
                                 (not necessarily this one), repeated in all 72 slots

  Value channels (even indices 0..14): a measured slot holds the mean of the
  readings in that hour.  A missing slot holds the last measured value of the
  same channel (LOCF); slots before the channel's first reading, and every slot
  of a channel never measured in the window, hold 0.  With `--impute zero`
  every missing slot holds 0 instead.  The mask channel right after each value
  channel (odd indices 1..15) tells the two apart, so real measurements are
  ``values[mask == 0]``.  Labs are drawn a few times a day, so their masks are
  mostly 1; vitals are mostly 0.

`stays_train.npy`, `stays_test.npy` -- int64, shape `(n_stays,)`: the MIMIC-IV
  `stay_id` of each row.

`subjects_train.npy`, `subjects_test.npy` -- int64, shape `(n_stays,)`: the
  MIMIC-IV `subject_id` of each row.  A subject with several stays repeats, but
  never appears in both splits.  About 75% of subjects are in train, drawn
  separately within mortality = 0 and mortality = 1.

`channels.json` -- run metadata: `channels` (the 20 names above, in order),
  `icd_codes` and `antibiotics` (the 72 labels behind channels 16 and 17),
  `n_train` / `n_test` (stays), `n_subjects_train` / `n_subjects_test`,
  `n_steps`, `granularity`, `time_axis`, `split`, `imputation`,
  `min_measurements_per_vital` and `dev`.

`annotations.json` -- per-stay labels that are NOT in the tensors, for
  conditioning or evaluation.  `stays` maps each stay_id (as a string) to a
  record holding: split, subject_id, hadm_id, demographics (gender, age, race,
  insurance, marital_status), admission details (admission_type /
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
    train = np.load(f"{out}/train.npy")                   # (n, 20, 72)
    meta = json.load(open(f"{out}/channels.json"))
    hr, hr_mask = train[:, 0], train[:, 1]                # (n, 72) each
    measured_hr = hr[hr_mask == 0]

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
import shutil
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Sequence

import narwhals as nw
import numpy as np
import polars as pl
from pyhealth.datasets import MIMIC4EHRDataset
from pyhealth.tasks import BaseTask

from mimic import (
    _ANTIBIOTIC_PATTERN,
    _EXCLUDED_ROUTES,
    LAB_ITEMS,
    N_STEPS,
    SIGNALS,
    VITAL_ITEMS,
    VITALS,
    channel_names,
    load_annotations,
)

LOG = logging.getLogger("mimic_pyhealth")

DEFAULT_EHR_ROOT = Path("/path/to/physionet.org/files/mimiciv/3.1")
DEFAULT_OUT_DIR = Path("/path/to/processed_mimic_pyhealth")

CONFIG_PATH = Path(__file__).with_name("mimic_pyhealth.yaml")
TABLES = ["diagnoses_icd", "prescriptions", "labevents", "chartevents"]

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


class TimeDiffHourlyMIMIC4(BaseTask):
    """One sample per ICU stay: the 72-hour signal grid, its mask and the statics.

    ``signals`` goes through PyHealth's ``TimeseriesProcessor`` and comes out
    imputed; ``mask`` is 1 where that slot had no measurement.
    """

    task_name: str = "TimeDiffHourlyMIMIC4"

    def __init__(
        self,
        top_icd: Sequence[str],
        top_antibiotics: Sequence[str],
        min_measurements: int = 70,
        impute_strategy: str = "forward_fill",
    ):
        self.top_icd = list(top_icd)
        self.top_antibiotics = list(top_antibiotics)
        self.min_measurements = min_measurements
        self.input_schema: Dict[str, Any] = {
            "signals": (
                "timeseries",
                {"sampling_rate": timedelta(hours=1), "impute_strategy": impute_strategy},
            ),
            "mask": "tensor",
            "icd": "tensor",
            "antibiotic": "tensor",
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
        diagnoses = patient.get_events(event_type="diagnoses_icd", return_df=True)
        icd = _top_indicator(set(diagnoses["diagnoses_icd/icd_code"].drop_nulls()), self.top_icd)
        drugs = patient.get_events(event_type="prescriptions", return_df=True)
        abx = _top_indicator(set(drugs["prescriptions/drug"].drop_nulls()), self.top_antibiotics)

        demographics = patient.get_events(event_type="patients")
        age = demographics[0].attr_dict.get("anchor_age") if demographics else None
        age = float(age) if age is not None else 0.0
        admissions = patient.get_events(event_type="admissions", return_df=True)
        # subject-level, as in the reference: died in any admission
        mortality = float(
            (admissions["admissions/hospital_expire_flag"].cast(pl.Float64, strict=False) == 1).any()
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
                    "mask": np.isnan(grid).astype(np.float32),
                    "icd": icd,
                    "antibiotic": abx,
                    "age": age,
                    "mortality": mortality,
                }
            )
        return samples


def stack_samples(samples) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`(n_stays, 20, 72)` tensor in `channel_names()` order, rows sorted by stay_id,
    with the stay_id and subject_id of each row."""
    rows, stays, subjects = [], [], []
    for i in range(len(samples)):
        s = samples[i]
        values = s["signals"].numpy().T  # (8, 72)
        mask = s["mask"].numpy().T
        channels = [c for k in range(len(SIGNALS)) for c in (values[k], mask[k])]
        channels += [
            s["icd"].numpy(),
            s["antibiotic"].numpy(),
            np.full(N_STEPS, float(s["age"])),
            np.full(N_STEPS, float(s["mortality"])),
        ]
        rows.append(np.stack(channels).astype(np.float32))
        stays.append(int(s["stay_id"]))
        subjects.append(int(s["patient_id"]))
    order = np.argsort(stays, kind="stable")
    return (
        np.stack(rows)[order],
        np.asarray(stays, dtype=np.int64)[order],
        np.asarray(subjects, dtype=np.int64)[order],
    )


def split_by_subject(
    tensor: np.ndarray,
    stay_ids: np.ndarray,
    subject_ids: np.ndarray,
    seed: int = 2023,
    train_frac: float = 0.75,
) -> dict[str, np.ndarray]:
    """75/25 split with all of a subject's stays on the same side, stratified on mortality.

    The mortality channel is subject-level, so stratifying subjects also
    stratifies stays; the stay ratio lands near 75/25 rather than exactly on it.
    """
    rng = np.random.default_rng(seed)
    subjects, first = np.unique(subject_ids, return_index=True)
    labels = tensor[first, -1, 0]
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
        top_icd, top_abx = rank_top_codes(dataset.global_event_df)
        task = TimeDiffHourlyMIMIC4(
            top_icd, top_abx, min_measurements=min_measurements, impute_strategy=impute_strategy
        )
        samples = dataset.set_task(task, num_workers=num_workers)
        if len(samples) == 0:
            raise RuntimeError("no ICU stay met the vital-sign count and 72-hour length criteria")
        tensor, stays, subjects = stack_samples(samples)
    finally:
        if samples is not None:
            samples.close()
        shutil.rmtree(dataset.cache_dir, ignore_errors=True)
        LOG.info("removed PyHealth cache %s", dataset.cache_dir)

    LOG.info("tensor: %s", tensor.shape)
    LOG.info("mortality rate %.3f", tensor[:, -1, 0].mean())

    parts = split_by_subject(tensor, stays, subjects, seed=seed)
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
                "source": "MIMIC-IV CSV via PyHealth, preprocessing of mimic.py",
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
                    " mask channels are 1 where a value was imputed"
                ),
                "dev": dev,
                "icd_codes": top_icd,
                "antibiotics": top_abx,
            },
            indent=2,
        )
    )
    written["channels"] = meta

    annotations = out_dir / "annotations.json"
    annotations.write_text(json.dumps(load_annotations(ehr_root, parts), indent=2))
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
