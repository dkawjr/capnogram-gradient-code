"""Rebuild pre-blood-gas features from local waveform and numeric CSV files."""

import argparse
import csv
import json
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
from features import FEATURE_COLUMNS, extract_bowing, extract_window_features


def read_table(path, required, unique):
    table = pd.read_csv(
        path, dtype={key: str for key in ("sample_id", "caseid", "subjectid")}
    )
    missing = set(required) - set(table)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    if table[unique].isna().any() or table[unique].duplicated().any():
        raise ValueError(f"{path}: {unique} must be present and unique")
    return table


def read_numeric(path):
    """Read explicitly timestamped monitor values, retaining 5-120 mmHg."""
    values = []
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        next(reader, None)
        for row in reader:
            try:
                time, value = float(row[0]), float(row[1])
            except (ValueError, IndexError):
                continue
            if np.isfinite(time) and 5 <= value <= 120:
                values.append((time, value))
    return np.asarray(values, dtype=float).reshape(-1, 2)


def preceding_median(numeric, blood_gas_time):
    """Inclusive 30-second window, with the study's two-decimal rounding."""
    time, value = numeric[:, 0], numeric[:, 1]
    selected = value[(time >= blood_gas_time - 30) & (time <= blood_gas_time)]
    return round(float(np.median(selected)), 2) if selected.size else np.nan


def read_waveform(path):
    """Read relative timestamps and restrict to [-60, 0] seconds."""
    table = pd.read_csv(path, usecols=["t", "co2"])
    time = pd.to_numeric(table.t, errors="coerce").to_numpy(float)
    co2 = pd.to_numeric(table.co2, errors="coerce").to_numpy(float)
    keep = (
        np.isfinite(time)
        & np.isfinite(co2)
        & (time >= -60)
        & (time <= 0)
        & (co2 >= 0)
        & (co2 <= 120)
    )
    time, co2 = time[keep], co2[keep]
    if len(time) and np.any(np.diff(time) <= 0):
        raise ValueError(f"{path}: waveform timestamps must increase strictly")
    return time, co2


def prepare(samples_path, cases_path, folds_path, numeric_map_path, waveform_root):
    samples = read_table(
        samples_path,
        ["sample_id", "caseid", "abg_dt", "paco2", "window_file"],
        "sample_id",
    )
    cases = read_table(cases_path, ["caseid", "subjectid"], "caseid")
    folds = read_table(folds_path, ["subjectid", "fold"], "subjectid")
    if not folds.fold.isin([1, 2, 3, 4, 5]).all():
        raise ValueError("Fold assignments must be integers from 1 to 5")
    numeric_map = read_table(numeric_map_path, ["caseid", "numeric_file"], "caseid")
    data = samples.merge(
        cases[["caseid", "subjectid"]], on="caseid", how="left", validate="many_to_one"
    )
    data = data.merge(
        folds[["subjectid", "fold"]], on="subjectid", how="left", validate="many_to_one"
    )
    data = data.merge(
        numeric_map[["caseid", "numeric_file"]],
        on="caseid",
        how="left",
        validate="many_to_one",
    )
    if data[["subjectid", "fold", "numeric_file"]].isna().any().any():
        raise ValueError("Incomplete patient, fold, or numeric-file mapping")
    output, accounting = [], []
    numeric_cache = {}
    for index, row in enumerate(data.itertuples(index=False), start=1):
        wave_path = Path(str(row.window_file).replace("\\", "/"))
        if not wave_path.is_absolute():
            wave_path = Path(waveform_root) / wave_path
        numeric_path = Path(str(row.numeric_file).replace("\\", "/"))
        if not numeric_path.is_absolute():
            numeric_path = Path(numeric_map_path).resolve().parent / numeric_path
        if row.caseid not in numeric_cache:
            numeric_cache[row.caseid] = read_numeric(numeric_path)
        if not np.isfinite(float(row.abg_dt)):
            raise ValueError(f"{row.sample_id}: missing blood-gas timestamp")
        etco2 = preceding_median(numeric_cache[row.caseid], float(row.abg_dt))
        time, co2 = read_waveform(wave_path)
        features, n_breaths, usable = extract_window_features(time, co2)
        paco2 = float(row.paco2)
        accepted = bool(usable == 1 and np.isfinite(etco2) and np.isfinite(paco2))
        accounting.append(
            dict(
                sample_id=row.sample_id,
                included=accepted,
                insufficient_breaths=not bool(usable),
                missing_etco2=not np.isfinite(etco2),
                missing_paco2=not np.isfinite(paco2),
                n_breaths=n_breaths,
            )
        )
        if accepted:
            bowing = extract_bowing(time, co2)
            finite_bowing = bowing[np.isfinite(bowing)]
            if len(bowing) != n_breaths:
                raise RuntimeError(
                    "Morphology and bowing use different accepted breaths"
                )
            output.append(
                dict(
                    sample_id=row.sample_id,
                    caseid=row.caseid,
                    subjectid=row.subjectid,
                    fold=int(row.fold),
                    paco2=paco2,
                    etco2_predraw=etco2,
                    g=paco2 - etco2,
                    n_breaths=n_breaths,
                    **features,
                    plateau_bowing=float(np.median(finite_bowing))
                    if len(finite_bowing)
                    else np.nan,
                    n_bowing_breaths=len(finite_bowing),
                )
            )
        if index % 500 == 0 or index == len(data):
            print(f"Extracted {index}/{len(data)} recordings", flush=True)
    result = (
        pd.DataFrame(output)
        .sort_values("sample_id", kind="stable")
        .reset_index(drop=True)
    )
    # The study fitted models after exporting and re-reading feature CSVs.
    # Preserve that parser boundary: last-bit changes can alter tree split ties.
    result[FEATURE_COLUMNS] = pd.read_csv(
        StringIO(result[FEATURE_COLUMNS].to_csv(index=False))
    )
    return result, pd.DataFrame(accounting)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("samples", "cases", "folds", "numeric-map", "waveform-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    table, accounting = prepare(
        args.samples, args.cases, args.folds, args.numeric_map, args.waveform_root
    )
    args.output.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output / "analysis_table.csv", index=False)
    accounting.to_csv(args.output / "exclusions.csv", index=False)
    summary = dict(
        source_recordings=len(accounting),
        recordings=len(table),
        operations=table.caseid.nunique(),
        individuals=table.subjectid.nunique(),
        accepted_breaths=int(table.n_breaths.sum()),
        bowing_breaths=int(table.n_bowing_breaths.sum()),
    )
    (args.output / "cohort.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
