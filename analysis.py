"""Primary regression, patient-separated validation, and patient bootstrap intervals."""

import argparse
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from features import FEATURE_COLUMNS
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)
from threadpoolctl import threadpool_limits

CUTS = (0, 5, 10, 15, 20)
NBOOT = 1000
MODELS = {
    "EtCO2 alone": ["etco2_predraw"],
    "Morphology alone": FEATURE_COLUMNS,
    "Full morphology + EtCO2": ["etco2_predraw"] + FEATURE_COLUMNS,
}


def load_data(path):
    """Validate the prepared table; never choose predictors from arbitrary columns."""
    # Preserve the prepared table's floats across CSV serialization.
    data = pd.read_csv(
        path,
        float_precision="round_trip",
        dtype={key: str for key in ("sample_id", "caseid", "subjectid")},
    )
    required = [
        "sample_id",
        "caseid",
        "subjectid",
        "fold",
        "paco2",
        "etco2_predraw",
        "n_breaths",
    ] + FEATURE_COLUMNS
    missing = set(required) - set(data)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if data.sample_id.duplicated().any() or data[required[:7]].isna().any().any():
        raise ValueError(
            "Duplicate samples or missing identifiers, folds, or target components"
        )
    if not np.isfinite(
        data[["paco2", "etco2_predraw", "fold", "n_breaths"]].to_numpy(float)
    ).all():
        raise ValueError("Non-finite target components, folds, or breath counts")
    if (
        set(data.fold) != {1, 2, 3, 4, 5}
        or data.groupby("subjectid").fold.nunique().max() != 1
    ):
        raise ValueError(
            "Require five frozen folds numbered 1-5 with each patient in one fold"
        )
    if data.groupby("caseid").subjectid.nunique().max() != 1:
        raise ValueError("An operation maps to more than one patient")
    if (data.n_breaths < 3).any() or (data.n_breaths % 1 != 0).any():
        raise ValueError(
            "Every included recording must contain at least three accepted breaths"
        )
    if not data.etco2_predraw.between(5, 120).all():
        raise ValueError("EtCO2 outside the numeric-track eligibility range")
    features = (
        data[FEATURE_COLUMNS].apply(pd.to_numeric, errors="raise").to_numpy(float)
    )
    if np.isinf(features).any():
        raise ValueError(
            "Infinite morphology input; use NaN for unavailable descriptors"
        )
    gradient = data.paco2 - data.etco2_predraw
    if "g" in data and not np.allclose(data.g, gradient, atol=1e-12, rtol=0):
        raise ValueError("Saved gradient does not equal PaCO2 minus the input EtCO2")
    data["g"] = gradient
    return data.sort_values("sample_id", kind="stable").reset_index(drop=True)


def bowing_summary(data):
    """Descriptive association across recordings, with patient bootstrap intervals."""
    if "plateau_bowing" not in data:
        return None
    keep = np.isfinite(data.plateau_bowing)
    subset = data.loc[keep]
    if not subset.plateau_bowing.between(0, 1).all():
        raise ValueError("Plateau bowing must lie between zero and one")
    x, y = subset.g.to_numpy(float), subset.plateau_bowing.to_numpy(float)
    rho = float(spearmanr(x, y).statistic)
    draws = [
        float(spearmanr(x[i], y[i]).statistic)
        for i in subject_bootstrap_indices(subset.subjectid.to_numpy(str), 20260819)
    ]
    groups = {}
    for label, mask in [
        ("<=5", x <= 5),
        (">5 to 10", (x > 5) & (x <= 10)),
        (">10 to 15", (x > 10) & (x <= 15)),
        (">15", x > 15),
    ]:
        groups[label] = dict(
            recordings=int(mask.sum()), median=float(np.median(y[mask]))
        )
    return dict(
        recordings=len(subset), rho=rho, rho_ci=percentile_ci(draws), groups=groups
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = load_data(args.data)
    predictions = {}
    with threadpool_limits(limits=1):
        for name, columns in MODELS.items():
            print(f"Fitting {name}", flush=True)
            predictions[name] = fixed_fold_oof(data, columns)
    y, pa, et = (
        data[column].to_numpy(float) for column in ("g", "paco2", "etco2_predraw")
    )
    result = {
        "cohort": dict(
            recordings=len(data),
            operations=data.caseid.nunique(),
            individuals=data.subjectid.nunique(),
            accepted_breaths=int(data.n_breaths.sum()),
        ),
        "versions": dict(
            python=platform.python_version(),
            numpy=np.__version__,
            pandas=pd.__version__,
            scipy=scipy.__version__,
            scikit_learn=sklearn.__version__,
        ),
        "points": {
            name: point_metrics(y, pa, et, values)
            for name, values in predictions.items()
        },
    }
    print("Calculating 1,000 patient bootstrap intervals", flush=True)
    result["bootstrap"] = bootstrap_all(
        y, pa, et, data.subjectid.to_numpy(str), predictions
    )
    result["bowing"] = bowing_summary(data)
    args.output.mkdir(parents=True, exist_ok=True)
    output = data[
        ["sample_id", "caseid", "subjectid", "fold", "paco2", "etco2_predraw", "g"]
    ].copy()
    for name, values in predictions.items():
        output[name] = values
    output.to_csv(args.output / "predictions.csv", index=False)
    (args.output / "results.json").write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(f"Results saved to {args.output}", flush=True)


def fixed_fold_oof(d: pd.DataFrame, columns: list[str]) -> np.ndarray:
    X = d[columns].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    y = d["g"].to_numpy(float)
    folds = d["fold"].to_numpy(int)
    groups = d["subjectid"].astype(str).to_numpy()
    pred = np.full(len(d), np.nan)
    for fold in sorted(np.unique(folds)):
        te = folds == fold
        tr = ~te
        if not set(groups[tr]).isdisjoint(set(groups[te])):
            raise RuntimeError(f"Subject leakage in fold {fold}")
        if np.isnan(X[tr]).all(axis=0).any():
            raise ValueError(f"A predictor is entirely missing in training fold {fold}")
        imp = SimpleImputer(strategy="median").fit(X[tr])
        model = HistGradientBoostingRegressor(random_state=0).fit(
            imp.transform(X[tr]), y[tr]
        )
        pred[te] = model.predict(imp.transform(X[te]))
    if not np.isfinite(pred).all():
        raise RuntimeError(f"Non-finite prediction for {columns}")
    return pred


def subject_bootstrap_indices(subjects: np.ndarray, seed: int):
    rng = np.random.default_rng(seed)
    unique = np.unique(subjects)
    locations = {s: np.flatnonzero(subjects == s) for s in unique}
    for _ in range(NBOOT):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        yield np.concatenate([locations[s] for s in sampled])


def percentile_ci(values) -> list[float]:
    values = np.asarray(values, float)
    return [float(x) for x in np.percentile(values[np.isfinite(values)], [2.5, 97.5])]


def point_metrics(
    y: np.ndarray, pa: np.ndarray, et: np.ndarray, pred: np.ndarray
) -> dict:
    out = {
        "gradient_r2": float(r2_score(y, pred)),
        "derived_paco2_r2": float(r2_score(pa, et + pred)),
        "derived_paco2_mae": float(mean_absolute_error(pa, et + pred)),
        "thresholds": {},
    }
    for cut in CUTS:
        binary = y > cut
        out["thresholds"][f"g>{cut}"] = {
            "positive_n": int(binary.sum()),
            "prevalence": float(binary.mean()),
            "auroc": float(roc_auc_score(binary, pred)),
            "auprc": float(average_precision_score(binary, pred)),
        }
    return out


def bootstrap_all(
    y: np.ndarray,
    pa: np.ndarray,
    et: np.ndarray,
    subjects: np.ndarray,
    predictions: dict[str, np.ndarray],
) -> dict:
    names = list(predictions)
    baseline_name = "EtCO2 alone"
    draws = {
        name: {
            "gradient_r2": [],
            "derived_paco2_r2": [],
            "derived_paco2_mae": [],
            "thresholds": {},
        }
        for name in names
    }
    deltas = {
        name: {
            "gradient_r2": [],
            "derived_paco2_r2": [],
            "mae_reduction": [],
            "thresholds": {},
        }
        for name in names
        if name != baseline_name
    }
    for name in names:
        for cut in CUTS:
            draws[name]["thresholds"][f"g>{cut}"] = {"auroc": [], "auprc": []}
    for name in deltas:
        for cut in CUTS:
            deltas[name]["thresholds"][f"g>{cut}"] = {
                "delta_auroc": [],
                "delta_auprc": [],
            }
    for ix in subject_bootstrap_indices(subjects, 20260811):
        yy, pp, ee = (y[ix], pa[ix], et[ix])
        base = predictions[baseline_name][ix]
        base_r2 = r2_score(yy, base)
        base_pa_r2 = r2_score(pp, ee + base)
        base_mae = mean_absolute_error(pp, ee + base)
        per_name = {}
        for name, pred_all in predictions.items():
            pred = pred_all[ix]
            rr = r2_score(yy, pred)
            prr = r2_score(pp, ee + pred)
            mae = mean_absolute_error(pp, ee + pred)
            draws[name]["gradient_r2"].append(rr)
            draws[name]["derived_paco2_r2"].append(prr)
            draws[name]["derived_paco2_mae"].append(mae)
            per_name[name] = (rr, prr, mae)
            if name != baseline_name:
                deltas[name]["gradient_r2"].append(rr - base_r2)
                deltas[name]["derived_paco2_r2"].append(prr - base_pa_r2)
                deltas[name]["mae_reduction"].append(base_mae - mae)
        for cut in CUTS:
            binary = yy > cut
            if np.unique(binary).size < 2:
                continue
            base_auc = roc_auc_score(binary, base)
            base_ap = average_precision_score(binary, base)
            for name, pred_all in predictions.items():
                pred = pred_all[ix]
                auc = roc_auc_score(binary, pred)
                ap = average_precision_score(binary, pred)
                key = f"g>{cut}"
                draws[name]["thresholds"][key]["auroc"].append(auc)
                draws[name]["thresholds"][key]["auprc"].append(ap)
                if name != baseline_name:
                    deltas[name]["thresholds"][key]["delta_auroc"].append(
                        auc - base_auc
                    )
                    deltas[name]["thresholds"][key]["delta_auprc"].append(ap - base_ap)
    out = {"arms": {}, "differences_vs_etco2": {}}
    for name in names:
        out["arms"][name] = {
            "gradient_r2_ci": percentile_ci(draws[name]["gradient_r2"]),
            "derived_paco2_r2_ci": percentile_ci(draws[name]["derived_paco2_r2"]),
            "derived_paco2_mae_ci": percentile_ci(draws[name]["derived_paco2_mae"]),
            "thresholds": {
                key: {
                    metric + "_ci": percentile_ci(values)
                    for metric, values in block.items()
                }
                for key, block in draws[name]["thresholds"].items()
            },
        }
    for name in deltas:
        out["differences_vs_etco2"][name] = {
            "gradient_r2_difference_ci": percentile_ci(deltas[name]["gradient_r2"]),
            "derived_paco2_r2_difference_ci": percentile_ci(
                deltas[name]["derived_paco2_r2"]
            ),
            "mae_reduction_ci": percentile_ci(deltas[name]["mae_reduction"]),
            "thresholds": {
                key: {
                    metric + "_ci": percentile_ci(values)
                    for metric, values in block.items()
                }
                for key, block in deltas[name]["thresholds"].items()
            },
        }
    return out


if __name__ == "__main__":
    main()
