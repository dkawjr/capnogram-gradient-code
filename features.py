"""Time-capnogram descriptors and descriptive plateau bowing.

Detection is smoothed; measurements are made on raw samples. The plateau
segment is fixed by timing, not by separately detecting physiological phase III.
"""

import math
import warnings

import numpy as np
from scipy.signal import savgol_filter as _savgol_filter

FS = 62.5
CO2_THRESH = 5.0
MIN_EXP_DUR = 0.5
MIN_BREATH_PERIOD = 1.5
MIN_GOOD_BREATHS = 3
MIN_PLATEAU_PTS = 4

PER_BREATH_FEATURES = [
    "phase2_slope",
    "phase3_slope",
    "phase3_slope_norm",
    "alpha_angle",
    "beta_angle",
    "etco2_wave",
    "insp_baseline",
    "plateau_curvature",
    "exp_time_frac",
    "rise_time",
    "area_ratio",
    "breath_period",
]
FEATURE_COLUMNS = [
    f"{feature}_{summary}"
    for feature in PER_BREATH_FEATURES
    for summary in ("med", "iqr")
]


def plateau_bowing(t, co2, start, end):
    """Fraction of plateau samples > endpoint reference + 1e-6 mmHg.

    The sample-index reference and numerical tolerance match the study.
    This is a fraction of samples, not an area or a measure of bow height.
    """
    segment_t = t[start : end + 1]
    segment_c = co2[start : end + 1]
    if segment_t.size < MIN_PLATEAU_PTS + 2:
        return np.nan
    peak = int(np.argmax(segment_c))
    if peak < 2:
        peak = max(2, int(0.5 * (segment_c.size - 1)))
    exp_t, exp_c = segment_t[: peak + 1], segment_c[: peak + 1]
    duration = exp_t[-1] - exp_t[0]
    if duration <= 0 or exp_t.size < MIN_PLATEAU_PTS + 2:
        return np.nan
    plateau = exp_c[exp_t >= exp_t[0] + 0.45 * duration]
    if plateau.size < MIN_PLATEAU_PTS:
        return np.nan
    reference = np.linspace(plateau[0], plateau[-1], plateau.size)
    return float(np.mean(plateau > reference + 1e-6))


def extract_bowing(t, co2):
    """Return one bowing value per accepted breath; never a model input."""
    values = []
    runs = segment_breaths(t, co2)
    for index, (start, end) in enumerate(runs):
        next_start = runs[index + 1][0] if index + 1 < len(runs) else None
        measured = _breath_features(t, co2, start, end, next_start)
        if np.isfinite(measured["phase3_slope"]) and np.isfinite(
            measured["etco2_wave"]
        ):
            values.append(plateau_bowing(t, co2, start, end))
    return np.asarray(values, dtype=float)


def _linfit_slope(x, y):
    """Fit a line to raw samples; return slope and intercept."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if x.size < 2 or np.ptp(x) == 0:
        return (np.nan, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            b, a = np.polyfit(x, y, 1)
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            return (np.nan, np.nan)
    return (float(b), float(a))


def _angle_between_slopes(m_lo, m_hi):
    """Slope-based alpha angle in degrees, with time in s and CO2 in mmHg."""
    if not (np.isfinite(m_lo) and np.isfinite(m_hi)):
        return np.nan
    theta2 = math.degrees(math.atan(m_lo)) if np.isfinite(m_lo) else 90.0
    theta3 = math.degrees(math.atan(m_hi))
    alpha = 180.0 - (theta2 - theta3)
    return float(alpha)


def _smooth(co2):
    """Smooth for breath detection only; descriptor measurements use raw samples."""
    n = co2.size
    if n < 7:
        return co2
    win = min(11, n if n % 2 == 1 else n - 1)
    return _savgol_filter(co2, win, 2)


def segment_breaths(t, co2, fs=FS):
    """Find above-5-mmHg runs lasting >=0.5 s, with onset spacing >=1.5 s."""
    t = np.asarray(t, float)
    co2 = np.asarray(co2, float)
    n = co2.size
    if n < 8:
        return []
    sm = _smooth(co2)
    above = sm > CO2_THRESH
    runs = []
    i = 0
    while i < n:
        if above[i]:
            j = i
            while j + 1 < n and above[j + 1]:
                j += 1
            runs.append((i, j))
            i = j + 1
        else:
            i += 1
    good = []
    last_start_t = -np.inf
    for s, e in runs:
        dur = t[e] - t[s]
        if dur < MIN_EXP_DUR:
            continue
        if t[s] - last_start_t < MIN_BREATH_PERIOD:
            continue
        good.append((s, e))
        last_start_t = t[s]
    return good


def _breath_features(t, co2, exp_s, exp_e, next_exp_s=None):
    """Measure one detected breath. Plateau: final 55% of onset-to-peak time."""
    feats = {k: np.nan for k in PER_BREATH_FEATURES}
    seg_t = t[exp_s : exp_e + 1]
    seg_c = co2[exp_s : exp_e + 1]
    if seg_t.size < MIN_PLATEAU_PTS + 2:
        return feats
    run_dur = seg_t[-1] - seg_t[0]
    if run_dur <= 0:
        return feats
    peak_i = int(np.argmax(seg_c))
    # Preserve the study's midpoint fallback for an early maximum.
    if peak_i < 2:
        peak_i = max(2, int(0.5 * (seg_c.size - 1)))
    etco2_wave = float(seg_c[peak_i])
    feats["etco2_wave"] = etco2_wave
    exp_t = seg_t[: peak_i + 1]
    exp_c = seg_c[: peak_i + 1]
    exp_dur = exp_t[-1] - exp_t[0]
    if exp_dur <= 0 or exp_t.size < MIN_PLATEAU_PTS + 2:
        return feats
    lo_lvl = 0.1 * etco2_wave
    hi_lvl70 = 0.7 * etco2_wave
    hi_lvl90 = 0.9 * etco2_wave
    rise_mask = (exp_c >= lo_lvl) & (exp_c <= hi_lvl70)
    if rise_mask.sum() >= 2:
        m2, _ = _linfit_slope(exp_t[rise_mask], exp_c[rise_mask])
    else:
        early = exp_t <= exp_t[0] + 0.3 * exp_dur
        m2, _ = (
            _linfit_slope(exp_t[early], exp_c[early])
            if early.sum() >= 2
            else (np.nan, np.nan)
        )
    feats["phase2_slope"] = m2
    idx_lo = np.argmax(exp_c >= lo_lvl)
    idx_hi = np.argmax(exp_c >= hi_lvl90)
    if idx_hi > idx_lo:
        feats["rise_time"] = float(exp_t[idx_hi] - exp_t[idx_lo])
    plateau_mask = exp_t >= exp_t[0] + 0.45 * exp_dur
    # Timing-based analysis segment; not a separately detected phase III onset.
    if plateau_mask.sum() >= MIN_PLATEAU_PTS:
        pt = exp_t[plateau_mask]
        pc = exp_c[plateau_mask]
        m3, b3 = _linfit_slope(pt, pc)
        feats["phase3_slope"] = m3
        if np.isfinite(etco2_wave) and etco2_wave > 1e-06 and np.isfinite(m3):
            feats["phase3_slope_norm"] = m3 / etco2_wave
        if pt.size >= 5 and np.ptp(pt) > 0:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    c2 = np.polyfit(pt - pt[0], pc, 2)[0]
                    feats["plateau_curvature"] = float(2.0 * c2)
                except (np.linalg.LinAlgError, ValueError, FloatingPointError):
                    feats["plateau_curvature"] = np.nan
    else:
        m3 = np.nan
    feats["alpha_angle"] = _angle_between_slopes(
        feats["phase2_slope"], feats["phase3_slope"]
    )
    abs_peak = exp_s + peak_i
    if next_exp_s is not None and next_exp_s > abs_peak:
        insp_t = t[abs_peak : next_exp_s + 1]
        insp_c = co2[abs_peak : next_exp_s + 1]
        if insp_c.size >= 2:
            feats["insp_baseline"] = float(np.min(insp_c))
            insp_dur = insp_t[-1] - insp_t[0]
            if insp_dur > 0:
                drop_mask = insp_t <= insp_t[0] + 0.5 * insp_dur
                if drop_mask.sum() >= 2:
                    m_down, _ = _linfit_slope(insp_t[drop_mask], insp_c[drop_mask])
                    if np.isfinite(m3) and np.isfinite(m_down):
                        th3 = math.degrees(math.atan(m3))
                        thd = math.degrees(math.atan(m_down))
                        # Historical column name; this is a slope-derived beta index.
                        feats["beta_angle"] = float(90.0 + (th3 - thd))
        feats["breath_period"] = float(t[next_exp_s] - t[exp_s])
        if feats["breath_period"] > 0:
            feats["exp_time_frac"] = float(exp_dur / feats["breath_period"])
    total_area = np.trapezoid(exp_c, exp_t)
    if total_area > 1e-06:
        pmask = exp_t >= exp_t[0] + 0.45 * exp_dur
        if pmask.sum() >= 2:
            plat_area = np.trapezoid(exp_c[pmask], exp_t[pmask])
            feats["area_ratio"] = float(plat_area / total_area)
    return feats


def _iqr(a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    if a.size < 2:
        return np.nan
    return float(np.percentile(a, 75) - np.percentile(a, 25))


def extract_window_features(t, co2, fs=FS):
    """Aggregate accepted breaths into 24 features; require at least three."""
    out = {c: np.nan for c in FEATURE_COLUMNS}
    t = np.asarray(t, float)
    co2 = np.asarray(co2, float)
    if t.size < 8 or not np.any(np.isfinite(co2)):
        return (out, 0, 0)
    runs = segment_breaths(t, co2, fs=fs)
    if len(runs) == 0:
        return (out, 0, 0)
    per_breath = []
    for bi, (s, e) in enumerate(runs):
        next_s = runs[bi + 1][0] if bi + 1 < len(runs) else None
        bf = _breath_features(t, co2, s, e, next_exp_s=next_s)
        if np.isfinite(bf["phase3_slope"]) and np.isfinite(bf["etco2_wave"]):
            per_breath.append(bf)
    n_breaths = len(per_breath)
    if n_breaths < MIN_GOOD_BREATHS:
        return (out, n_breaths, 0)
    for f in PER_BREATH_FEATURES:
        vals = np.array([b[f] for b in per_breath], float)
        vals = vals[np.isfinite(vals)]
        if vals.size >= 1:
            out[f + "_med"] = float(np.median(vals))
        if vals.size >= 2:
            out[f + "_iqr"] = _iqr(vals)
    return (out, n_breaths, 1)
