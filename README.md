# Capnogram morphology and the arterial-to-end-tidal CO2 gradient

Minimal reproduction code for the study's pre-blood-gas analysis. It compares
EtCO2 alone, waveform morphology alone, and morphology plus EtCO2. Plateau
bowing is calculated separately as a descriptive waveform measure, not as an
input to these models.

This is a code-only release. Study recordings and patient-level tables are not
included. The required inputs are described in [INPUTS.md](INPUTS.md); those
files must accompany reviewer access or be obtained separately. Editors and
reviewers can request the study-specific input tables and secondary-analysis
code from the corresponding author using the manuscript contact details.
This package does not download the full database or reconstruct the initial source-cohort
selection from all VitalDB operations.

## Install

Use Python 3.12.13. From this directory:

```text
python -m venv .venv
```

Activate with `.venv\Scripts\Activate.ps1` in PowerShell, or
`source .venv/bin/activate` on macOS/Linux. Then run:

```text
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

The tests use synthetic data and require no database access. The study replay
was checked on Windows with the pinned environment. Floating-point behavior
can differ across platforms; do not assume exact cross-platform agreement.

## Reproduce

Starting with local waveform windows, monitor numeric tracks, the selected
blood-gas records, and the patient/fold mappings:

```text
python prepare_data.py --samples data/samples.csv --cases data/cases.csv --folds data/folds.csv --numeric-map data/numeric_map.csv --waveform-root data --output data/prepared
python analysis.py --data data/prepared/analysis_table.csv --output results
```

If the prepared analysis table is supplied directly, only the second command
is needed. Output files contain the exclusion accounting, cohort counts,
individual out-of-fold predictions, aggregate metrics, and 95% confidence
intervals.

## Analysis

- Morphology uses waveform samples from -60 to 0 seconds; numeric EtCO2 is the
  median from -30 to 0 seconds, relative to the recorded blood-gas timestamp.
  Both endpoints are included. No later input data are used.
- The target is the signed gradient, PaCO2 minus that same EtCO2 value.
  PaCO2 is never a predictor. Derived PaCO2 equals EtCO2 plus predicted gradient.
- Each of 12 waveform descriptors is summarized by its median and interquartile
  range, giving 24 morphology inputs. At least three accepted breaths are required.
- Five patient-grouped folds are held fixed. Median imputation is fitted only
  within each training fold. The model is HistGradientBoostingRegressor with
  scikit-learn 1.6.1 defaults, random_state=0, and no hyperparameter tuning.
- One regression model per fold supplies the scores at all five gradient
  thresholds: >0, >5, >10, >15, and >20 mmHg. AUPRC is computed as average
  precision. Higher AUROC, AUPRC, and R2 are better; lower MAE is better.
- Confidence intervals use 1,000 patient bootstrap resamples without refitting,
  with seed 20260811. Each resampled patient retains all their recordings.
  Paired model differences use the same resamples. Bowing correlation uses the
  study's separate bootstrap seed, 20260819.

The source cohort has 8,470 recordings. The preceding-data requirements exclude
14: ten fail the three-breath requirement only, two lack valid EtCO2 only, and
two meet both criteria. The resulting cohort contains 8,456 recordings from
3,273 operations in 3,168 individuals, with 122,941 accepted breaths.

The original feature CSV export/import step is retained. The prepared table is
then read with round-trip float precision. This matters because tiny numeric
changes can alter tree split ties; neither features nor predictions should be
manually rounded before fitting.

## Plateau bowing

For each accepted breath, the measurement begins at the first sample at or
after 45% of the detected expiratory-onset-to-peak interval, retaining its final
55%. This is a fixed timing rule, not independent detection of physiological
phase III. The implementation retains the study's fallback for an early peak.

Bowing is the fraction of segment samples above the straight endpoint
reference, with a 1e-6-mmHg numerical tolerance. It is not the shaded area or
the height above the line. Values are summarized by the median within each
recording. The waveform peak is distinct from the separately recorded monitor
EtCO2 median used in the model and target.

## Scope

Included: waveform extraction, bowing, the three main regression models,
discrimination, continuous-gradient prediction, derived PaCO2 error, and
patient-bootstrap uncertainty. Comparator models, SHAP, calibration,
decision-curve analysis, subgroup analyses, longitudinal analyses, and figure
production are not included. This is not a complete reproduction of every
manuscript result and is not software for clinical use.

The recorded timestamp does not establish the exact sampling time or a clinical
warning lead time. Keep the manuscript's methods, limitations, and disclosures
with any interpretation of these results.

The software retains the existing MIT license. Data remain subject to their
source license and attribution requirements; the software license does not
license the data.
