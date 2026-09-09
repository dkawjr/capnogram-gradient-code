# Input files

Use the study's selected, de-identified VitalDB records and local waveform and
numeric exports. The software makes no network requests. Obtain database files
through the data source's authorized distribution, not by scraping its API.
No raw recordings, clinical outcomes, or subject-level tables are bundled here.

## Starting from waveform windows

`samples.csv` has one row per selected blood-gas recording:

| Column | Meaning |
|---|---|
| sample_id | Unique, stable recording identifier; preserve as a string |
| caseid | VitalDB operation identifier, matching cases.csv |
| abg_dt | Recorded blood-gas time in seconds relative to case recording start |
| paco2 | PaCO2 from the matched arterial blood gas, in mmHg |
| window_file | Local CSV path, absolute or relative to --waveform-root |

For exact study reproduction, use the 8,470 source-cohort records. This source
selection is an input, not a selection inferred from the new model's results.

Each waveform CSV has columns `t,co2`. Time is in seconds **relative to the
blood-gas timestamp**, and CO2 is in mmHg. Use the original 62.5-Hz waveform
samples and their timestamps, not screenshots or interpolated monitor trends.
The original waveform export omitted invalid values outside 0-120 mmHg and
rounded relative timestamps to four decimal places. The loader removes
non-finite values and uses only -60 <= t <= 0; it never interpolates gaps.
Waveform measurements after detection use the unsmoothed samples.

`cases.csv` must contain unique `caseid` values and their `subjectid` mappings.
Several operations may map to one individual. Extra clinical columns are ignored.

`folds.csv` has one row per `subjectid`, with a `fold` integer from 1 to 5.
Use the study's saved patient assignments. Changing folds changes the analysis;
do not split by operation or recording. All operations from a patient must
remain together. These assignments must cover the source-cohort patients.

`numeric_map.csv` contains unique `caseid` values and a `numeric_file` path for
each. Relative paths are resolved from the directory containing numeric_map.csv.
Each numeric CSV has a header followed by two columns: time in seconds relative
to case start, and the Primus/ETCO2 value in mmHg. Timestamps must be explicit.
The parser retains values from 5 to 120 mmHg, takes the median in the inclusive
30 seconds ending at abg_dt, and rounds it to two decimal places as in the
study. Missing preceding EtCO2 excludes that observation.

Missing files or incomplete mappings stop preparation rather than silently
reducing the cohort. Missing usable measurements within a present file are
recorded in `exclusions.csv`.

## Starting from a prepared table

`analysis_table.csv` requires sample_id, caseid, subjectid, fold, paco2,
etco2_predraw, n_breaths, and all 24 feature columns listed in features.py.
The optional `g` column is checked against paco2 - etco2_predraw. The optional
`plateau_bowing` column enables the descriptive correlation.

Preserve the file's full precision. Missing individual descriptors may be NaN;
the model imputes them using training data only. Missing target components,
infinite features, fewer than three accepted breaths, duplicate sample IDs,
or patient overlap across folds cause an error.

The intended data flow is:

```text
selected blood gases + raw waveform windows + monitor numeric tracks
    -> prepare_data.py -> analysis_table.csv + exclusions.csv
    -> analysis.py -> predictions.csv + results.json
```

The code cannot establish correct source selection or timing for arbitrary
user-supplied tables. For reviewer access, provide the exact selected-record
manifest, mappings, and signal files above, or the exact prepared table with
its extraction provenance. The source files and their access arrangements are
separate from this code-only archive.
