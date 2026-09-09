# Verification

Checked on 9 September 2026 in a clean Windows environment using Python
3.12.13 and the dependencies pinned in requirements.txt.

- Re-extracted all 8,470 source-cohort waveform windows and their preceding
  monitor EtCO2 values from local study files.
- Reproduced the 14 exclusions and the final 8,456 recordings, 3,273 operations,
  3,168 individuals, and 122,941 accepted breaths.
- Matched all 202,944 morphology-feature entries and their missing-value pattern
  exactly after the documented feature CSV parsing step. Breath counts and
  numeric EtCO2 values also matched exactly.
- Refitted the three regression models across the five saved patient folds.
  Every prediction agreed with the frozen study manifest within 1e-12 mmHg.
- Reproduced the frozen study metrics and confidence intervals within a
  1e-10 tolerance, including the 1,000-resample paired comparisons
  and the separate plateau-bowing correlation interval.
- Passed 15 synthetic tests, Python compilation, and standard lint checks.

These checks establish reproduction of the included analyses from the available
local study inputs. They do not establish external clinical validity,
cross-platform bitwise identity, or reconstruction of the original cohort
selection from a fresh full-database download. Database acquisition and the
source-cohort input files remain separate from this code-only release.
