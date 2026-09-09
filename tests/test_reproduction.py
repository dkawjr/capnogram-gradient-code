"""Synthetic checks; no study data or network access required."""

import tempfile
import unittest
from pathlib import Path

import analysis
import numpy as np
import pandas as pd
from features import (
    FEATURE_COLUMNS,
    extract_bowing,
    extract_window_features,
    plateau_bowing,
)
from prepare_data import preceding_median, read_numeric, read_table, read_waveform
from sklearn.impute import SimpleImputer


def waveform():
    time = np.arange(-60, 0.001, 0.016)
    phase = (time + 60) % 4
    co2 = np.interp(phase, [0, 0.3, 0.6, 1.5, 2.5, 2.9, 4], [0, 0, 30, 33, 34, 0, 0])
    return time, co2


def example_table():
    rows = []
    for index in range(10):
        rows.append(
            dict(
                sample_id=str(index),
                caseid=str(index),
                subjectid=str(index),
                fold=index % 5 + 1,
                paco2=40 + index,
                etco2_predraw=34,
                n_breaths=14,
                **{key: float(index) for key in FEATURE_COLUMNS},
            )
        )
    return pd.DataFrame(rows)


class ReproductionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def save(self, table, name="data.csv"):
        path = self.root / name
        table.to_csv(path, index=False)
        return path

    def test_feature_count_and_order(self):
        self.assertEqual(len(FEATURE_COLUMNS), 24)
        self.assertEqual(
            analysis.MODELS["Full morphology + EtCO2"],
            ["etco2_predraw"] + FEATURE_COLUMNS,
        )
        self.assertNotIn("plateau_bowing", sum(analysis.MODELS.values(), []))
        self.assertNotIn("paco2", sum(analysis.MODELS.values(), []))

    def test_breath_acceptance_matches_bowing(self):
        time, co2 = waveform()
        features, count, usable = extract_window_features(time, co2)
        bow = extract_bowing(time, co2)
        self.assertEqual(usable, 1)
        self.assertGreaterEqual(count, 3)
        self.assertEqual(count, len(bow))
        self.assertTrue(np.isfinite(list(features.values())).all())
        self.assertTrue(((bow >= 0) & (bow <= 1)).all())

    def test_minimum_breath_gate(self):
        time, co2 = waveform()
        features, count, usable = extract_window_features(time[:250], co2[:250])
        self.assertLess(count, 3)
        self.assertEqual(usable, 0)
        self.assertTrue(np.isnan(list(features.values())).all())

    def test_bowing_is_fraction_not_area(self):
        time = np.arange(21, dtype=float)
        straight = 10 + time
        self.assertEqual(plateau_bowing(time, straight, 0, 20), 0)
        bowed = straight.copy()
        bowed[10:20] += 0.2
        expected = 10 / 12  # segment indices 9..20, ten points strictly above
        self.assertAlmostEqual(plateau_bowing(time, bowed, 0, 20), expected)
        bowed[10:20] += 0.2
        self.assertAlmostEqual(plateau_bowing(time, bowed, 0, 20), expected)

    def test_future_waveform_is_excluded(self):
        path = self.save(
            pd.DataFrame(
                {"t": [-61, -60, -0.016, 0, 0.016], "co2": [50, 2, 30, 31, 120]}
            )
        )
        time, co2 = read_waveform(path)
        np.testing.assert_array_equal(time, [-60, -0.016, 0])
        np.testing.assert_array_equal(co2, [2, 30, 31])

    def test_numeric_window_boundaries_and_validity(self):
        values = np.array([[69.9, 100], [70, 30], [80, 32], [100, 34], [100.1, 120]])
        self.assertEqual(preceding_median(values, 100), 32)
        self.assertTrue(np.isnan(preceding_median(values, 0)))
        path = self.save(
            pd.DataFrame({"time": [1, 2, 3, 4], "value": [0, 5, 120, 121]})
        )
        np.testing.assert_array_equal(read_numeric(path)[:, 1], [5, 120])

    def test_duplicate_join_keys_rejected(self):
        path = self.save(pd.DataFrame({"sample_id": ["a", "a"]}))
        with self.assertRaises(ValueError):
            read_table(path, ["sample_id"], "sample_id")

    def test_patient_overlap_rejected(self):
        data = example_table()
        data.loc[1, "subjectid"] = data.loc[0, "subjectid"]
        with self.assertRaises(ValueError):
            analysis.load_data(self.save(data))

    def test_changed_target_rejected(self):
        data = example_table()
        data["g"] = 99
        with self.assertRaises(ValueError):
            analysis.load_data(self.save(data))

    def test_missing_breath_count_rejected(self):
        data = example_table()
        data.loc[1, "n_breaths"] = np.nan
        with self.assertRaises(ValueError):
            analysis.load_data(self.save(data))

    def test_training_only_imputation(self):
        imputer = SimpleImputer(strategy="median").fit([[1], [3], [np.nan]])
        np.testing.assert_array_equal(
            imputer.transform([[1000], [np.nan]]).ravel(), [1000, 2]
        )

    def test_bootstrap_keeps_patient_records_together(self):
        groups = np.array(["a", "a", "b", "c", "c", "c"])
        first = list(analysis.subject_bootstrap_indices(groups, 12))[:10]
        again = list(analysis.subject_bootstrap_indices(groups, 12))[:10]
        for index, repeat in zip(first, again):
            np.testing.assert_array_equal(index, repeat)
            counts = np.bincount(index, minlength=6)
            self.assertEqual(counts[0], counts[1])
            self.assertEqual(counts[3], counts[4])
            self.assertEqual(counts[4], counts[5])

    def test_negative_gradients_remain_controls(self):
        gradient = np.array([-4, 0, 5, 6, 11, 16, 21], dtype=float)
        metrics = analysis.point_metrics(
            gradient, gradient + 30, np.full(7, 30), gradient
        )
        self.assertEqual(metrics["thresholds"]["g>5"]["positive_n"], 4)
        self.assertEqual(metrics["thresholds"]["g>5"]["auroc"], 1)

    def test_expected_check_detects_changes(self):
        with self.assertRaises(ValueError):
            analysis.check_expected({"value": 0.7}, {"value": 0.8})

    def test_prepared_csv_preserves_feature_floats(self):
        data = example_table()
        value = float("0.12345678912345678")
        data[FEATURE_COLUMNS[0]] = value
        loaded = analysis.load_data(self.save(data))
        self.assertEqual(loaded[FEATURE_COLUMNS[0]].iloc[0], value)

    def test_fixed_fold_prediction_reproducible(self):
        data = analysis.load_data(self.save(example_table()))
        from threadpoolctl import threadpool_limits

        with threadpool_limits(limits=1):
            first = analysis.fixed_fold_oof(data, ["etco2_predraw"])
            second = analysis.fixed_fold_oof(data, ["etco2_predraw"])
        np.testing.assert_array_equal(first, second)


if __name__ == "__main__":
    unittest.main()
