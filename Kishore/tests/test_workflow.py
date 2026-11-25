from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

import morphology
from data_io import CLASSES, ROOT, digest, extract_table, groups_for
from experiment import fit_experiment, split_plan, validate_predictions, verify_plan
from main import smoke_data
from morphology import FEATURES, Settings, extract, gini, m20
from predictor import load_model, predict_image


def profile(sx=8, sy=8):
    y, x = np.indices((128, 128))
    return .08 + .8 * np.exp(-.5 * (((x - 63.5) / sx) ** 2 + ((y - 63.5) / sy) ** 2))


class FeatureTests(unittest.TestCase):
    def test_blank_malformed_and_low_signal(self):
        self.assertEqual(extract(np.ones((128, 128)))[1], ["blank"])
        for image in (np.ones(128), np.zeros((40, 40, 4)),
                      np.full((40, 40), np.nan), np.full((40, 40), 256)):
            self.assertEqual(extract(image)[1], ["malformed"])
        self.assertIn("low_signal", extract(profile() * .001)[1])

    def test_deterministic_and_reasonable_concentration(self):
        values, flags, _ = extract(profile())
        self.assertEqual((values, flags), extract(profile())[:2])
        self.assertTrue(np.isfinite(list(values.values())).all())
        self.assertGreater(values["concentration"], 1.8)
        self.assertLess(values["concentration"], 2.4)

    def test_asymmetry(self):
        symmetric = profile()
        asymmetric = symmetric.copy()
        asymmetric[60:68, 76:83] += .25
        self.assertLess(extract(symmetric)[0]["asymmetry"], .01)
        self.assertGreater(extract(asymmetric)[0]["asymmetry"], .04)

    def test_ellipticity(self):
        self.assertLess(extract(profile())[0]["ellipticity"], .025)
        self.assertGreater(extract(profile(13, 4))[0]["ellipticity"], .6)

    def test_gini_m20(self):
        self.assertAlmostEqual(gini(np.ones(8)), 0)
        self.assertAlmostEqual(gini(np.array([0, 0, 2])), 1)
        self.assertTrue(np.isnan(gini(np.zeros(4))))
        self.assertAlmostEqual(m20(np.array([5, 3, 2]), np.array([1, 2, 3])),
                               np.log10(5 / 17))
        self.assertTrue(np.isnan(m20(np.ones(3), np.zeros(3))))
        self.assertTrue(np.isnan(m20(np.array([9, 1]), np.array([0, 1]))))

    def test_shared_settings_and_model_round_trip(self):
        settings = Settings(size=96)
        images = [profile(5 + i, 6) for i in range(10)]
        frame = pd.DataFrame([extract(im, settings)[0] for im in images], columns=FEATURES)
        pipeline = Pipeline([("forest", RandomForestClassifier(n_estimators=10, random_state=42))])
        pipeline.fit(frame, np.arange(10))
        bundle = {"pipeline": pipeline, "settings": asdict(settings), "feature_order": FEATURES,
                  "class_mapping": dict(enumerate(CLASSES)),
                  "feature_code_sha256": digest(Path(morphology.__file__))}
        with tempfile.TemporaryDirectory(dir=ROOT / "cache") as folder:
            path = Path(folder) / "model.joblib"
            joblib.dump(bundle, path)
            restored = load_model(path)
            before = predict_image(images[0], bundle)[0]
            after = predict_image(images[0], restored)[0]
            self.assertEqual(before, after)
            self.assertEqual(after["features"], frame.iloc[0].to_dict())
            self.assertAlmostEqual(sum(after["probabilities"].values()), 1)
            self.assertIsNone(predict_image(np.zeros((32, 32)), restored)[0]["class_id"])


class WorkflowTests(unittest.TestCase):
    def test_group_transitivity_and_split_separation(self):
        labels = np.repeat(np.arange(10), 30)
        hashes = [str(i) for i in range(len(labels))]
        hashes[1] = hashes[0]
        ra = np.linspace(.00005, 359, len(labels))
        ra[2] = 359.99999
        groups, audit = groups_for(hashes, ra, np.zeros(len(labels)))
        self.assertEqual(audit["joint_non_singleton_groups"], 1)
        np.testing.assert_array_equal(groups[:3], np.zeros(3))
        table = pd.DataFrame(dict(label=labels, row_id=np.arange(len(labels)),
                                  group_id=groups, usable=True))
        plan = split_plan(table)
        verify_plan(table, plan)
        self.assertEqual(plan, split_plan(table))
        dev = set(plan["development"])
        validation = [i for fold in plan["outer"] for i in fold["validation"]]
        self.assertEqual(len(validation), len(dev))
        self.assertEqual(set(validation), dev)

    def test_smoke_predictions_recompute_and_cache_reuses(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "cache") as folder:
            out = Path(folder)
            fixture = out / "synthetic.h5"
            smoke_data(fixture)
            table, provenance = extract_table(fixture, out / "cache", workers=1, official=False)
            cached, repeat = extract_table(fixture, out / "cache", workers=1, official=False)
            self.assertEqual(provenance, repeat)
            pd.testing.assert_frame_equal(table, cached, check_exact=False, atol=1e-14)
            fit_experiment(table, provenance, out, workers=1, smoke=True)
            validate_predictions(out)
            model = load_model(out / "model.joblib")
            self.assertEqual(model["mode"], "synthetic_smoke")
            prediction = pd.read_csv(out / "test_predictions.csv")
            xs = table.set_index("row_id").loc[prediction.row_id, list(FEATURES)]
            np.testing.assert_array_equal(model["pipeline"].predict(xs), prediction.forest)
            path = out / "test_predictions.csv"
            path.write_text(path.read_text() + "\n")
            with self.assertRaises(AssertionError):
                validate_predictions(out)


if __name__ == "__main__":
    unittest.main()
