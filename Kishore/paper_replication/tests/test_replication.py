import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

from catalogs import AUTHOR_FILE, GZ_FILE, angle_degrees, grouping, k_value, prepare, select
from common import FEATURES, ROOT
from evaluation import model_pipeline, split_plan, splits, verify_plan
from image_features import (Settings, asymmetry, butterworth, concentration, entropy,
                            extract_prepared, g2, gradient_moment, smoothness)
from main import synthetic_table


class CatalogTests(unittest.TestCase):
    def test_sparse_repeat_group_stratification_with_different_k_bands(self):
        size = 1200
        labels = np.arange(size) % 2
        groups = np.arange(size)
        groups[2] = groups[0]
        bands = np.arange(size) % 3
        frame = pd.DataFrame({"label": labels, "group_id": groups})
        strata = labels * 3 + bands
        folds = splits(frame, 5, 42, strata)
        for train, val in folds:
            self.assertEqual(0 in val, 2 in val)
            self.assertLessEqual(abs(np.count_nonzero(labels[val] == 0) - 120), 2)
            self.assertEqual(set(groups[train]) & set(groups[val]), set())

    def test_actual_gz1_sexagesimal_format(self):
        np.testing.assert_allclose(angle_degrees(pd.Series(["00:00:00.41", "12:00:00.00"]), True),
                                   [15*.41/3600, 180])
        np.testing.assert_allclose(angle_degrees(pd.Series(["-00:30:00.0", "+10:22:25.7"]), False),
                                   [-.5, 10+22/60+25.7/3600])

    def test_released_archive_shapes_and_cache_round_trip(self):
        objid = "587700000000000001"
        sample = pd.DataFrame([{**dict.fromkeys(FEATURES, .123), "OBJID": objid,
                                "K": 15, "Error": 0, "ML2classes": 1}])
        votes = pd.DataFrame([{"OBJID": objid, "ELLIPTICAL": 1, "SPIRAL": 0, "UNCERTAIN": 0}])
        with tempfile.TemporaryDirectory(dir=ROOT / "cache") as temp:
            raw = Path(temp)
            with ZipFile(raw / AUTHOR_FILE, "w") as archive:
                archive.writestr("author.csv", sample.to_csv(index=False))
            votes.to_csv(raw / GZ_FILE, index=False)
            rows, provenance = prepare(raw, raw / "features")
            cached, second = prepare(raw, raw / "features")
            self.assertEqual(provenance, second)
            self.assertEqual(rows.objid.iloc[0], cached.objid.iloc[0])
            self.assertEqual(cached.label.iloc[0], 0)
            self.assertEqual(cached.selection_k10.iloc[0], "included")
            self.assertNotIn("ML2classes", cached)

    def test_exact_join_flags_and_paper_cuts(self):
        base = 587700000000000000
        ids = [str(base+i) for i in range(7)]
        author = pd.DataFrame({name: ["0.23"]*7 for name in FEATURES}).assign(
            OBJID=ids, K=["20", "20", "10", "5", "25", "18", "22"], Error=["0","0","0","1","0","0","0"],
            GL="unused", b=["31","32","40","35","34","29","33"])
        votes = pd.DataFrame({"OBJID": ids[:-1], "ELLIPTICAL": list("100111"),
                              "SPIRAL": list("010000"), "UNCERTAIN": list("001000")})
        rows, _ = select(author, votes)
        self.assertEqual(rows.objid.tolist(), ids)
        self.assertEqual(rows.label.tolist(), [0, 1, -1, 0, 0, 0, -1])
        self.assertEqual(rows.selection_k20.tolist(),
                         ["included", "included", "no_defined_gz1_label", "cymorph_error",
                          "included", "outside_b_selection", "no_gz1_match"])

    def test_k_units_and_invalid(self):
        np.testing.assert_allclose(k_value([10, 8], [2, 0]), [100, np.nan], equal_nan=True)
        self.assertAlmostEqual(float(k_value(10, 1.584, "pixel", "arcsec")), 25)

    def test_identity_transitivity_across_ra_wrap(self):
        frame = pd.DataFrame(dict(objid=["a","a","b","c"], image_sha256=["1","2","3","3"],
                                  ra=[.00005, 40, 359.99999, 120], dec=[0,0,0,0]))
        groups, audit = grouping(frame)
        np.testing.assert_array_equal(groups, [0,0,0,0])
        self.assertEqual(audit["one_arcsec_pairs"], 1)

    def test_all_split_boundaries_and_training_only_transform(self):
        table, _ = synthetic_table()
        plan = split_plan(table, 42)
        verify_plan(table, plan)
        data = table.set_index("row_id")
        fold = plan["5"]["outer"][0]
        inner = fold["inner"][0]
        tr = data.loc[inner["train"]]
        estimator = model_pipeline("svm", 42, smoke=True).fit(tr[list(FEATURES)], tr.label)
        np.testing.assert_allclose(estimator["scaler"].mean_, tr[list(FEATURES)].mean())
        self.assertFalse(np.allclose(estimator["scaler"].mean_, data.loc[fold["train"], list(FEATURES)].mean()))
        train_scaled = estimator["scaler"].transform(tr[list(FEATURES)])
        np.testing.assert_allclose(estimator["rbf"].components_,
                                   train_scaled[estimator["rbf"].component_indices_])


class ImageTests(unittest.TestCase):
    def test_concentration_interpolation_and_undefined(self):
        self.assertAlmostEqual(concentration(np.arange(14, dtype=float), 5), np.log10(.75/.35))
        self.assertTrue(np.isnan(concentration(np.arange(8), 5)))
        self.assertTrue(np.isnan(concentration(np.zeros(20), 5)))

    def test_entropy(self):
        mask = np.ones((2, 2), dtype=bool)
        self.assertAlmostEqual(entropy(np.array([[1, 2], [3, 4]]), mask, 4), 1)
        self.assertAlmostEqual(entropy(np.ones((2, 2)), mask, 4), 0)
        self.assertTrue(np.isnan(entropy(np.zeros((2, 2)), mask, 4)))

    def test_rank_rotation_and_smoothing(self):
        y, x = np.indices((24, 24))
        profile = np.exp(-((x-11.5)**2+(y-11.5)**2)/40.)
        mask = np.ones(profile.shape, dtype=bool)
        self.assertLess(asymmetry(profile, mask, 2), .005)
        self.assertGreater(asymmetry(profile+.4*(x > 12), mask, 2), .15)
        noisy = profile + .25*(-1.)**(x+y)
        smoothed = butterworth(noisy, 10, Settings())
        self.assertLess(np.std(smoothed-profile), np.std(noisy-profile))
        self.assertTrue(0 <= smoothness(noisy, noisy, mask, 10, 2, Settings()) <= 2)

    def test_rosa_gradient_equation_and_author_conventions(self):
        # Two equal opposed survivors have zero vector sum. Parallel unit
        # survivors give unit alignment. The population includes removed vectors.
        self.assertAlmostEqual(gradient_moment(np.array([[1.,0],[-1.,0]]), 4), 1)
        self.assertAlmostEqual(gradient_moment(np.array([[1.,0],[1.,0]]), 4), .5)
        y, x = np.indices((11, 11))
        mask = np.ones(x.shape, dtype=bool)
        self.assertEqual(g2((x-5)**2+(y-5)**2, mask, 5, 5, Settings()), 0)
        self.assertAlmostEqual(g2(x+y, mask, 5, 5, Settings()), 1, places=6)
        self.assertEqual(g2(x, mask, 5, 5, Settings()), 0)
        self.assertTrue(np.isnan(g2(np.ones(x.shape), mask, 5, 5, Settings())))

    def test_prepared_determinism_and_scientific_image_contract(self):
        y, x = np.indices((20, 20))
        image = x*x + x*y + y
        data = dict(survey="SDSS-DR7", band="r", rp_pixels=5., fwhm_pixels=2., max_rad_pixels=9.,
                    cx_pixels=10., cy_pixels=10., cumulative_flux=np.arange(15, dtype=float))
        for name in ("clean_g", "raw_h", "enhanced_a", "sky_subtracted_s", "filter_input_s"):
            data[name] = image
        for name in ("mask_g", "mask_h", "mask_a", "mask_s"):
            data[name] = np.ones(image.shape, dtype=bool)
        with tempfile.TemporaryDirectory(dir=ROOT / "cache") as temp:
            path = Path(temp) / "image.npz"
            np.savez(path, **data)
            self.assertEqual(extract_prepared(path), extract_prepared(path))
            np.savez(path, **{**data, "band": "rgb"})
            with self.assertRaises(ValueError):
                extract_prepared(path)


if __name__ == "__main__":
    unittest.main()
