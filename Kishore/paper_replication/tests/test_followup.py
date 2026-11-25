"""Regression coverage for complete-case policy and frozen splits."""
import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

import numpy as np
import pandas as pd

from common import FEATURES, ROOT, write_json
from followup import POLICY, apply_policy, feature_conditions, prediction, restrict_plan
from main import synthetic_table
from evaluation import record, split_plan


class FollowupTests(unittest.TestCase):
    def test_all_confirmed_codes_and_feature_domains(self):
        for key in FEATURES:
            for code in POLICY["sentinels"]:
                independent = {name: .5 for name in FEATURES}
                independent[key] = code
                a, b, c = feature_conditions(pd.DataFrame([independent]))
                self.assertEqual(a.sum().sum(), 1)
                self.assertTrue(a.at[0, key])
                self.assertFalse(b.any().any() or c.any().any())
        good = pd.DataFrame([dict(c=0, a=-1, s=2, g2=2, h=1),
                             dict(c=1, a=-.5, s=-.2, g2=0, h=0)])
        self.assertFalse(any(v.any().any() for v in feature_conditions(good)))
        bad = pd.DataFrame([dict(c=-1e-5, a=-1.01, s=2.01, g2=2.01, h=1.01)])
        self.assertTrue(feature_conditions(bad)[2].all().all())

    def test_inference_abstains(self):
        bundle = {"measurement_policy": POLICY}
        values = dict(c=-9999.99, a=-.2, s=0, g2=1, h=None)
        self.assertEqual(prediction(bundle, values, 1),
                         {"class": None, "flags": ["sentinel:c", "nonfinite:h", "author_error"]})

    def test_filter_retains_all_split_memberships(self):
        table, _ = synthetic_table()
        table[list(FEATURES)] = 1 / (1 + np.exp(-table[list(FEATURES)]))
        plan = split_plan(table, 42)
        table.loc[table.row_id.eq(plan["5"]["test"][0]), "c"] = -9999.999999
        copy, audit = apply_policy(table)
        corrected = restrict_plan(plan, copy)
        self.assertEqual(corrected["5"]["development"], plan["5"]["development"])
        self.assertEqual(corrected["5"]["test"], plan["5"]["test"][1:])
        self.assertEqual(corrected["5"]["outer"], plan["5"]["outer"])
        self.assertEqual(corrected["5"]["final_tuning"], plan["5"]["final_tuning"])
        self.assertEqual(audit["by_cut"]["5"]["overlap_patterns"], {"c": 1})

    def test_metadata_boundaries_and_exact_string_ids(self):
        ids = ["587700000000000001", "587700000000000002", "587700000000000003",
               "587700000000000004"]
        table = pd.DataFrame({f: [.5]*4 for f in FEATURES})
        table = table.assign(objid=ids, label=[0, 1, 0, 1], error=0, k=[5, 10, 20, 20],
                             label_provenance="GZ1 Table 2", b=30)
        for k in (5, 10, 20):
            table[f"selection_k{k}"] = np.where(table.k.ge(k), "included", "below_k")
        meta = pd.DataFrame({"z": [.03, .031, .05, .1], "petromag_r": [16, 17.779, 17.78, 15]}, index=ids)
        final, info = apply_policy(table, meta)
        self.assertEqual(final.loc[final.selection_k5.eq("included"), "objid"].tolist(), [ids[1]])
        self.assertEqual(info["metadata"]["by_cut"]["5"]["unmatched"]["rows"], 0)

    def test_incompatible_checkpoint_rejected_before_fit(self):
        with TemporaryDirectory(dir=ROOT / "cache") as folder:
            out = Path(folder)
            write_json(out / "dt_outer1.json", {"fingerprint": "as_released"})
            with self.assertRaisesRegex(ValueError, "Checkpoint incompatible"):
                record(out, "dt_outer1", "complete_case", None, None, None, 42, 1, True)


if __name__ == "__main__":
    unittest.main()
