import unittest

import pandas as pd

from validate_dr7 import FIELDS, coverage, inspect_export


class ExportTests(unittest.TestCase):
    def test_conflicts_reject_whole_exact_id(self):
        conflict, neighbor = "587732702328717358", "587732702328717359"
        rows = pd.DataFrame([(conflict, ".04", "16"), (conflict, ".08", "16"),
                             (neighbor, ".06", "17"), (neighbor, "0.060", "17.0")],
                            columns=FIELDS)
        clean, rejected, audit = inspect_export(rows)
        self.assertEqual(rejected, {conflict})
        self.assertEqual(clean.objid.tolist(), [neighbor])
        self.assertEqual(audit["duplicate_ids"], 2)
        self.assertEqual(coverage(rows.objid, clean.objid)["unmatched_rows"], 2)

    def test_missing_and_nonfinite_are_recorded(self):
        rows = pd.DataFrame([("123", "", "inf")], columns=FIELDS)
        clean, rejected, audit = inspect_export(rows)
        self.assertEqual(clean.objid.tolist(), ["123"])
        self.assertFalse(rejected)
        self.assertEqual(audit["missing"]["z"], 1)
        self.assertEqual(audit["nonfinite_or_unparseable"], {"z": 1, "petromag_r": 1})

    def test_scientific_notation_is_invalid_for_ids(self):
        rows = pd.DataFrame([("5.877e17", ".05", "17")], columns=FIELDS)
        with self.assertRaisesRegex(ValueError, "non-decimal"):
            inspect_export(rows)
