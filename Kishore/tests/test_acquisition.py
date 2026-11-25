import hashlib
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import acquisition


class AcquisitionTests(unittest.TestCase):
    def test_partial_is_validated_and_promoted_without_fetching(self):
        with tempfile.TemporaryDirectory(dir=acquisition.ROOT / "cache") as folder:
            dataset = Path(folder) / "file.h5"
            partial = dataset.with_suffix(".h5.part")
            partial.write_bytes(b"fixture")
            checksum = hashlib.sha256(b"fixture").hexdigest()
            with patch.object(acquisition, "DATA", dataset), patch.object(acquisition, "SHA256", checksum):
                with patch.object(acquisition.subprocess, "run") as curl:
                    acquisition.download()
                    acquisition.download()
                    curl.assert_not_called()
                self.assertFalse(partial.exists())
                dataset.write_bytes(b"tampered")
                with self.assertRaises(ValueError):
                    acquisition.verify()

    def test_range_rejection_restarts_download_and_checks_integrity(self):
        with tempfile.TemporaryDirectory(dir=acquisition.ROOT / "cache") as folder:
            dataset = Path(folder) / "file.h5"
            partial = dataset.with_suffix(".h5.part")
            partial.write_bytes(b"partial")
            checksum = hashlib.sha256(b"complete").hexdigest()
            calls = []

            def fetch(command, check=False):
                calls.append(command)
                if "--continue-at" in command:
                    return subprocess.CompletedProcess(command, 33)
                partial.write_bytes(b"complete")
                return subprocess.CompletedProcess(command, 0)

            with patch.object(acquisition, "DATA", dataset), patch.object(acquisition, "SHA256", checksum):
                with patch.object(acquisition.subprocess, "run", side_effect=fetch):
                    acquisition.download()
                self.assertEqual(dataset.read_bytes(), b"complete")
                self.assertFalse(partial.exists())
                self.assertIn("--continue-at", calls[0])
                self.assertNotIn("--continue-at", calls[1])


if __name__ == "__main__":
    unittest.main()
