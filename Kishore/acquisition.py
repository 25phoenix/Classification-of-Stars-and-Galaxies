"""Download and verify the official file without third-party Python dependencies."""
import fcntl
import hashlib
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data/raw/Galaxy10_DECals.h5"
URL = "https://zenodo.org/records/10845026/files/Galaxy10_DECals.h5"
SHA256 = "19aefc477c41bb7f77ff07599a6b82a038dc042f889a111b0d4d98bb755c1571"


def digest(path):
    with path.open("rb") as file:
        h = hashlib.sha256()
        for block in iter(lambda: file.read(8 * 1024 * 1024), b""):
            h.update(block)
        return h.hexdigest()


def verify():
    if not DATA.is_file():
        raise FileNotFoundError(f"Missing {DATA}. Run the download command.")
    if digest(DATA) != SHA256:
        raise ValueError(f"Official SHA-256 mismatch for {DATA}")


def download():
    DATA.parent.mkdir(parents=True, exist_ok=True)
    with DATA.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        _download_locked()


def _download_locked():
    if DATA.exists():
        verify()
        return
    part = DATA.with_suffix(".h5.part")
    if not part.exists() or digest(part) != SHA256:
        command = ["curl", "--fail", "--location", "--retry", "4", "--retry-delay", "2",
                   "--connect-timeout", "20", "--max-time", "3600", "--output", str(part)]
        resumed = part.exists() and part.stat().st_size > 0
        result = subprocess.run(command + (["--continue-at", "-"] if resumed else []) + [URL])
        if resumed and result.returncode in (22, 33, 36):
            if not part.exists() or digest(part) != SHA256:
                subprocess.run(command + [URL], check=True)
        else:
            result.check_returncode()
    if digest(part) != SHA256:
        raise ValueError(f"Download checksum mismatch. Inspect or move {part} before retrying.")
    part.replace(DATA)
