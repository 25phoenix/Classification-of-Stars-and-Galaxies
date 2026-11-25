"""Stage both public catalogs from ordinary WSL without Python packages."""
import gzip
from pathlib import Path
import shutil
import subprocess
from zipfile import BadZipFile, ZipFile

ROOT = Path(__file__).resolve().parent / "data/raw"
AUTHOR = "Barchi19_Morph-catalog_670k-galaxies.zip"
GZ1 = "GalaxyZoo1_DR_table2.csv.gz"
FULL_RELEASE = "https://data.mendeley.com/public-api/zip/tg7c985c2n/download/1"
GZ1_URL = "https://galaxy-zoo-1.s3.amazonaws.com/" + GZ1


def curl(url, destination):
    command = ["curl", "-fL", "--retry", "3", "--connect-timeout", "12", "--max-time",
               "600", "-o", str(destination)]
    if destination.is_file() and destination.stat().st_size:
        result = subprocess.run(command + ["-C", "-", url])
        if result.returncode not in (22, 33, 36):
            result.check_returncode()
            return
    subprocess.run(command + [url], check=True)


def validate_zip(path):
    with ZipFile(path) as archive:
        if archive.testzip() is not None:
            raise ValueError(f"ZIP CRC failed: {path}")
        files = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if len(files) != 1:
            raise ValueError(f"Unexpected author ZIP entries: {archive.namelist()}")


def download_author(root):
    final = root / AUTHOR
    if final.is_file():
        validate_zip(final)
        return
    outer = root / "mendeley-v1.zip.part"
    curl(FULL_RELEASE, outer)
    partial = final.with_suffix(".zip.part")
    with ZipFile(outer) as archive:
        nested = [name for name in archive.namelist() if Path(name).name == AUTHOR]
        if len(nested) == 1:
            with archive.open(nested[0]) as source, partial.open("wb") as dest:
                shutil.copyfileobj(source, dest)
        else:
            partial = outer
    validate_zip(partial)
    partial.replace(final)
    if outer.exists():
        outer.unlink()


def validate_gz(path):
    with gzip.open(path, "rb") as stream:
        header = stream.readline().upper()
        if not all(x in header for x in (b"OBJID", b"ELLIPTICAL", b"SPIRAL", b"UNCERTAIN")):
            raise ValueError("Unexpected GZ1 header")
        while stream.read(1024 * 1024):
            pass


def download_gz1(root):
    final = root / GZ1
    if final.is_file():
        validate_gz(final)
        return
    partial = final.with_suffix(".gz.part")
    curl(GZ1_URL, partial)
    validate_gz(partial)
    partial.replace(final)


if __name__ == "__main__":
    ROOT.mkdir(parents=True, exist_ok=True)
    try:
        download_author(ROOT)
        download_gz1(ROOT)
    except (OSError, ValueError, BadZipFile, subprocess.CalledProcessError) as error:
        raise SystemExit(str(error)) from None
    print(f"Validated both archive containers under {ROOT}")
