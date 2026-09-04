"""Download the authors' convenience FI-2010 decimal-precision Setup-2 archive.

This is a convenience mirror from the DeepLOB authors' GitHub repository, not the
canonical FI-2010 host.

Integrity comes primarily from pinning the URL to an immutable commit SHA: GitHub serves
content-addressed blobs, so that pin fixes the bytes. The size check below is a secondary
guard against truncated or interrupted downloads, not a substitute for the pin.

Extracting the four text files needs roughly 1 GB of free disk space.
For the complete dataset and the z-score variant use Fairdata/ETSIN.
"""

from __future__ import annotations

import argparse
import urllib.request
import zipfile
from pathlib import Path

AUTHOR_DATA_COMMIT = "d45844b022209bd9d7985de97076f2e80c5144dc"
EXPECTED_ARCHIVE_BYTES = 56_278_154
URL = (
    "https://raw.githubusercontent.com/zcakhaa/"
    "DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books/"
    f"{AUTHOR_DATA_COMMIT}/data/data.zip"
)


def _safe_extract(zf: zipfile.ZipFile, destination: Path) -> None:
    root = destination.resolve()
    for member in zf.infolist():
        target = (destination / member.filename).resolve()
        if root not in target.parents and target != root:
            raise ValueError(f"Unsafe archive path: {member.filename}")
    zf.extractall(destination)


def _validate_archive(path: Path) -> None:
    size = path.stat().st_size
    if size != EXPECTED_ARCHIVE_BYTES:
        raise RuntimeError(
            f"Unexpected author archive size: {size} bytes; "
            f"expected {EXPECTED_ARCHIVE_BYTES}. Refusing extraction."
        )
    if not zipfile.is_zipfile(path):
        raise RuntimeError("Downloaded author archive is not a valid ZIP file")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/raw")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    archive = out / "deeplob_author_data.zip"
    partial = archive.with_suffix(".zip.partial")
    try:
        urllib.request.urlretrieve(URL, partial)
        _validate_archive(partial)
        partial.replace(archive)
    finally:
        partial.unlink(missing_ok=True)
    with zipfile.ZipFile(archive) as zf:
        _safe_extract(zf, out)
    archive.unlink(missing_ok=True)
    print(f"Extracted pinned author Setup-2 files to {out}")


if __name__ == "__main__":
    main()
