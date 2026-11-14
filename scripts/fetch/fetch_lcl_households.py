#!/usr/bin/env python3
"""Extract a deterministic household subset from the Low Carbon London dataset.

The full LCL file is ~800 MB zipped and ~10 GB unzipped (167 M rows), far too
large to keep in the repo.  This script streams the CSV out of the zip without
ever writing the unzipped file to disk, keeps a reproducible sample of households
plus the benchmark target (MAC000033), and writes:

* ``data/exogenous/neighbors/lcl_households_2013.csv.gz``  (time x household matrix)
* ``data/exogenous/metadata/lcl_households.csv``           (tariff per household)

ACORN is deliberately not extracted: it is a postcode-level geodemographic label
(CACI classification) rather than a property of the connection, so it is not used
as a covariate anywhere in the exogenous ladder.  The household-information file
that would carry it is a separate LCL product and is not downloaded.
"""

import argparse
import io
import sys
import zipfile
import zlib
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.exogenous.config import METADATA_DIR, NEIGHBOR_DIR  # noqa: E402

LCL_URL = "https://data.london.gov.uk/download/vqm0d/3527bf39-d93e-4071-8451-df2ade1ea4f2/LCL-FullData.zip"
DEFAULT_TARGET = "MAC000033"
CHUNK_ROWS = 2_000_000


def download(url, destination, chunk_size=1 << 20):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        print(f"using existing archive {destination} ({destination.stat().st_size / 1e6:.0f} MB)")
        return destination
    print(f"downloading {url} -> {destination}")
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        written = 0
        with open(destination, "wb") as handle:
            for chunk in response.iter_content(chunk_size=chunk_size):
                handle.write(chunk)
                written += len(chunk)
                if total:
                    print(f"\r  {written / 1e6:6.1f} / {total / 1e6:6.1f} MB", end="")
    print()
    return destination


def pick_csv_member(archive):
    names = [n for n in archive.namelist() if n.lower().endswith(".csv")]
    if not names:
        raise RuntimeError("no CSV member inside the archive")
    return max(names, key=lambda n: archive.getinfo(n).file_size)


class _KeepAliveStream:
    """Minimal binary stream that keeps the owning ZipFile open."""

    def __init__(self, archive, handle):
        self._archive = archive
        self._handle = handle

    def read(self, size=-1):
        return self._handle.read(size)

    def readinto(self, buffer):
        data = self._handle.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)

    def readable(self):
        return True

    def close(self):
        try:
            self._handle.close()
        finally:
            self._archive.close()


def open_member(zip_path, member):
    """Binary stream for one zip member, without extracting it to disk.

    The LCL archive uses Deflate64, which Python's ``zipfile`` cannot decompress
    (``NotImplementedError`` from ``ZipFile.open``), so fall back to streaming
    through ``unzip -p``.
    """
    import subprocess

    archive = zipfile.ZipFile(zip_path)
    try:
        return io.BufferedReader(_KeepAliveStream(archive, archive.open(member)))
    except NotImplementedError:
        archive.close()

    process = subprocess.Popen(
        ["unzip", "-p", str(zip_path), member],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    return process.stdout


def keep_household(lcl_id, modulus, target):
    if lcl_id == target:
        return True
    return zlib.crc32(str(lcl_id).encode()) % modulus == 0


def extract(zip_path, year, modulus, target):
    frames = []
    tariffs = {}
    kept_rows = 0

    with zipfile.ZipFile(zip_path) as archive:
        member = pick_csv_member(archive)
        size = archive.getinfo(member).file_size
    print(f"streaming {member} ({size / 1e9:.1f} GB uncompressed)")

    stream = open_member(zip_path, member)
    # The published header carries a trailing space ("KWH/hh (per half hour) "),
    # so select by position instead of by name.
    reader = pd.read_csv(
        io.TextIOWrapper(stream, encoding="utf-8"),
        header=0,
        usecols=[0, 1, 2, 3],
        chunksize=CHUNK_ROWS,
        on_bad_lines="skip",
    )
    for index, chunk in enumerate(reader, start=1):
        chunk.columns = ["LCLid", "stdorToU", "DateTime", "kwh_raw"]
        chunk = chunk[chunk["LCLid"].map(lambda x: keep_household(x, modulus, target))]
        if not chunk.empty:
            chunk = chunk.assign(DateTime=pd.to_datetime(chunk["DateTime"], errors="coerce"))
            chunk = chunk.dropna(subset=["DateTime"])
            chunk = chunk[chunk["DateTime"].dt.year == year]
            for lcl_id, group in chunk.groupby("LCLid"):
                tariffs[lcl_id] = group["stdorToU"].mode().iloc[0]
            frames.append(chunk[["LCLid", "DateTime", "kwh_raw"]])
            kept_rows += len(chunk)
        if index % 5 == 0:
            print(f"  chunk {index:3d}: kept {kept_rows:,} rows", flush=True)

    if not frames:
        raise RuntimeError("no rows survived the filter; is the archive the LCL full data?")

    combined = pd.concat(frames, ignore_index=True)
    combined.columns = ["LCLid", "DateTime", "kwh"]
    combined = combined.drop_duplicates(subset=["LCLid", "DateTime"])
    matrix = combined.pivot(index="DateTime", columns="LCLid", values="kwh").sort_index()
    return matrix, tariffs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip-path", type=Path, default=NEIGHBOR_DIR.parent / "raw" / "LCL-FullData.zip")
    parser.add_argument("--year", type=int, default=2013)
    parser.add_argument("--modulus", type=int, default=97,
                        help="Keep a household when crc32(LCLid) %% modulus == 0 (~1/modulus of 5,567).")
    parser.add_argument("--target", default=DEFAULT_TARGET)
    parser.add_argument("--keep-zip", action="store_true")
    args = parser.parse_args(argv)

    zip_path = download(LCL_URL, args.zip_path)
    matrix, tariffs = extract(zip_path, args.year, args.modulus, args.target)

    NEIGHBOR_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    matrix_path = NEIGHBOR_DIR / f"lcl_households_{args.year}.csv.gz"
    matrix.to_csv(matrix_path, float_format="%.4f")
    print(f"matrix {matrix.shape[0]:,} rows x {matrix.shape[1]} households -> {matrix_path.relative_to(PROJECT_ROOT)}")

    metadata = pd.DataFrame(
        [{"entity_id": lcl_id, "tariff_type": tariff} for lcl_id, tariff in sorted(tariffs.items())]
    )
    metadata_path = METADATA_DIR / "lcl_households.csv"
    metadata.to_csv(metadata_path, index=False)
    print(f"metadata {len(metadata)} households -> {metadata_path.relative_to(PROJECT_ROOT)}")
    print(f"target {args.target} present: {args.target in matrix.columns}")
    print("ACORN is not extracted: geodemographic label, excluded from the covariate set by design.")

    if not args.keep_zip:
        zip_path.unlink()
        print(f"removed {zip_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
