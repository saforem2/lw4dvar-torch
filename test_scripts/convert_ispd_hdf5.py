#!/usr/bin/env python3
"""Convert public ISPD v4.7 hourly HDF5 files to lw4dvar's psobs text format.

The solver expects ``psobs1_YYYYMMDDHH.txt`` rows with an eight-character
ignored prefix followed by observation type, longitude, latitude, elevation,
unused value, surface pressure (hPa), unused value, and error (hPa).

ISPD v4.7 is the public source, not a byte-for-byte copy of the older
``psobs1`` archive used by historical runs. This conversion keeps records
with NCEP observation type 181 and a finite, physical surface pressure. It
uses 1 hPa observation error, matching the existing ``psobs1`` inputs.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import h5py
import numpy as np


SOURCE_URL = (
    "https://osdf-director.osg-htc.org/ncar/gdex/d132002/"
    "ispdv4.7_datafiles"
)
DOI = "10.5065/9EYR-TY90"
TYPE_CODE = 181
HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"


def ensure_source(source: Path, timestamp: datetime, download_missing: bool) -> None:
    if source.is_file():
        with source.open("rb") as handle:
            if handle.read(8) == HDF5_MAGIC:
                return
        raise ValueError(f"{source}: not an HDF5 file")
    if not download_missing:
        raise FileNotFoundError(source)
    source.parent.mkdir(parents=True, exist_ok=True)
    stamp = timestamp.strftime("%Y%m%d%H")
    url = f"{SOURCE_URL}/{timestamp.year}/{timestamp:%m}/{stamp}.h5"
    partial = source.with_suffix(source.suffix + ".part")
    subprocess.run(
        ["curl", "-fLsS", "--retry", "5", "--retry-all-errors",
         "--connect-timeout", "20", "--max-time", "180", "-o", str(partial), url],
        check=True,
    )
    with partial.open("rb") as handle:
        if handle.read(8) != HDF5_MAGIC:
            partial.unlink(missing_ok=True)
            raise ValueError(f"{url}: response was not an HDF5 file")
    partial.replace(source)


def convert_file(source: Path, target: Path, timestamp: datetime, error_hpa: float) -> int:
    with h5py.File(source, "r") as h5:
        observations = h5["Data/Observations/Observations"][:]
        types = h5["Data/Observations/ObservationTypes"][:]
        locations = h5["Data/SpatialTemporalLocation/SpatialTemporalLocation"][:]

    keys = ("Timestamp", "Unique Observation Number")
    for name, table in (("types", types), ("locations", locations)):
        if len(table) != len(observations) or any(
            not np.array_equal(observations[key], table[key]) for key in keys
        ):
            raise ValueError(f"{source}: observation IDs/timestamps do not align with {name}")

    stamp = timestamp.strftime("%Y%m%d%H")
    # Each hourly HDF5 file can contain observations from any minute in that
    # hour. The filename already selects the hour; do not discard non-:00
    # reports when writing the six-hour solver inputs.
    type_code = types["NCEP Observation Type Code"]
    pressure = observations["Observed Surface Pressure"]
    valid = (
        (type_code == TYPE_CODE)
        & np.isfinite(pressure)
        & (pressure > 0.0)
        & (pressure < 2000.0)
    )
    selected = np.flatnonzero(valid)
    if not len(selected):
        raise ValueError(f"{source}: no valid type-{TYPE_CODE} surface pressures for {stamp}")

    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w") as output:
        for row_number, i in enumerate(selected, start=1):
            lon = float(locations["Longitude"][i])
            lat = float(locations["Latitude"][i])
            elev = float(locations["Elevation"][i])
            if not all(np.isfinite(v) for v in (lon, lat, elev)):
                raise ValueError(f"{source}: non-finite location for record {i}")
            output.write(
                f"{row_number:08d} {TYPE_CODE:3d} {lon:8.2f} {lat:7.2f} "
                f"{elev:7.1f} {0.0:6.2f} {float(pressure[i]):8.1f} "
                f"{0.0:7.4f} {error_hpa:.2f}\n"
            )
    return len(selected)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="root containing YYYY/MM/*.h5")
    parser.add_argument("--output", type=Path, required=True, help="destination for psobs1_*.txt")
    parser.add_argument("--start", default="2015-01-01T00", help="first valid time, UTC")
    parser.add_argument("--end", default="2015-03-06T18", help="last valid time, UTC")
    parser.add_argument("--interval-hours", type=int, default=6)
    parser.add_argument("--error-hpa", type=float, default=1.0)
    parser.add_argument(
        "--download-missing", action="store_true",
        help="fetch missing hourly HDF5 files from the public NCAR GDEX OSDF mirror",
    )
    args = parser.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%dT%H").replace(tzinfo=timezone.utc)
    end = datetime.strptime(args.end, "%Y-%m-%dT%H").replace(tzinfo=timezone.utc)
    if args.interval_hours <= 0 or end < start:
        parser.error("require a positive interval and end >= start")

    counts: dict[str, int] = {}
    timestamp = start
    while timestamp <= end:
        stamp = timestamp.strftime("%Y%m%d%H")
        source = args.input / str(timestamp.year) / timestamp.strftime("%m") / f"{stamp}.h5"
        target = args.output / f"psobs1_{stamp}.txt"
        ensure_source(source, timestamp, args.download_missing)
        counts[stamp] = convert_file(source, target, timestamp, args.error_hpa)
        timestamp += timedelta(hours=args.interval_hours)

    manifest = {
        "source_dataset": "International Surface Pressure Databank v4.7",
        "source_doi": DOI,
        "source_url": SOURCE_URL,
        "time_range_utc": [args.start, args.end],
        "interval_hours": args.interval_hours,
        "selection": {
            "ncep_observation_type_code": TYPE_CODE,
            "surface_pressure_hpa": "finite and 0 < value < 2000",
            "observation_error_hpa": args.error_hpa,
        },
        "counts_by_time": counts,
        "total_observations": sum(counts.values()),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "ispdv4.7_conversion_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(f"converted={len(counts)} files observations={manifest['total_observations']}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
