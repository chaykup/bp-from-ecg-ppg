"""Build the feature table from all four Parts and save it as data/processed/features.parquet.

Usage (from the repo root):
    python src/build_table.py              # full run, all records in all four Parts
    python src/build_table.py --limit 100  # first 100 records of each Part, for testing

Records that are copies of an earlier record (see load.find_duplicates) are skipped.

Also writes data/processed/build_log.json with the window rejection counts, failed records, dropped
duplicates and records flagged by the ECG peak-count check.
"""
import argparse
import json
import time
import warnings
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from features import FEATURE_COLS, extract_features
import config
from load import find_duplicates, load_record, n_records
from signals import REASONS, ecg_peak_count_check, process_record, window_record

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "processed"
PARTS = [f"Part_{p}" for p in range(1, 5)]


def process_one(job: tuple[Path, int]) -> dict:
    """Load, filter, window and featurize one record. Never raises: failures come back as an error string."""
    part, i = job
    record_id = f"{part.stem}_Patient_{i}"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")     # NeuroKit/SciPy warnings on noisy records; failures are caught below
            record_id, ppg, abp, ecg = load_record(part, i)
            rec = process_record(ppg, abp, ecg)
            windows, counts = window_record(record_id, rec)
            rows = [extract_features(w) for w in windows]
            check = ecg_peak_count_check(rec["ecg"], rec["r_peaks"])
        return {"record_id": record_id, "rows": rows, "counts": counts, "ecg_flagged": check["flagged"], "error": None}
    except Exception as e:
        return {"record_id": record_id, "rows": [], "counts": Counter(), "ecg_flagged": False, "error": repr(e)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=None, help="records per Part (default: all)")
    parser.add_argument("--workers", type=int, default=None, help="parallel processes (default: all CPUs)")
    parser.add_argument("--out", type=Path, default=OUT_DIR / "features.parquet")
    args = parser.parse_args()

    jobs = []
    for name in PARTS:
        part = RAW_DIR / f"{name}.mat"
        n = n_records(part)
        jobs += [(part, i) for i in range(min(n, args.limit) if args.limit else n)]
    n_scanned = len(jobs)

    # Drop records that are copies of an earlier one, so the same recording can't land in both train and test
    duplicates = find_duplicates(jobs, config.WINDOW)
    jobs = [(part, i) for part, i in jobs if f"{part.stem}_Patient_{i}" not in duplicates]
    print(f"{n_scanned} records from {len(PARTS)} Parts, {len(duplicates)} duplicates dropped, {len(jobs)} to process")

    rows, counts, failed, flagged = [], Counter(), [], []
    start = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for k, res in enumerate(pool.map(process_one, jobs, chunksize=8), 1):
            rows.extend(res["rows"])
            counts.update(res["counts"])
            if res["error"]:
                failed.append({"record_id": res["record_id"], "error": res["error"]})
            if res["ecg_flagged"]:
                flagged.append(res["record_id"])
            if k % 500 == 0 or k == len(jobs):
                print(f"    {k}/{len(jobs)} records, {len(rows)} windows kept, {time.time() - start:.0f} s", flush=True)

    # Build the DataFrame once at the end; appending in the loop is very slow
    df = pd.DataFrame(rows, columns=["record_id", "part", "window_idx", *FEATURE_COLS, "sbp", "dbp"])
    assert df["record_id"].notna().all() and df[["sbp", "dbp"]].notna().all().all(), "labels must never be NaN"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)

    total = sum(counts.values())
    log = {
        "records_scanned": n_scanned,
        "records_duplicate_dropped": len(duplicates),
        "records": len(jobs),
        "records_failed": len(failed),
        "records_ecg_peak_count_flagged": len(flagged),
        "windows_total": total,
        "windows": {r: counts[r] for r in ("kept",) + REASONS},
        "failed": failed,
        "ecg_peak_count_flagged": flagged,
        "duplicates_dropped": duplicates,   # {dropped copy: kept original}
    }
    with open(args.out.parent / "build_log.json", "w") as f:
        json.dump(log, f, indent=2)

    # Summary of the table
    print(f"\nWrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB) in {time.time() - start:.0f} s")
    print(f"{len(jobs)} records, {len(failed)} failed, {len(flagged)} flagged by the ECG peak-count check")
    print(f"{total} windows:")
    for r in ("kept",) + REASONS:
        print(f"    {r:18s} {counts[r]:7d}  ({counts[r] / max(total, 1):.1%})")
    print()
    df.info()
    print("\n", df.describe().T.round(3).to_string())
    print("\nNaN per column:\n", df.isna().sum()[lambda s: s > 0].to_string() or "    none")
    print("\nWindows per record:\n", df.groupby("record_id").size().describe().round(1).to_string())


if __name__ == "__main__":
    main()
