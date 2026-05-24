#!/usr/bin/env python3
"""
Download large dataset files that are excluded from git due to size limits.

Usage
-----
    python download_data.py              # download everything
    python download_data.py --list       # show what will be downloaded
    python download_data.py --dataset movielens  # download a specific dataset

Datasets
--------
  movielens   MovieLens 20M  (~190 MB download, ~620 MB extracted)
              ratings.csv, movies.csv, links.csv, tags.csv
              Source: https://grouplens.org/datasets/movielens/20m/
"""

import argparse
import hashlib
import os
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent / "training_data"

# ── Dataset registry ──────────────────────────────────────────────────────────
DATASETS = {
    "movielens": {
        "description": "MovieLens 20M — 22 million ratings for 27,000 movies",
        "url": "https://files.grouplens.org/datasets/movielens/ml-20m.zip",
        # Expected files after extraction (relative to DATA_DIR)
        "files": ["ratings.csv", "movies.csv", "links.csv", "tags.csv"],
        # MD5 of the zip from GroupLens (verified 2016-01-29 release)
        "md5": "cd245b17a1ae2cc31bb14903e1204af3",
        # Inside the zip, files live under this subfolder
        "zip_subfolder": "ml-20m",
    }
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def md5(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while buf := f.read(chunk):
            h.update(buf)
    return h.hexdigest()


def progress_hook(block_num, block_size, total_size):
    """Simple download progress bar."""
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100, downloaded * 100 // total_size)
        mb_done = downloaded / 1_048_576
        mb_total = total_size / 1_048_576
        bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
        print(f"\r  [{bar}] {pct:3d}%  {mb_done:.0f}/{mb_total:.0f} MB", end="", flush=True)
    else:
        print(f"\r  Downloaded {downloaded / 1_048_576:.0f} MB …", end="", flush=True)


def download_dataset(name: str, info: dict, force: bool = False) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = DATA_DIR / f"{name}.zip"

    # ── Check if files already exist ─────────────────────────────────────────
    missing = [f for f in info["files"] if not (DATA_DIR / f).exists()]
    if not missing and not force:
        print(f"✅  {name}: all files already present — skipping.")
        print(f"    (use --force to re-download)")
        return

    if missing:
        print(f"📥  {name}: missing files → {missing}")
    else:
        print(f"📥  {name}: force re-download requested")

    # ── Download ──────────────────────────────────────────────────────────────
    print(f"    URL : {info['url']}")
    print(f"    Dest: {zip_path}")
    urllib.request.urlretrieve(info["url"], zip_path, reporthook=progress_hook)
    print()  # newline after progress bar

    # ── Verify MD5 (if provided) ──────────────────────────────────────────────
    if "md5" in info:
        print("    Verifying checksum …", end="", flush=True)
        actual = md5(zip_path)
        if actual != info["md5"]:
            print(f"\n❌  MD5 mismatch!\n    expected: {info['md5']}\n    got     : {actual}")
            print("    The file may be corrupt or the dataset was updated upstream.")
            print("    Proceeding anyway — if extraction fails, delete the zip and retry.")
        else:
            print(" OK")

    # ── Extract ───────────────────────────────────────────────────────────────
    print(f"    Extracting to {DATA_DIR} …")
    with zipfile.ZipFile(zip_path, "r") as zf:
        subfolder = info.get("zip_subfolder", "")
        for target_file in info["files"]:
            zip_member = f"{subfolder}/{target_file}" if subfolder else target_file
            # Find the member (some zips use different separators)
            members = zf.namelist()
            match = next((m for m in members if m.endswith(f"/{target_file}") or m == target_file), None)
            if match is None:
                print(f"    ⚠️  {target_file} not found inside zip — skipping")
                continue
            # Extract to a temp location then move to DATA_DIR
            zf.extract(match, DATA_DIR / "_tmp_extract")
            extracted = DATA_DIR / "_tmp_extract" / match
            dest = DATA_DIR / target_file
            shutil.move(str(extracted), str(dest))
            size_mb = dest.stat().st_size / 1_048_576
            print(f"    ✓  {target_file}  ({size_mb:.1f} MB)")

    # ── Cleanup ───────────────────────────────────────────────────────────────
    if (DATA_DIR / "_tmp_extract").exists():
        shutil.rmtree(DATA_DIR / "_tmp_extract")
    zip_path.unlink(missing_ok=True)
    print(f"✅  {name}: done.\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Download large datasets for the ML curriculum notebooks."
    )
    parser.add_argument(
        "--dataset", "-d",
        choices=list(DATASETS.keys()),
        help="Download only this dataset (default: all)",
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List datasets and their status, then exit",
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Re-download even if files already exist",
    )
    args = parser.parse_args()

    targets = {args.dataset: DATASETS[args.dataset]} if args.dataset else DATASETS

    if args.list:
        print("\nAvailable datasets:\n")
        for name, info in targets.items():
            print(f"  {name}")
            print(f"    {info['description']}")
            for fname in info["files"]:
                path = DATA_DIR / fname
                status = f"✅ present ({path.stat().st_size / 1_048_576:.1f} MB)" if path.exists() else "❌ missing"
                print(f"    • {fname}: {status}")
            print()
        return

    print()
    for name, info in targets.items():
        download_dataset(name, info, force=args.force)

    print("All done! You can now run the recommendation system notebooks.")
    print(f"Data location: {DATA_DIR}\n")


if __name__ == "__main__":
    main()
