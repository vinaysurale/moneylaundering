"""
╔══════════════════════════════════════════════════════════════╗
║  ELLIPTIC BITCOIN DATASET — AUTOMATED DATA RETRIEVAL        ║
║  AML Detection Protocol v2.0                                ║
╚══════════════════════════════════════════════════════════════╝

Downloads and extracts the Elliptic Bitcoin Dataset.
Primary: Kaggle API  |  Fallback: HuggingFace Hub direct download.
"""

import os
import sys
import zipfile
import hashlib
import time
import requests
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────
DATA_DIR = Path(__file__).parent / "data"
KAGGLE_DATASET = "ellipticco/elliptic-data-set"

HUGGINGFACE_BASE = (
    "https://huggingface.co/datasets/yhoma/elliptic-bitcoin-dataset"
    "/resolve/main"
)
HF_FILES = {
    "elliptic_txs_features.csv": f"{HUGGINGFACE_BASE}/elliptic_txs_features.csv",
    "elliptic_txs_classes.csv": f"{HUGGINGFACE_BASE}/elliptic_txs_classes.csv",
    "elliptic_txs_edgelist.csv": f"{HUGGINGFACE_BASE}/elliptic_txs_edgelist.csv",
}

EXPECTED_FILES = [
    "elliptic_txs_features.csv",
    "elliptic_txs_classes.csv",
    "elliptic_txs_edgelist.csv",
]


def log(msg: str, level: str = "INFO"):
    ts = time.strftime("%H:%M:%S")
    prefix = {"INFO": "✓", "WARN": "⚠", "ERROR": "✗", "STEP": "►"}
    print(f"  [{ts}] {prefix.get(level, '·')} {msg}")


def sha256_short(filepath: Path) -> str:
    """Return first 12 chars of SHA-256 hash for integrity check."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def files_already_present() -> bool:
    """Check if all expected CSVs already exist in DATA_DIR."""
    return all((DATA_DIR / f).exists() for f in EXPECTED_FILES)


def download_via_kaggle() -> bool:
    """Attempt download via Kaggle API. Returns True on success."""
    log("Attempting Kaggle API download...", "STEP")
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        log("Kaggle auth successful")

        zip_path = DATA_DIR / "elliptic-data-set.zip"
        api.dataset_download_files(KAGGLE_DATASET, path=str(DATA_DIR), unzip=False)
        log(f"Downloaded ZIP to {zip_path}")

        # Extract
        with zipfile.ZipFile(str(zip_path), "r") as zf:
            zf.extractall(str(DATA_DIR))
        log("Extracted ZIP contents")

        # Clean up zip
        zip_path.unlink(missing_ok=True)

        # Kaggle sometimes nests files in a subfolder — flatten if needed
        _flatten_nested_csvs()
        return True

    except Exception as e:
        log(f"Kaggle download failed: {e}", "WARN")
        return False


def download_via_huggingface() -> bool:
    """Fallback: direct HTTP download from HuggingFace Hub."""
    log("Falling back to HuggingFace Hub download...", "STEP")
    try:
        for filename, url in HF_FILES.items():
            dest = DATA_DIR / filename
            if dest.exists():
                log(f"  {filename} already exists, skipping")
                continue

            log(f"  Downloading {filename}...")
            max_retries = 3
            for attempt in range(1, max_retries + 1):
                try:
                    resp = requests.get(url, stream=True, timeout=120)
                    resp.raise_for_status()
                    total = int(resp.headers.get("content-length", 0))
                    downloaded = 0
                    with open(dest, "wb") as f:
                        for chunk in resp.iter_content(chunk_size=65536):
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total > 0:
                                pct = downloaded / total * 100
                                print(
                                    f"\r    [{downloaded / 1e6:.1f}/{total / 1e6:.1f} MB] "
                                    f"{pct:.0f}%",
                                    end="",
                                    flush=True,
                                )
                    print()  # newline after progress
                    log(f"  ✓ {filename} downloaded ({downloaded / 1e6:.1f} MB)")
                    break
                except (requests.RequestException, IOError) as net_err:
                    log(
                        f"  Attempt {attempt}/{max_retries} failed: {net_err}",
                        "WARN",
                    )
                    if attempt == max_retries:
                        raise
                    time.sleep(2 ** attempt)

        return True
    except Exception as e:
        log(f"HuggingFace download failed: {e}", "ERROR")
        return False


def _flatten_nested_csvs():
    """If Kaggle extracted into a subfolder, move CSVs up to DATA_DIR."""
    for root, dirs, files in os.walk(DATA_DIR):
        root_path = Path(root)
        if root_path == DATA_DIR:
            continue
        for f in files:
            if f.endswith(".csv"):
                src = root_path / f
                dst = DATA_DIR / f
                if not dst.exists():
                    src.rename(dst)
                    log(f"  Moved {f} from nested dir to data/")


def validate():
    """Validate all files exist and print integrity hashes."""
    log("Running validation checks...", "STEP")
    all_ok = True
    for fname in EXPECTED_FILES:
        fpath = DATA_DIR / fname
        if fpath.exists():
            size_mb = fpath.stat().st_size / 1e6
            sha = sha256_short(fpath)
            log(f"  {fname}: {size_mb:.1f} MB | SHA256: {sha}...")
        else:
            log(f"  {fname}: MISSING!", "ERROR")
            all_ok = False
    return all_ok


def main():
    print()
    print("  ╔══════════════════════════════════════════════════╗")
    print("  ║  ELLIPTIC DATASET RETRIEVAL PROTOCOL             ║")
    print("  ╚══════════════════════════════════════════════════╝")
    print()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if files_already_present():
        log("All data files already present — skipping download.")
        validate()
        return

    # Try Kaggle first, then HuggingFace
    success = download_via_kaggle() or download_via_huggingface()

    if not success:
        log("FATAL: Could not download dataset from any source.", "ERROR")
        sys.exit(1)

    if not validate():
        log("FATAL: Validation failed — missing files.", "ERROR")
        sys.exit(1)

    print()
    log("DATA RETRIEVAL COMPLETE ✓", "STEP")
    print()


if __name__ == "__main__":
    main()
