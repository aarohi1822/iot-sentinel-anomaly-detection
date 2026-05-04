from __future__ import annotations

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path


TELEMANOM_DATA_URL = "https://s3-us-west-2.amazonaws.com/telemanom/data.zip"
LABELS_URL = "https://raw.githubusercontent.com/khundman/telemanom/master/labeled_anomalies.csv"
HF_BASE_URL = "https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/main"


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request) as response, destination.open("wb") as output:
        shutil.copyfileobj(response, output)


def download_aggregate_smap(output_dir: Path) -> None:
    files = {
        "SMAP_train.npy": f"{HF_BASE_URL}/SMAP/SMAP_train.npy",
        "SMAP_test.npy": f"{HF_BASE_URL}/SMAP/SMAP_test.npy",
        "SMAP_test_label.npy": f"{HF_BASE_URL}/SMAP/SMAP_test_label.npy",
        "labeled_anomalies.csv": LABELS_URL,
    }
    for filename, url in files.items():
        download_file(url, output_dir / filename)


def prepare_dataset(output_dir: str | Path, force: bool = False, keep_archive: bool = False) -> Path:
    output_dir = Path(output_dir)
    train_dir = output_dir / "train"
    test_dir = output_dir / "test"
    labels_path = output_dir / "labeled_anomalies.csv"

    if train_dir.exists() and test_dir.exists() and labels_path.exists() and not force:
        return output_dir

    archive_path = output_dir / "data.zip"
    extract_dir = output_dir / "_extract"

    if output_dir.exists() and force:
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        download_file(TELEMANOM_DATA_URL, archive_path)
        download_file(LABELS_URL, labels_path)

        if extract_dir.exists():
            shutil.rmtree(extract_dir)
        extract_dir.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(extract_dir)

        extracted_root = extract_dir / "data" if (extract_dir / "data").exists() else extract_dir
        shutil.copytree(extracted_root / "train", train_dir, dirs_exist_ok=True)
        shutil.copytree(extracted_root / "test", test_dir, dirs_exist_ok=True)
    except Exception as exc:
        print(f"Telemanom host unavailable ({exc}). Falling back to SMAP aggregate mirror.")
        download_aggregate_smap(output_dir)
    finally:
        if extract_dir.exists():
            shutil.rmtree(extract_dir)
        if not keep_archive and archive_path.exists():
            archive_path.unlink()

    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the public NASA SMAP/MSL dataset.")
    parser.add_argument("--output-dir", default="data/nasa_telemanom")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--keep-archive", action="store_true")
    args = parser.parse_args()

    dataset_dir = prepare_dataset(
        output_dir=args.output_dir,
        force=args.force,
        keep_archive=args.keep_archive,
    )
    print(f"NASA dataset ready at {dataset_dir}")


if __name__ == "__main__":
    main()
