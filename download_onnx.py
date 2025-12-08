#!/usr/bin/env python
import os
import shutil
import sys
import zipfile
from pathlib import Path
from urllib.request import urlopen

DEFAULT_URLS = [
    "https://github.com/the-database/MangaJaNai/releases/download/3.0.0/IllustrationJaNai_V3denoise_onnx.zip",
    "https://github.com/the-database/MangaJaNai/releases/download/3.0.0/IllustrationJaNai_V3detail_onnx.zip",
]

ONNX_DIR = Path("./onnx").resolve()
ONNX_DIR.mkdir(parents=True, exist_ok=True)


def download_zip(url: str, dest_dir: Path) -> Path:
    """
    Download a zip file from `url` to `dest_dir` and return the local path.
    """
    filename = url.rsplit("/", 1)[-1] or "model.zip"
    dest_path = dest_dir / filename

    print(f"\nDownloading:\n  {url}\n  -> {dest_path}")
    with urlopen(url) as resp, open(dest_path, "wb") as f:
        shutil.copyfileobj(resp, f)

    print(f"Downloaded: {dest_path}")
    return dest_path


def extract_zip(zip_path: Path, extract_to: Path) -> None:
    """
    Extract `zip_path` into `extract_to`.
    """
    print(f"Extracting:\n  {zip_path}\n  -> {extract_to}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)
    print("Extraction complete.")


def main(urls: list[str]) -> None:
    tmp_dir = ONNX_DIR / "tmp_downloads"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    for url in urls:
        try:
            zip_path = download_zip(url, tmp_dir)
            extract_zip(zip_path, ONNX_DIR)
        except Exception as e:
            print(f"ERROR processing {url}: {e}")

    try:
        shutil.rmtree(tmp_dir)
        print(f"\nCleaned up temp directory: {tmp_dir}")
    except Exception as e:
        print(f"\nCould not remove temp directory {tmp_dir}: {e}")

    print("\nAll done.")


if __name__ == "__main__":
    main(DEFAULT_URLS)
