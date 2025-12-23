import os

import numpy as np
import pyvips
import re
import requests
from urllib.parse import urlparse, unquote
from mangajanaitrt.console import console

def is_url(path: str) -> bool:
    return path.startswith(("http://", "https://"))


def guess_extension_from_content_type(content_type: str | None) -> str:
    if not content_type:
        return ".png"

    content_type = content_type.lower().split(";")[0].strip()
    mapping = {
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/jpg": ".jpg",
        "image/webp": ".webp",
        "image/avif": ".avif",
        "image/bmp": ".bmp",
        "image/tiff": ".tiff",
        "image/gif": ".gif",
    }
    return mapping.get(content_type, ".png")


def guess_extension_from_magic(data: bytes) -> str:
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    elif data[:2] == b"\xff\xd8":
        return ".jpg"
    elif data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ".webp"
    elif data[:4] == b"GIF8":
        return ".gif"
    elif data[:2] in (b"BM",):
        return ".bmp"
    elif data[:4] in (b"II\x2a\x00", b"MM\x00\x2a"):
        return ".tiff"

    elif len(data) >= 12 and data[4:8] == b"ftyp":
        brand = data[8:12]
        if brand in (b"avif", b"avis", b"mif1"):
            return ".avif"
    return ".png"


def extract_filename_from_url(url: str) -> str:
    parsed = urlparse(url)
    path = unquote(parsed.path)
    basename = os.path.basename(path) if path else ""

    if not basename or basename.startswith("?"):
        safe_name = re.sub(r"[^\w\-.]", "_", parsed.netloc + parsed.path)
        basename = safe_name[:50] if len(safe_name) > 50 else safe_name

    name_without_ext = os.path.splitext(basename)[0]
    name_without_ext = re.sub(r"[^\w\-.]", "_", name_without_ext)

    return name_without_ext if name_without_ext else "downloaded_image"


def download_image_to_temp(url: str, temp_dir: str) -> str:
    console.print(f"[dim]Downloading: {url}[/]")

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    response = requests.get(url, headers=headers, timeout=30, stream=True)
    response.raise_for_status()

    content = response.content

    content_type = response.headers.get("Content-Type")
    ext = guess_extension_from_content_type(content_type)

    if ext == ".png" and content_type not in ("image/png",):
        ext = guess_extension_from_magic(content)

    base_name = extract_filename_from_url(url)
    filename = f"{base_name}{ext}"

    temp_path = os.path.join(temp_dir, filename)

    counter = 1
    while os.path.exists(temp_path):
        temp_path = os.path.join(temp_dir, f"{base_name}_{counter}{ext}")
        counter += 1

    with open(temp_path, "wb") as f:
        f.write(content)

    return temp_path


def load_image(path: str) -> tuple[np.ndarray, np.ndarray | None]:

    img_peek = pyvips.Image.new_from_file(path, access="sequential")
    needs_alpha = img_peek.bands in (2, 4)

    # need random access to extract alpha
    access_mode = "random" if needs_alpha else "sequential"
    img = pyvips.Image.new_from_file(path, access=access_mode, fail=True)

    alpha = None

    if img.bands == 1:
        img = img.colourspace("srgb")
    elif img.bands == 2:
        alpha_band = img.extract_band(1)
        img = img.extract_band(0).colourspace("srgb")
        alpha = np.ndarray(
            buffer=alpha_band.write_to_memory(),
            dtype=np.uint8,
            shape=[alpha_band.height, alpha_band.width],
        )
    elif img.bands == 4:
        alpha_band = img.extract_band(3)
        img = img.extract_band(0, n=3).icc_transform("srgb")
        alpha = np.ndarray(
            buffer=alpha_band.write_to_memory(),
            dtype=np.uint8,
            shape=[alpha_band.height, alpha_band.width],
        )
    elif img.bands == 3:
        img = img.icc_transform("srgb")
    else:
        img = img.colourspace("srgb")

    rgb = np.ndarray(
        buffer=img.write_to_memory(), dtype=np.uint8, shape=[img.height, img.width, 3]
    )

    return rgb, alpha


def save_image(
    arr: np.ndarray,
    path: str,
    quality: int = 95,
    webp_lossless: bool = False,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    height, width, bands = arr.shape
    img = pyvips.Image.new_from_memory(arr.tobytes(), width, height, bands, "uchar")

    q = int(quality)
    if q < 1:
        q = 1
    elif q > 100:
        q = 100

    ext = os.path.splitext(path)[1].lower()

    if ext == ".png":
        img.pngsave(path, compression=6)

    elif ext in (".jpg", ".jpeg"):
        if bands == 4:
            img = img.flatten(background=[255, 255, 255])
        img.jpegsave(path, Q=q)

    elif ext == ".webp":
        if webp_lossless:
            img.webpsave(path, lossless=True)
        else:
            img.webpsave(path, Q=q)

    elif ext == ".avif":
        try:
            img.heifsave(path, Q=q, compression="av1")
        except AttributeError as e:
            raise RuntimeError(
                "AVIF output requested, but this libvips build "
                "does not support heif/avif (heifsave missing)."
            ) from e

    elif ext in (".tiff", ".tif"):
        img.tiffsave(path, compression="lzw")

    else:
        img.pngsave(path, compression=6)
