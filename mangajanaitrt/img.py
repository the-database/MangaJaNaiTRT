import os

import numpy as np
import pyvips


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
