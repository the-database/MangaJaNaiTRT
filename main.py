import argparse
import os
import tempfile
import threading
import time
import warnings
from configparser import ConfigParser
from pathlib import Path
from queue import Queue

warnings.filterwarnings("ignore", message="CUDA path could not be detected")

import numpy as np
import pyvips
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from mangajanaitrt.console import console, dbg
from mangajanaitrt.img import is_url, load_image, save_image, get_output_path, filter_existing_outputs, collect_input_files
from mangajanaitrt.trt_upscaler import TensorRTUpscaler
from mangajanaitrt.vram_monitor import VRAMMonitor


def parse_hw_tuple(s: str) -> tuple[int, int]:
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 2:
        raise ValueError(f"Invalid H,W tuple: {s!r}")
    return int(parts[0]), int(parts[1])


def load_config(config_path: str | Path = "config.ini"):
    cfg = ConfigParser(interpolation=None)
    read_files = cfg.read(config_path, encoding="utf-8")
    if not read_files:
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # [paths]
    input_path = cfg.get("paths", "input_path")
    input_onnx = cfg.get("paths", "input_onnx")
    output_dir = cfg.get("paths", "output_dir")

    engine_cache_dir_raw = cfg.get("paths", "engine_cache_dir", fallback="").strip()
    engine_cache_dir: str | None
    if not engine_cache_dir_raw or engine_cache_dir_raw.lower() in {"none", "null"}:
        engine_cache_dir = None
    else:
        engine_cache_dir = engine_cache_dir_raw

    # [tiling]
    dynamic_shape_min = parse_hw_tuple(cfg.get("tiling", "dynamic_shape_min"))
    dynamic_shape_opt = parse_hw_tuple(cfg.get("tiling", "dynamic_shape_opt"))
    dynamic_shape_max = parse_hw_tuple(cfg.get("tiling", "dynamic_shape_max"))
    tile_align = cfg.getint("tiling", "tile_align", fallback=16)
    tile_overlap = cfg.getint("tiling", "tile_overlap", fallback=16)

    # [trt]
    use_fp16 = cfg.getboolean("trt", "use_fp16", fallback=False)
    use_bf16 = cfg.getboolean("trt", "use_bf16", fallback=True)
    use_strong_types = cfg.getboolean("trt", "use_strong_types", fallback=False)
    batch_size = cfg.getint("trt", "batch_size", fallback=1)
    device_id = cfg.getint("trt", "device_id", fallback=0)
    trt_workspace_gb = cfg.getint("trt", "workspace_gb", fallback=4)
    trt_opt_level = cfg.getint("trt", "opt_level", fallback=3)

    # [runtime]
    prefetch_images = cfg.getint("runtime", "prefetch_images", fallback=1)
    save_queue_maxsize = cfg.getint("runtime", "save_queue_maxsize", fallback=4)
    num_save_threads = cfg.getint("runtime", "num_save_threads", fallback=2)

    # [output]
    if cfg.has_section("output"):
        output_format = cfg.get("output", "format", fallback="png").strip().lower()
        note = cfg.get("output", "note", fallback="mangajanai")
        quality = cfg.getint("output", "quality", fallback=95)
        webp_lossless = cfg.getboolean("output", "webp_lossless", fallback=False)
        skip_existing = cfg.getboolean("output", "skip_existing", fallback=False)
    else:
        output_format = "png"
        note = "mangajanai"
        quality = 95
        webp_lossless = False
        skip_existing = False

    if output_format == "jpeg":
        output_format = "jpg"

    valid_formats = {"png", "jpg", "webp", "avif"}
    if output_format not in valid_formats:
        raise ValueError(
            f"Invalid output format {output_format!r}; must be one of {sorted(valid_formats)}"
        )

    return {
        "INPUT_PATH": input_path,
        "INPUT_ONNX": input_onnx,
        "OUTPUT_DIR": output_dir,
        "ENGINE_CACHE_DIR": engine_cache_dir,
        "DYNAMIC_SHAPE_MIN": dynamic_shape_min,
        "DYNAMIC_SHAPE_OPT": dynamic_shape_opt,
        "DYNAMIC_SHAPE_MAX": dynamic_shape_max,
        "TILE_ALIGN": tile_align,
        "TILE_OVERLAP": tile_overlap,
        "USE_FP16": use_fp16,
        "USE_BF16": use_bf16,
        "USE_STRONG_TYPES": use_strong_types,
        "BATCH_SIZE": batch_size,
        "DEVICE_ID": device_id,
        "TRT_WORKSPACE_GB": trt_workspace_gb,
        "TRT_OPT_LEVEL": trt_opt_level,
        "NOTE": note,
        "PREFETCH_IMAGES": prefetch_images,
        "SAVE_QUEUE_MAXSIZE": save_queue_maxsize,
        "NUM_SAVE_THREADS": num_save_threads,
        "OUTPUT_FORMAT": output_format,
        "QUALITY": quality,
        "WEBP_LOSSLESS": webp_lossless,
        "SKIP_EXISTING": skip_existing,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TensorRT upscaler front-end with INI config."
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="config.ini",
        help="Path to config INI file (default: config.ini)",
    )
    return parser.parse_args()


def format_hms(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(seconds, 60.0)
    if m < 60:
        return f"{int(m)}m {s:04.1f}s"
    h, m = divmod(m, 60.0)
    return f"{int(h)}h {int(m):02d}m {s:04.1f}s"


def filename_for_bar(name: str, width: int = 28) -> str:
    if len(name) > width:
        keep = width - 3
        head = keep // 2
        tail = keep - head
        return name[:head] + "..." + name[-tail:]
    return name.ljust(width)


def make_progress() -> Progress:
    return Progress(
        TextColumn("{task.percentage:>5.1f}%"),
        BarColumn(bar_width=24),
        MofNCompleteColumn(),
        TextColumn("Elapsed:"),
        TimeElapsedColumn(),
        TextColumn(" ETA:"),
        TimeRemainingColumn(),
        TextColumn(" • {task.fields[filename]}"),
        console=console,
        refresh_per_second=5,
        transient=False,
    )


def upscale_alpha(alpha: np.ndarray, scale: int) -> np.ndarray:
    h, w = alpha.shape
    img = pyvips.Image.new_from_memory(alpha.tobytes(), w, h, 1, "uchar")
    # Use bicubic (cubic) interpolation for smooth alpha edges
    img = img.resize(scale, kernel="cubic")
    return np.ndarray(
        buffer=img.write_to_memory(), dtype=np.uint8, shape=[img.height, img.width]
    )


def image_writer_thread(
        q: "Queue[tuple[np.ndarray, np.ndarray | None, str] | None]",
        quality: int,
        webp_lossless: bool,
        scale: int,
) -> None:
    while True:
        item = q.get()
        if item is None:
            q.put(None)
            break
        rgb, alpha, out_path = item
        if alpha is not None:
            result_alpha = upscale_alpha(alpha, scale)
            out_arr = np.dstack([rgb, result_alpha])
        else:
            out_arr = rgb
        save_image(out_arr, out_path, quality=quality, webp_lossless=webp_lossless)


def image_loader_thread(
        files: list[str],
        job_q: "Queue[tuple[str, np.ndarray, np.ndarray | None] | None]",
) -> None:
    for path in files:
        try:
            rgb, alpha = load_image(path)
            job_q.put((path, rgb, alpha))
        except Exception as e:
            console.print(f"[red]Failed to load {path}: {e}[/]")
    job_q.put(None)


def main() -> None:
    args = parse_args()
    total_t0 = time.perf_counter()

    cfg = load_config(args.config)
    vram_monitor = VRAMMonitor(device_id=cfg["DEVICE_ID"])
    vram_monitor.start()

    if not os.path.isfile(cfg["INPUT_ONNX"]):
        raise FileNotFoundError(f"ONNX not found: {cfg['INPUT_ONNX']}")

    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".avif"}

    # Create temp directory for URL downloads if needed
    temp_dir = None
    input_is_url = is_url(cfg["INPUT_PATH"])

    if input_is_url:
        temp_dir = tempfile.mkdtemp(prefix="trt_upscale_")
        console.print(f"[dim]Using temp directory: {temp_dir}[/]")

    try:
        files = collect_input_files(cfg["INPUT_PATH"], exts, temp_dir)
        if not files:
            raise RuntimeError(f"No images in {cfg['INPUT_PATH']}")

        os.makedirs(cfg["OUTPUT_DIR"], exist_ok=True)

        total_found = len(files)
        console.print(f"[bold]Found {total_found} image(s)[/]")

        # Filter out files with existing outputs if skip_existing is enabled
        skipped_count = 0
        if cfg["SKIP_EXISTING"]:
            files, skipped_count = filter_existing_outputs(
                files, cfg["OUTPUT_DIR"], cfg["NOTE"], cfg["OUTPUT_FORMAT"]
            )
            if skipped_count > 0:
                console.print(
                    f"[yellow]Skipping {skipped_count} file(s) with existing outputs[/]"
                )
            if not files:
                console.print("[green]All outputs already exist, nothing to do.[/]")
                vram_monitor.stop()
                return

        console.print("\n[bold]Initializing TensorRT...[/]")
        upscaler = TensorRTUpscaler(
            onnx_path=cfg["INPUT_ONNX"],
            batch_size=cfg["BATCH_SIZE"],
            use_fp16=cfg["USE_FP16"],
            use_bf16=cfg["USE_BF16"],
            device_id=cfg["DEVICE_ID"],
            engine_cache_dir=cfg["ENGINE_CACHE_DIR"],
            shape_min=cfg["DYNAMIC_SHAPE_MIN"],
            shape_opt=cfg["DYNAMIC_SHAPE_OPT"],
            shape_max=cfg["DYNAMIC_SHAPE_MAX"],
            tile_align=cfg["TILE_ALIGN"],
            builder_opt_level=cfg["TRT_OPT_LEVEL"],
            trt_workspace_gb=cfg["TRT_WORKSPACE_GB"],
            use_strong_types=cfg["USE_STRONG_TYPES"],
        )

        num_savers = cfg["NUM_SAVE_THREADS"]
        save_q: Queue[tuple[np.ndarray, np.ndarray | None, str] | None] = Queue(
            maxsize=cfg["SAVE_QUEUE_MAXSIZE"]
        )
        writers = []
        for _ in range(num_savers):
            w = threading.Thread(
                target=image_writer_thread,
                args=(save_q, cfg["QUALITY"], cfg["WEBP_LOSSLESS"], upscaler.scale),
                daemon=True,
            )
            w.start()
            writers.append(w)

        job_q: Queue[tuple[str, np.ndarray, np.ndarray | None] | None] = Queue(
            maxsize=cfg["PREFETCH_IMAGES"]
        )
        loader = threading.Thread(
            target=image_loader_thread, args=(files, job_q), daemon=True
        )
        loader.start()

        total = len(files)
        progress = make_progress()
        console.print(
            f"\n[bold]Processing {total} images with {num_savers} saver thread(s)...[/]\n"
        )

        with progress:
            task_id = progress.add_task("upscaling", total=total, filename="")
            done = 0
            while done < total:
                item = job_q.get()
                if item is None:
                    break
                path, rgb, alpha = item
                progress.update(
                    task_id, filename=filename_for_bar(os.path.basename(path))
                )

                t0 = time.perf_counter()
                result_rgb = upscaler.upscale_image(rgb, overlap=cfg["TILE_OVERLAP"])

                if alpha is not None:
                    result_alpha = upscale_alpha(alpha, upscaler.scale)
                    result = np.dstack([result_rgb, result_alpha])
                else:
                    result = result_rgb

                dbg(f"{os.path.basename(path)}: {time.perf_counter() - t0:.2f}s")

                out_path = get_output_path(
                    path, cfg["OUTPUT_DIR"], cfg["NOTE"], cfg["OUTPUT_FORMAT"]
                )

                save_q.put((result_rgb, alpha, out_path))
                progress.update(task_id, advance=1)
                done += 1

        save_q.put(None)
        for w in writers:
            w.join()
        loader.join(timeout=1.0)

    finally:
        # Cleanup temp directory if we created one
        if temp_dir is not None:
            import shutil

            try:
                shutil.rmtree(temp_dir)
                console.print("[dim]Cleaned up temp directory[/]")
            except Exception as e:
                console.print(f"[yellow]Warning: Failed to cleanup temp dir: {e}[/]")

    vram_stats = None
    if vram_monitor is not None and vram_monitor.enabled:
        vram_stats = vram_monitor.stop()

    total_runtime = time.perf_counter() - total_t0
    console.print("\n[bold]===== Summary =====[/]")
    console.print(f"Images processed : {total}")
    if skipped_count > 0:
        console.print(f"Images skipped   : {skipped_count}")
    console.print(f"Total run time   : {format_hms(total_runtime)}")
    if total > 0:
        console.print(f"Average time     : {total_runtime / total:.2f}s / image")
        console.print(
            f"Throughput       : {total / (total_runtime / 60.0):.1f} images / minute"
        )

    if vram_stats:
        console.print(f"VRAM baseline    : {vram_stats['base_used_gib']:.2f} GiB")
        console.print(f"VRAM peak (abs)  : {vram_stats['peak_abs_gib']:.2f} GiB")
        console.print(f"VRAM peak (delta): {vram_stats['peak_delta_gib']:.2f} GiB")


if __name__ == "__main__":
    main()