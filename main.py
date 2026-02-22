import argparse
import os
import tempfile
import threading
import time
import warnings
from configparser import ConfigParser
from pathlib import Path
from queue import Queue, Empty

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
from mangajanaitrt.img import (
    is_url,
    load_image,
    save_image,
    get_output_path,
    filter_existing_outputs,
    collect_input_files,
)
from mangajanaitrt.trt_upscaler import TensorRTUpscaler
from mangajanaitrt.vram_monitor import MultiGPUVRAMMonitor


def parse_hw_tuple(s: str) -> tuple[int, int]:
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 2:
        raise ValueError(f"Invalid H,W tuple: {s!r}")
    return int(parts[0]), int(parts[1])


def parse_device_ids(s: str) -> list[int]:
    """Parse comma-separated device IDs like '0,1,2,3' or just '0'."""
    parts = [p.strip() for p in s.split(",")]
    return [int(p) for p in parts if p]


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
    trt_workspace_gb = cfg.getint("trt", "workspace_gb", fallback=4)
    trt_opt_level = cfg.getint("trt", "opt_level", fallback=3)

    # Multi-GPU: parse device_ids as comma-separated list
    device_ids_raw = cfg.get("trt", "device_id", fallback="0")
    device_ids = parse_device_ids(device_ids_raw)

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
        "DEVICE_IDS": device_ids,
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
        description="Multi-GPU TensorRT upscaler with INI config."
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
        TextColumn(" • {task.fields[status]}"),
        console=console,
        refresh_per_second=5,
        transient=False,
    )


def upscale_alpha(alpha: np.ndarray, scale: int) -> np.ndarray:
    h, w = alpha.shape
    img = pyvips.Image.new_from_memory(alpha.tobytes(), w, h, 1, "uchar")
    img = img.resize(scale, kernel="cubic")
    return np.ndarray(
        buffer=img.write_to_memory(), dtype=np.uint8, shape=[img.height, img.width]
    )


def image_writer_thread(
    save_queue: "Queue[tuple[np.ndarray, np.ndarray | None, str, int] | None]",
    quality: int,
    webp_lossless: bool,
    scale: int,
) -> None:
    """Writer thread that saves completed images to disk."""
    while True:
        item = save_queue.get()
        if item is None:
            save_queue.put(None)  # propagate sentinel
            break
        rgb, alpha, out_path, _scale = item
        if alpha is not None:
            result_alpha = upscale_alpha(alpha, scale)
            out_arr = np.dstack([rgb, result_alpha])
        else:
            out_arr = rgb
        save_image(out_arr, out_path, quality=quality, webp_lossless=webp_lossless)


def image_loader_thread(
    files: list[str],
    job_queue: "Queue[tuple[str, np.ndarray, np.ndarray | None] | None]",
    num_gpus: int,
) -> None:
    """Loader thread that reads images and feeds them to the job queue."""
    for path in files:
        try:
            rgb, alpha = load_image(path)
            job_queue.put((path, rgb, alpha))
        except Exception as e:
            console.print(f"[red]Failed to load {path}: {e}[/]")

    # Send sentinel for each GPU worker
    for _ in range(num_gpus):
        job_queue.put(None)


def gpu_worker_thread(
    device_id: int,
    job_queue: "Queue[tuple[str, np.ndarray, np.ndarray | None] | None]",
    result_queue: "Queue[tuple[str, np.ndarray, np.ndarray | None, int] | None]",
    cfg: dict,
    ready_event: threading.Event,
    init_error: list,
    engine_build_lock: threading.Lock,
) -> None:
    """
    GPU worker thread. Each thread:
    1. Initializes its own TensorRT engine on the assigned GPU
    2. Pulls images from the shared job queue
    3. Pushes results to the result queue
    """
    try:
        # TensorRT builder is not thread-safe - serialize engine builds
        with engine_build_lock:
            console.print(f"  [dim]GPU {device_id}: initializing...[/]")
            upscaler = TensorRTUpscaler(
                onnx_path=cfg["INPUT_ONNX"],
                batch_size=cfg["BATCH_SIZE"],
                use_fp16=cfg["USE_FP16"],
                use_bf16=cfg["USE_BF16"],
                device_id=device_id,
                engine_cache_dir=cfg["ENGINE_CACHE_DIR"],
                shape_min=cfg["DYNAMIC_SHAPE_MIN"],
                shape_opt=cfg["DYNAMIC_SHAPE_OPT"],
                shape_max=cfg["DYNAMIC_SHAPE_MAX"],
                tile_align=cfg["TILE_ALIGN"],
                builder_opt_level=cfg["TRT_OPT_LEVEL"],
                trt_workspace_gb=cfg["TRT_WORKSPACE_GB"],
                use_strong_types=cfg["USE_STRONG_TYPES"],
            )
        scale = upscaler.scale
        ready_event.set()
    except Exception as e:
        init_error.append((device_id, e))
        ready_event.set()
        return

    overlap = cfg["TILE_OVERLAP"]

    while True:
        try:
            item = job_queue.get(timeout=0.5)
        except Empty:
            continue

        if item is None:
            break

        path, rgb, alpha = item

        try:
            t0 = time.perf_counter()
            result_rgb = upscaler.upscale_image(rgb, overlap=overlap)
            dbg(f"[GPU {device_id}] {os.path.basename(path)}: {time.perf_counter() - t0:.2f}s")

            result_queue.put((path, result_rgb, alpha, scale))
        except Exception as e:
            console.print(f"[red][GPU {device_id}] Failed to upscale {path}: {e}[/]")


def result_collector_thread(
    result_queue: "Queue[tuple[str, np.ndarray, np.ndarray | None, int] | None]",
    save_queue: "Queue[tuple[np.ndarray, np.ndarray | None, str, int] | None]",
    cfg: dict,
    progress: Progress,
    task_id,
    total: int,
    num_gpus: int,
    done_counter: list,
) -> None:
    """
    Collector thread that:
    1. Receives results from GPU workers
    2. Updates progress bar
    3. Forwards to save queue
    """
    done = 0
    active_gpus = num_gpus

    while done < total and active_gpus > 0:
        try:
            item = result_queue.get(timeout=0.5)
        except Empty:
            continue

        if item is None:
            active_gpus -= 1
            continue

        path, result_rgb, alpha, scale = item

        out_path = get_output_path(
            path, cfg["OUTPUT_DIR"], cfg["NOTE"], cfg["OUTPUT_FORMAT"]
        )

        save_queue.put((result_rgb, alpha, out_path, scale))

        done += 1
        done_counter[0] = done
        progress.update(
            task_id,
            advance=1,
            status=f"GPU×{num_gpus} | {filename_for_bar(os.path.basename(path), 20)}",
        )


def main() -> None:
    args = parse_args()
    total_t0 = time.perf_counter()

    cfg = load_config(args.config)
    device_ids = cfg["DEVICE_IDS"]
    num_gpus = len(device_ids)

    console.print(f"[bold cyan]Multi-GPU mode: {num_gpus} GPU(s) - devices {device_ids}[/]")

    # Initialize multi-GPU VRAM monitor
    vram_monitor = MultiGPUVRAMMonitor(device_ids=device_ids)
    vram_monitor.start()

    if not os.path.isfile(cfg["INPUT_ONNX"]):
        raise FileNotFoundError(f"ONNX not found: {cfg['INPUT_ONNX']}")

    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".avif"}

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

        total = len(files)

        # === Initialize GPU workers ===
        console.print(f"\n[bold]Initializing TensorRT on {num_gpus} GPU(s)...[/]")

        # Shared queues
        # Job queue: loader -> GPU workers (prefetch per GPU for good saturation)
        job_queue: Queue[tuple[str, np.ndarray, np.ndarray | None] | None] = Queue(
            maxsize=cfg["PREFETCH_IMAGES"] * num_gpus
        )
        # Result queue: GPU workers -> collector
        result_queue: Queue[tuple[str, np.ndarray, np.ndarray | None, int] | None] = Queue(
            maxsize=num_gpus * 2
        )
        # Save queue: collector -> writers
        save_queue: Queue[tuple[np.ndarray, np.ndarray | None, str, int] | None] = Queue(
            maxsize=cfg["SAVE_QUEUE_MAXSIZE"]
        )

        # TensorRT builder is not thread-safe, so serialize engine builds
        engine_build_lock = threading.Lock()

        # Start GPU workers
        gpu_workers = []
        ready_events = []
        init_errors: list[tuple[int, Exception]] = []

        for dev_id in device_ids:
            ready = threading.Event()
            ready_events.append(ready)
            worker = threading.Thread(
                target=gpu_worker_thread,
                args=(dev_id, job_queue, result_queue, cfg, ready, init_errors, engine_build_lock),
                daemon=True,
            )
            worker.start()
            gpu_workers.append(worker)

        # Wait for all GPUs to initialize
        for i, ready in enumerate(ready_events):
            ready.wait()
            if not init_errors:
                console.print(f"  [green]GPU {device_ids[i]} ready[/]")

        if init_errors:
            for dev_id, err in init_errors:
                console.print(f"[red]GPU {dev_id} init failed: {err}[/]")
            raise RuntimeError("One or more GPUs failed to initialize")

        # Get scale from first worker (they're all identical)
        # We need to query it - for now assume scale is determined, use a workaround
        # by checking the engine. We'll pass scale through result queue items.

        # Start image loader
        loader = threading.Thread(
            target=image_loader_thread,
            args=(files, job_queue, num_gpus),
            daemon=True,
        )
        loader.start()

        # Start save threads (scale them with GPU count)
        num_savers = max(cfg["NUM_SAVE_THREADS"], num_gpus)
        writers = []

        # We need the scale - peek it from first result or use a placeholder
        # Actually, we pass scale in each result item, so writers get it per-image
        # But upscale_alpha needs scale. Let's create a modified writer that uses per-item scale.

        # For simplicity, we'll determine scale after first GPU initializes
        # The scale is in the result tuple, so writers use it from there.

        # Modified writer that accepts scale per item:
        def image_writer_thread_v2(
            sq: "Queue[tuple[np.ndarray, np.ndarray | None, str, int] | None]",
            quality: int,
            webp_lossless: bool,
        ) -> None:
            while True:
                item = sq.get()
                if item is None:
                    sq.put(None)
                    break
                rgb, alpha, out_path, scale = item
                if alpha is not None:
                    result_alpha = upscale_alpha(alpha, scale)
                    out_arr = np.dstack([rgb, result_alpha])
                else:
                    out_arr = rgb
                save_image(out_arr, out_path, quality=quality, webp_lossless=webp_lossless)

        for _ in range(num_savers):
            w = threading.Thread(
                target=image_writer_thread_v2,
                args=(save_queue, cfg["QUALITY"], cfg["WEBP_LOSSLESS"]),
                daemon=True,
            )
            w.start()
            writers.append(w)

        # Progress and collection
        progress = make_progress()
        done_counter = [0]

        console.print(
            f"\n[bold]Processing {total} images with {num_gpus} GPU(s) and {num_savers} saver thread(s)...[/]\n"
        )

        with progress:
            task_id = progress.add_task("upscaling", total=total, status="starting...")

            collector = threading.Thread(
                target=result_collector_thread,
                args=(
                    result_queue,
                    save_queue,
                    cfg,
                    progress,
                    task_id,
                    total,
                    num_gpus,
                    done_counter,
                ),
                daemon=True,
            )
            collector.start()

            # Wait for collection to complete
            collector.join()

        # Signal writers to stop
        save_queue.put(None)
        for w in writers:
            w.join()

        # Wait for GPU workers (they should have exited after getting None sentinels)
        for worker in gpu_workers:
            worker.join(timeout=2.0)

        loader.join(timeout=1.0)

    finally:
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
    console.print(f"GPUs used        : {num_gpus} ({device_ids})")
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
        console.print("\n[bold]VRAM Usage per GPU:[/]")
        for dev_id, stats in vram_stats.items():
            console.print(
                f"  GPU {dev_id}: baseline {stats['base_used_gib']:.2f} GiB, "
                f"peak {stats['peak_abs_gib']:.2f} GiB, "
                f"delta {stats['peak_delta_gib']:.2f} GiB"
            )


if __name__ == "__main__":
    main()