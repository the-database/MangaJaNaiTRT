import os
from pathlib import Path

import numpy as np

from mangajanaitrt.console import console, dbg
from mangajanaitrt.tile_info import TileInfo


class TensorRTUpscaler:
    def __init__(
        self,
        onnx_path: str,
        batch_size: int = 1,
        use_fp16: bool = True,
        use_bf16: bool = False,
        device_id: int = 0,
        engine_cache_dir: str | None = None,
        # (width, height)
        shape_min: tuple[int, int] = (64, 64),
        shape_opt: tuple[int, int] = (256, 256),
        shape_max: tuple[int, int] = (512, 512),
        tile_align: int = 1,
        builder_opt_level: int = 3,
        trt_workspace_gb: int = 4,
    ) -> None:
        try:
            import tensorrt as trt
        except ImportError:
            raise ImportError(
                "TensorRT not found. Install with:\n  pip install tensorrt\n"
            )

        try:
            import cupy as cp
        except ImportError:
            raise ImportError(
                "CuPy not found. Install with:\n  pip install cupy-cuda12x\n"
            )

        self.trt = trt
        self.cp = cp

        self.onnx_path = onnx_path
        self.batch_size = batch_size
        self.use_fp16 = use_fp16
        self.use_bf16 = use_bf16
        self.device_id = device_id
        self.engine_cache_dir = engine_cache_dir or os.path.dirname(onnx_path)

        self.shape_min: tuple[int, int] = shape_min
        self.shape_opt: tuple[int, int] = shape_opt
        self.shape_max: tuple[int, int] = shape_max

        self.tile_align = tile_align
        self.builder_opt_level = builder_opt_level
        self.trt_workspace_gb = trt_workspace_gb

        self._validate_dynamic_shapes()

        self.is_dynamic = (
            self.shape_min != self.shape_opt or self.shape_opt != self.shape_max
        )

        cp.cuda.Device(device_id).use()

        self.logger = trt.Logger(trt.Logger.WARNING)
        self.engine = self._get_engine()

        self.input_name = None
        self.output_name = None
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_name = name
            else:
                self.output_name = name

        self.context = self.engine.create_execution_context()

        opt_w, opt_h = self.shape_opt
        input_shape = (batch_size, 3, opt_h, opt_w)
        if not self.context.set_input_shape(self.input_name, input_shape):
            raise RuntimeError(
                f"Failed to set input shape {input_shape} for {self.input_name}"
            )

        out_shape = self.context.get_tensor_shape(
            self.output_name
        )
        if len(out_shape) != 4:
            raise RuntimeError(
                f"Unexpected output shape {out_shape} for {self.output_name}"
            )

        in_h, in_w = input_shape[2], input_shape[3]
        out_h, out_w = out_shape[2], out_shape[3]

        if out_h % in_h != 0 or out_w % in_w != 0:
            raise RuntimeError(
                f"Cannot infer integer scale: input ({in_h}, {in_w}) -> output ({out_h}, {out_w})"
            )

        inferred_scale_h = out_h // in_h
        inferred_scale_w = out_w // in_w

        if inferred_scale_h != inferred_scale_w:
            raise RuntimeError(
                f"Asymmetric scale not supported: H scale {inferred_scale_h}, W scale {inferred_scale_w}"
            )

        self.scale = inferred_scale_h

        self._current_shape = (opt_h, opt_w)

        self.stream = cp.cuda.Stream(non_blocking=True)

        console.print("[green]TensorRT engine loaded[/]")
        engine_type = "dynamic" if self.is_dynamic else "static"
        console.print(f"  Engine type: {engine_type}")
        console.print(
            f"  Shape range (WxH): {self.shape_min} -> {self.shape_opt} -> {self.shape_max}"
        )
        console.print(f"  Scale: {self.scale}x, Align: {self.tile_align}")

    def _validate_dynamic_shapes(self) -> None:
        min_w, min_h = self.shape_min
        opt_w, opt_h = self.shape_opt
        max_w, max_h = self.shape_max

        if not (min_w <= opt_w <= max_w):
            raise ValueError(
                f"Invalid dynamic width range: min={min_w}, opt={opt_w}, max={max_w}. "
                f"Must satisfy min <= opt <= max."
            )
        if not (min_h <= opt_h <= max_h):
            raise ValueError(
                f"Invalid dynamic height range: min={min_h}, opt={opt_h}, max={max_h}. "
                f"Must satisfy min <= opt <= max."
            )

        align = self.tile_align
        if align > 1:
            for name, (w, h) in [
                ("min", self.shape_min),
                ("opt", self.shape_opt),
                ("max", self.shape_max),
            ]:
                if w % align != 0 or h % align != 0:
                    raise ValueError(
                        f"Dynamic shape {name}=({w}, {h}) not aligned to TILE_ALIGN={align}"
                    )

    def _get_engine_path(self) -> str:
        onnx_name = Path(self.onnx_path).stem
        precision = "bf16" if self.use_bf16 else ("fp16" if self.use_fp16 else "fp32")

        min_w, min_h = self.shape_min
        opt_w, opt_h = self.shape_opt
        max_w, max_h = self.shape_max

        if self.is_dynamic:
            shape_str = f"dyn_{min_w}x{min_h}_{opt_w}x{opt_h}_{max_w}x{max_h}"
        else:
            shape_str = f"{opt_w}x{opt_h}"

        opt_str = f"opt{self.builder_opt_level}"

        return os.path.join(
            self.engine_cache_dir,
            f"{onnx_name}_{shape_str}_b{self.batch_size}_{precision}_{opt_str}.engine",
        )

    def _get_engine(self):
        engine_path = self._get_engine_path()

        if os.path.exists(engine_path):
            console.print(f"[cyan]Loading cached engine: {Path(engine_path).name}[/]")
            with open(engine_path, "rb") as f:
                runtime = self.trt.Runtime(self.logger)
                return runtime.deserialize_cuda_engine(f.read())

        console.print(
            "[yellow]Building TensorRT engine (may take several minutes)...[/]"
        )
        engine = self._build_engine()

        os.makedirs(self.engine_cache_dir, exist_ok=True)
        with open(engine_path, "wb") as f:
            f.write(engine.serialize())
        console.print(f"[green]Engine cached: {Path(engine_path).name}[/]")

        return engine

    def _build_engine(self):
        trt = self.trt

        old_severity = self.logger.min_severity
        self.logger.min_severity = trt.Logger.INFO
        try:
            builder = trt.Builder(self.logger)
            network = builder.create_network(
                1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            )
            parser = trt.OnnxParser(network, self.logger)

            with open(self.onnx_path, "rb") as f:
                if not parser.parse(f.read()):
                    for i in range(parser.num_errors):
                        console.print(f"[red]Parse Error: {parser.get_error(i)}[/]")
                    raise RuntimeError("Failed to parse ONNX model")

            config = builder.create_builder_config()

            config.set_memory_pool_limit(
                trt.MemoryPoolType.WORKSPACE, self.trt_workspace_gb << 30
            )

            try:
                config.builder_optimization_level = self.builder_opt_level
                console.print(f"  Builder opt level: {self.builder_opt_level}")
            except AttributeError as e:
                console.print(e)
                console.print(
                    "[yellow]  Builder opt level not supported by this TRT version[/]"
                )

            if self.use_bf16 and builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.BF16)
                console.print("  Using BF16 precision")
            elif self.use_fp16 and builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
                console.print("  Using FP16 precision")
            else:
                console.print("  Using FP32 precision")

            profile = builder.create_optimization_profile()
            input_tensor = network.get_input(0)

            min_w, min_h = self.shape_min
            opt_w, opt_h = self.shape_opt
            max_w, max_h = self.shape_max

            min_shape = (self.batch_size, 3, min_h, min_w)
            opt_shape = (self.batch_size, 3, opt_h, opt_w)
            max_shape = (self.batch_size, 3, max_h, max_w)

            profile.set_shape(input_tensor.name, min_shape, opt_shape, max_shape)
            config.add_optimization_profile(profile)

            if self.is_dynamic:
                console.print("  Building dynamic engine:")
                console.print(f"    Min shape (NCHW): {min_shape}")
                console.print(f"    Opt shape (NCHW): {opt_shape}")
                console.print(f"    Max shape (NCHW): {max_shape}")
            else:
                console.print(f"  Building static engine for shape (NCHW): {opt_shape}")

            serialized = builder.build_serialized_network(network, config)
            if serialized is None:
                raise RuntimeError("Failed to build TensorRT engine")

            runtime = trt.Runtime(self.logger)
            return runtime.deserialize_cuda_engine(serialized)
        finally:
            self.logger.min_severity = old_severity

    def _extract_tile(self, img_float: np.ndarray, tile_info: TileInfo) -> np.ndarray:
        tile = img_float[
            tile_info.src_y : tile_info.src_y + tile_info.src_h,
            tile_info.src_x : tile_info.src_x + tile_info.src_w,
        ]

        if tile_info.pad_bottom > 0 or tile_info.pad_right > 0:
            tile = np.pad(
                tile,
                ((0, tile_info.pad_bottom), (0, tile_info.pad_right), (0, 0)),
                mode="reflect",
            )

        # HWC -> CHW
        return tile.transpose(2, 0, 1)

    def _set_input_shape(self, h: int, w: int) -> None:
        if (h, w) == self._current_shape:
            return

        input_shape = (self.batch_size, 3, h, w)
        if not self.context.set_input_shape(self.input_name, input_shape):
            raise RuntimeError(f"Failed to set input shape {input_shape}")

        self._current_shape = (h, w)

    def upscale_image(self, img: np.ndarray, overlap: int = 16) -> np.ndarray:
        cp = self.cp
        h, w = img.shape[:2]
        out_h, out_w = h * self.scale, w * self.scale

        tiles = compute_tiles(
            img_h=h,
            img_w=w,
            overlap=overlap,
            scale=self.scale,
            tile_align=self.tile_align,
            shape_min=self.shape_min,
            shape_opt=self.shape_opt,
            shape_max=self.shape_max,
        )
        num_tiles = len(tiles)

        dbg(f"Image {h}x{w} -> {out_h}x{out_w}, {num_tiles} tiles")

        img_float = img.astype(np.float32) / 255.0

        # single tile
        if num_tiles == 1:
            tile_info = tiles[0]
            infer_h, infer_w = tile_info.infer_h, tile_info.infer_w

            tile_data = self._extract_tile(
                img_float, tile_info
            )

            input_buf = cp.empty(
                (self.batch_size, 3, infer_h, infer_w), dtype=cp.float32
            )
            buf_out_h = infer_h * self.scale
            buf_out_w = infer_w * self.scale
            output_buf = cp.empty(
                (self.batch_size, 3, buf_out_h, buf_out_w), dtype=cp.float32
            )

            if self.is_dynamic:
                self._set_input_shape(infer_h, infer_w)

            input_buf[0] = cp.asarray(tile_data)

            self.context.set_tensor_address(self.input_name, input_buf.data.ptr)
            self.context.set_tensor_address(self.output_name, output_buf.data.ptr)
            self.context.execute_async_v3(self.stream.ptr)
            self.stream.synchronize()

            result_chw = cp.asnumpy(output_buf[0])
            result = result_chw.transpose(1, 2, 0)

            if tile_info.pad_bottom > 0:
                result = result[: -tile_info.pad_bottom * self.scale, :, :]
            if tile_info.pad_right > 0:
                result = result[:, : -tile_info.pad_right * self.scale, :]

            return np.clip(result * 255.0, 0, 255).astype(np.uint8)

        # multi tile

        output = np.zeros((out_h, out_w, 3), dtype=np.float32)
        weight_sum = np.zeros((out_h, out_w, 1), dtype=np.float32)

        blend_masks: dict = {}

        input_buffer_cache: dict[tuple[int, int], cp.ndarray] = {}
        output_buffer_cache: dict[tuple[int, int], cp.ndarray] = {}

        for _tile_idx, tile_info in enumerate(tiles):
            tile_data = self._extract_tile(img_float, tile_info)
            infer_h, infer_w = tile_info.infer_h, tile_info.infer_w
            shape_key = (infer_h, infer_w)

            if shape_key not in input_buffer_cache:
                input_buffer_cache[shape_key] = cp.empty(
                    (self.batch_size, 3, infer_h, infer_w), dtype=cp.float32
                )
                buf_out_h = infer_h * self.scale
                buf_out_w = infer_w * self.scale
                output_buffer_cache[shape_key] = cp.empty(
                    (self.batch_size, 3, buf_out_h, buf_out_w), dtype=cp.float32
                )

            input_buf = input_buffer_cache[shape_key]
            output_buf = output_buffer_cache[shape_key]

            if self.is_dynamic:
                self._set_input_shape(infer_h, infer_w)

            input_buf[0] = cp.asarray(tile_data)

            self.context.set_tensor_address(self.input_name, input_buf.data.ptr)
            self.context.set_tensor_address(self.output_name, output_buf.data.ptr)
            self.context.execute_async_v3(self.stream.ptr)
            self.stream.synchronize()

            result = cp.asnumpy(output_buf[0])

            self._accumulate_tile(result, tile_info, output, weight_sum, blend_masks)

        weight_sum = np.maximum(weight_sum, 1e-8)
        output = output / weight_sum

        return np.clip(output * 255.0, 0, 255).astype(np.uint8)

    def _accumulate_tile(
        self,
        result_chw: np.ndarray,
        tile_info: TileInfo,
        output: np.ndarray,
        weight_sum: np.ndarray,
        blend_masks: dict,
    ) -> None:
        result = result_chw.transpose(1, 2, 0)

        if tile_info.pad_bottom > 0:
            result = result[: -tile_info.pad_bottom * self.scale, :, :]
        if tile_info.pad_right > 0:
            result = result[:, : -tile_info.pad_right * self.scale, :]

        mask_key = (
            result.shape[0],
            result.shape[1],
            tile_info.blend_top,
            tile_info.blend_bottom,
            tile_info.blend_left,
            tile_info.blend_right,
        )
        if mask_key not in blend_masks:
            blend_masks[mask_key] = create_blend_mask(
                result.shape[0], result.shape[1], tile_info
            )[:, :, np.newaxis]

        mask = blend_masks[mask_key]

        y1, y2 = tile_info.dst_y, tile_info.dst_y + tile_info.dst_h
        x1, x2 = tile_info.dst_x, tile_info.dst_x + tile_info.dst_w

        output[y1:y2, x1:x2] += result * mask
        weight_sum[y1:y2, x1:x2] += mask


def create_blend_mask(h: int, w: int, tile_info: TileInfo) -> np.ndarray:
    mask = np.ones((h, w), dtype=np.float32)

    if tile_info.blend_top > 0:
        ramp = np.linspace(0, 1, tile_info.blend_top, dtype=np.float32)
        mask[: tile_info.blend_top, :] *= ramp[:, np.newaxis]

    if tile_info.blend_bottom > 0:
        ramp = np.linspace(1, 0, tile_info.blend_bottom, dtype=np.float32)
        mask[-tile_info.blend_bottom :, :] *= ramp[:, np.newaxis]

    if tile_info.blend_left > 0:
        ramp = np.linspace(0, 1, tile_info.blend_left, dtype=np.float32)
        mask[:, : tile_info.blend_left] *= ramp[np.newaxis, :]

    if tile_info.blend_right > 0:
        ramp = np.linspace(1, 0, tile_info.blend_right, dtype=np.float32)
        mask[:, -tile_info.blend_right :] *= ramp[np.newaxis, :]

    return mask


def compute_tiles(
    img_h: int,
    img_w: int,
    overlap: int,
    scale: int,
    tile_align: int,
    # (width, height)
    shape_min: tuple[int, int],
    shape_opt: tuple[int, int],
    shape_max: tuple[int, int],
) -> list[TileInfo]:
    min_w, min_h = shape_min
    opt_w, opt_h = shape_opt
    max_w, max_h = shape_max

    tile_h = compute_optimal_tile_size(img_h, min_h, max_h, opt_h)
    tile_w = compute_optimal_tile_size(img_w, min_w, max_w, opt_w)

    tiles = []
    step_h = tile_h - overlap
    step_w = tile_w - overlap

    n_tiles_h = max(1, (img_h - overlap + step_h - 1) // step_h)
    n_tiles_w = max(1, (img_w - overlap + step_w - 1) // step_w)

    for ty in range(n_tiles_h):
        for tx in range(n_tiles_w):
            src_y = min(ty * step_h, max(0, img_h - tile_h))
            src_x = min(tx * step_w, max(0, img_w - tile_w))

            actual_h = min(tile_h, img_h - src_y)
            actual_w = min(tile_w, img_w - src_x)

            infer_h = align_up(actual_h, tile_align)
            infer_w = align_up(actual_w, tile_align)

            infer_h = max(infer_h, min_h)
            infer_w = max(infer_w, min_w)

            tiles.append(
                TileInfo(
                    src_y=src_y,
                    src_x=src_x,
                    src_h=actual_h,
                    src_w=actual_w,
                    infer_h=infer_h,
                    infer_w=infer_w,
                    pad_bottom=infer_h - actual_h,
                    pad_right=infer_w - actual_w,
                    dst_y=src_y * scale,
                    dst_x=src_x * scale,
                    dst_h=actual_h * scale,
                    dst_w=actual_w * scale,
                    blend_top=(overlap // 2 if ty > 0 else 0) * scale,
                    blend_bottom=(overlap // 2 if ty < n_tiles_h - 1 else 0) * scale,
                    blend_left=(overlap // 2 if tx > 0 else 0) * scale,
                    blend_right=(overlap // 2 if tx < n_tiles_w - 1 else 0) * scale,
                )
            )

    return tiles


def compute_optimal_tile_size(
    img_size: int,
    min_size: int,
    max_size: int,
    align: int,
) -> int:
    if align > 1:
        aligned_max = align_down(max_size, align)
    else:
        aligned_max = max_size

    if img_size <= aligned_max:
        if align > 1:
            tile = align_up(img_size, align)
        else:
            tile = img_size

        tile = max(min_size, min(tile, aligned_max))
        return tile

    return max(min_size, aligned_max)


def align_up(x: int, align: int) -> int:
    if align <= 1:
        return x
    return ((x + align - 1) // align) * align


def align_down(x: int, align: int) -> int:
    if align <= 1:
        return x
    return (x // align) * align
