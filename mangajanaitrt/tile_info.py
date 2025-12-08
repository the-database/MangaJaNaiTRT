from dataclasses import dataclass


@dataclass
class TileInfo:
    src_y: int
    src_x: int
    src_h: int
    src_w: int
    infer_h: int
    infer_w: int
    pad_bottom: int
    pad_right: int
    dst_y: int
    dst_x: int
    dst_h: int
    dst_w: int
    blend_top: int
    blend_bottom: int
    blend_left: int
    blend_right: int
