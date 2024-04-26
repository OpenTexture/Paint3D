import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, List


@dataclass
class RenderConfig:
    """ Parameters for the Mesh Renderer """
    grid_size: int = 2048
    radius: float = 1.5
    look_at_height = 0.25
    base_theta: float = 60
    # Suzanne
    fov_para: float = np.pi / 3 * 0.9  # 0.61 or 0.8 for Orthographic ; np.pi / 3 for Pinhole
    remove_mesh_part_names: List[str] = field(default_factory=["MI_CH_Top"].copy)
    remove_unsupported_buffers: List[str] = field(default_factory=["filamat"].copy)
    n_views: int = 24  # 16
    # Additional views to use before rotating around shape
    views_before: List[Tuple[float, float]] = field(default_factory=list)
    # Additional views to use after rotating around shape
    views_after: List[Tuple[float, float]] = field(default_factory=[[180, 30], [180, 150]].copy)
    # Whether to alternate between the rotating views from the different sides
    alternate_views: bool = True
    calcu_uncolored_mode: str = "WarpGrid"  # FACE_ID, DIFF, WarpGrid
    projection_mode: str = "Pinhole"  # Pinhole, Orthographic
    texture_interpolation_mode: str = 'bilinear'
    texture_default_color: List[float] = field(default_factory=[0.8, 0.1, 0.8].copy)
    texturify_blend_alpha: float = 1.0
    render_angle_thres: float = 68
    # Suzanne
    views_init: List[float] = field(default_factory=[0, 23].copy)
    views_inpaint: List[Tuple[float, float]] = field(default_factory=[(5, 6), (24, 25)].copy)


@dataclass
class GuideConfig:
    """ Parameters defining the guidance """
    shape_path: str = "xxx"
    # init texture map
    initial_texture: Path = None
    texture_resolution: List[int] = field(default_factory=[1024, 1024].copy)  # h w
    # Append direction to text prompts
    append_direction: bool = True
    # mesh在归一化后缩放的比例
    shape_scale: float = 0.6
    # Threshold for defining refine regions
    z_update_thr: float = 0.2
    # Some more strict masking for projecting back
    strict_projection: bool = True
    # force run xatlas for mesh
    force_run_xatlas: bool = False


@dataclass
class OptimConfig:
    """ Parameters for the optimization process """
    seed: int = 0
    lr: float = 1e-2
    train_step: int = 200


@dataclass
class LogConfig:
    exp_path = "xxx"
    full_eval_size: int = 100
    cache_path: str = "paint3d_cache"


@dataclass
class TrainConfig:
    """ The main configuration for the coach trainer """
    log: LogConfig = field(default_factory=LogConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    guide: GuideConfig = field(default_factory=GuideConfig)


