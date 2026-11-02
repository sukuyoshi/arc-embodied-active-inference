from __future__ import annotations

import math
import os
import sys
import types
import numpy as np
import torch

try:
    import cv2
except ModuleNotFoundError:
    sys.modules["cv2"] = types.ModuleType("cv2")

_GLOBAL_SIMULATION_APP = None


def ensure_isaaclab_simulation_app(headless: bool = True):
    global _GLOBAL_SIMULATION_APP
    if _GLOBAL_SIMULATION_APP is None and "omni.kit.app" not in sys.modules:
        from isaaclab.app import AppLauncher

        launcher = AppLauncher(headless=bool(headless))
        _GLOBAL_SIMULATION_APP = launcher.app
    return _GLOBAL_SIMULATION_APP


ensure_isaaclab_simulation_app(headless=True)

from isaac_warp_3d_engine import get_global_isaac_warp_engine
import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.terrains.height_field.utils import height_field_to_mesh
from isaaclab.utils import configclass
import isaaclab_tasks.manager_based.classic.humanoid.mdp as mdp
from isaaclab_assets.robots.humanoid import HUMANOID_CFG

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
if _MODULE_DIR not in sys.path:
    sys.path.insert(0, _MODULE_DIR)

from agi_homeostasis_curiosity_mdp import (
    CURIOSITY_NOVEL_BIOME_REWARD,
    compute_biome_quadrant_index,
    compute_local_and_arena_xy,
    compute_spinal_cpg_and_vestibular_joints_21d,
    ensure_homeostasis_curiosity_buffers,
    reset_homeostasis_and_curiosity,
    reward_biome_quadrant_curiosity,
    reward_homeostatic_vitality,
    termination_hp_depleted,
)
from agi_sensorimotor_puzzle_mdp import (
    Sensorimotor11DActionCfg,
    VocalSignal2DActionCfg,
    _quat_wxyz_to_yaw_rad,
    ensure_sensorimotor_puzzle_buffers,
    execute_single_env_toy_manipulation,
    obs_nearest_toy_puzzle_attributes_6d,
    obs_sensorimotor_puzzle_state_12d,
    reset_sensorimotor_puzzle_state,
    reward_toy_sensorimotor_puzzle,
)


def compute_gaussian_wave_height_numpy(
    x_coord: np.ndarray,
    y_coord: np.ndarray,
    terrain_shift_x: float = 0.0,
    peak1_center: tuple[float, float] = (30.0, 30.0),
    peak1_height: float = 4.5,
    peak1_variance: float = 280.0,
    peak2_center: tuple[float, float] = (70.0, 68.0),
    peak2_height: float = 5.2,
    peak2_variance: float = 320.0,
    abyss1_center: tuple[float, float] = (68.0, 22.0),
    abyss2_center: tuple[float, float] = (82.0, 32.0),
    abyss_depth: float = 2.4,
    abyss_variance: float = 10.0,
    wave_amplitude: float = 1.4,
    wave_frequency: float = 0.08,
    base_elevation: float = 1.5,
    min_height: float = 0.0,
    max_height: float = 12.0,
) -> np.ndarray:
    xs = x_coord + float(terrain_shift_x)
    peak1 = float(peak1_height) * np.exp(
        -((xs - float(peak1_center[0])) ** 2 + (y_coord - float(peak1_center[1])) ** 2)
        / float(peak1_variance)
    )
    peak2 = float(peak2_height) * np.exp(
        -((xs - float(peak2_center[0])) ** 2 + (y_coord - float(peak2_center[1])) ** 2)
        / float(peak2_variance)
    )
    sink1 = float(abyss_depth) * np.exp(
        -((xs - float(abyss1_center[0])) ** 2 + (y_coord - float(abyss1_center[1])) ** 2)
        / float(abyss_variance)
    )
    sink2 = float(abyss_depth) * np.exp(
        -((xs - float(abyss2_center[0])) ** 2 + (y_coord - float(abyss2_center[1])) ** 2)
        / float(abyss_variance)
    )
    wave = (
        float(wave_amplitude)
        * np.sin(float(wave_frequency) * xs)
        * np.cos(float(wave_frequency) * y_coord)
    )
    return np.clip(
        peak1 + peak2 - sink1 - sink2 + wave + float(base_elevation),
        float(min_height),
        float(max_height),
    )


def compute_gaussian_wave_height_torch(
    x_coord: torch.Tensor,
    y_coord: torch.Tensor,
    terrain_shift_x: float = 0.0,
    peak1_center: tuple[float, float] = (30.0, 30.0),
    peak1_height: float = 4.5,
    peak1_variance: float = 280.0,
    peak2_center: tuple[float, float] = (70.0, 68.0),
    peak2_height: float = 5.2,
    peak2_variance: float = 320.0,
    abyss1_center: tuple[float, float] = (68.0, 22.0),
    abyss2_center: tuple[float, float] = (82.0, 32.0),
    abyss_depth: float = 2.4,
    abyss_variance: float = 10.0,
    wave_amplitude: float = 1.4,
    wave_frequency: float = 0.08,
    base_elevation: float = 1.5,
    min_height: float = 0.0,
    max_height: float = 12.0,
) -> torch.Tensor:
    xs = x_coord + float(terrain_shift_x)
    peak1 = float(peak1_height) * torch.exp(
        -((xs - float(peak1_center[0])) ** 2 + (y_coord - float(peak1_center[1])) ** 2)
        / float(peak1_variance)
    )
    peak2 = float(peak2_height) * torch.exp(
        -((xs - float(peak2_center[0])) ** 2 + (y_coord - float(peak2_center[1])) ** 2)
        / float(peak2_variance)
    )
    sink1 = float(abyss_depth) * torch.exp(
        -((xs - float(abyss1_center[0])) ** 2 + (y_coord - float(abyss1_center[1])) ** 2)
        / float(abyss_variance)
    )
    sink2 = float(abyss_depth) * torch.exp(
        -((xs - float(abyss2_center[0])) ** 2 + (y_coord - float(abyss2_center[1])) ** 2)
        / float(abyss_variance)
    )
    wave = (
        float(wave_amplitude)
        * torch.sin(float(wave_frequency) * xs)
        * torch.cos(float(wave_frequency) * y_coord)
    )
    return torch.clamp(
        peak1 + peak2 - sink1 - sink2 + wave + float(base_elevation),
        float(min_height),
        float(max_height),
    )


@height_field_to_mesh
def gaussian_hills_wave_terrain(difficulty: float, cfg: HfGaussianHillsWaveTerrainCfg) -> np.ndarray:
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    x_local = np.linspace(0.0, float(cfg.size[0]), width_pixels, dtype=np.float32)
    y_local = np.linspace(0.0, float(cfg.size[1]), length_pixels, dtype=np.float32)

    row_offset_x = 50.0 if (cfg.auto_row_offset_from_difficulty and float(difficulty) >= 0.5) else float(cfg.quadrant_offset[0])
    col_offset_y = float(cfg.quadrant_offset[1])

    x_arena = x_local + row_offset_x
    y_arena = y_local + col_offset_y
    xx, yy = np.meshgrid(x_arena, y_arena, indexing="ij")

    z_meters = compute_gaussian_wave_height_numpy(
        xx,
        yy,
        terrain_shift_x=cfg.terrain_shift_x,
        peak1_center=cfg.peak1_center,
        peak1_height=cfg.peak1_height,
        peak1_variance=cfg.peak1_variance,
        peak2_center=cfg.peak2_center,
        peak2_height=cfg.peak2_height,
        peak2_variance=cfg.peak2_variance,
        abyss1_center=cfg.abyss1_center,
        abyss2_center=cfg.abyss2_center,
        abyss_depth=cfg.abyss_depth,
        abyss_variance=cfg.abyss_variance,
        wave_amplitude=cfg.wave_amplitude,
        wave_frequency=cfg.wave_frequency,
        base_elevation=cfg.base_elevation,
        min_height=cfg.min_height,
        max_height=cfg.max_height,
    )
    hf_discrete = z_meters / float(cfg.vertical_scale)
    return np.rint(hf_discrete).astype(np.int16)


@configclass
class HfGaussianHillsWaveTerrainCfg(terrain_gen.HfTerrainBaseCfg):
    function = gaussian_hills_wave_terrain
    quadrant_offset: tuple[float, float] = (0.0, 0.0)
    auto_row_offset_from_difficulty: bool = False
    terrain_shift_x: float = 0.0
    peak1_center: tuple[float, float] = (30.0, 30.0)
    peak1_height: float = 4.5
    peak1_variance: float = 280.0
    peak2_center: tuple[float, float] = (70.0, 68.0)
    peak2_height: float = 5.2
    peak2_variance: float = 320.0
    abyss1_center: tuple[float, float] = (68.0, 22.0)
    abyss2_center: tuple[float, float] = (82.0, 32.0)
    abyss_depth: float = 2.4
    abyss_variance: float = 10.0
    wave_amplitude: float = 1.4
    wave_frequency: float = 0.08
    base_elevation: float = 1.5
    min_height: float = 0.0
    max_height: float = 12.0


FOUR_BIOMES_GRID_CFG = TerrainGeneratorCfg(
    size=(100.0, 100.0),
    border_width=0.0,
    num_rows=1,
    num_cols=1,
    horizontal_scale=0.25,
    vertical_scale=0.005,
    slope_threshold=0.75,
    curriculum=False,
    use_cache=False,
    seed=42,
    sub_terrains={
        "unified_four_biomes_arena": HfGaussianHillsWaveTerrainCfg(
            proportion=1.0,
            quadrant_offset=(0.0, 0.0),
            auto_row_offset_from_difficulty=False,
            border_width=0.0,
        ),
    },
)


BIOME_CENTERS_LOCAL_XY: tuple[tuple[float, float], ...] = (
    (-25.0, -25.0),
    (25.0, -25.0),
    (-25.0, 25.0),
    (25.0, 25.0),
)

DEFAULT_FOOD_LOCAL_XY: tuple[tuple[float, float], ...] = (
    (-20.0, -20.0),
    (20.0, -20.0),
    (-20.0, 20.0),
    (20.0, 18.0),
)

DEFAULT_PREDATOR_LOCAL_XY: tuple[tuple[float, float], ...] = (
    (-32.0, -15.0),
    (32.0, -15.0),
    (-32.0, 15.0),
    (32.0, 15.0),
)

DEFAULT_TOY_SLOT_LOCAL_XY: tuple[tuple[float, float], ...] = (
    (-22.0, -22.0),
    (22.0, -22.0),
    (-22.0, 22.0),
    (22.0, 22.0),
)

DEFAULT_TOY_BLOCK_LOCAL_XY: tuple[tuple[float, float], ...] = (
    (-17.5, -22.0),
    (26.5, -22.0),
    (-17.5, 22.0),
    (26.5, 22.0),
)

DEFAULT_METEOR_LOCAL_XYZ: tuple[tuple[float, float, float], ...] = (
    (-28.0, -28.0, 22.0),
    (-16.0, -30.0, 26.0),
)

DEFAULT_ABYSS_LOCAL_XY: tuple[tuple[float, float], ...] = (
    (18.0, -28.0),
    (32.0, -18.0),
)

DEFAULT_VORTEX_LOCAL_XY: tuple[tuple[float, float], ...] = (
    (-28.0, 28.0),
    (-16.0, 32.0),
)

DEFAULT_SIGNPOST_LOCAL_XY: tuple[tuple[float, float], ...] = (
    (18.0, 28.0),
    (32.0, 28.0),
)

DEFAULT_OBSTACLE_LOCAL_XY: tuple[tuple[float, float], ...] = (
    (-14.0, -34.0),
    (14.0, -34.0),
    (-14.0, 34.0),
    (14.0, 34.0),
)


def _local_terrain_elevation(x_local: float, y_local: float) -> float:
    return float(
        compute_gaussian_wave_height_numpy(
            np.asarray(float(x_local) + 50.0, dtype=np.float32),
            np.asarray(float(y_local) + 50.0, dtype=np.float32),
        )
    )


SAFE_STUDY_ROOM_BASE_Z: float = _local_terrain_elevation(50.0, 0.0)

SAFE_STUDY_ROOM_BOUNDS_LOCAL_XYZ: tuple[float, float, float, float, float, float] = (
    50.0,
    74.0,
    -12.0,
    12.0,
    SAFE_STUDY_ROOM_BASE_Z,
    SAFE_STUDY_ROOM_BASE_Z + 4.0,
)
SAFE_STUDY_ROOM_DOORWAY_LOCAL_XYZ: tuple[float, float, float] = (50.0, 0.0, SAFE_STUDY_ROOM_BASE_Z)
SAFE_STUDY_ROOM_DESK_LOCAL_XYZ: tuple[float, float, float] = (62.0, 0.0, SAFE_STUDY_ROOM_BASE_Z + 0.78)
SAFE_STUDY_ROOM_CHAIR_LOCAL_XYZ: tuple[float, float, float] = (62.0, -0.30, SAFE_STUDY_ROOM_BASE_Z + 0.45)

SAFE_ROOM_BOUNDS_LOCAL = SAFE_STUDY_ROOM_BOUNDS_LOCAL_XYZ
SAFE_ROOM_DESK_LOCAL_XYZ = SAFE_STUDY_ROOM_DESK_LOCAL_XYZ
SAFE_ROOM_CHAIR_LOCAL_XYZ = SAFE_STUDY_ROOM_CHAIR_LOCAL_XYZ
TOY_PROXIMITY_SCALE: float = 10.0

SAFE_ROOM_RIGID_OBJECT_KEYS: tuple[str, ...] = (
    "safe_room_floor",
    "safe_room_wall_north",
    "safe_room_wall_south",
    "safe_room_wall_east",
    "safe_room_wall_west_north",
    "safe_room_wall_west_south",
    "safe_room_desk",
    "safe_room_chair",
)

NON_TOY_RIGID_OBJECT_KEYS: tuple[str, ...] = (
    "food_0",
    "food_1",
    "food_2",
    "food_3",
    "predator_0",
    "predator_1",
    "predator_2",
    "predator_3",
    "meteor_0",
    "meteor_1",
    "abyss_hazard_0",
    "abyss_hazard_1",
    "vortex_center_0",
    "vortex_center_1",
    "signpost_panel_0",
    "signpost_panel_1",
    "obstacle_0",
    "obstacle_1",
    "obstacle_2",
    "obstacle_3",
)


def _make_food_rigid_cfg(prim_name: str, x_local: float, y_local: float) -> RigidObjectCfg:
    z_ground = _local_terrain_elevation(x_local, y_local)
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{prim_name}",
        spawn=sim_utils.SphereCfg(
            radius=0.45,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=False, disable_gravity=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.85, 0.25)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(float(x_local), float(y_local), z_ground + 0.5),
        ),
    )


def _make_predator_rigid_cfg(prim_name: str, x_local: float, y_local: float) -> RigidObjectCfg:
    z_ground = _local_terrain_elevation(x_local, y_local)
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{prim_name}",
        spawn=sim_utils.SphereCfg(
            radius=0.65,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=False, disable_gravity=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=4.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.90, 0.15, 0.15)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(float(x_local), float(y_local), z_ground + 0.65),
        ),
    )


def _make_toy_block_rigid_cfg(
    prim_name: str,
    x_local: float,
    y_local: float,
    diffuse_color: tuple[float, float, float] = (0.95, 0.50, 0.10),
) -> RigidObjectCfg:
    z_ground = _local_terrain_elevation(x_local, y_local)
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{prim_name}",
        spawn=sim_utils.CuboidCfg(
            size=(1.4, 1.4, 1.1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=False, disable_gravity=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=2.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=diffuse_color),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(float(x_local), float(y_local), z_ground + 0.60),
        ),
    )


def _make_toy_slot_rigid_cfg(
    prim_name: str,
    x_local: float,
    y_local: float,
    diffuse_color: tuple[float, float, float] = (0.55, 0.10, 0.20),
) -> RigidObjectCfg:
    z_ground = _local_terrain_elevation(x_local, y_local)
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{prim_name}",
        spawn=sim_utils.CuboidCfg(
            size=(1.8, 1.8, 0.9),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=False, disable_gravity=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=8.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=diffuse_color),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(float(x_local), float(y_local), z_ground + 0.45),
        ),
    )


def _make_meteor_rigid_cfg(prim_name: str, x_local: float, y_local: float, z_sky: float) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{prim_name}",
        spawn=sim_utils.SphereCfg(
            radius=1.2,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=False, disable_gravity=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=12.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.90, 0.20, 0.75)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(float(x_local), float(y_local), float(z_sky)),
        ),
    )


def _make_abyss_rigid_cfg(prim_name: str, x_local: float, y_local: float) -> RigidObjectCfg:
    z_ground = _local_terrain_elevation(x_local, y_local)
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{prim_name}",
        spawn=sim_utils.CylinderCfg(
            radius=2.5,
            height=0.10,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=50.0),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.30, 0.05, 0.40)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(float(x_local), float(y_local), z_ground + 0.05),
        ),
    )


def _make_vortex_center_rigid_cfg(prim_name: str, x_local: float, y_local: float) -> RigidObjectCfg:
    z_ground = _local_terrain_elevation(x_local, y_local)
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{prim_name}",
        spawn=sim_utils.SphereCfg(
            radius=0.5,
            visible=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(float(x_local), float(y_local), z_ground + 2.0),
        ),
    )


def _make_signpost_panel_rigid_cfg(prim_name: str, x_local: float, y_local: float) -> RigidObjectCfg:
    z_ground = _local_terrain_elevation(x_local, y_local)
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{prim_name}",
        spawn=sim_utils.CuboidCfg(
            size=(1.8, 1.8, 1.4),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=15.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.50, 0.10)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(float(x_local), float(y_local), z_ground + 0.70),
        ),
    )


def _make_obstacle_pillar_rigid_cfg(prim_name: str, x_local: float, y_local: float) -> RigidObjectCfg:
    z_ground = _local_terrain_elevation(x_local, y_local)
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{prim_name}",
        spawn=sim_utils.CuboidCfg(
            size=(3.2, 3.2, 2.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=40.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.85, 0.10)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(float(x_local), float(y_local), z_ground + 1.25),
        ),
    )


def _make_static_cuboid_asset_cfg(
    prim_name: str,
    size_xyz: tuple[float, float, float],
    pos_xyz: tuple[float, float, float],
    diffuse_color: tuple[float, float, float],
    collision_enabled: bool = True,
) -> AssetBaseCfg | RigidObjectCfg:
    if collision_enabled:
        return RigidObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/{prim_name}",
            spawn=sim_utils.CuboidCfg(
                size=size_xyz,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
                mass_props=sim_utils.MassPropertiesCfg(mass=500.0),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=diffuse_color),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=pos_xyz),
        )
    return AssetBaseCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{prim_name}",
        spawn=sim_utils.CuboidCfg(
            size=size_xyz,
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=diffuse_color),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=pos_xyz),
    )


@configclass
class FourBiomeHumanoidSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=FOUR_BIOMES_GRID_CFG,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    biome_floor_0 = _make_static_cuboid_asset_cfg(
        "BiomeFloor0",
        (12.0, 12.0, 0.06),
        (-25.0, -25.0, _local_terrain_elevation(-25.0, -25.0) + 0.04),
        (0.20, 0.45, 0.25),
        collision_enabled=False,
    )
    biome_floor_1 = _make_static_cuboid_asset_cfg(
        "BiomeFloor1",
        (12.0, 12.0, 0.06),
        (25.0, -25.0, _local_terrain_elevation(25.0, -25.0) + 0.04),
        (0.45, 0.40, 0.20),
        collision_enabled=False,
    )
    biome_floor_2 = _make_static_cuboid_asset_cfg(
        "BiomeFloor2",
        (12.0, 12.0, 0.06),
        (-25.0, 25.0, _local_terrain_elevation(-25.0, 25.0) + 0.04),
        (0.20, 0.35, 0.50),
        collision_enabled=False,
    )
    biome_floor_3 = _make_static_cuboid_asset_cfg(
        "BiomeFloor3",
        (12.0, 12.0, 0.06),
        (25.0, 25.0, _local_terrain_elevation(25.0, 25.0) + 0.04),
        (0.40, 0.25, 0.45),
        collision_enabled=False,
    )

    food_0: RigidObjectCfg = _make_food_rigid_cfg("Food0", *DEFAULT_FOOD_LOCAL_XY[0])
    food_1: RigidObjectCfg = _make_food_rigid_cfg("Food1", *DEFAULT_FOOD_LOCAL_XY[1])
    food_2: RigidObjectCfg = _make_food_rigid_cfg("Food2", *DEFAULT_FOOD_LOCAL_XY[2])
    food_3: RigidObjectCfg = _make_food_rigid_cfg("Food3", *DEFAULT_FOOD_LOCAL_XY[3])

    predator_0: RigidObjectCfg = _make_predator_rigid_cfg("Predator0", *DEFAULT_PREDATOR_LOCAL_XY[0])
    predator_1: RigidObjectCfg = _make_predator_rigid_cfg("Predator1", *DEFAULT_PREDATOR_LOCAL_XY[1])
    predator_2: RigidObjectCfg = _make_predator_rigid_cfg("Predator2", *DEFAULT_PREDATOR_LOCAL_XY[2])
    predator_3: RigidObjectCfg = _make_predator_rigid_cfg("Predator3", *DEFAULT_PREDATOR_LOCAL_XY[3])

    toy_block_0: RigidObjectCfg = _make_toy_block_rigid_cfg(
        "ToyBlock0", *DEFAULT_TOY_BLOCK_LOCAL_XY[0], diffuse_color=(0.95, 0.50, 0.10)
    )
    toy_block_1: RigidObjectCfg = _make_toy_block_rigid_cfg(
        "ToyBlock1", *DEFAULT_TOY_BLOCK_LOCAL_XY[1], diffuse_color=(0.00, 0.45, 0.85)
    )
    toy_block_2: RigidObjectCfg = _make_toy_block_rigid_cfg(
        "ToyBlock2", *DEFAULT_TOY_BLOCK_LOCAL_XY[2], diffuse_color=(0.18, 0.80, 0.25)
    )
    toy_block_3: RigidObjectCfg = _make_toy_block_rigid_cfg(
        "ToyBlock3", *DEFAULT_TOY_BLOCK_LOCAL_XY[3], diffuse_color=(0.35, 0.80, 0.95)
    )

    toy_slot_0: RigidObjectCfg = _make_toy_slot_rigid_cfg(
        "ToySlot0", *DEFAULT_TOY_SLOT_LOCAL_XY[0], diffuse_color=(0.55, 0.10, 0.20)
    )
    toy_slot_1: RigidObjectCfg = _make_toy_slot_rigid_cfg(
        "ToySlot1", *DEFAULT_TOY_SLOT_LOCAL_XY[1], diffuse_color=(0.55, 0.10, 0.20)
    )
    toy_slot_2: RigidObjectCfg = _make_toy_slot_rigid_cfg(
        "ToySlot2", *DEFAULT_TOY_SLOT_LOCAL_XY[2], diffuse_color=(0.55, 0.10, 0.20)
    )
    toy_slot_3: RigidObjectCfg = _make_toy_slot_rigid_cfg(
        "ToySlot3", *DEFAULT_TOY_SLOT_LOCAL_XY[3], diffuse_color=(0.55, 0.10, 0.20)
    )

    meteor_0: RigidObjectCfg = _make_meteor_rigid_cfg("Meteor0", *DEFAULT_METEOR_LOCAL_XYZ[0])
    meteor_1: RigidObjectCfg = _make_meteor_rigid_cfg("Meteor1", *DEFAULT_METEOR_LOCAL_XYZ[1])

    abyss_hazard_0: RigidObjectCfg = _make_abyss_rigid_cfg("AbyssHazard0", *DEFAULT_ABYSS_LOCAL_XY[0])
    abyss_hazard_1: RigidObjectCfg = _make_abyss_rigid_cfg("AbyssHazard1", *DEFAULT_ABYSS_LOCAL_XY[1])

    vortex_center_0: RigidObjectCfg = _make_vortex_center_rigid_cfg("VortexCenter0", *DEFAULT_VORTEX_LOCAL_XY[0])
    vortex_center_1: RigidObjectCfg = _make_vortex_center_rigid_cfg("VortexCenter1", *DEFAULT_VORTEX_LOCAL_XY[1])

    signpost_panel_0: RigidObjectCfg = _make_signpost_panel_rigid_cfg("SignpostPanel0", *DEFAULT_SIGNPOST_LOCAL_XY[0])
    signpost_panel_1: RigidObjectCfg = _make_signpost_panel_rigid_cfg("SignpostPanel1", *DEFAULT_SIGNPOST_LOCAL_XY[1])

    obstacle_0: RigidObjectCfg = _make_obstacle_pillar_rigid_cfg("Obstacle0", *DEFAULT_OBSTACLE_LOCAL_XY[0])
    obstacle_1: RigidObjectCfg = _make_obstacle_pillar_rigid_cfg("Obstacle1", *DEFAULT_OBSTACLE_LOCAL_XY[1])
    obstacle_2: RigidObjectCfg = _make_obstacle_pillar_rigid_cfg("Obstacle2", *DEFAULT_OBSTACLE_LOCAL_XY[2])
    obstacle_3: RigidObjectCfg = _make_obstacle_pillar_rigid_cfg("Obstacle3", *DEFAULT_OBSTACLE_LOCAL_XY[3])

    safe_room_floor = _make_static_cuboid_asset_cfg(
        "SafeRoomFloor",
        (24.0, 24.0, 0.2),
        (62.0, 0.0, SAFE_STUDY_ROOM_BASE_Z - 0.10),
        (0.35, 0.80, 0.95),
    )
    safe_room_wall_north = _make_static_cuboid_asset_cfg(
        "SafeRoomWallNorth",
        (24.0, 0.3, 4.0),
        (62.0, 12.0, SAFE_STUDY_ROOM_BASE_Z + 2.0),
        (0.65, 0.65, 0.68),
    )
    safe_room_wall_south = _make_static_cuboid_asset_cfg(
        "SafeRoomWallSouth",
        (24.0, 0.3, 4.0),
        (62.0, -12.0, SAFE_STUDY_ROOM_BASE_Z + 2.0),
        (0.65, 0.65, 0.68),
    )
    safe_room_wall_east = _make_static_cuboid_asset_cfg(
        "SafeRoomWallEast",
        (0.3, 24.0, 4.0),
        (74.0, 0.0, SAFE_STUDY_ROOM_BASE_Z + 2.0),
        (0.65, 0.65, 0.68),
    )
    safe_room_wall_west_north = _make_static_cuboid_asset_cfg(
        "SafeRoomWallWestNorth",
        (0.3, 9.5, 4.0),
        (50.0, 7.25, SAFE_STUDY_ROOM_BASE_Z + 2.0),
        (0.65, 0.65, 0.68),
    )
    safe_room_wall_west_south = _make_static_cuboid_asset_cfg(
        "SafeRoomWallWestSouth",
        (0.3, 9.5, 4.0),
        (50.0, -7.25, SAFE_STUDY_ROOM_BASE_Z + 2.0),
        (0.65, 0.65, 0.68),
    )
    safe_room_desk = _make_static_cuboid_asset_cfg(
        "SafeRoomDesk",
        (1.6, 0.50, 0.06),
        (62.0, 0.25, SAFE_STUDY_ROOM_BASE_Z + 0.75),
        (0.55, 0.38, 0.20),
        collision_enabled=True,
    )
    safe_room_chair = _make_static_cuboid_asset_cfg(
        "SafeRoomChair",
        (0.75, 0.42, 0.62),
        (62.0, -0.68, SAFE_STUDY_ROOM_BASE_Z + 0.23),
        (0.30, 0.30, 0.35),
        collision_enabled=True,
    )

    robot: ArticulationCfg = HUMANOID_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 4.35),
            joint_pos={".*": 0.0},
        ),
    )

    left_eye_raycaster: RayCasterCfg = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso",
        offset=RayCasterCfg.OffsetCfg(pos=(0.12, 0.032, 0.18)),
        ray_alignment="yaw",
        pattern_cfg=patterns.LidarPatternCfg(
            channels=1,
            vertical_fov_range=(-6.0, -6.0),
            horizontal_fov_range=(-56.0, 56.0),
            horizontal_res=16.0,
        ),
        max_distance=60.0,
        drift_range=(0.0, 0.0),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    right_eye_raycaster: RayCasterCfg = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso",
        offset=RayCasterCfg.OffsetCfg(pos=(0.12, -0.032, 0.18)),
        ray_alignment="yaw",
        pattern_cfg=patterns.LidarPatternCfg(
            channels=1,
            vertical_fov_range=(-6.0, -6.0),
            horizontal_fov_range=(-56.0, 56.0),
            horizontal_res=16.0,
        ),
        max_distance=60.0,
        drift_range=(0.0, 0.0),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


def _extract_env_origins(env: ManagerBasedRLEnv, num_envs: int, device: torch.device | str) -> torch.Tensor:
    if hasattr(env, "scene"):
        if hasattr(env.scene, "env_origins") and env.scene.env_origins is not None:
            return env.scene.env_origins.to(device=device, dtype=torch.float32)
        if isinstance(env.scene, dict) and env.scene.get("env_origins") is not None:
            return env.scene["env_origins"].to(device=device, dtype=torch.float32)
    return torch.zeros((num_envs, 3), dtype=torch.float32, device=device)


def _build_local_anchor_tensor(
    xy_pairs: tuple[tuple[float, float], ...],
    z_offset: float,
    device: torch.device | str,
) -> torch.Tensor:
    coords = []
    for x_loc, y_loc in xy_pairs:
        z_loc = _local_terrain_elevation(x_loc, y_loc) + float(z_offset)
        coords.append((float(x_loc), float(y_loc), float(z_loc)))
    return torch.tensor(coords, dtype=torch.float32, device=device)


def _collect_scene_rigid_positions_w(
    env: ManagerBasedRLEnv,
    override_attr: str,
    entity_keys: tuple[str, ...],
    device: torch.device | str,
) -> torch.Tensor:
    override_val = getattr(env, override_attr, None)
    if override_val is not None:
        return override_val.to(device=device, dtype=torch.float32)
    collected = [
        env.scene[key].data.root_pos_w.to(device=device, dtype=torch.float32)
        for key in entity_keys
    ]
    return torch.stack(collected, dim=1)


def sync_safe_study_room_colliders(env: ManagerBasedRLEnv, env_ids: torch.Tensor | None = None) -> None:
    room_keys = (
        "safe_room_floor",
        "safe_room_wall_north",
        "safe_room_wall_south",
        "safe_room_wall_east",
        "safe_room_wall_west_north",
        "safe_room_wall_west_south",
        "safe_room_desk",
        "safe_room_chair",
    )
    robot = env.scene["robot"]
    device = robot.data.root_pos_w.device
    num_envs = int(robot.data.root_pos_w.shape[0])
    if env_ids is None:
        ids = torch.arange(num_envs, dtype=torch.long, device=device)
    else:
        ids = env_ids.to(device=device, dtype=torch.long)
    if ids.numel() == 0:
        return
    env_origins = _extract_env_origins(env, num_envs, device)
    zero_vel = torch.zeros((ids.numel(), 6), dtype=torch.float32, device=device)
    for key in room_keys:
        if hasattr(env.scene, "keys") and key not in env.scene.keys():
            continue
        obj = env.scene[key]
        if not hasattr(obj, "write_root_pose_to_sim"):
            continue
        default_pose = obj.data.default_root_state[ids, :7].clone()
        default_pose[:, :3] += env_origins[ids]
        if key == "safe_room_chair":
            default_pose[:, 1] = env_origins[ids, 1] - 0.68
            default_pose[:, 2] = env_origins[ids, 2] + float(SAFE_STUDY_ROOM_BASE_Z) + 0.23
        obj.write_root_pose_to_sim(default_pose, env_ids=ids)
        obj.write_root_velocity_to_sim(zero_vel, env_ids=ids)


def reset_scene_rigid_objects(env: ManagerBasedRLEnv, env_ids: torch.Tensor | None) -> None:
    robot = env.scene["robot"]
    device = robot.data.root_pos_w.device
    num_envs = int(robot.data.root_pos_w.shape[0])

    if env_ids is None:
        ids = torch.arange(num_envs, dtype=torch.long, device=device)
    else:
        ids = env_ids.to(device=device, dtype=torch.long)

    if ids.numel() == 0:
        return

    env_origins = _extract_env_origins(env, num_envs, device)
    zero_vel = torch.zeros((ids.numel(), 6), dtype=torch.float32, device=device)

    mode_str = str(getattr(env, "unpredictable_mode", "standard"))
    for key in NON_TOY_RIGID_OBJECT_KEYS:
        obj = env.scene[key]
        default_pose = obj.data.default_root_state[ids, :7].clone()
        default_pose[:, :3] += env_origins[ids]
        init_vel = zero_vel.clone()
        if key == "safe_room_chair":
            default_pose[:, 1] = env_origins[ids, 1] - 0.68
            default_pose[:, 2] = env_origins[ids, 2] + float(SAFE_STUDY_ROOM_BASE_Z) + 0.23
        elif mode_str != "standard":
            if key == "meteor_0":
                default_pose[:, 0] = env_origins[ids, 0] + 2.2
                default_pose[:, 1] = env_origins[ids, 1] + 2.0
                default_pose[:, 2] = env_origins[ids, 2] + 11.5
                init_vel[:, 2] = -12.0
            elif key == "meteor_1":
                default_pose[:, 0] = env_origins[ids, 0] - 2.5
                default_pose[:, 1] = env_origins[ids, 1] + 2.2
                default_pose[:, 2] = env_origins[ids, 2] + 14.0
                init_vel[:, 2] = -12.0
            elif key == "predator_0":
                default_pose[:, 0] = env_origins[ids, 0] + 7.5
                default_pose[:, 1] = env_origins[ids, 1] + 6.5
        obj.write_root_pose_to_sim(default_pose, env_ids=ids)
        obj.write_root_velocity_to_sim(init_vel, env_ids=ids)
    sync_safe_study_room_colliders(env, env_ids=ids)


def reset_humanoid_at_study_room_desk(env: ManagerBasedRLEnv, env_ids: torch.Tensor | None = None) -> None:
    ensure_sensorimotor_puzzle_buffers(env)
    sync_safe_study_room_colliders(env, env_ids=env_ids)
    robot = env.scene["robot"]
    device = robot.data.root_pos_w.device
    num_envs = int(robot.data.root_pos_w.shape[0])
    if env_ids is None:
        ids = torch.arange(num_envs, dtype=torch.long, device=device)
    else:
        ids = env_ids.to(device=device, dtype=torch.long)
    if ids.numel() == 0:
        return
    env_origins = _extract_env_origins(env, num_envs, device)
    desk_root_pose = torch.zeros((ids.numel(), 7), dtype=torch.float32, device=device)
    desk_root_pose[:, 0] = env_origins[ids, 0] + float(SAFE_STUDY_ROOM_CHAIR_LOCAL_XYZ[0])
    desk_root_pose[:, 1] = env_origins[ids, 1] + float(SAFE_STUDY_ROOM_CHAIR_LOCAL_XYZ[1])
    desk_root_pose[:, 2] = env_origins[ids, 2] + float(SAFE_STUDY_ROOM_BASE_Z) + 1.19
    half_yaw = 0.25 * math.pi
    desk_root_pose[:, 3] = float(math.cos(half_yaw))
    desk_root_pose[:, 4] = 0.0
    desk_root_pose[:, 5] = 0.0
    desk_root_pose[:, 6] = float(math.sin(half_yaw))
    zero_root_vel = torch.zeros((ids.numel(), 6), dtype=torch.float32, device=device)
    robot.write_root_pose_to_sim(desk_root_pose, env_ids=ids)
    robot.write_root_velocity_to_sim(zero_root_vel, env_ids=ids)
    joint_pos = robot.data.default_joint_pos[ids].clone()
    joint_vel = torch.zeros_like(joint_pos)
    if joint_pos.shape[1] >= 21:
        joint_pos[:, 1] = 0.18
        joint_pos[:, 2] = -0.62
        joint_pos[:, 3] = -0.15
        joint_pos[:, 4] = -0.35
        joint_pos[:, 7] = -0.55
        joint_pos[:, 8] = -0.45
        joint_pos[:, 11] = -0.35
        joint_pos[:, 14] = -0.35
        joint_pos[:, 15] = -0.55
        joint_pos[:, 16] = -0.55
        joint_pos[:, 17] = 0.20
        joint_pos[:, 19] = 0.20
    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=ids)
    robot.set_joint_position_target(joint_pos, env_ids=ids)
    env.agent_gaze_yaw_offset_rad[ids] = 0.0
    env.agent_gaze_pitch_offset_rad[ids] = -0.95
    if hasattr(env, "scene") and hasattr(env, "sim"):
        phys_dt = float(getattr(env, "physics_dt", 1.0 / 120.0))
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        env.scene.update(phys_dt)


def get_biome_centers_w(env: ManagerBasedRLEnv) -> torch.Tensor:
    robot = env.scene["robot"]
    device = robot.data.root_pos_w.device
    num_envs = int(robot.data.root_pos_w.shape[0])
    if hasattr(env, "biome_centers_w") and env.biome_centers_w is not None:
        return env.biome_centers_w.to(device=device, dtype=torch.float32)
    env_origins = _extract_env_origins(env, num_envs, device)
    local_centers = _build_local_anchor_tensor(BIOME_CENTERS_LOCAL_XY, z_offset=0.0, device=device)
    return env_origins.unsqueeze(1) + local_centers.unsqueeze(0)


def get_food_positions_w(env: ManagerBasedRLEnv) -> torch.Tensor:
    device = env.scene["robot"].data.root_pos_w.device
    return _collect_scene_rigid_positions_w(
        env,
        "food_pos_w",
        ("food_0", "food_1", "food_2", "food_3"),
        device,
    )


def get_predator_positions_w(env: ManagerBasedRLEnv) -> torch.Tensor:
    device = env.scene["robot"].data.root_pos_w.device
    return _collect_scene_rigid_positions_w(
        env,
        "predator_pos_w",
        ("predator_0", "predator_1", "predator_2", "predator_3"),
        device,
    )


def get_toy_block_positions_w(env: ManagerBasedRLEnv) -> torch.Tensor:
    device = env.scene["robot"].data.root_pos_w.device
    return _collect_scene_rigid_positions_w(
        env,
        "toy_block_pos_w",
        ("toy_block_0", "toy_block_1", "toy_block_2", "toy_block_3"),
        device,
    )


def get_toy_slot_positions_w(env: ManagerBasedRLEnv) -> torch.Tensor:
    device = env.scene["robot"].data.root_pos_w.device
    return _collect_scene_rigid_positions_w(
        env,
        "toy_slot_pos_w",
        ("toy_slot_0", "toy_slot_1", "toy_slot_2", "toy_slot_3"),
        device,
    )


def get_disaster_and_obstacle_positions_w(env: ManagerBasedRLEnv) -> torch.Tensor:
    device = env.scene["robot"].data.root_pos_w.device
    entity_keys = (
        "meteor_0",
        "meteor_1",
        "abyss_hazard_0",
        "abyss_hazard_1",
        "vortex_center_0",
        "vortex_center_1",
        "signpost_panel_0",
        "signpost_panel_1",
        "obstacle_0",
        "obstacle_1",
        "obstacle_2",
        "obstacle_3",
    )
    return _collect_scene_rigid_positions_w(
        env,
        "disaster_pos_w",
        entity_keys,
        device,
    )


def compute_toy_block_slot_mutual_distances(env: ManagerBasedRLEnv) -> torch.Tensor:
    block_pos_w = get_toy_block_positions_w(env)
    slot_pos_w = get_toy_slot_positions_w(env)
    return torch.linalg.norm(slot_pos_w - block_pos_w, dim=-1)


def compute_egocentric_relative_vectors(
    root_pos_w: torch.Tensor,
    root_quat_w: torch.Tensor,
    target_pos_w: torch.Tensor,
    gaze_yaw_offset_rad: torch.Tensor | None = None,
    fov_half_rad: float = 1.05,
    max_vis_dist: float = 38.0,
    apply_visual_frustum_mask: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    delta_w = target_pos_w - root_pos_w.unsqueeze(1)
    qw = root_quat_w[:, 0]
    qx = root_quat_w[:, 1]
    qy = root_quat_w[:, 2]
    qz = root_quat_w[:, 3]
    yaw = torch.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    if gaze_yaw_offset_rad is not None and gaze_yaw_offset_rad.shape[0] == yaw.shape[0]:
        yaw = yaw + gaze_yaw_offset_rad.to(device=yaw.device, dtype=torch.float32)
    cos_yaw = torch.cos(yaw).unsqueeze(1)
    sin_yaw = torch.sin(yaw).unsqueeze(1)
    dx_ego = delta_w[:, :, 0] * cos_yaw + delta_w[:, :, 1] * sin_yaw
    dy_ego = -delta_w[:, :, 0] * sin_yaw + delta_w[:, :, 1] * cos_yaw
    dz_ego = delta_w[:, :, 2]
    ego_vec = torch.stack([dx_ego, dy_ego, dz_ego], dim=-1)
    dist = torch.linalg.norm(ego_vec, dim=-1)
    bearing = torch.atan2(dy_ego, dx_ego)
    if apply_visual_frustum_mask:
        in_fov = (torch.abs(bearing) <= float(fov_half_rad)) & (dist <= float(max_vis_dist))
        ego_vec = ego_vec * in_fov.to(dtype=torch.float32).unsqueeze(-1)
        dist = torch.where(in_fov, dist, torch.full_like(dist, float(max_vis_dist) * 2.5))
        bearing = bearing * in_fov.to(dtype=torch.float32)
    return ego_vec, dist, bearing


def _encode_top2_egocentric_targets_8d(
    root_pos_w: torch.Tensor,
    root_quat_w: torch.Tensor,
    target_pos_w: torch.Tensor,
    pos_scale: float = 50.0,
    prox_range: float = 35.0,
    gaze_yaw_offset_rad: torch.Tensor | None = None,
) -> torch.Tensor:
    ego_vec, dist, bearing = compute_egocentric_relative_vectors(
        root_pos_w,
        root_quat_w,
        target_pos_w,
        gaze_yaw_offset_rad=gaze_yaw_offset_rad,
        fov_half_rad=1.05,
        max_vis_dist=prox_range,
        apply_visual_frustum_mask=True,
    )
    sorted_dist, sorted_idx = torch.sort(dist, dim=1)
    idx_0 = sorted_idx[:, 0:1].unsqueeze(-1).expand(-1, 1, 3)
    idx_1 = sorted_idx[:, 1:2].unsqueeze(-1).expand(-1, 1, 3) if target_pos_w.shape[1] > 1 else idx_0
    nearest_ego = torch.gather(ego_vec, 1, idx_0).squeeze(1)
    second_ego = torch.gather(ego_vec, 1, idx_1).squeeze(1)
    nearest_bearing = torch.gather(bearing, 1, sorted_idx[:, 0:1])
    vis_0 = (sorted_dist[:, 0:1] <= float(prox_range)).to(dtype=torch.float32)
    vis_1 = (sorted_dist[:, 1:2] <= float(prox_range)).to(dtype=torch.float32) if target_pos_w.shape[1] > 1 else vis_0
    norm_0 = torch.clamp(nearest_ego / float(pos_scale), -1.0, 1.0) * vis_0
    norm_1 = torch.clamp(second_ego / float(pos_scale), -1.0, 1.0) * vis_1
    prox_0 = torch.clamp(1.0 - sorted_dist[:, 0:1] / float(prox_range), 0.0, 1.0) * vis_0
    bear_0 = torch.clamp(nearest_bearing / math.pi, -1.0, 1.0) * vis_0
    return torch.cat([norm_0, norm_1, prox_0, bear_0], dim=-1)


def obs_egocentric_predators_8d(env: ManagerBasedRLEnv) -> torch.Tensor:
    ensure_sensorimotor_puzzle_buffers(env)
    robot = env.scene["robot"]
    pred_pos_w = get_predator_positions_w(env)
    return _encode_top2_egocentric_targets_8d(
        robot.data.root_pos_w,
        robot.data.root_quat_w,
        pred_pos_w,
        pos_scale=50.0,
        prox_range=35.0,
        gaze_yaw_offset_rad=env.agent_gaze_yaw_offset_rad,
    )


def obs_egocentric_foods_8d(env: ManagerBasedRLEnv) -> torch.Tensor:
    ensure_sensorimotor_puzzle_buffers(env)
    robot = env.scene["robot"]
    food_pos_w = get_food_positions_w(env)
    return _encode_top2_egocentric_targets_8d(
        robot.data.root_pos_w,
        robot.data.root_quat_w,
        food_pos_w,
        pos_scale=50.0,
        prox_range=35.0,
        gaze_yaw_offset_rad=env.agent_gaze_yaw_offset_rad,
    )


def obs_egocentric_biome_centers_8d(env: ManagerBasedRLEnv) -> torch.Tensor:
    ensure_sensorimotor_puzzle_buffers(env)
    robot = env.scene["robot"]
    biome_pos_w = get_biome_centers_w(env)
    biome_ego, _, _ = compute_egocentric_relative_vectors(
        robot.data.root_pos_w,
        robot.data.root_quat_w,
        biome_pos_w,
        gaze_yaw_offset_rad=env.agent_gaze_yaw_offset_rad,
        fov_half_rad=1.40,
        max_vis_dist=65.0,
        apply_visual_frustum_mask=True,
    )
    num_envs = int(robot.data.root_pos_w.shape[0])
    return torch.clamp(biome_ego[:, :, :2] / 50.0, -1.0, 1.0).reshape(num_envs, 8)


def obs_egocentric_biome_vertical_2d(env: ManagerBasedRLEnv) -> torch.Tensor:
    ensure_sensorimotor_puzzle_buffers(env)
    robot = env.scene["robot"]
    biome_pos_w = get_biome_centers_w(env)
    biome_ego, biome_dist, _ = compute_egocentric_relative_vectors(
        robot.data.root_pos_w,
        robot.data.root_quat_w,
        biome_pos_w,
        gaze_yaw_offset_rad=env.agent_gaze_yaw_offset_rad,
        fov_half_rad=1.40,
        max_vis_dist=65.0,
        apply_visual_frustum_mask=True,
    )
    min_dist, min_idx = torch.min(biome_dist, dim=1, keepdim=True)
    nearest_dz = torch.gather(biome_ego[:, :, 2], 1, min_idx)
    dz_norm = torch.clamp(nearest_dz / 10.0, -1.0, 1.0)
    prox_norm = torch.clamp(1.0 - min_dist / 50.0, 0.0, 1.0)
    return torch.cat([dz_norm, prox_norm], dim=-1)


def obs_egocentric_somatic_7d(env: ManagerBasedRLEnv) -> torch.Tensor:
    ensure_homeostasis_curiosity_buffers(env)
    robot = env.scene["robot"]
    root_pos_w = robot.data.root_pos_w
    root_lin_vel_w = robot.data.root_lin_vel_w
    device = root_pos_w.device
    num_envs = int(root_pos_w.shape[0])
    env_origins = _extract_env_origins(env, num_envs, device)
    local_pos = root_pos_w - env_origins
    arena_x = local_pos[:, 0] + 50.0
    arena_y = local_pos[:, 1] + 50.0
    terrain_shift = float(getattr(env, "terrain_shift_x", 0.0))
    ground_z = compute_gaussian_wave_height_torch(arena_x, arena_y, terrain_shift_x=terrain_shift)
    alt_norm = torch.clamp((local_pos[:, 2] - ground_z) / 10.0, 0.0, 1.0)
    speed_norm = torch.clamp(torch.linalg.norm(root_lin_vel_w[:, :2], dim=-1) / 5.2, 0.0, 1.0)
    vz_norm = torch.clamp(root_lin_vel_w[:, 2] / 5.0, -1.0, 1.0)
    return torch.stack(
        [
            env.agent_hp.to(device=device, dtype=torch.float32),
            env.agent_energy.to(device=device, dtype=torch.float32),
            env.agent_pain.to(device=device, dtype=torch.float32),
            env.agent_surprise.to(device=device, dtype=torch.float32),
            speed_norm,
            vz_norm,
            alt_norm,
        ],
        dim=-1,
    )


def obs_egocentric_toy_blocks_8d(env: ManagerBasedRLEnv) -> torch.Tensor:
    ensure_sensorimotor_puzzle_buffers(env)
    robot = env.scene["robot"]
    block_pos_w = get_toy_block_positions_w(env)
    ego_vec, _, _ = compute_egocentric_relative_vectors(
        robot.data.root_pos_w,
        robot.data.root_quat_w,
        block_pos_w,
        gaze_yaw_offset_rad=env.agent_gaze_yaw_offset_rad,
        fov_half_rad=1.05,
        max_vis_dist=38.0,
        apply_visual_frustum_mask=True,
    )
    num_envs = int(robot.data.root_pos_w.shape[0])
    return torch.clamp(ego_vec[:, :, :2] / 50.0, -1.0, 1.0).reshape(num_envs, 8)


def obs_egocentric_toy_slots_8d(env: ManagerBasedRLEnv) -> torch.Tensor:
    ensure_sensorimotor_puzzle_buffers(env)
    robot = env.scene["robot"]
    slot_pos_w = get_toy_slot_positions_w(env)
    ego_vec, _, _ = compute_egocentric_relative_vectors(
        robot.data.root_pos_w,
        robot.data.root_quat_w,
        slot_pos_w,
        gaze_yaw_offset_rad=env.agent_gaze_yaw_offset_rad,
        fov_half_rad=1.05,
        max_vis_dist=38.0,
        apply_visual_frustum_mask=True,
    )
    num_envs = int(robot.data.root_pos_w.shape[0])
    return torch.clamp(ego_vec[:, :, :2] / 50.0, -1.0, 1.0).reshape(num_envs, 8)


def obs_toy_block_slot_mutual_distances_12d(env: ManagerBasedRLEnv) -> torch.Tensor:
    ensure_sensorimotor_puzzle_buffers(env)
    robot = env.scene["robot"]
    root_pos_w = robot.data.root_pos_w
    root_quat_w = robot.data.root_quat_w
    block_pos_w = get_toy_block_positions_w(env)
    slot_pos_w = get_toy_slot_positions_w(env)
    _, b_dist, b_bear = compute_egocentric_relative_vectors(
        root_pos_w,
        root_quat_w,
        block_pos_w,
        gaze_yaw_offset_rad=env.agent_gaze_yaw_offset_rad,
        fov_half_rad=1.05,
        max_vis_dist=38.0,
        apply_visual_frustum_mask=False,
    )
    vis_mask = ((torch.abs(b_bear) <= 1.05) & (b_dist <= 38.0)).to(dtype=torch.float32)
    mutual_dist = torch.linalg.norm(slot_pos_w - block_pos_w, dim=-1)
    mutual_dist_norm = torch.clamp(mutual_dist / TOY_PROXIMITY_SCALE, 0.0, 1.0) * vis_mask
    mutual_prox_norm = torch.clamp(1.0 - mutual_dist / TOY_PROXIMITY_SCALE, 0.0, 1.0) * vis_mask
    agent_block_dist = torch.linalg.norm(block_pos_w - root_pos_w.unsqueeze(1), dim=-1)
    agent_block_dist_norm = torch.clamp(agent_block_dist / 50.0, 0.0, 1.0) * vis_mask
    return torch.cat([mutual_dist_norm, mutual_prox_norm, agent_block_dist_norm], dim=-1)


def obs_egocentric_disasters_16d(env: ManagerBasedRLEnv) -> torch.Tensor:
    ensure_sensorimotor_puzzle_buffers(env)
    robot = env.scene["robot"]
    root_pos_w = robot.data.root_pos_w
    root_quat_w = robot.data.root_quat_w
    gaze_off = env.agent_gaze_yaw_offset_rad
    all_dis_pos_w = get_disaster_and_obstacle_positions_w(env)

    meteor_pos = all_dis_pos_w[:, 0:2, :]
    abyss_pos = all_dis_pos_w[:, 2:4, :]
    vortex_pos = all_dis_pos_w[:, 4:6, :]
    signpost_pos = all_dis_pos_w[:, 6:8, :]
    obstacle_pos = all_dis_pos_w[:, 8:12, :]

    m_ego, m_dist, _ = compute_egocentric_relative_vectors(
        root_pos_w, root_quat_w, meteor_pos, gaze_yaw_offset_rad=gaze_off, fov_half_rad=1.15, max_vis_dist=38.0, apply_visual_frustum_mask=True
    )
    m_min_d, m_idx = torch.min(m_dist, dim=1, keepdim=True)
    m_near = torch.gather(m_ego, 1, m_idx.unsqueeze(-1).expand(-1, 1, 3)).squeeze(1)
    m_vec = torch.cat(
        [torch.clamp(m_near / 50.0, -1.0, 1.0), torch.clamp(1.0 - m_min_d / 35.0, 0.0, 1.0)],
        dim=-1,
    )

    a_ego, a_dist, _ = compute_egocentric_relative_vectors(
        root_pos_w, root_quat_w, abyss_pos, gaze_yaw_offset_rad=gaze_off, fov_half_rad=1.05, max_vis_dist=30.0, apply_visual_frustum_mask=True
    )
    a_min_d, a_idx = torch.min(a_dist, dim=1, keepdim=True)
    a_near = torch.gather(a_ego[:, :, :2], 1, a_idx.unsqueeze(-1).expand(-1, 1, 2)).squeeze(1)
    a_vec = torch.cat(
        [torch.clamp(a_near / 50.0, -1.0, 1.0), torch.clamp(1.0 - a_min_d / 25.0, 0.0, 1.0)],
        dim=-1,
    )

    v_ego, v_dist, _ = compute_egocentric_relative_vectors(
        root_pos_w, root_quat_w, vortex_pos, gaze_yaw_offset_rad=gaze_off, fov_half_rad=1.05, max_vis_dist=35.0, apply_visual_frustum_mask=True
    )
    v_min_d, v_idx = torch.min(v_dist, dim=1, keepdim=True)
    v_near = torch.gather(v_ego[:, :, :2], 1, v_idx.unsqueeze(-1).expand(-1, 1, 2)).squeeze(1)
    v_vec = torch.cat(
        [torch.clamp(v_near / 50.0, -1.0, 1.0), torch.clamp(1.0 - v_min_d / 30.0, 0.0, 1.0)],
        dim=-1,
    )

    s_ego, s_dist, s_bear = compute_egocentric_relative_vectors(
        root_pos_w, root_quat_w, signpost_pos, gaze_yaw_offset_rad=gaze_off, fov_half_rad=1.05, max_vis_dist=30.0, apply_visual_frustum_mask=True
    )
    s_min_d, s_idx = torch.min(s_dist, dim=1, keepdim=True)
    s_near = torch.gather(s_ego[:, :, :2], 1, s_idx.unsqueeze(-1).expand(-1, 1, 2)).squeeze(1)
    s_near_b = torch.gather(s_bear, 1, s_idx)
    s_vec = torch.cat(
        [
            torch.clamp(s_near / 50.0, -1.0, 1.0),
            torch.clamp(1.0 - s_min_d / 25.0, 0.0, 1.0),
            torch.clamp(s_near_b / math.pi, -1.0, 1.0),
        ],
        dim=-1,
    )

    o_ego, o_dist, _ = compute_egocentric_relative_vectors(
        root_pos_w, root_quat_w, obstacle_pos, gaze_yaw_offset_rad=gaze_off, fov_half_rad=1.05, max_vis_dist=35.0, apply_visual_frustum_mask=True
    )
    _, o_idx = torch.min(o_dist, dim=1, keepdim=True)
    o_near = torch.gather(o_ego[:, :, :2], 1, o_idx.unsqueeze(-1).expand(-1, 1, 2)).squeeze(1)
    o_vec = torch.clamp(o_near / 50.0, -1.0, 1.0)

    return torch.cat([m_vec, a_vec, v_vec, s_vec, o_vec], dim=-1)


def obs_egocentric_disasters_8d(env: ManagerBasedRLEnv) -> torch.Tensor:
    dis_16d = obs_egocentric_disasters_16d(env)
    return torch.cat(
        [
            dis_16d[:, 0:2],
            dis_16d[:, 4:6],
            dis_16d[:, 7:9],
            dis_16d[:, 10:12],
        ],
        dim=-1,
    )


def _compute_eye_ray_starts_and_dirs(
    env: ManagerBasedRLEnv,
    sensor_key: str,
    eye_offset_y: float,
    num_rays: int = 8,
    max_distance: float = 60.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ensure_sensorimotor_puzzle_buffers(env)
    robot = env.scene["robot"]
    root_pos_w = robot.data.root_pos_w
    root_quat_w = robot.data.root_quat_w
    device = root_pos_w.device
    num_envs = int(root_pos_w.shape[0])

    sensor = None
    if hasattr(env, "scene"):
        if hasattr(env.scene, "sensors") and sensor_key in env.scene.sensors:
            sensor = env.scene.sensors[sensor_key]
        elif isinstance(env.scene, dict) and sensor_key in env.scene:
            sensor = env.scene[sensor_key]
        else:
            try:
                sensor = env.scene[sensor_key]
            except Exception:
                sensor = None

    qw = root_quat_w[:, 0]
    qx = root_quat_w[:, 1]
    qy = root_quat_w[:, 2]
    qz = root_quat_w[:, 3]
    yaw = torch.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    cos_y = torch.cos(yaw)
    sin_y = torch.sin(yaw)

    ox = 0.12 * cos_y - float(eye_offset_y) * sin_y
    oy = 0.12 * sin_y + float(eye_offset_y) * cos_y
    oz = torch.full_like(ox, 0.55)
    eye_pos = root_pos_w + torch.stack([ox, oy, oz], dim=-1)
    if sensor is not None and hasattr(sensor, "_ray_starts_w"):
        ray_starts_w = sensor._ray_starts_w.to(device=device, dtype=torch.float32)
    else:
        ray_starts_w = eye_pos.unsqueeze(1).expand(num_envs, num_rays, 3)

    gaze_yaw = yaw + env.agent_gaze_yaw_offset_rad.to(device=device, dtype=torch.float32)
    gaze_pitch = torch.clamp(
        -6.0 * (math.pi / 180.0) + env.agent_gaze_pitch_offset_rad.to(device=device, dtype=torch.float32),
        -0.85,
        0.65,
    )

    horiz_deg = torch.linspace(-56.0, 56.0, num_rays, device=device, dtype=torch.float32)
    horiz_rad = horiz_deg * (math.pi / 180.0)
    ray_yaw = gaze_yaw.unsqueeze(1) + horiz_rad.unsqueeze(0)
    cos_v = torch.cos(gaze_pitch).unsqueeze(1)
    sin_v = torch.sin(gaze_pitch).unsqueeze(1).expand_as(ray_yaw)
    ray_dirs_w = torch.stack(
        [
            cos_v * torch.cos(ray_yaw),
            cos_v * torch.sin(ray_yaw),
            sin_v,
        ],
        dim=-1,
    )
    terrain_dist = torch.full((num_envs, num_rays), max_distance, dtype=torch.float32, device=device)
    if sensor is not None and hasattr(sensor, "data") and hasattr(sensor.data, "ray_hits_w"):
        ray_hits_w = sensor.data.ray_hits_w.to(device=device, dtype=torch.float32)
        if ray_hits_w.shape[1] == num_rays:
            raw_d = torch.linalg.norm(ray_hits_w - ray_starts_w, dim=-1)
            terrain_dist = torch.where(
                torch.isfinite(raw_d),
                torch.clamp(raw_d, 0.1, max_distance),
                terrain_dist,
            )
    return ray_starts_w, ray_dirs_w, terrain_dist


def _collect_retinal_scene_spheres(env: ManagerBasedRLEnv, device: torch.device | str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ensure_sensorimotor_puzzle_buffers(env)
    num_envs = int(env.scene["robot"].data.root_pos_w.shape[0])

    food_pos = get_food_positions_w(env)
    pred_pos = get_predator_positions_w(env)
    block_pos = get_toy_block_positions_w(env)
    slot_pos = get_toy_slot_positions_w(env)
    dis_pos = get_disaster_and_obstacle_positions_w(env)
    obj_pos = torch.cat([food_pos, pred_pos, block_pos, slot_pos, dis_pos], dim=1)

    radii_list = (
        [0.45] * 4
        + [0.65] * 4
        + [0.85] * 4
        + [0.95] * 4
        + [1.20] * 2
        + [2.50] * 2
        + [1.20] * 2
        + [1.10] * 2
        + [1.80] * 4
    )
    obj_radii = torch.tensor(radii_list, dtype=torch.float32, device=device).unsqueeze(0).expand(num_envs, -1)

    food_rgb = torch.tensor([0.15, 0.85, 0.25], dtype=torch.float32, device=device).view(1, 1, 3).expand(num_envs, 4, 3)
    pred_rgb = torch.tensor([0.90, 0.15, 0.15], dtype=torch.float32, device=device).view(1, 1, 3).expand(num_envs, 4, 3)
    block_rgb = env.toy_block_rgb.to(device=device, dtype=torch.float32)
    slot_rgb = torch.tensor([0.55, 0.10, 0.20], dtype=torch.float32, device=device).view(1, 1, 3).expand(num_envs, 4, 3)
    meteor_rgb = torch.tensor([0.90, 0.20, 0.75], dtype=torch.float32, device=device).view(1, 1, 3).expand(num_envs, 2, 3)
    abyss_rgb = torch.tensor([0.30, 0.05, 0.40], dtype=torch.float32, device=device).view(1, 1, 3).expand(num_envs, 2, 3)
    vortex_rgb = torch.tensor([0.20, 0.75, 0.95], dtype=torch.float32, device=device).view(1, 1, 3).expand(num_envs, 2, 3)
    sign_rgb = torch.tensor([0.95, 0.50, 0.10], dtype=torch.float32, device=device).view(1, 1, 3).expand(num_envs, 2, 3)
    obs_rgb = torch.tensor([0.95, 0.85, 0.10], dtype=torch.float32, device=device).view(1, 1, 3).expand(num_envs, 4, 3)

    obj_rgb = torch.cat(
        [food_rgb, pred_rgb, block_rgb, slot_rgb, meteor_rgb, abyss_rgb, vortex_rgb, sign_rgb, obs_rgb],
        dim=1,
    )
    return obj_pos, obj_radii, obj_rgb


def _intersect_rays_with_scene_objects(
    ray_starts_w: torch.Tensor,
    ray_dirs_w: torch.Tensor,
    terrain_dist: torch.Tensor,
    obj_pos: torch.Tensor,
    obj_radii: torch.Tensor,
    obj_rgb: torch.Tensor,
    max_distance: float = 60.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = ray_starts_w.device
    oc = obj_pos.unsqueeze(1) - ray_starts_w.unsqueeze(2)
    t_proj = torch.sum(oc * ray_dirs_w.unsqueeze(2), dim=-1)
    oc_sq = torch.sum(oc * oc, dim=-1)
    perp_sq = torch.clamp(oc_sq - t_proj * t_proj, min=0.0)
    rad_sq = obj_radii.unsqueeze(1) ** 2
    hit_mask = (t_proj > 0.1) & (perp_sq <= rad_sq)
    t_hit = torch.where(
        hit_mask,
        torch.clamp(t_proj - torch.sqrt(torch.clamp(rad_sq - perp_sq, min=0.0)), min=0.1),
        torch.full_like(t_proj, max_distance + 10.0),
    )
    min_obj_t, min_obj_idx = torch.min(t_hit, dim=-1)

    terrain_rgb = torch.tensor([0.32, 0.42, 0.28], dtype=torch.float32, device=device).view(1, 1, 3).expand_as(ray_starts_w)
    idx_exp = min_obj_idx.unsqueeze(-1).expand(-1, -1, 3)
    nearest_obj_rgb = torch.gather(obj_rgb, 1, idx_exp)

    use_obj = min_obj_t < terrain_dist
    final_dist = torch.where(use_obj, min_obj_t, terrain_dist)
    final_rgb = torch.where(use_obj.unsqueeze(-1), nearest_obj_rgb, terrain_rgb)
    return final_dist, final_rgb


def obs_binocular_retinal_raycast_16d(env: ManagerBasedRLEnv) -> torch.Tensor:
    device = env.scene["robot"].data.root_pos_w.device
    max_dist = 60.0
    l_starts, l_dirs, l_terr_d = _compute_eye_ray_starts_and_dirs(
        env, "left_eye_raycaster", eye_offset_y=0.035, num_rays=8, max_distance=max_dist
    )
    r_starts, r_dirs, r_terr_d = _compute_eye_ray_starts_and_dirs(
        env, "right_eye_raycaster", eye_offset_y=-0.035, num_rays=8, max_distance=max_dist
    )
    obj_pos, obj_radii, obj_rgb = _collect_retinal_scene_spheres(env, device)

    d_left, rgb_left = _intersect_rays_with_scene_objects(
        l_starts, l_dirs, l_terr_d, obj_pos, obj_radii, obj_rgb, max_distance=max_dist
    )
    d_right, rgb_right = _intersect_rays_with_scene_objects(
        r_starts, r_dirs, r_terr_d, obj_pos, obj_radii, obj_rgb, max_distance=max_dist
    )

    num_envs = int(d_left.shape[0])
    num_rays = int(d_left.shape[1])
    mid = max(1, num_rays // 2)

    mean_rgb_3d = 0.5 * (torch.mean(rgb_left, dim=1) + torch.mean(rgb_right, dim=1))
    foveal_rgb_3d = 0.5 * (
        torch.mean(rgb_left[:, mid - 1 : mid + 1, :], dim=1)
        + torch.mean(rgb_right[:, mid - 1 : mid + 1, :], dim=1)
    )

    l_prox = torch.clamp(1.0 - d_left / max_dist, 0.0, 1.0)
    r_prox = torch.clamp(1.0 - d_right / max_dist, 0.0, 1.0)
    if num_rays >= 4:
        l_sectors = l_prox.view(num_envs, 4, -1).mean(dim=-1)
        r_sectors = r_prox.view(num_envs, 4, -1).mean(dim=-1)
    else:
        l_sectors = l_prox[:, :4]
        r_sectors = r_prox[:, :4]

    disparity = torch.clamp((1.0 / torch.clamp(d_left, min=0.25)) - (1.0 / torch.clamp(d_right, min=0.25)), -1.0, 1.0)
    mean_disp = torch.mean(disparity, dim=1, keepdim=True)
    foveal_disp = torch.mean(torch.abs(disparity[:, mid - 1 : mid + 1]), dim=1, keepdim=True)

    return torch.cat(
        [
            mean_rgb_3d,
            foveal_rgb_3d,
            l_sectors,
            r_sectors,
            mean_disp,
            foveal_disp,
        ],
        dim=-1,
    )


def compute_egocentric_33d_observation(env: ManagerBasedRLEnv) -> torch.Tensor:
    ego_pred_8d = obs_egocentric_predators_8d(env)
    ego_food_8d = obs_egocentric_foods_8d(env)
    ego_biome_8d = obs_egocentric_biome_centers_8d(env)
    ego_biome_vert_2d = obs_egocentric_biome_vertical_2d(env)
    ego_somatic_7d = obs_egocentric_somatic_7d(env)
    return torch.cat(
        [ego_pred_8d, ego_food_8d, ego_biome_8d, ego_biome_vert_2d, ego_somatic_7d],
        dim=-1,
    )


def compute_toy_and_disaster_36d_observation(env: ManagerBasedRLEnv) -> torch.Tensor:
    ego_blocks_8d = obs_egocentric_toy_blocks_8d(env)
    ego_slots_8d = obs_egocentric_toy_slots_8d(env)
    mutual_12d = obs_toy_block_slot_mutual_distances_12d(env)
    ego_disasters_8d = obs_egocentric_disasters_8d(env)
    return torch.cat(
        [ego_blocks_8d, ego_slots_8d, mutual_12d, ego_disasters_8d],
        dim=-1,
    )


@configclass
class ActionsCfg:
    joint_effort = mdp.JointEffortActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale={
            ".*_waist.*": 67.5,
            ".*_upper_arm.*": 67.5,
            "pelvis": 67.5,
            ".*_lower_arm": 45.0,
            ".*_thigh:0": 45.0,
            ".*_thigh:1": 135.0,
            ".*_thigh:2": 45.0,
            ".*_shin": 90.0,
            ".*_foot.*": 22.5,
        },
    )
    sensorimotor_11d = Sensorimotor11DActionCfg(asset_name="robot")
    vocal_signal_2d = VocalSignal2DActionCfg(asset_name="robot")


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        binocular_retinal = ObsTerm(func=obs_binocular_retinal_raycast_16d)
        egocentric_predators = ObsTerm(func=obs_egocentric_predators_8d)
        egocentric_foods = ObsTerm(func=obs_egocentric_foods_8d)
        egocentric_biome_centers = ObsTerm(func=obs_egocentric_biome_centers_8d)
        egocentric_biome_vertical = ObsTerm(func=obs_egocentric_biome_vertical_2d)
        egocentric_somatic = ObsTerm(func=obs_egocentric_somatic_7d)
        egocentric_toy_blocks = ObsTerm(func=obs_egocentric_toy_blocks_8d)
        egocentric_toy_slots = ObsTerm(func=obs_egocentric_toy_slots_8d)
        toy_block_slot_mutual_distances = ObsTerm(func=obs_toy_block_slot_mutual_distances_12d)
        egocentric_disasters = ObsTerm(func=obs_egocentric_disasters_16d)
        sensorimotor_puzzle = ObsTerm(func=obs_sensorimotor_puzzle_state_12d)
        sensorimotor_puzzle_attributes = ObsTerm(func=obs_nearest_toy_puzzle_attributes_6d)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class ProprioceptionCfg(ObsGroup):
        base_height = ObsTerm(func=mdp.base_pos_z)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.25)
        joint_pos_norm = ObsTerm(func=mdp.joint_pos_limit_normalized)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.1)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    proprioception: ProprioceptionCfg = ProprioceptionCfg()


@configclass
class EventCfg:
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={"pose_range": {}, "velocity_range": {}},
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.1, 0.1),
            "velocity_range": (-0.05, 0.05),
        },
    )

    reset_scene_objects = EventTerm(
        func=reset_scene_rigid_objects,
        mode="reset",
    )

    reset_homeostasis = EventTerm(
        func=reset_homeostasis_and_curiosity,
        mode="reset",
    )

    reset_puzzle = EventTerm(
        func=reset_sensorimotor_puzzle_state,
        mode="reset",
    )


@configclass
class RewardsCfg:
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    homeostatic_vitality = RewTerm(func=reward_homeostatic_vitality, weight=1.0)
    biome_quadrant_curiosity = RewTerm(
        func=reward_biome_quadrant_curiosity,
        weight=1.0,
        params={"bonus_value": CURIOSITY_NOVEL_BIOME_REWARD},
    )
    toy_sensorimotor_puzzle = RewTerm(
        func=reward_toy_sensorimotor_puzzle,
        weight=1.0,
    )


def termination_time_out(env: ManagerBasedRLEnv) -> torch.Tensor:
    if bool(getattr(env, "manual_mortality_mode", False)):
        return torch.zeros((env.num_envs,), dtype=torch.bool, device=env.device)
    return mdp.time_out(env)


def termination_torso_height(env: ManagerBasedRLEnv, minimum_height: float = 0.8) -> torch.Tensor:
    if bool(getattr(env, "manual_mortality_mode", False)):
        return torch.zeros((env.num_envs,), dtype=torch.bool, device=env.device)
    return mdp.root_height_below_minimum(env, minimum_height=minimum_height)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=termination_time_out, time_out=True)
    torso_height = DoneTerm(func=termination_torso_height, params={"minimum_height": 0.8})
    hp_depleted = DoneTerm(func=termination_hp_depleted, params={"min_hp": 0.0})


@configclass
class FourBiomeHumanoidEnvCfg(ManagerBasedRLEnvCfg):
    scene: FourBiomeHumanoidSceneCfg = FourBiomeHumanoidSceneCfg(
        num_envs=1,
        env_spacing=100.0,
        clone_in_fabric=False,
    )
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 16.0
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physics_material.static_friction = 1.0
        self.sim.physics_material.dynamic_friction = 1.0
        self.sim.physics_material.restitution = 0.0


_GLOBAL_MANAGER_RL_ENV: ManagerBasedRLEnv | None = globals().get("_GLOBAL_MANAGER_RL_ENV", None)


def create_four_biome_humanoid_env(num_envs: int = 1, device: str = "cuda:0") -> ManagerBasedRLEnv:
    global _GLOBAL_MANAGER_RL_ENV
    env_cfg = FourBiomeHumanoidEnvCfg()
    env_cfg.scene.num_envs = int(num_envs)
    env_cfg.sim.device = str(device)
    env = ManagerBasedRLEnv(cfg=env_cfg)
    env.reset()
    _GLOBAL_MANAGER_RL_ENV = env
    return env


def get_or_create_four_biome_humanoid_env(num_envs: int = 4, device: str = "cuda:0") -> ManagerBasedRLEnv:
    global _GLOBAL_MANAGER_RL_ENV
    if _GLOBAL_MANAGER_RL_ENV is None and globals().get("_ACTIVE_ISAACLAB_ENV", None) is not None:
        _GLOBAL_MANAGER_RL_ENV = globals().get("_ACTIVE_ISAACLAB_ENV")
    if _GLOBAL_MANAGER_RL_ENV is not None:
        return _GLOBAL_MANAGER_RL_ENV
    return create_four_biome_humanoid_env(num_envs=max(4, int(num_envs)), device=device)


class IsaacLabUnified3DArena:
    def __init__(
        self,
        pop_size: int = 4,
        n_food: int = 38,
        n_pred: int = 5,
        n_obs: int = 8,
        n_meteors: int = 6,
        seed: int | None = None,
        device: str = "cuda:0",
    ):
        self.rl_env = get_or_create_four_biome_humanoid_env(num_envs=max(4, int(pop_size)), device=device)
        self.rl_env.manual_mortality_mode = False
        self.device = self.rl_env.device
        self.pop_size = min(int(pop_size), int(self.rl_env.num_envs))
        self.n_food = int(n_food)
        self.n_pred = int(n_pred)
        self.n_obs = int(n_obs)
        self.n_meteors = int(n_meteors)
        self.n_toys = 4
        self.world_size = 100.0
        self.dt = float(self.rl_env.step_dt)
        self.unpredictable_mode = "standard"
        self.procedural_cycle = 0
        self.current_hazard_dna = {}
        self.threat_colors = [2, 6]
        self.obstacle_colors = [4, 9]
        self.agent_body_logits = torch.zeros((self.pop_size, 11), dtype=torch.float32, device=self.device)
        self.agent_joint_effort_21d = torch.zeros((self.pop_size, 21), dtype=torch.float32, device=self.device)
        self.last_binocular_retinal_bundle = None
        get_global_isaac_warp_engine().bound_isaaclab_env = self.rl_env
        if seed is not None:
            torch.manual_seed(int(seed))
        self.reset_world("standard")

    @property
    def step_idx(self) -> int:
        return int(getattr(self.rl_env, "common_step_counter", 0))

    @property
    def pos(self) -> torch.Tensor:
        robot = self.rl_env.scene["robot"]
        origins = _extract_env_origins(self.rl_env, self.rl_env.num_envs, self.device)
        p = (robot.data.root_pos_w - origins)[: self.pop_size].clone()
        p[:, 0] = torch.clamp(p[:, 0] + 50.0, 0.0, self.world_size)
        p[:, 1] = torch.clamp(p[:, 1] + 50.0, 0.0, self.world_size)
        return p

    @property
    def vel(self) -> torch.Tensor:
        robot = self.rl_env.scene["robot"]
        return robot.data.root_lin_vel_w[: self.pop_size].clone()

    @property
    def yaw(self) -> torch.Tensor:
        return _quat_wxyz_to_yaw_rad(self.rl_env.scene["robot"].data.root_quat_w[: self.pop_size])

    @property
    def gaze_yaw_offset(self) -> torch.Tensor:
        ensure_sensorimotor_puzzle_buffers(self.rl_env)
        return self.rl_env.agent_gaze_yaw_offset_rad[: self.pop_size]

    @property
    def gaze_pitch_offset(self) -> torch.Tensor:
        ensure_sensorimotor_puzzle_buffers(self.rl_env)
        return self.rl_env.agent_gaze_pitch_offset_rad[: self.pop_size]

    @property
    def foveal_hit_verified(self) -> torch.Tensor:
        ensure_sensorimotor_puzzle_buffers(self.rl_env)
        return self.rl_env.agent_foveal_hit_verified[: self.pop_size]

    @property
    def hp(self) -> torch.Tensor:
        ensure_homeostasis_curiosity_buffers(self.rl_env)
        return self.rl_env.agent_hp[: self.pop_size]

    @hp.setter
    def hp(self, val: torch.Tensor):
        ensure_homeostasis_curiosity_buffers(self.rl_env)
        self.rl_env.agent_hp[: self.pop_size] = val[: self.pop_size]

    @property
    def energy(self) -> torch.Tensor:
        ensure_homeostasis_curiosity_buffers(self.rl_env)
        return self.rl_env.agent_energy[: self.pop_size]

    @energy.setter
    def energy(self, val: torch.Tensor):
        ensure_homeostasis_curiosity_buffers(self.rl_env)
        self.rl_env.agent_energy[: self.pop_size] = val[: self.pop_size]

    @property
    def pain(self) -> torch.Tensor:
        ensure_homeostasis_curiosity_buffers(self.rl_env)
        return self.rl_env.agent_pain[: self.pop_size]

    @pain.setter
    def pain(self, val: torch.Tensor):
        ensure_homeostasis_curiosity_buffers(self.rl_env)
        self.rl_env.agent_pain[: self.pop_size] = val[: self.pop_size]

    @property
    def surprise(self) -> torch.Tensor:
        ensure_homeostasis_curiosity_buffers(self.rl_env)
        return self.rl_env.agent_surprise[: self.pop_size]

    @surprise.setter
    def surprise(self, val: torch.Tensor):
        ensure_homeostasis_curiosity_buffers(self.rl_env)
        self.rl_env.agent_surprise[: self.pop_size] = val[: self.pop_size]

    @property
    def alive(self) -> torch.Tensor:
        ensure_homeostasis_curiosity_buffers(self.rl_env)
        return self.rl_env.agent_alive[: self.pop_size]

    @alive.setter
    def alive(self, val: torch.Tensor):
        ensure_homeostasis_curiosity_buffers(self.rl_env)
        self.rl_env.agent_alive[: self.pop_size] = val[: self.pop_size]

    @property
    def active_env_id(self) -> int:
        return int(getattr(self, "_active_env_id", 0))

    @active_env_id.setter
    def active_env_id(self, val: int) -> None:
        self._active_env_id = max(0, min(int(val), self.pop_size - 1))

    def _env_origin(self, env_i: int) -> torch.Tensor:
        num_envs = int(self.rl_env.scene["robot"].data.root_pos_w.shape[0])
        origins = _extract_env_origins(self.rl_env, num_envs, self.device)
        return origins[env_i]

    @property
    def food_pos(self) -> torch.Tensor:
        env_i = self.active_env_id
        fp = (get_food_positions_w(self.rl_env)[env_i] - self._env_origin(env_i)).clone()
        fp[:, 0] = fp[:, 0] + 50.0
        fp[:, 1] = fp[:, 1] + 50.0
        return fp

    @property
    def pred_pos(self) -> torch.Tensor:
        env_i = self.active_env_id
        pp = (get_predator_positions_w(self.rl_env)[env_i] - self._env_origin(env_i)).clone()
        pp[:, 0] = pp[:, 0] + 50.0
        pp[:, 1] = pp[:, 1] + 50.0
        return pp

    @property
    def obs_pos(self) -> torch.Tensor:
        env_i = self.active_env_id
        dis = get_disaster_and_obstacle_positions_w(self.rl_env)[env_i, 8:12, :] - self._env_origin(env_i)
        op = dis.clone()
        op[:, 0] = op[:, 0] + 50.0
        op[:, 1] = op[:, 1] + 50.0
        return op

    @property
    def meteors(self) -> torch.Tensor:
        env_i = self.active_env_id
        dis = get_disaster_and_obstacle_positions_w(self.rl_env)[env_i, 0:2, :] - self._env_origin(env_i)
        mp = dis.clone()
        mp[:, 0] = mp[:, 0] + 50.0
        mp[:, 1] = mp[:, 1] + 50.0
        return mp

    @property
    def toy_block_pos(self) -> torch.Tensor:
        env_i = self.active_env_id
        tb = (get_toy_block_positions_w(self.rl_env)[env_i] - self._env_origin(env_i)).clone()
        tb[:, 0] = tb[:, 0] + 50.0
        tb[:, 1] = tb[:, 1] + 50.0
        return tb

    @property
    def toy_slot_pos(self) -> torch.Tensor:
        env_i = self.active_env_id
        ts = (get_toy_slot_positions_w(self.rl_env)[env_i] - self._env_origin(env_i)).clone()
        ts[:, 0] = ts[:, 0] + 50.0
        ts[:, 1] = ts[:, 1] + 50.0
        return ts

    @property
    def toy_type(self) -> torch.Tensor:
        ensure_sensorimotor_puzzle_buffers(self.rl_env)
        return self.rl_env.toy_type

    @property
    def toy_block_color(self) -> torch.Tensor:
        ensure_sensorimotor_puzzle_buffers(self.rl_env)
        return self.rl_env.toy_block_color[self.active_env_id]

    @property
    def toy_slot_color(self) -> torch.Tensor:
        ensure_sensorimotor_puzzle_buffers(self.rl_env)
        return self.rl_env.toy_sample_val_color[self.active_env_id]

    @property
    def toy_sample_key_color(self) -> torch.Tensor:
        ensure_sensorimotor_puzzle_buffers(self.rl_env)
        return self.rl_env.toy_sample_key_color[self.active_env_id]

    @property
    def toy_sample_val_color(self) -> torch.Tensor:
        ensure_sensorimotor_puzzle_buffers(self.rl_env)
        return self.rl_env.toy_sample_val_color[self.active_env_id]

    @property
    def toy_slot_yaw_rad(self) -> torch.Tensor:
        ensure_sensorimotor_puzzle_buffers(self.rl_env)
        return self.rl_env.toy_slot_yaw_rad[self.active_env_id]

    @property
    def toy_block_yaw_rad(self) -> torch.Tensor:
        ensure_sensorimotor_puzzle_buffers(self.rl_env)
        return self.rl_env.toy_block_yaw_rad[self.active_env_id]

    @property
    def toy_insertion_depth(self) -> torch.Tensor:
        ensure_sensorimotor_puzzle_buffers(self.rl_env)
        return self.rl_env.toy_insertion_depth[self.active_env_id]

    @property
    def toys_solved_total(self) -> int:
        return int(getattr(self.rl_env, "toys_solved_total", 0))

    @property
    def curiosity_explorations_total(self) -> int:
        return int(getattr(self.rl_env, "curiosity_explorations_total", 0))

    @property
    def threat_observations_total(self) -> int:
        return int(getattr(self.rl_env, "threat_observations_total", 0))

    @property
    def biome_dispersion_entropy(self) -> float:
        return float(getattr(self.rl_env, "biome_dispersion_entropy", 0.0))

    @property
    def sensorimotor_chunk_hits_total(self) -> int:
        return int(getattr(self.rl_env, "sensorimotor_chunk_hits_total", 0))

    @property
    def cerebellar_corrections_total(self) -> int:
        return int(getattr(self.rl_env, "cerebellar_corrections_total", 0))

    @property
    def discovered_sensorimotor_chunks(self) -> dict:
        return getattr(self.rl_env, "discovered_sensorimotor_chunks", {})

    @property
    def gait_phase_rad(self) -> torch.Tensor:
        ensure_homeostasis_curiosity_buffers(self.rl_env)
        return self.rl_env.agent_gait_phase_rad[: self.pop_size]

    @property
    def torso_upright_cos(self) -> torch.Tensor:
        ensure_homeostasis_curiosity_buffers(self.rl_env)
        return self.rl_env.agent_torso_upright_cos[: self.pop_size]

    @property
    def gait_coordination_score(self) -> torch.Tensor:
        ensure_homeostasis_curiosity_buffers(self.rl_env)
        return self.rl_env.agent_gait_coordination_score[: self.pop_size]

    @property
    def bipedal_gait_cycles_total(self) -> int:
        return int(getattr(self.rl_env, "bipedal_gait_cycles_total", 0))

    @property
    def vestibular_balance_corrections_total(self) -> int:
        return int(getattr(self.rl_env, "vestibular_balance_corrections_total", 0))

    @property
    def hand_contact_verified(self) -> torch.Tensor:
        ensure_sensorimotor_puzzle_buffers(self.rl_env)
        return self.rl_env.agent_hand_contact_verified[: self.pop_size]

    @property
    def active_hand_pos_w(self) -> torch.Tensor:
        ensure_sensorimotor_puzzle_buffers(self.rl_env)
        return self.rl_env.agent_active_hand_pos_w[: self.pop_size]

    @property
    def hand_to_block_dist_m(self) -> torch.Tensor:
        ensure_sensorimotor_puzzle_buffers(self.rl_env)
        return self.rl_env.agent_hand_to_block_dist_m[: self.pop_size]

    @property
    def physical_hand_contacts_total(self) -> int:
        return int(getattr(self.rl_env, "physical_hand_contacts_total", 0))

    @property
    def upper_limb_reach_movements_total(self) -> int:
        return int(getattr(self.rl_env, "upper_limb_reach_movements_total", 0))

    def sample_procedural_hazard_dna(self, cycle_idx: int = 0) -> dict:
        c = int(cycle_idx)
        phase = float(c) * 0.75 + 0.35
        dna = {
            "cycle_id": c,
            "pred_speed": round(2.2 + 0.6 * abs(math.sin(phase)), 3),
            "meteor_vz": round(-4.5 - 1.2 * abs(math.cos(phase)), 3),
            "vortex_pull": round(0.70 + 0.30 * abs(math.sin(phase * 1.3)), 3),
            "abyss_pull": round(0.55 + 0.25 * abs(math.cos(phase * 0.9)), 3),
            "mimic_ratio": round(0.20 + 0.15 * abs(math.sin(phase * 0.5)), 3),
            "terrain_shift_x": round(4.0 * math.sin(phase * 0.6), 3),
            "signpost_freq": round(0.15 + 0.10 * abs(math.cos(phase * 1.1)), 3),
        }
        self.rl_env.terrain_shift_x = float(dna["terrain_shift_x"])
        self.rl_env.current_hazard_dna = dict(dna)
        return dna

    def terrain_height(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        shift_x = float(getattr(self.rl_env, "terrain_shift_x", 0.0))
        return compute_gaussian_wave_height_torch(x, y, terrain_shift_x=shift_x)

    def reset_world(self, unpredictable_mode: str | None = None, keep_discoveries: bool = False):
        if unpredictable_mode is not None:
            self.unpredictable_mode = str(unpredictable_mode)
        saved_chunks = dict(self.discovered_sensorimotor_chunks) if keep_discoveries else {}
        saved_hits = self.sensorimotor_chunk_hits_total if keep_discoveries else 0
        saved_cereb = self.cerebellar_corrections_total if keep_discoveries else 0
        saved_gait = self.bipedal_gait_cycles_total if keep_discoveries else 0
        saved_vest = self.vestibular_balance_corrections_total if keep_discoveries else 0
        saved_contacts = self.physical_hand_contacts_total if keep_discoveries else 0
        saved_reaches = self.upper_limb_reach_movements_total if keep_discoveries else 0
        self.rl_env.unpredictable_mode = self.unpredictable_mode
        self.rl_env.reset()
        self.rl_env.unpredictable_mode = self.unpredictable_mode
        all_ids = torch.arange(self.rl_env.num_envs, dtype=torch.long, device=self.device)
        reset_scene_rigid_objects(self.rl_env, all_ids)
        reset_homeostasis_and_curiosity(self.rl_env, all_ids, mark_initial_quadrant=True)
        reset_sensorimotor_puzzle_state(self.rl_env, all_ids)
        self.rl_env.threat_observations_total = 0
        self.rl_env.biome_dispersion_entropy = 0.0
        if self.unpredictable_mode == "infinite_procedural_chaos":
            self.current_hazard_dna = self.sample_procedural_hazard_dna(self.procedural_cycle)
            self.procedural_cycle += 1
        else:
            self.current_hazard_dna = self.sample_procedural_hazard_dna(0)
        if keep_discoveries:
            self.rl_env.discovered_sensorimotor_chunks = saved_chunks
            self.rl_env.sensorimotor_chunk_hits_total = saved_hits
            self.rl_env.cerebellar_corrections_total = saved_cereb
            self.rl_env.bipedal_gait_cycles_total = saved_gait
            self.rl_env.vestibular_balance_corrections_total = saved_vest
            self.rl_env.physical_hand_contacts_total = saved_contacts
            self.rl_env.upper_limb_reach_movements_total = saved_reaches
        self.agent_body_logits.zero_()
        self.agent_joint_effort_21d.zero_()

    def reset_at_study_room_desk(self, env_ids: torch.Tensor | None = None) -> None:
        reset_humanoid_at_study_room_desk(self.rl_env, env_ids=env_ids)
        self.agent_body_logits.zero_()
        self.agent_joint_effort_21d.zero_()

    def get_proprioception_83d(self) -> torch.Tensor:
        ensure_homeostasis_curiosity_buffers(self.rl_env)
        robot = self.rl_env.scene["robot"]
        N = self.pop_size
        base_lin_vel = robot.data.root_lin_vel_b[:N]
        base_ang_vel = robot.data.root_ang_vel_b[:N]
        projected_gravity = robot.data.projected_gravity_b[:N]
        joint_pos = (robot.data.joint_pos - robot.data.default_joint_pos)[:N]
        joint_vel = robot.data.joint_vel[:N]
        raw_act = self.rl_env.action_manager.action
        if raw_act.shape[-1] >= 21:
            last_action_21d = raw_act[:N, :21]
        else:
            last_action_21d = torch.zeros((N, 21), dtype=torch.float32, device=self.device)
        homeostasis_5d = torch.stack(
            [
                self.rl_env.agent_hp[:N],
                self.rl_env.agent_energy[:N],
                self.rl_env.agent_pain[:N],
                self.rl_env.agent_surprise[:N],
                self.rl_env.agent_alive[:N].to(dtype=torch.float32),
            ],
            dim=-1,
        )
        _, arena_xy = compute_local_and_arena_xy(self.rl_env)
        quad_idx = compute_biome_quadrant_index(arena_xy[:N, 0], arena_xy[:N, 1])
        biome_id_4d = torch.nn.functional.one_hot(quad_idx, num_classes=4).to(dtype=torch.float32)
        env_indices = torch.arange(N, dtype=torch.long, device=self.device)
        cur_visit = self.rl_env.agent_visited_quadrants[env_indices, quad_idx].unsqueeze(-1)
        cur_res = self.rl_env.quadrant_resource_level[env_indices, quad_idx].unsqueeze(-1)
        cur_novelty_2d = torch.cat([1.0 - cur_visit, cur_res], dim=-1)
        return torch.cat(
            [
                base_lin_vel,
                base_ang_vel,
                projected_gravity,
                joint_pos,
                joint_vel,
                last_action_21d,
                homeostasis_5d,
                biome_id_4d,
                cur_novelty_2d,
            ],
            dim=-1,
        )

    def get_policy_observations_111d(self) -> torch.Tensor:
        prop_83 = self.get_proprioception_83d()
        ret_16 = obs_binocular_retinal_raycast_16d(self.rl_env)[: self.pop_size]
        puz_12 = obs_sensorimotor_puzzle_state_12d(self.rl_env)[: self.pop_size]
        return torch.cat([prop_83, ret_16, puz_12], dim=-1)

    def render_first_person_retinal_grids(self, retina_h: int = 9, retina_w: int = 9):
        N = self.pop_size
        food_cols = getattr(self.rl_env, "food_colors", torch.tensor([3, 3, 3, 3], dtype=torch.int32, device=self.device))
        box_groups = [
            (self.obs_pos, (3.2, 3.2, 2.5), 0.0, 4),
            (self.food_pos, (1.6, 1.6, 1.2), 0.0, food_cols),
            (self.pred_pos, (2.2, 2.2, 1.8), 0.0, 2),
            (self.meteors, (2.4, 2.4, 2.0), 0.0, 6),
            (self.toy_slot_pos, (1.8, 1.8, 0.9), self.toy_slot_yaw_rad, 9),
            (self.toy_block_pos, (1.4, 1.4, 1.1), self.toy_block_yaw_rad, 7),
        ]
        env_i = self.active_env_id
        dis_local = get_disaster_and_obstacle_positions_w(self.rl_env)[env_i] - self._env_origin(env_i)
        ab_3d = dis_local[2:4].clone()
        ab_3d[:, 0] += 50.0
        ab_3d[:, 1] += 50.0
        vx_3d = dis_local[4:6].clone()
        vx_3d[:, 0] += 50.0
        vx_3d[:, 1] += 50.0
        pn_3d = dis_local[6:8].clone()
        pn_3d[:, 0] += 50.0
        pn_3d[:, 1] += 50.0
        box_groups.append((ab_3d, (3.0, 3.0, 1.2), 0.0, 6))
        box_groups.append((vx_3d, (2.5, 2.5, 1.5), 0.0, 6))
        panel_safe = bool(torch.any(self.rl_env.agent_signpost_cued[:N]).item())
        pn_col = 7 if panel_safe else 9
        box_groups.append((pn_3d, (1.8, 1.8, 1.4), 0.0, pn_col))

        warp_engine = get_global_isaac_warp_engine()
        warp_engine.bound_isaaclab_env = self.rl_env
        prev_lum = getattr(self, "prev_left_retinal_lum", None)
        bino_out = warp_engine.render_arena_population_binocular_gpu(
            self.pos,
            self.yaw,
            box_groups,
            retina_h=int(retina_h),
            retina_w=int(retina_w),
            prev_left_lum_t=prev_lum,
            agent_gaze_yaw_offset_t=self.gaze_yaw_offset,
            agent_gaze_pitch_offset_t=self.gaze_pitch_offset,
        )
        self.prev_left_retinal_lum = bino_out["left_lum"].detach().clone()
        self.last_binocular_retinal_bundle = bino_out

        retinas = bino_out["left_color"].clone()
        z_buffers = bino_out["reconstructed_depth"].clone()
        horizon_r = int(retina_h) // 2
        center_c = int(retina_w) // 2
        ground_z = self.terrain_height(self.pos[:, 0], self.pos[:, 1])
        high_ground = (ground_z > 5.0).unsqueeze(-1).unsqueeze(-1)
        ground_fill = torch.where(
            high_ground.expand(N, int(retina_h) - horizon_r - 1, int(retina_w)),
            torch.full((N, int(retina_h) - horizon_r - 1, int(retina_w)), 5, dtype=torch.int32, device=self.device),
            torch.full((N, int(retina_h) - horizon_r - 1, int(retina_w)), 1, dtype=torch.int32, device=self.device),
        )
        retinas[:, horizon_r + 1 :, :] = torch.where(
            retinas[:, horizon_r + 1 :, :] == 0,
            ground_fill,
            retinas[:, horizon_r + 1 :, :],
        )

        ray_range = 32.0
        prox_map = torch.clamp(1.0 - z_buffers / ray_range, 0.0, 1.0)
        if isinstance(food_cols, torch.Tensor) and food_cols.ndim > 0:
            if food_cols.numel() == N:
                fc = food_cols.view(N, 1, 1).to(self.device)
            else:
                fc = food_cols[0].to(self.device)
            is_food_mask = (retinas == fc).to(torch.float32)
        else:
            is_food_mask = (retinas == int(food_cols)).to(torch.float32)
        tc_tensor = torch.tensor(self.threat_colors, device=self.device)
        oc_tensor = torch.tensor(self.obstacle_colors, device=self.device)
        is_threat_mask = torch.isin(retinas, tc_tensor).to(torch.float32)
        is_obs_mask = torch.isin(retinas, oc_tensor).to(torch.float32)

        r_grid = torch.arange(int(retina_h), device=self.device).view(1, int(retina_h), 1)
        c_grid = torch.arange(int(retina_w), device=self.device).view(1, 1, int(retina_w))
        fov_mask = (((r_grid - horizon_r).abs() <= 1) & ((c_grid - center_c).abs() <= 1)).to(torch.float32)
        per_mask = 1.0 - fov_mask

        foveal_salience = torch.zeros((N, 4), dtype=torch.float32, device=self.device)
        peripheral_salience = torch.zeros((N, 4), dtype=torch.float32, device=self.device)

        foveal_salience[:, 0] = torch.clamp((prox_map * is_food_mask * fov_mask).sum(dim=(1, 2)), 0.0, 1.0)
        foveal_salience[:, 1] = torch.clamp((prox_map * is_threat_mask * fov_mask).sum(dim=(1, 2)), 0.0, 1.0)
        foveal_salience[:, 2] = torch.clamp((prox_map * is_obs_mask * fov_mask).sum(dim=(1, 2)), 0.0, 1.0)

        peripheral_salience[:, 0] = torch.clamp((prox_map * is_food_mask * per_mask).sum(dim=(1, 2)), 0.0, 1.0)
        peripheral_salience[:, 1] = torch.clamp((prox_map * is_threat_mask * per_mask).sum(dim=(1, 2)), 0.0, 1.0)
        peripheral_salience[:, 2] = torch.clamp((prox_map * is_obs_mask * per_mask).sum(dim=(1, 2)), 0.0, 1.0)

        fovea_patch = retinas[:, horizon_r - 1 : horizon_r + 2, center_c - 1 : center_c + 2]
        foveal_salience[:, 3] = (fovea_patch != 0).float().mean(dim=(1, 2))
        peripheral_salience[:, 3] = (retinas != 0).float().mean(dim=(1, 2))
        return retinas, foveal_salience, peripheral_salience

    def extract_retinal_16d_observations(self, retinas=None, foveal_salience=None, peripheral_salience=None) -> torch.Tensor:
        if retinas is None or foveal_salience is None or peripheral_salience is None:
            retinas, foveal_salience, peripheral_salience = self.render_first_person_retinal_grids()
        N, rh, rw = retinas.shape
        g_scale = float(max(rh, rw, 1))
        r_coords = torch.arange(rh, dtype=torch.float32, device=self.device).view(1, rh, 1).expand(N, rh, rw)
        c_coords = torch.arange(rw, dtype=torch.float32, device=self.device).view(1, 1, rw).expand(N, rh, rw)
        food_cols = getattr(self.rl_env, "food_colors", torch.tensor([3, 3, 3, 3], dtype=torch.int32, device=self.device))
        if isinstance(food_cols, torch.Tensor) and food_cols.ndim > 0:
            if food_cols.numel() == N:
                fc = food_cols.view(N, 1, 1).to(self.device)
            else:
                fc = food_cols[0].to(self.device)
            food_mask = (retinas == fc).to(torch.float32)
        else:
            food_mask = (retinas == int(food_cols)).to(torch.float32)
        tc_tensor = torch.tensor(self.threat_colors, device=self.device)
        oc_tensor = torch.tensor(self.obstacle_colors, device=self.device)
        threat_mask = torch.isin(retinas, tc_tensor).to(torch.float32)
        obs_mask = torch.isin(retinas, oc_tensor).to(torch.float32)
        ray_range = 32.0
        if self.last_binocular_retinal_bundle is not None and self.last_binocular_retinal_bundle["reconstructed_depth"].shape == retinas.shape:
            rec_z = self.last_binocular_retinal_bundle["reconstructed_depth"]
            prox_map = torch.clamp(1.0 - rec_z / ray_range, 0.0, 1.0)
        else:
            prox_map = torch.ones_like(food_mask)
        f_sum = food_mask.sum(dim=(1, 2)) + 1e-6
        has_food = (food_mask.sum(dim=(1, 2)) > 0.5).float()
        dx_t = (((food_mask * c_coords).sum(dim=(1, 2)) / f_sum - rw * 0.5) / g_scale) * has_food
        dy_t = (((food_mask * r_coords).sum(dim=(1, 2)) / f_sum - rh * 0.5) / g_scale) * has_food
        dz_t = ((food_mask * prox_map).sum(dim=(1, 2)) / f_sum) * has_food
        ratio_w = (food_mask.amax(dim=1).sum(dim=1) / g_scale) * has_food
        ratio_h = (food_mask.amax(dim=2).sum(dim=1) / g_scale) * has_food
        t_sum = threat_mask.sum(dim=(1, 2)) + 1e-6
        has_threat = (threat_mask.sum(dim=(1, 2)) > 0.5).float()
        dx_h = (((threat_mask * c_coords).sum(dim=(1, 2)) / t_sum - rw * 0.5) / g_scale) * has_threat
        dy_h = (((threat_mask * r_coords).sum(dim=(1, 2)) / t_sum - rh * 0.5) / g_scale) * has_threat
        dz_h = ((threat_mask * prox_map).sum(dim=(1, 2)) / t_sum) * has_threat
        contact_cue = ((foveal_salience[:, 0] > 0.35) | (foveal_salience[:, 1] > 0.35)).float()
        depth_occlude_cue = ((obs_mask.sum(dim=(1, 2)) > 0.5) & ((has_food > 0.5) | (has_threat > 0.5))).float()
        aspect_mismatch = (ratio_w - ratio_h).abs()
        ground_z = self.terrain_height(self.pos[:, 0], self.pos[:, 1])
        gravity_cue = (((self.pos[:, 2] - ground_z) > 0.35) | (torch.isin(retinas[:, : rh // 2, :], tc_tensor).any(dim=(1, 2)))).float()
        enclosure_cue = (peripheral_salience[:, 2] > 0.25).float()
        raycaster_16d = obs_binocular_retinal_raycast_16d(self.rl_env)[:N]
        mean_disp = raycaster_16d[:, 14]
        foveal_disp = raycaster_16d[:, 15]
        foveal_prox = prox_map[:, rh // 4 : max(rh // 4 + 1, 3 * rh // 4), rw // 4 : max(rw // 4 + 1, 3 * rw // 4)].mean(dim=(1, 2))
        return torch.stack([
            dx_t, dy_t, dz_t,
            dx_h, dy_h, dz_h,
            ratio_w, ratio_h,
            contact_cue, depth_occlude_cue, aspect_mismatch, gravity_cue, enclosure_cue,
            mean_disp, foveal_disp, foveal_prox,
        ], dim=-1)

    def get_observations(self) -> torch.Tensor:
        return compute_egocentric_33d_observation(self.rl_env)[: self.pop_size]

    def step_toy_sensorimotor_body(
        self,
        ag_idx: int,
        t_idx: int,
        t_kind: int,
        motor_u_ag: torch.Tensor,
        vocal_s_ag: torch.Tensor,
        enforce_proximity: bool = True,
    ):
        ensure_sensorimotor_puzzle_buffers(self.rl_env)
        env_i = min(int(ag_idx), int(self.rl_env.num_envs) - 1)
        self.rl_env.agent_body_logits[env_i] = self.agent_body_logits[int(ag_idx)]

        num_e = self.rl_env.num_envs
        motor_u_full = torch.zeros((num_e, 3), dtype=torch.float32, device=self.device)
        vocal_s_full = torch.zeros((num_e, 2), dtype=torch.float32, device=self.device)
        motor_u_full[env_i] = motor_u_ag[int(ag_idx)].to(device=self.device, dtype=torch.float32)
        vocal_s_full[env_i] = vocal_s_ag[int(ag_idx)].to(device=self.device, dtype=torch.float32)

        res = execute_single_env_toy_manipulation(
            self.rl_env,
            env_i,
            int(t_idx),
            motor_u_full,
            vocal_s_full,
        )
        return res

    def step_physics(
        self,
        motor_u: torch.Tensor,
        vocal_s: torch.Tensor,
        thought_tax: torch.Tensor,
        comm_enabled: bool = True,
        joint_effort_21d: torch.Tensor | None = None,
    ):
        ensure_homeostasis_curiosity_buffers(self.rl_env)
        ensure_sensorimotor_puzzle_buffers(self.rl_env)
        num_e = self.rl_env.num_envs
        N = self.pop_size
        self.rl_env.comm_enabled = bool(comm_enabled)
        self.rl_env.unpredictable_mode = self.unpredictable_mode

        tax_full = torch.zeros((num_e,), dtype=torch.float32, device=self.device)
        cmd_full = torch.zeros((num_e, 3), dtype=torch.float32, device=self.device)
        actions_34d = torch.zeros((num_e, 34), dtype=torch.float32, device=self.device)
        eff_src = joint_effort_21d if joint_effort_21d is not None else self.agent_joint_effort_21d

        for k in range(num_e):
            src = k % N
            tax_full[k] = thought_tax[src]
            cmd_full[k] = motor_u[src, :3]
            actions_34d[k, 21:32] = self.agent_body_logits[src]
            actions_34d[k, 32:34] = vocal_s[src]

        self.rl_env.agent_locomotion_cmd_3d.copy_(cmd_full)
        self.rl_env._locomotion_cmd_step = int(getattr(self.rl_env, "common_step_counter", 0))
        robot = self.rl_env.scene["robot"]
        reach_timer = getattr(self.rl_env, "agent_reach_active_timer", None)
        reach_targets = getattr(self.rl_env, "agent_reach_joint_targets_21d", None)
        reach_mask = (reach_timer > 0) & self.rl_env.agent_alive if reach_timer is not None else None
        cpg_joints_21d = compute_spinal_cpg_and_vestibular_joints_21d(
            cmd_full[:, 0],
            cmd_full[:, 1],
            cmd_full[:, 2],
            self.rl_env.agent_gait_phase_rad,
            proj_g_b=robot.data.projected_gravity_b,
            ang_vel_b=robot.data.root_ang_vel_b,
            reach_targets_21d=reach_targets,
            reach_active_mask=reach_mask,
        )

        for k in range(num_e):
            src = k % N
            if eff_src is not None and src < eff_src.shape[0]:
                actions_34d[k, 0:21] = torch.clamp(0.65 * cpg_joints_21d[k] + 0.35 * eff_src[src], -1.0, 1.0)
            else:
                actions_34d[k, 0:21] = cpg_joints_21d[k]

        self.rl_env.thought_metabolic_tax = tax_full
        self.rl_env.step(actions_34d)

        _, arena_xy = compute_local_and_arena_xy(self.rl_env)
        quad_idx = compute_biome_quadrant_index(arena_xy[:N, 0], arena_xy[:N, 1])
        counts = torch.bincount(quad_idx, minlength=4).to(dtype=torch.float32)
        probs = counts / torch.clamp(counts.sum(), min=1.0)
        valid_p = probs[probs > 0.0]
        entropy_val = float((-torch.sum(valid_p * torch.log2(valid_p))).item())
        self.rl_env.biome_dispersion_entropy = entropy_val

        pred_d = torch.linalg.norm(self.pos[:, :2].unsqueeze(1) - self.pred_pos[:, :2].unsqueeze(0), dim=-1)
        near_threat = (pred_d.min(dim=1).values < 18.0) & self.alive
        self.rl_env.threat_observations_total = int(getattr(self.rl_env, "threat_observations_total", 0)) + int(
            near_threat.sum().item()
        )

        total_dmg = self.rl_env.agent_last_damage[:N].clone()
        delta_e = self.rl_env.agent_last_delta_energy[:N].clone()
        total_metabolic_cost = self.rl_env.agent_last_metabolic_cost[:N].clone()
        return total_dmg, delta_e, total_metabolic_cost


IsaacStyle3DPhysicsArena = IsaacLabUnified3DArena
