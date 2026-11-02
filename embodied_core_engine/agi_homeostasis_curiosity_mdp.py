from __future__ import annotations

import math
import torch


INITIAL_HP: float = 1.0
INITIAL_ENERGY: float = 0.56
BASAL_ATP_METABOLIC_COST: float = 0.0028
MOTOR_ATP_METABOLIC_COEFF: float = 0.0007
STARVATION_ENERGY_THRESHOLD: float = 0.03
STARVATION_HP_DAMAGE: float = 0.022
METABOLIC_DEFICIT_HP_DECAY: float = 0.0022
PREDATOR_CONTACT_RADIUS: float = 3.0
PREDATOR_MAX_DAMAGE: float = 0.24
PREDATOR_PURSUIT_RADIUS: float = 22.0
PREDATOR_PURSUIT_SPEED: float = 4.2
PREDATOR_PATROL_SPEED: float = 1.8
PREDATOR_PATROL_ORBIT_RADIUS: float = 8.0
WALL_BOUNDARY_MARGIN: float = 1.5
WALL_BOUNDARY_DAMAGE: float = 0.07
FOOD_FORAGE_RADIUS: float = 3.6
SOLO_FOOD_ENERGY_GAIN: float = 0.30
COOP_FOOD_ENERGY_GAIN: float = 0.44
QUADRANT_RESOURCE_REGEN: float = 0.008
QUADRANT_RESOURCE_DEPLETION: float = 0.045
QUADRANT_RESOURCE_MIN: float = 0.25
QUADRANT_RESOURCE_MAX: float = 1.0
VISITED_QUADRANT_DECAY: float = 0.992
NOVEL_QUADRANT_THRESHOLD: float = 0.35
CURIOSITY_NOVEL_BIOME_REWARD: float = 0.12
SAFETY_GATE_MIN_THRESHOLD: float = 0.25

METEOR_BLAST_RADIUS: float = 6.2
METEOR_BLAST_MAX_DAMAGE: float = 0.30
METEOR_IMPACT_HEIGHT_MARGIN: float = 1.50
ABYSS_HAZARD_RADIUS: float = 5.0
ABYSS_HAZARD_DAMAGE: float = 0.25
ABYSS_PULL_STRENGTH: float = 0.65
ABYSS_DOWNWARD_PULL_VZ: float = 1.15
VORTEX_PULL_RADIUS: float = 16.0
VORTEX_PULL_STRENGTH: float = 0.85
VORTEX_CORE_DAMAGE_RADIUS: float = 2.8
VORTEX_CORE_MAX_DAMAGE: float = 0.08
VORTEX_ORBIT_RADIUS: float = 2.5
VORTEX_ORBIT_ANGULAR_SPEED: float = 0.04
SIGNPOST_CUE_RADIUS: float = 4.5
MIMIC_FOOD_DAMAGE: float = 0.22

BIOME_QUADRANT_CENTERS_LOCAL_XY: tuple[tuple[float, float], ...] = (
    (-25.0, -25.0),
    (25.0, -25.0),
    (-25.0, 25.0),
    (25.0, 25.0),
)

VORTEX_ANCHORS_LOCAL_XY: tuple[tuple[float, float], ...] = (
    (-28.0, 28.0),
    (-16.0, 32.0),
)


def _compute_ground_height_torch(arena_x: torch.Tensor, arena_y: torch.Tensor, terrain_shift_x: float = 0.0) -> torch.Tensor:
    xs = arena_x + float(terrain_shift_x)
    peak1 = 4.5 * torch.exp(-((xs - 30.0) ** 2 + (arena_y - 30.0) ** 2) / 280.0)
    peak2 = 5.2 * torch.exp(-((xs - 70.0) ** 2 + (arena_y - 68.0) ** 2) / 320.0)
    wave = 1.4 * torch.sin(0.08 * xs) * torch.cos(0.08 * arena_y)
    sink1 = 2.4 * torch.exp(-((xs - 68.0) ** 2 + (arena_y - 22.0) ** 2) / 10.0)
    sink2 = 2.4 * torch.exp(-((xs - 82.0) ** 2 + (arena_y - 32.0) ** 2) / 10.0)
    return torch.clamp(peak1 + peak2 + wave + 1.5 - sink1 - sink2, 0.0, 12.0)


def is_in_safe_study_room(arena_xy: torch.Tensor) -> torch.Tensor:
    ax = arena_xy[:, 0]
    ay = arena_xy[:, 1]
    in_doorway = (ax >= 98.0) & (ax <= 102.0) & (torch.abs(ay - 50.0) <= 2.5)
    in_room = (ax >= 100.0) & (ax <= 123.7) & (torch.abs(ay - 50.0) <= 11.7)
    return in_doorway | in_room


def _resolve_env_origins(env, num_envs: int, device: torch.device | str) -> torch.Tensor:
    if hasattr(env, "scene"):
        if hasattr(env.scene, "env_origins") and env.scene.env_origins is not None:
            return env.scene.env_origins.to(device=device, dtype=torch.float32)
        if isinstance(env.scene, dict) and env.scene.get("env_origins") is not None:
            return env.scene["env_origins"].to(device=device, dtype=torch.float32)
    return torch.zeros((num_envs, 3), dtype=torch.float32, device=device)


def _resolve_scene_entity_positions_w(
    env,
    attr_name: str,
    scene_prefix: str,
    count: int,
    device: torch.device | str,
) -> torch.Tensor:
    override_tensor = getattr(env, attr_name, None)
    if override_tensor is not None:
        return override_tensor.to(device=device, dtype=torch.float32)
    collected = [
        env.scene[f"{scene_prefix}_{idx}"].data.root_pos_w.to(device=device, dtype=torch.float32)
        for idx in range(count)
    ]
    return torch.stack(collected, dim=1)


def compute_local_and_arena_xy(env) -> tuple[torch.Tensor, torch.Tensor]:
    robot = env.scene["robot"]
    root_pos_w = robot.data.root_pos_w
    device = root_pos_w.device
    num_envs = int(root_pos_w.shape[0])
    env_origins = _resolve_env_origins(env, num_envs, device)
    local_pos = root_pos_w - env_origins
    local_xy = local_pos[:, :2]
    arena_xy = local_xy + 50.0
    return local_xy, arena_xy


def compute_biome_quadrant_index(arena_x: torch.Tensor, arena_y: torch.Tensor) -> torch.Tensor:
    q_x = (arena_x >= 50.0).to(dtype=torch.long)
    q_y = (arena_y >= 50.0).to(dtype=torch.long)
    return torch.clamp(q_x + 2 * q_y, 0, 3)


def ensure_homeostasis_curiosity_buffers(env) -> None:
    robot = env.scene["robot"]
    device = robot.data.root_pos_w.device
    num_envs = int(robot.data.root_pos_w.shape[0])

    if not hasattr(env, "agent_hp") or env.agent_hp is None or env.agent_hp.shape[0] != num_envs:
        env.agent_hp = torch.full((num_envs,), INITIAL_HP, dtype=torch.float32, device=device)
    if not hasattr(env, "agent_energy") or env.agent_energy is None or env.agent_energy.shape[0] != num_envs:
        env.agent_energy = torch.full((num_envs,), INITIAL_ENERGY, dtype=torch.float32, device=device)
    if not hasattr(env, "agent_pain") or env.agent_pain is None or env.agent_pain.shape[0] != num_envs:
        env.agent_pain = torch.zeros((num_envs,), dtype=torch.float32, device=device)
    if not hasattr(env, "agent_surprise") or env.agent_surprise is None or env.agent_surprise.shape[0] != num_envs:
        env.agent_surprise = torch.zeros((num_envs,), dtype=torch.float32, device=device)
    if not hasattr(env, "agent_alive") or env.agent_alive is None or env.agent_alive.shape[0] != num_envs:
        env.agent_alive = torch.ones((num_envs,), dtype=torch.bool, device=device)
    if not hasattr(env, "agent_last_damage") or env.agent_last_damage is None or env.agent_last_damage.shape[0] != num_envs:
        env.agent_last_damage = torch.zeros((num_envs,), dtype=torch.float32, device=device)
    if not hasattr(env, "agent_last_delta_energy") or env.agent_last_delta_energy is None or env.agent_last_delta_energy.shape[0] != num_envs:
        env.agent_last_delta_energy = torch.zeros((num_envs,), dtype=torch.float32, device=device)
    if not hasattr(env, "agent_last_metabolic_cost") or env.agent_last_metabolic_cost is None or env.agent_last_metabolic_cost.shape[0] != num_envs:
        env.agent_last_metabolic_cost = torch.zeros((num_envs,), dtype=torch.float32, device=device)
    if not hasattr(env, "agent_survival_safety_gate") or env.agent_survival_safety_gate is None or env.agent_survival_safety_gate.shape[0] != num_envs:
        env.agent_survival_safety_gate = torch.ones((num_envs,), dtype=torch.float32, device=device)
    if not hasattr(env, "agent_visited_quadrants") or env.agent_visited_quadrants is None or env.agent_visited_quadrants.shape != (num_envs, 4):
        env.agent_visited_quadrants = torch.zeros((num_envs, 4), dtype=torch.float32, device=device)
    if not hasattr(env, "quadrant_resource_level") or env.quadrant_resource_level is None or env.quadrant_resource_level.shape != (num_envs, 4):
        env.quadrant_resource_level = torch.ones((num_envs, 4), dtype=torch.float32, device=device)
    if not hasattr(env, "last_modulated_curiosity_bonus") or env.last_modulated_curiosity_bonus is None or env.last_modulated_curiosity_bonus.shape[0] != num_envs:
        env.last_modulated_curiosity_bonus = torch.zeros((num_envs,), dtype=torch.float32, device=device)
    if not hasattr(env, "agent_signpost_cued") or env.agent_signpost_cued is None or env.agent_signpost_cued.shape[0] != num_envs:
        env.agent_signpost_cued = torch.zeros((num_envs,), dtype=torch.bool, device=device)
    if not hasattr(env, "agent_gait_phase_rad") or env.agent_gait_phase_rad is None or env.agent_gait_phase_rad.shape != (num_envs,):
        env.agent_gait_phase_rad = torch.zeros((num_envs,), dtype=torch.float32, device=device)
    if not hasattr(env, "agent_gait_cadence_hz") or env.agent_gait_cadence_hz is None or env.agent_gait_cadence_hz.shape != (num_envs,):
        env.agent_gait_cadence_hz = torch.full((num_envs,), 1.8, dtype=torch.float32, device=device)
    if not hasattr(env, "agent_locomotion_cmd_3d") or env.agent_locomotion_cmd_3d is None or env.agent_locomotion_cmd_3d.shape != (num_envs, 3):
        env.agent_locomotion_cmd_3d = torch.zeros((num_envs, 3), dtype=torch.float32, device=device)
    if not hasattr(env, "agent_cpg_target_joints_21d") or env.agent_cpg_target_joints_21d is None or env.agent_cpg_target_joints_21d.shape != (num_envs, 21):
        env.agent_cpg_target_joints_21d = torch.zeros((num_envs, 21), dtype=torch.float32, device=device)
    if not hasattr(env, "agent_cerebellar_joint_torques_21d") or env.agent_cerebellar_joint_torques_21d is None or env.agent_cerebellar_joint_torques_21d.shape != (num_envs, 21):
        env.agent_cerebellar_joint_torques_21d = torch.zeros((num_envs, 21), dtype=torch.float32, device=device)
    if not hasattr(env, "agent_torso_upright_cos") or env.agent_torso_upright_cos is None or env.agent_torso_upright_cos.shape != (num_envs,):
        env.agent_torso_upright_cos = torch.ones((num_envs,), dtype=torch.float32, device=device)
    if not hasattr(env, "agent_gait_coordination_score") or env.agent_gait_coordination_score is None or env.agent_gait_coordination_score.shape != (num_envs,):
        env.agent_gait_coordination_score = torch.ones((num_envs,), dtype=torch.float32, device=device)
    if not hasattr(env, "bipedal_gait_cycles_total"):
        env.bipedal_gait_cycles_total = 0
    if not hasattr(env, "vestibular_balance_corrections_total"):
        env.vestibular_balance_corrections_total = 0
    if not hasattr(env, "_locomotion_cmd_step"):
        env._locomotion_cmd_step = -1
    if not hasattr(env, "curiosity_explorations_total"):
        env.curiosity_explorations_total = 0
    if not hasattr(env, "_last_homeostasis_step_counter"):
        env._last_homeostasis_step_counter = -1
    if not hasattr(env, "_last_pre_physics_world_step"):
        env._last_pre_physics_world_step = -1
    if not hasattr(env, "_world_orbit_tick"):
        env._world_orbit_tick = 0


def reset_homeostasis_and_curiosity(
    env,
    env_ids: torch.Tensor | None,
    mark_initial_quadrant: bool = False,
) -> None:
    ensure_homeostasis_curiosity_buffers(env)
    robot = env.scene["robot"]
    device = robot.data.root_pos_w.device
    num_envs = int(robot.data.root_pos_w.shape[0])

    if env_ids is None:
        ids = torch.arange(num_envs, dtype=torch.long, device=device)
    else:
        ids = env_ids.to(device=device, dtype=torch.long)

    if ids.numel() == 0:
        return

    env.agent_hp[ids] = INITIAL_HP
    env.agent_energy[ids] = INITIAL_ENERGY
    env.agent_pain[ids] = 0.0
    env.agent_surprise[ids] = 0.0
    env.agent_alive[ids] = True
    env.agent_last_damage[ids] = 0.0
    env.agent_last_delta_energy[ids] = 0.0
    env.agent_last_metabolic_cost[ids] = 0.0
    env.agent_survival_safety_gate[ids] = 1.0
    env.quadrant_resource_level[ids] = 1.0
    env.agent_visited_quadrants[ids] = 0.0
    env.last_modulated_curiosity_bonus[ids] = 0.0
    env.agent_signpost_cued[ids] = False
    env.agent_gait_phase_rad[ids] = 0.0
    env.agent_gait_cadence_hz[ids] = 1.8
    env.agent_locomotion_cmd_3d[ids] = 0.0
    env.agent_cpg_target_joints_21d[ids] = 0.0
    env.agent_cerebellar_joint_torques_21d[ids] = 0.0
    env.agent_torso_upright_cos[ids] = 1.0
    env.agent_gait_coordination_score[ids] = 1.0

    if mark_initial_quadrant:
        _, arena_xy = compute_local_and_arena_xy(env)
        init_q = compute_biome_quadrant_index(arena_xy[ids, 0], arena_xy[ids, 1])
        env.agent_visited_quadrants[ids, init_q] = 1.0


def compute_atp_metabolic_cost(
    root_lin_vel_w: torch.Tensor,
    actions: torch.Tensor | None = None,
    thought_metabolic_tax: torch.Tensor | float = 0.0,
    basal_rate: float = BASAL_ATP_METABOLIC_COST,
    effort_coeff: float = MOTOR_ATP_METABOLIC_COEFF,
) -> torch.Tensor:
    speed_xy = torch.linalg.norm(root_lin_vel_w[:, :2], dim=-1)
    speed_effort = torch.clamp(speed_xy / 3.9, 0.0, 2.0)
    jump_effort = torch.clamp(root_lin_vel_w[:, 2] / 5.2, 0.0, 1.0)

    if actions is not None and actions.numel() > 0:
        act_effort = torch.mean(torch.abs(actions.to(device=root_lin_vel_w.device, dtype=torch.float32)), dim=-1)
    else:
        act_effort = torch.zeros_like(speed_effort)

    physical_metabolic_cost = float(basal_rate) + float(effort_coeff) * (speed_effort + jump_effort + act_effort)

    if isinstance(thought_metabolic_tax, torch.Tensor):
        tax_tensor = thought_metabolic_tax.to(device=root_lin_vel_w.device, dtype=torch.float32)
    else:
        tax_tensor = torch.full_like(physical_metabolic_cost, float(thought_metabolic_tax))

    return physical_metabolic_cost + tax_tensor


def _respawn_consumed_foods(
    env,
    near_food: torch.Tensor,
    root_pos_w: torch.Tensor,
    env_origins: torch.Tensor,
    terrain_shift: float,
) -> None:
    if bool(getattr(env, "disable_food_respawn", False)):
        return
    if getattr(env, "food_pos_w", None) is not None:
        return
    if not hasattr(env, "scene"):
        return

    device = root_pos_w.device
    for f_idx in range(4):
        consumed_ids = torch.where(near_food[:, f_idx])[0]
        if consumed_ids.numel() == 0:
            continue
        f_key = f"food_{f_idx}"
        if f_key not in env.scene.keys() if hasattr(env.scene, "keys") else not hasattr(env.scene, f_key):
            continue
        food_obj = env.scene[f_key]
        n_c = int(consumed_ids.numel())
        x_min, x_max = (-42.0, -8.0) if (f_idx % 2 == 0) else (8.0, 42.0)
        y_min, y_max = (-42.0, -8.0) if (f_idx // 2 == 0) else (8.0, 42.0)
        rand_x = torch.empty((n_c,), dtype=torch.float32, device=device).uniform_(x_min, x_max)
        rand_y = torch.empty((n_c,), dtype=torch.float32, device=device).uniform_(y_min, y_max)

        ag_local_xy = root_pos_w[consumed_ids, :2] - env_origins[consumed_ids, :2]
        sep_xy = torch.stack([rand_x, rand_y], dim=-1) - ag_local_xy
        too_close = torch.linalg.norm(sep_xy, dim=-1) < (FOOD_FORAGE_RADIUS + 3.0)
        if torch.any(too_close):
            cx, cy = BIOME_QUADRANT_CENTERS_LOCAL_XY[f_idx]
            alt_x = torch.where(ag_local_xy[:, 0] >= cx, torch.full_like(rand_x, x_min + 2.5), torch.full_like(rand_x, x_max - 2.5))
            alt_y = torch.where(ag_local_xy[:, 1] >= cy, torch.full_like(rand_y, y_min + 2.5), torch.full_like(rand_y, y_max - 2.5))
            rand_x = torch.where(too_close, alt_x, rand_x)
            rand_y = torch.where(too_close, alt_y, rand_y)

        z_ground = _compute_ground_height_torch(rand_x + 50.0, rand_y + 50.0, terrain_shift_x=terrain_shift)
        new_pose = food_obj.data.default_root_state[consumed_ids, :7].clone()
        new_pose[:, 0] = env_origins[consumed_ids, 0] + rand_x
        new_pose[:, 1] = env_origins[consumed_ids, 1] + rand_y
        new_pose[:, 2] = env_origins[consumed_ids, 2] + z_ground + 0.50
        new_pose[:, 3] = 1.0
        new_pose[:, 4:7] = 0.0
        zero_vel = torch.zeros((n_c, 6), dtype=torch.float32, device=device)
        food_obj.write_root_pose_to_sim(new_pose, env_ids=consumed_ids)
        food_obj.write_root_velocity_to_sim(zero_vel, env_ids=consumed_ids)


def compute_foraging_energy_and_resource_update(
    root_pos_w: torch.Tensor,
    food_pos_w: torch.Tensor,
    env_origins: torch.Tensor,
    alive: torch.Tensor,
    quadrant_resource_level: torch.Tensor,
    nutritious_food_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ag_food_dist = torch.linalg.norm(root_pos_w.unsqueeze(1) - food_pos_w, dim=-1)
    near_food = (ag_food_dist < FOOD_FORAGE_RADIUS) & alive.unsqueeze(1)

    if nutritious_food_mask is not None:
        nutritious_near = near_food & nutritious_food_mask
    else:
        nutritious_near = near_food

    food_local_xy = food_pos_w[:, :, :2] - env_origins[:, :2].unsqueeze(1)
    food_arena_xy = food_local_xy + 50.0
    food_quad = compute_biome_quadrant_index(food_arena_xy[:, :, 0], food_arena_xy[:, :, 1])

    food_res_scale = torch.gather(quadrant_resource_level, 1, food_quad)
    delta_e = (nutritious_near.to(dtype=torch.float32) * food_res_scale).sum(dim=1) * SOLO_FOOD_ENERGY_GAIN

    updated_resource = torch.clamp(
        quadrant_resource_level + QUADRANT_RESOURCE_REGEN,
        QUADRANT_RESOURCE_MIN,
        QUADRANT_RESOURCE_MAX,
    )
    if torch.any(nutritious_near):
        for q_idx in range(4):
            harvested_in_q = (nutritious_near & (food_quad == q_idx)).sum(dim=1).to(dtype=torch.float32)
            updated_resource[:, q_idx] = torch.clamp(
                updated_resource[:, q_idx] - QUADRANT_RESOURCE_DEPLETION * harvested_in_q,
                QUADRANT_RESOURCE_MIN,
                QUADRANT_RESOURCE_MAX,
            )

    return delta_e, updated_resource, near_food


def compute_environmental_and_starvation_damage(
    root_pos_w: torch.Tensor,
    pred_pos_w: torch.Tensor,
    arena_xy: torch.Tensor,
    energy: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    in_safe_room = is_in_safe_study_room(arena_xy)
    out_bound = (
        (arena_xy[:, 0] < WALL_BOUNDARY_MARGIN)
        | (arena_xy[:, 0] > 100.0 - WALL_BOUNDARY_MARGIN)
        | (arena_xy[:, 1] < WALL_BOUNDARY_MARGIN)
        | (arena_xy[:, 1] > 100.0 - WALL_BOUNDARY_MARGIN)
    ) & (~in_safe_room)
    wall_dmg = torch.where(
        out_bound,
        torch.full_like(energy, WALL_BOUNDARY_DAMAGE),
        torch.zeros_like(energy),
    )

    ag_pred_dist = torch.linalg.norm(root_pos_w.unsqueeze(1) - pred_pos_w, dim=-1)
    min_pred_d = ag_pred_dist.min(dim=1).values
    pred_hit = (min_pred_d < PREDATOR_CONTACT_RADIUS) & (~in_safe_room)
    pred_dmg = torch.where(
        pred_hit,
        PREDATOR_MAX_DAMAGE * (PREDATOR_CONTACT_RADIUS - min_pred_d) / PREDATOR_CONTACT_RADIUS,
        torch.zeros_like(energy),
    )

    starvation_dmg = torch.where(
        energy <= STARVATION_ENERGY_THRESHOLD,
        torch.full_like(energy, STARVATION_HP_DAMAGE),
        torch.zeros_like(energy),
    )

    total_dmg = wall_dmg + pred_dmg + starvation_dmg
    return total_dmg, wall_dmg, pred_dmg, starvation_dmg


def _quat_wxyz_to_yaw_local(quat_wxyz: torch.Tensor) -> torch.Tensor:
    w = quat_wxyz[:, 0]
    x = quat_wxyz[:, 1]
    y = quat_wxyz[:, 2]
    z = quat_wxyz[:, 3]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return torch.atan2(siny_cosp, cosy_cosp)


def _yaw_pitch_roll_to_quat_wxyz(yaw: torch.Tensor, pitch: torch.Tensor, roll: torch.Tensor) -> torch.Tensor:
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return torch.stack([w, x, y, z], dim=-1)


def compute_spinal_cpg_and_vestibular_joints_21d(
    u_fwd: torch.Tensor,
    u_turn: torch.Tensor,
    u_jump: torch.Tensor,
    phase_rad: torch.Tensor,
    proj_g_b: torch.Tensor | None = None,
    ang_vel_b: torch.Tensor | None = None,
    reach_targets_21d: torch.Tensor | None = None,
    reach_active_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    device = u_fwd.device
    num_envs = int(u_fwd.shape[0])
    q_target = torch.zeros((num_envs, 21), dtype=torch.float32, device=device)

    phi_r = phase_rad
    phi_l = phase_rad + math.pi
    sin_r = torch.sin(phi_r)
    cos_r = torch.cos(phi_r)
    sin_l = torch.sin(phi_l)
    cos_l = torch.cos(phi_l)

    stride_amp = torch.clamp(torch.abs(u_fwd) + 0.20 * torch.abs(u_turn), 0.0, 1.0)
    dir_sign = torch.where(u_fwd >= -0.05, torch.ones_like(u_fwd), -torch.ones_like(u_fwd))
    swing_r = torch.clamp(sin_r, min=0.0)
    swing_l = torch.clamp(sin_l, min=0.0)
    jump_pos = torch.clamp(u_jump, min=0.0)

    if proj_g_b is not None and proj_g_b.shape == (num_envs, 3):
        gx = proj_g_b[:, 0]
        gy = proj_g_b[:, 1]
    else:
        gx = torch.zeros_like(u_fwd)
        gy = torch.zeros_like(u_fwd)

    if ang_vel_b is not None and ang_vel_b.shape == (num_envs, 3):
        wx = ang_vel_b[:, 0]
        wy = ang_vel_b[:, 1]
        wz = ang_vel_b[:, 2]
    else:
        wx = torch.zeros_like(u_fwd)
        wy = torch.zeros_like(u_fwd)
        wz = torch.zeros_like(u_fwd)

    vest_roll = -0.55 * gy - 0.12 * wx
    vest_pitch = -0.65 * gx - 0.15 * wy
    vest_yaw = -0.10 * wz

    q_target[:, 0] = torch.clamp(vest_roll + 0.08 * stride_amp * cos_r, -0.75, 0.75)
    q_target[:, 1] = torch.clamp(vest_pitch + 0.12 * u_fwd, -0.75, 0.75)
    q_target[:, 2] = torch.clamp(-0.45 * stride_amp * dir_sign * sin_r, -1.20, 1.20)
    q_target[:, 3] = torch.clamp(-0.12 - 0.08 * stride_amp * cos_r, -0.95, 0.95)
    q_target[:, 4] = torch.clamp(-0.45 * stride_amp * dir_sign * sin_l, -1.20, 1.20)
    q_target[:, 5] = torch.clamp(0.12 + 0.08 * stride_amp * cos_l, -0.95, 0.95)
    q_target[:, 6] = torch.clamp(-0.16 * stride_amp * sin_r + 0.30 * u_turn + vest_yaw, -0.80, 0.80)
    q_target[:, 7] = torch.clamp(-0.35 - 0.25 * stride_amp * swing_l, -1.45, 0.20)
    q_target[:, 8] = torch.clamp(-0.35 - 0.25 * stride_amp * swing_r, -1.45, 0.20)
    q_target[:, 9] = torch.clamp(-0.08 * stride_amp * cos_r + 0.55 * vest_roll + 0.12 * u_turn, -0.65, 0.65)
    q_target[:, 10] = torch.clamp(0.28 * u_turn - 0.06 * stride_amp * sin_r, -0.65, 0.65)
    q_target[:, 11] = torch.clamp(0.55 * stride_amp * dir_sign * sin_r + 0.55 * vest_pitch + 0.30 * jump_pos, -1.25, 1.25)
    q_target[:, 12] = torch.clamp(0.08 * stride_amp * cos_l + 0.55 * vest_roll - 0.12 * u_turn, -0.65, 0.65)
    q_target[:, 13] = torch.clamp(0.28 * u_turn - 0.06 * stride_amp * sin_l, -0.65, 0.65)
    q_target[:, 14] = torch.clamp(0.55 * stride_amp * dir_sign * sin_l + 0.55 * vest_pitch + 0.30 * jump_pos, -1.25, 1.25)
    q_target[:, 15] = torch.clamp(-0.65 * stride_amp * swing_r - 0.12 - 0.45 * jump_pos, -1.50, 0.10)
    q_target[:, 16] = torch.clamp(-0.65 * stride_amp * swing_l - 0.12 - 0.45 * jump_pos, -1.50, 0.10)
    q_target[:, 17] = torch.clamp(-0.32 * stride_amp * dir_sign * sin_r + 0.30 * swing_r + 0.60 * vest_pitch + 0.35 * jump_pos, -0.85, 0.85)
    q_target[:, 18] = torch.clamp(0.08 * stride_amp * cos_r + 0.50 * vest_roll, -0.60, 0.60)
    q_target[:, 19] = torch.clamp(-0.32 * stride_amp * dir_sign * sin_l + 0.30 * swing_l + 0.60 * vest_pitch + 0.35 * jump_pos, -0.85, 0.85)
    q_target[:, 20] = torch.clamp(-0.08 * stride_amp * cos_l + 0.50 * vest_roll, -0.60, 0.60)

    if (
        reach_targets_21d is not None
        and reach_active_mask is not None
        and reach_targets_21d.shape == (num_envs, 21)
        and torch.any(reach_active_mask)
    ):
        r_mask = reach_active_mask.unsqueeze(-1).expand(num_envs, 9)
        q_target[:, :9] = torch.where(r_mask, reach_targets_21d[:, :9], q_target[:, :9])

    return q_target


def step_spinal_cpg_and_bipedal_locomotion(env) -> None:
    ensure_homeostasis_curiosity_buffers(env)
    robot = env.scene["robot"]
    root_pos_w = robot.data.root_pos_w
    device = root_pos_w.device
    num_envs = int(root_pos_w.shape[0])

    cur_step = int(getattr(env, "common_step_counter", -1))
    is_new_env_step = int(getattr(env, "_cpg_last_phase_step", -2)) != cur_step
    if int(getattr(env, "_locomotion_cmd_step", -2)) == cur_step:
        u_fwd = env.agent_locomotion_cmd_3d[:, 0]
        u_turn = env.agent_locomotion_cmd_3d[:, 1]
        u_jump = env.agent_locomotion_cmd_3d[:, 2]
    elif hasattr(env, "action_manager") and env.action_manager.action.shape[-1] >= 21:
        raw_j = env.action_manager.action[:, :21]
        u_fwd = torch.clamp(0.50 * (raw_j[:, 11] - raw_j[:, 14]) + 0.50 * raw_j[:, 3:9].mean(dim=-1), -1.0, 1.0)
        u_turn = torch.clamp(0.50 * (raw_j[:, 0] + raw_j[:, 6]) + 0.50 * (raw_j[:, 10] + raw_j[:, 13]), -1.0, 1.0)
        u_jump = torch.clamp(0.50 * raw_j[:, 9:15].mean(dim=-1) + 0.50 * (raw_j[:, 17] + raw_j[:, 19]), -1.0, 1.0)
        env.agent_locomotion_cmd_3d[:, 0] = u_fwd
        env.agent_locomotion_cmd_3d[:, 1] = u_turn
        env.agent_locomotion_cmd_3d[:, 2] = u_jump
    else:
        u_fwd = torch.zeros((num_envs,), dtype=torch.float32, device=device)
        u_turn = torch.zeros((num_envs,), dtype=torch.float32, device=device)
        u_jump = torch.zeros((num_envs,), dtype=torch.float32, device=device)

    loco_drive_mag = torch.abs(u_fwd) + torch.abs(u_turn) + torch.clamp(u_jump, min=0.0)
    loco_active = (loco_drive_mag > 1e-3) & env.agent_alive

    step_dt = float(getattr(env, "step_dt", 1.0 / 30.0))
    cadence_hz = 1.6 + 1.8 * torch.clamp(torch.abs(u_fwd) + 0.35 * torch.abs(u_turn), 0.0, 1.0)
    env.agent_gait_cadence_hz[:] = cadence_hz
    if is_new_env_step:
        env._cpg_last_phase_step = cur_step
        delta_phase = torch.where(
            loco_active,
            2.0 * math.pi * cadence_hz * max(step_dt, 1.0 / 60.0),
            torch.zeros_like(cadence_hz),
        )
        new_phase_unwrapped = env.agent_gait_phase_rad + delta_phase
        completed_cycles = torch.floor(new_phase_unwrapped / (2.0 * math.pi)).to(dtype=torch.long)
        env.bipedal_gait_cycles_total = int(getattr(env, "bipedal_gait_cycles_total", 0)) + int(
            torch.clamp(completed_cycles, min=0).sum().item()
        )
        env.agent_gait_phase_rad[:] = torch.remainder(new_phase_unwrapped, 2.0 * math.pi)

    proj_g_b = robot.data.projected_gravity_b
    ang_vel_b = robot.data.root_ang_vel_b
    reach_timer = getattr(env, "agent_reach_active_timer", None)
    reach_targets = getattr(env, "agent_reach_joint_targets_21d", None)
    if reach_timer is not None and reach_timer.shape == (num_envs,):
        reach_active_mask = (reach_timer > 0) & env.agent_alive
        if is_new_env_step:
            env.agent_reach_active_timer[:] = torch.clamp(reach_timer - 1, min=0)
    else:
        reach_active_mask = torch.zeros((num_envs,), dtype=torch.bool, device=device)

    q_target_21d = compute_spinal_cpg_and_vestibular_joints_21d(
        u_fwd,
        u_turn,
        u_jump,
        env.agent_gait_phase_rad,
        proj_g_b=proj_g_b,
        ang_vel_b=ang_vel_b,
        reach_targets_21d=reach_targets,
        reach_active_mask=reach_active_mask,
    )
    env.agent_cpg_target_joints_21d[:] = q_target_21d

    default_q = robot.data.default_joint_pos[:, :21]
    q_cur = robot.data.joint_pos[:, :21] - default_q
    dq_cur = robot.data.joint_vel[:, :21]
    pd_torque = torch.tanh(2.4 * (q_target_21d - q_cur) - 0.18 * dq_cur)
    env.agent_cerebellar_joint_torques_21d[:] = pd_torque

    robot.set_joint_position_target(default_q + q_target_21d)
    if hasattr(env, "action_manager") and env.action_manager.action.shape[-1] >= 21:
        brain_j = torch.tanh(env.action_manager.action[:, :21])
        unit_effort = torch.clamp(0.55 * pd_torque + 0.45 * brain_j, -1.0, 1.0)
    else:
        unit_effort = pd_torque
    effort_scale = torch.tensor(
        [
            67.5, 67.5, 67.5, 67.5, 67.5, 67.5, 67.5, 45.0, 45.0,
            45.0, 135.0, 45.0, 45.0, 135.0, 45.0, 90.0, 90.0, 22.5, 22.5, 22.5, 22.5,
        ],
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)
    robot.set_joint_effort_target(unit_effort * effort_scale)

    anti_phase_diff = torch.abs(q_target_21d[:, 11] - q_target_21d[:, 14])
    tracking_err = torch.mean(torch.abs(q_target_21d[:, 9:21] - q_cur[:, 9:21]), dim=-1)
    gait_coord = torch.clamp(
        0.55 + 0.45 * torch.exp(-1.5 * tracking_err) * (0.4 + 0.6 * torch.clamp(anti_phase_diff, 0.0, 1.0)),
        0.35,
        1.0,
    )
    env.agent_gait_coordination_score[:] = gait_coord

    tilt_mag = torch.linalg.norm(proj_g_b[:, :2], dim=-1)
    ang_mag = torch.linalg.norm(ang_vel_b[:, :2], dim=-1)
    upright_cos = torch.clamp(-proj_g_b[:, 2], -1.0, 1.0)
    if is_new_env_step:
        needs_vest = ((tilt_mag > 0.06) | (ang_mag > 0.30)) & env.agent_alive
        env.vestibular_balance_corrections_total = int(getattr(env, "vestibular_balance_corrections_total", 0)) + int(
            needs_vest.sum().item()
        )

    env.agent_torso_upright_cos[:] = upright_cos


def pre_physics_step_world_dynamics(env) -> None:
    ensure_homeostasis_curiosity_buffers(env)
    step_spinal_cpg_and_bipedal_locomotion(env)
    robot = env.scene["robot"]
    root_pos_w = robot.data.root_pos_w
    device = root_pos_w.device
    num_envs = int(root_pos_w.shape[0])
    env_origins = _resolve_env_origins(env, num_envs, device)
    local_xy, arena_xy = compute_local_and_arena_xy(env)
    in_safe_room = is_in_safe_study_room(arena_xy)
    ag_quad = compute_biome_quadrant_index(arena_xy[:, 0], arena_xy[:, 1])
    terrain_shift = float(getattr(env, "terrain_shift_x", 0.0))

    env._world_orbit_tick = int(getattr(env, "_world_orbit_tick", 0)) + 1
    tick_val = float(env._world_orbit_tick)

    if not bool(getattr(env, "disable_predator_pursuit", False)) and getattr(env, "predator_pos_w", None) is None:
        for p_idx in range(4):
            pred_obj = env.scene[f"predator_{p_idx}"]
            p_pos_w = pred_obj.data.root_pos_w
            p_vel_w = pred_obj.data.root_state_w[:, 7:13]
            to_ag_xy = root_pos_w[:, :2] - p_pos_w[:, :2]
            dist_ag = torch.linalg.norm(to_ag_xy, dim=-1, keepdim=True)
            dir_ag = to_ag_xy / (dist_ag + 1e-5)

            cx, cy = BIOME_QUADRANT_CENTERS_LOCAL_XY[p_idx]
            orbit_angle = 0.035 * tick_val + float(p_idx) * (0.5 * math.pi)
            patrol_target_local = torch.tensor(
                [
                    cx + PREDATOR_PATROL_ORBIT_RADIUS * math.cos(orbit_angle),
                    cy + PREDATOR_PATROL_ORBIT_RADIUS * math.sin(orbit_angle),
                ],
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0).expand(num_envs, 2)
            p_local_xy = p_pos_w[:, :2] - env_origins[:, :2]
            to_patrol = patrol_target_local - p_local_xy
            dist_patrol = torch.linalg.norm(to_patrol, dim=-1, keepdim=True)
            dir_patrol = to_patrol / (dist_patrol + 1e-5)

            pursue_mask = (
                ((ag_quad == p_idx) | (dist_ag.squeeze(-1) < 14.0))
                & (dist_ag.squeeze(-1) < PREDATOR_PURSUIT_RADIUS)
                & env.agent_alive
                & (~in_safe_room)
            ).unsqueeze(-1)
            desired_vel_xy = torch.where(
                pursue_mask,
                PREDATOR_PURSUIT_SPEED * dir_ag,
                PREDATOR_PATROL_SPEED * dir_patrol,
            )
            p_forces = torch.zeros((num_envs, 1, 3), dtype=torch.float32, device=device)
            p_torques = torch.zeros((num_envs, 1, 3), dtype=torch.float32, device=device)
            p_forces[:, 0, 0:2] = torch.clamp(24.0 * (desired_vel_xy - p_vel_w[:, 0:2]), -95.0, 95.0)
            pred_obj.set_external_force_and_torque(p_forces, p_torques)

    if not bool(getattr(env, "disable_vortex_orbit", False)) and getattr(env, "vortex_pos_w", None) is None:
        for v_idx in range(2):
            v_obj = env.scene[f"vortex_center_{v_idx}"]
            ax_loc, ay_loc = VORTEX_ANCHORS_LOCAL_XY[v_idx]
            phase = VORTEX_ORBIT_ANGULAR_SPEED * tick_val + float(v_idx) * math.pi
            vx_loc = float(ax_loc + VORTEX_ORBIT_RADIUS * math.cos(phase))
            vy_loc = float(ay_loc + VORTEX_ORBIT_RADIUS * math.sin(phase))
            vx_t = torch.full((num_envs,), vx_loc + 50.0, dtype=torch.float32, device=device)
            vy_t = torch.full((num_envs,), vy_loc + 50.0, dtype=torch.float32, device=device)
            vz_ground = _compute_ground_height_torch(vx_t, vy_t, terrain_shift_x=terrain_shift)
            v_pos_now = v_obj.data.root_pos_w
            v_vel_now = v_obj.data.root_state_w[:, 7:10]
            target_v_pos = torch.stack(
                [
                    env_origins[:, 0] + vx_loc,
                    env_origins[:, 1] + vy_loc,
                    env_origins[:, 2] + vz_ground + 2.0,
                ],
                dim=-1,
            )
            v_forces = torch.zeros((num_envs, 1, 3), dtype=torch.float32, device=device)
            v_torques = torch.zeros((num_envs, 1, 3), dtype=torch.float32, device=device)
            v_forces[:, 0, :] = torch.clamp(45.0 * (target_v_pos - v_pos_now) - 8.0 * v_vel_now, -160.0, 160.0)
            v_obj.set_external_force_and_torque(v_forces, v_torques)

    robot_ext_forces = torch.zeros((num_envs, 1, 3), dtype=torch.float32, device=device)
    robot_ext_torques = torch.zeros((num_envs, 1, 3), dtype=torch.float32, device=device)
    has_env_force = False

    vortex_pos_w = _resolve_scene_entity_positions_w(env, "vortex_pos_w", "vortex_center", 2, device)
    vortex_diff_xy = root_pos_w[:, :2].unsqueeze(1) - vortex_pos_w[:, :, :2]
    vortex_dist = torch.linalg.norm(vortex_diff_xy, dim=-1)
    min_v_dist, min_v_idx = torch.min(vortex_dist, dim=1)
    in_vortex = (min_v_dist < VORTEX_PULL_RADIUS) & (min_v_dist > 0.4) & env.agent_alive & (~in_safe_room)
    if torch.any(in_vortex):
        v_ids = torch.where(in_vortex)[0]
        gather_idx = min_v_idx.unsqueeze(1).unsqueeze(2).expand(-1, 1, 2)
        nearest_diff = torch.gather(vortex_diff_xy, 1, gather_idx).squeeze(1)
        dx = nearest_diff[v_ids, 0] / min_v_dist[v_ids]
        dy = nearest_diff[v_ids, 1] / min_v_dist[v_ids]
        pull_scale = VORTEX_PULL_STRENGTH * (VORTEX_PULL_RADIUS - min_v_dist[v_ids]) / VORTEX_PULL_RADIUS
        dv_x = (-0.35 * dx - 0.85 * dy) * pull_scale
        dv_y = (-0.35 * dy + 0.85 * dx) * pull_scale
        robot_ext_forces[v_ids, 0, 0] = robot_ext_forces[v_ids, 0, 0] + 185.0 * dv_x
        robot_ext_forces[v_ids, 0, 1] = robot_ext_forces[v_ids, 0, 1] + 185.0 * dv_y
        robot_ext_torques[v_ids, 0, 2] = robot_ext_torques[v_ids, 0, 2] + 28.0 * pull_scale
        has_env_force = True

    abyss_pos_w = _resolve_scene_entity_positions_w(env, "abyss_pos_w", "abyss_hazard", 2, device)
    abyss_diff_xy = abyss_pos_w[:, :, :2] - root_pos_w[:, :2].unsqueeze(1)
    abyss_dist = torch.linalg.norm(abyss_diff_xy, dim=-1)
    min_a_dist, min_a_idx = torch.min(abyss_dist, dim=1)
    in_abyss_pull = (min_a_dist < ABYSS_HAZARD_RADIUS) & (min_a_dist > 0.15) & env.agent_alive & (~in_safe_room)
    if torch.any(in_abyss_pull):
        a_ids = torch.where(in_abyss_pull)[0]
        gather_a = min_a_idx.unsqueeze(1).unsqueeze(2).expand(-1, 1, 2)
        to_abyss = torch.gather(abyss_diff_xy, 1, gather_a).squeeze(1)
        dir_ax = to_abyss[a_ids, 0] / min_a_dist[a_ids]
        dir_ay = to_abyss[a_ids, 1] / min_a_dist[a_ids]
        a_scale = ABYSS_PULL_STRENGTH * (ABYSS_HAZARD_RADIUS - min_a_dist[a_ids]) / ABYSS_HAZARD_RADIUS
        robot_ext_forces[a_ids, 0, 0] = robot_ext_forces[a_ids, 0, 0] + 210.0 * dir_ax * a_scale
        robot_ext_forces[a_ids, 0, 1] = robot_ext_forces[a_ids, 0, 1] + 210.0 * dir_ay * a_scale
        robot_ext_forces[a_ids, 0, 2] = robot_ext_forces[a_ids, 0, 2] - 195.0 * ABYSS_DOWNWARD_PULL_VZ
        has_env_force = True

    if has_env_force:
        robot.set_external_force_and_torque(robot_ext_forces, robot_ext_torques, body_ids=[0])

    env._pre_physics_world_done_for_step = True
    if hasattr(env, "common_step_counter"):
        env._last_pre_physics_world_step = int(env.common_step_counter)


def _step_disaster_physics_and_hazard_damage(
    env,
    root_pos_w: torch.Tensor,
    env_origins: torch.Tensor,
    arena_xy: torch.Tensor,
    near_food: torch.Tensor,
) -> torch.Tensor:
    device = root_pos_w.device
    num_envs = int(root_pos_w.shape[0])
    hazard_dmg = torch.zeros((num_envs,), dtype=torch.float32, device=device)
    terrain_shift = float(getattr(env, "terrain_shift_x", 0.0))
    in_safe_room = is_in_safe_study_room(arena_xy)

    pre_step_ran = bool(getattr(env, "_pre_physics_world_done_for_step", False))
    env._pre_physics_world_done_for_step = False

    for m_idx in range(2):
        m_key = f"meteor_{m_idx}"
        meteor_obj = env.scene[m_key]
        m_pos_w = meteor_obj.data.root_pos_w
        m_local = m_pos_w - env_origins
        m_arena_x = m_local[:, 0] + 50.0
        m_arena_y = m_local[:, 1] + 50.0
        m_ground_z = _compute_ground_height_torch(m_arena_x, m_arena_y, terrain_shift_x=terrain_shift)
        impacted = m_local[:, 2] <= (m_ground_z + METEOR_IMPACT_HEIGHT_MARGIN)
        if torch.any(impacted):
            dist_xy = torch.linalg.norm(root_pos_w[:, :2] - m_pos_w[:, :2], dim=-1)
            blast_hit = impacted & (dist_xy < METEOR_BLAST_RADIUS) & (~in_safe_room)
            blast_dmg = torch.where(
                blast_hit,
                METEOR_BLAST_MAX_DAMAGE * (METEOR_BLAST_RADIUS - dist_xy) / METEOR_BLAST_RADIUS,
                torch.zeros_like(hazard_dmg),
            )
            hazard_dmg = hazard_dmg + blast_dmg

            imp_ids = torch.where(impacted)[0]
            n_imp = int(imp_ids.numel())
            respawn_pose = meteor_obj.data.default_root_state[imp_ids, :7].clone()
            target_rx = root_pos_w[imp_ids, 0] - env_origins[imp_ids, 0]
            target_ry = root_pos_w[imp_ids, 1] - env_origins[imp_ids, 1]
            rand_mx = target_rx + torch.empty((n_imp,), dtype=torch.float32, device=device).uniform_(-6.5, 6.5)
            rand_my = target_ry + torch.empty((n_imp,), dtype=torch.float32, device=device).uniform_(-6.5, 6.5)
            rand_mz = torch.empty((n_imp,), dtype=torch.float32, device=device).uniform_(11.0, 15.0)
            respawn_pose[:, 0] = env_origins[imp_ids, 0] + rand_mx
            respawn_pose[:, 1] = env_origins[imp_ids, 1] + rand_my
            respawn_pose[:, 2] = env_origins[imp_ids, 2] + rand_mz
            respawn_vel = torch.zeros((n_imp, 6), dtype=torch.float32, device=device)
            respawn_vel[:, 2] = -12.0
            meteor_obj.write_root_pose_to_sim(respawn_pose, env_ids=imp_ids)
            meteor_obj.write_root_velocity_to_sim(respawn_vel, env_ids=imp_ids)

    abyss_pos_w = _resolve_scene_entity_positions_w(env, "abyss_pos_w", "abyss_hazard", 2, device)
    ag_abyss_dist_xy = torch.linalg.norm(root_pos_w[:, :2].unsqueeze(1) - abyss_pos_w[:, :, :2], dim=-1)
    min_abyss_d = ag_abyss_dist_xy.min(dim=1).values
    ag_local_z = root_pos_w[:, 2] - env_origins[:, 2]
    ag_ground_z = _compute_ground_height_torch(arena_xy[:, 0], arena_xy[:, 1], terrain_shift_x=terrain_shift)
    in_abyss = (min_abyss_d < ABYSS_HAZARD_RADIUS) & ((ag_local_z - ag_ground_z) < 2.6) & (~in_safe_room)
    hazard_dmg = hazard_dmg + torch.where(
        in_abyss,
        torch.full_like(hazard_dmg, ABYSS_HAZARD_DAMAGE),
        torch.zeros_like(hazard_dmg),
    )

    vortex_pos_w = _resolve_scene_entity_positions_w(env, "vortex_pos_w", "vortex_center", 2, device)
    vortex_diff_xy = root_pos_w[:, :2].unsqueeze(1) - vortex_pos_w[:, :, :2]
    vortex_dist = torch.linalg.norm(vortex_diff_xy, dim=-1)
    min_v_dist, min_v_idx = torch.min(vortex_dist, dim=1)
    if not pre_step_ran:
        in_vortex = (min_v_dist < VORTEX_PULL_RADIUS) & (min_v_dist > 0.4) & env.agent_alive & (~in_safe_room)
        if torch.any(in_vortex):
            v_ids = torch.where(in_vortex)[0]
            gather_idx = min_v_idx.unsqueeze(1).unsqueeze(2).expand(-1, 1, 2)
            nearest_diff = torch.gather(vortex_diff_xy, 1, gather_idx).squeeze(1)
            dx = nearest_diff[v_ids, 0] / min_v_dist[v_ids]
            dy = nearest_diff[v_ids, 1] / min_v_dist[v_ids]
            pull_scale = VORTEX_PULL_STRENGTH * (VORTEX_PULL_RADIUS - min_v_dist[v_ids]) / VORTEX_PULL_RADIUS
            dv_x = (-0.35 * dx - 0.85 * dy) * pull_scale
            dv_y = (-0.35 * dy + 0.85 * dx) * pull_scale
            robot = env.scene["robot"]
            v_forces = torch.zeros(( int(v_ids.numel()), 1, 3), dtype=torch.float32, device=device)
            v_torques = torch.zeros((int(v_ids.numel()), 1, 3), dtype=torch.float32, device=device)
            v_forces[:, 0, 0] = 185.0 * dv_x
            v_forces[:, 0, 1] = 185.0 * dv_y
            v_torques[:, 0, 2] = 28.0 * pull_scale
            robot.set_external_force_and_torque(v_forces, v_torques, body_ids=[0], env_ids=v_ids)

    vortex_core_hit = (min_v_dist < VORTEX_CORE_DAMAGE_RADIUS) & (~in_safe_room)
    hazard_dmg = hazard_dmg + torch.where(
        vortex_core_hit,
        VORTEX_CORE_MAX_DAMAGE * (VORTEX_CORE_DAMAGE_RADIUS - min_v_dist) / VORTEX_CORE_DAMAGE_RADIUS,
        torch.zeros_like(hazard_dmg),
    )

    signpost_pos_w = _resolve_scene_entity_positions_w(env, "signpost_pos_w", "signpost_panel", 2, device)
    to_sign_xy = signpost_pos_w[:, :, :2] - root_pos_w[:, :2].unsqueeze(1)
    sign_dist_xy = torch.linalg.norm(to_sign_xy, dim=-1)
    min_sign_d, min_sign_idx = torch.min(sign_dist_xy, dim=1)
    robot = env.scene["robot"]
    rq = robot.data.root_quat_w
    body_yaw = torch.atan2(
        2.0 * (rq[:, 0] * rq[:, 3] + rq[:, 1] * rq[:, 2]),
        1.0 - 2.0 * (rq[:, 2] * rq[:, 2] + rq[:, 3] * rq[:, 3]),
    )
    gaze_off = getattr(env, "agent_gaze_yaw_offset_rad", None)
    if gaze_off is not None and gaze_off.shape == (num_envs,):
        gaze_yaw = body_yaw + gaze_off.to(device=device, dtype=torch.float32)
    else:
        gaze_yaw = body_yaw
    sign_gather = min_sign_idx.unsqueeze(1).unsqueeze(2).expand(-1, 1, 2)
    near_to_sign = torch.gather(to_sign_xy, 1, sign_gather).squeeze(1)
    sign_bearing = torch.atan2(near_to_sign[:, 1], near_to_sign[:, 0])
    sign_rel_yaw = torch.atan2(torch.sin(sign_bearing - gaze_yaw), torch.cos(sign_bearing - gaze_yaw))
    foveated_signpost = (min_sign_d < SIGNPOST_CUE_RADIUS) & (torch.abs(sign_rel_yaw) <= 0.85)
    env.agent_signpost_cued = env.agent_signpost_cued | foveated_signpost

    if not bool(getattr(env, "disable_mimic_food_hazard", False)):
        ate_uncued_mimic = near_food[:, 3] & (~env.agent_signpost_cued) & (~in_safe_room)
        hazard_dmg = hazard_dmg + torch.where(
            ate_uncued_mimic,
            torch.full_like(hazard_dmg, MIMIC_FOOD_DAMAGE),
            torch.zeros_like(hazard_dmg),
        )

    if torch.any(near_food):
        _respawn_consumed_foods(env, near_food, root_pos_w, env_origins, terrain_shift)

    return hazard_dmg


def compute_survival_safety_gate(
    hp: torch.Tensor,
    energy: torch.Tensor,
    pain: torch.Tensor,
    dmg: torch.Tensor,
    alive: torch.Tensor,
) -> torch.Tensor:
    hp_gate = torch.clamp((hp - 0.35) / 0.65, 0.0, 1.0)
    energy_gate = torch.clamp((energy - 0.22) / 0.78, 0.0, 1.0)
    pain_gate = torch.exp(-4.5 * (pain + dmg))
    return hp_gate * energy_gate * pain_gate * alive.to(dtype=torch.float32)


def step_homeostasis_metabolism_and_damage(env) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ensure_homeostasis_curiosity_buffers(env)
    robot = env.scene["robot"]
    root_pos_w = robot.data.root_pos_w
    root_lin_vel_w = robot.data.root_lin_vel_w
    device = root_pos_w.device
    num_envs = int(root_pos_w.shape[0])

    env_origins = _resolve_env_origins(env, num_envs, device)
    _, arena_xy = compute_local_and_arena_xy(env)
    food_pos_w = _resolve_scene_entity_positions_w(env, "food_pos_w", "food", 4, device)
    pred_pos_w = _resolve_scene_entity_positions_w(env, "predator_pos_w", "predator", 4, device)

    signpost_pos_w = _resolve_scene_entity_positions_w(env, "signpost_pos_w", "signpost_panel", 2, device)
    sign_dist_xy = torch.linalg.norm(root_pos_w[:, :2].unsqueeze(1) - signpost_pos_w[:, :, :2], dim=-1)
    env.agent_signpost_cued = env.agent_signpost_cued | (sign_dist_xy.min(dim=1).values < SIGNPOST_CUE_RADIUS)

    nutritious_mask = torch.ones((num_envs, 4), dtype=torch.bool, device=device)
    if not bool(getattr(env, "disable_mimic_food_hazard", False)):
        nutritious_mask[:, 3] = env.agent_signpost_cued

    actions = None
    if hasattr(env, "action_manager") and hasattr(env.action_manager, "action"):
        actions = env.action_manager.action
    elif hasattr(env, "last_actions"):
        actions = env.last_actions

    thought_tax = getattr(env, "thought_metabolic_tax", 0.0)
    total_metabolic_cost = compute_atp_metabolic_cost(
        root_lin_vel_w=root_lin_vel_w,
        actions=actions,
        thought_metabolic_tax=thought_tax,
    )

    delta_e, updated_resource, near_food = compute_foraging_energy_and_resource_update(
        root_pos_w=root_pos_w,
        food_pos_w=food_pos_w,
        env_origins=env_origins,
        alive=env.agent_alive,
        quadrant_resource_level=env.quadrant_resource_level,
        nutritious_food_mask=nutritious_mask,
    )
    env.quadrant_resource_level = updated_resource

    env.agent_energy = torch.clamp(env.agent_energy + delta_e - total_metabolic_cost, 0.0, 1.0)

    base_dmg, _, _, _ = compute_environmental_and_starvation_damage(
        root_pos_w=root_pos_w,
        pred_pos_w=pred_pos_w,
        arena_xy=arena_xy,
        energy=env.agent_energy,
    )
    disaster_dmg = _step_disaster_physics_and_hazard_damage(
        env=env,
        root_pos_w=root_pos_w,
        env_origins=env_origins,
        arena_xy=arena_xy,
        near_food=near_food,
    )
    total_dmg = base_dmg + disaster_dmg

    env.agent_survival_safety_gate = compute_survival_safety_gate(
        hp=env.agent_hp,
        energy=env.agent_energy,
        pain=env.agent_pain,
        dmg=total_dmg,
        alive=env.agent_alive,
    )

    env.agent_pain = 1.0 - torch.exp(-2.5 * total_dmg)
    env.agent_hp = torch.clamp(
        env.agent_hp - total_dmg - METABOLIC_DEFICIT_HP_DECAY * (1.0 - env.agent_energy),
        min=0.0,
        max=1.0,
    )
    env.agent_alive = env.agent_hp > 0.0
    env.agent_last_damage = total_dmg
    env.agent_last_delta_energy = delta_e
    env.agent_last_metabolic_cost = total_metabolic_cost

    if hasattr(env, "common_step_counter"):
        env._last_homeostasis_step_counter = int(env.common_step_counter)

    return total_dmg, delta_e, total_metabolic_cost


def _sync_step_homeostasis_if_needed(env) -> None:
    ensure_homeostasis_curiosity_buffers(env)
    if hasattr(env, "common_step_counter"):
        cur_step = int(env.common_step_counter)
        if int(getattr(env, "_last_homeostasis_step_counter", -1)) != cur_step:
            step_homeostasis_metabolism_and_damage(env)


def reward_biome_quadrant_curiosity(
    env,
    bonus_value: float = CURIOSITY_NOVEL_BIOME_REWARD,
    visited_threshold: float = NOVEL_QUADRANT_THRESHOLD,
    visit_decay: float = VISITED_QUADRANT_DECAY,
    require_safety_gate: bool = True,
    modulate_by_gate: bool = False,
) -> torch.Tensor:
    _sync_step_homeostasis_if_needed(env)
    robot = env.scene["robot"]
    device = robot.data.root_pos_w.device
    num_envs = int(robot.data.root_pos_w.shape[0])

    _, arena_xy = compute_local_and_arena_xy(env)
    ag_quad = compute_biome_quadrant_index(arena_xy[:, 0], arena_xy[:, 1])
    env_indices = torch.arange(num_envs, dtype=torch.long, device=device)

    env.agent_visited_quadrants = env.agent_visited_quadrants * float(visit_decay)
    cur_visit_val = env.agent_visited_quadrants[env_indices, ag_quad]

    if require_safety_gate:
        gate_ok = env.agent_survival_safety_gate > SAFETY_GATE_MIN_THRESHOLD
    else:
        gate_ok = env.agent_alive

    entered_novel_biome = (cur_visit_val < float(visited_threshold)) & gate_ok
    quad_res = env.quadrant_resource_level[env_indices, ag_quad]

    modulated_bonus = torch.where(
        entered_novel_biome,
        float(bonus_value) * env.agent_survival_safety_gate * quad_res,
        torch.zeros((num_envs,), dtype=torch.float32, device=device),
    )
    raw_bonus = torch.where(
        entered_novel_biome,
        torch.full((num_envs,), float(bonus_value), dtype=torch.float32, device=device),
        torch.zeros((num_envs,), dtype=torch.float32, device=device),
    )

    env.agent_visited_quadrants[env_indices, ag_quad] = 1.0
    env.last_modulated_curiosity_bonus = modulated_bonus
    env.curiosity_explorations_total = int(getattr(env, "curiosity_explorations_total", 0)) + int(
        entered_novel_biome.sum().item()
    )

    return modulated_bonus if modulate_by_gate else raw_bonus


def reward_homeostatic_vitality(env) -> torch.Tensor:
    _sync_step_homeostasis_if_needed(env)
    vitality = (
        0.50 * env.agent_hp
        + 0.35 * env.agent_energy
        - 0.25 * env.agent_pain
    )
    return vitality * env.agent_alive.to(dtype=torch.float32)


def termination_hp_depleted(env, min_hp: float = 0.0) -> torch.Tensor:
    _sync_step_homeostasis_if_needed(env)
    if bool(getattr(env, "manual_mortality_mode", False)):
        return torch.zeros_like(env.agent_hp, dtype=torch.bool)
    return env.agent_hp <= float(min_hp)
