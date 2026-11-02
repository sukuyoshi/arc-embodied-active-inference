from __future__ import annotations

import math
import torch

from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from agi_homeostasis_curiosity_mdp import (
    compute_survival_safety_gate,
    ensure_homeostasis_curiosity_buffers,
    pre_physics_step_world_dynamics,
    step_spinal_cpg_and_bipedal_locomotion,
)

GAZE_OPS: tuple[str, ...] = (
    "gaze_salient_target",
    "gaze_corner_sample",
    "gaze_next_matching_attr",
    "gaze_count_objects",
)

WM_OPS: tuple[str, ...] = (
    "wm_store_attr",
    "wm_store_sample_pair",
    "wm_store_count",
)

FINGER_OPS: tuple[str, ...] = (
    "finger_step_until_contact",
    "finger_paint_from_wm",
    "finger_apply_to_gaze",
    "finger_step_n_from_wm",
)

TOY_ENTITY_BLOCK_KEYS: tuple[str, ...] = (
    "toy_block_0",
    "toy_block_1",
    "toy_block_2",
    "toy_block_3",
)

TOY_ENTITY_SLOT_KEYS: tuple[str, ...] = (
    "toy_slot_0",
    "toy_slot_1",
    "toy_slot_2",
    "toy_slot_3",
)

TOY_TOUCH_RADIUS: float = 4.2
TOY_SAFETY_GATE_MIN: float = 0.20
TOY_TOUCH_REWARD_COEFF: float = 0.04
TOY_SOLVE_REWARD_COEFF: float = 0.36
TOY_GROUP_STEP_REWARD_COEFF: float = 0.11
TOY_SLOT_SOLVE_DISTANCE: float = 2.4
TOY_YAW_TOLERANCE_RAD: float = 0.22
TOY_HABITUATION_PENALTY: float = 1.85
TOY_SINGLE_STEP_ADVANCE_M: float = 2.85
TOY_GROUP_STEP_ADVANCE_M: float = 1.05

ARC_PALETTE_RGB: tuple[tuple[float, float, float], ...] = (
    (0.10, 0.10, 0.10),
    (0.00, 0.45, 0.85),
    (0.90, 0.15, 0.15),
    (0.18, 0.80, 0.25),
    (0.95, 0.85, 0.10),
    (0.60, 0.60, 0.65),
    (0.90, 0.20, 0.75),
    (0.95, 0.50, 0.10),
    (0.35, 0.80, 0.95),
    (0.55, 0.10, 0.20),
)


def _yaw_to_quat_wxyz(yaw_rad: torch.Tensor) -> torch.Tensor:
    half_yaw = 0.5 * yaw_rad
    qw = torch.cos(half_yaw)
    qx = torch.zeros_like(half_yaw)
    qy = torch.zeros_like(half_yaw)
    qz = torch.sin(half_yaw)
    return torch.stack([qw, qx, qy, qz], dim=-1)


def _quat_wxyz_to_yaw_rad(quat_wxyz: torch.Tensor) -> torch.Tensor:
    qw = quat_wxyz[..., 0]
    qx = quat_wxyz[..., 1]
    qy = quat_wxyz[..., 2]
    qz = quat_wxyz[..., 3]
    return torch.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def _wrap_angle_rad(angle_rad: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle_rad), torch.cos(angle_rad))


def _lookup_palette_rgb(color_indices: torch.Tensor) -> torch.Tensor:
    palette = torch.tensor(ARC_PALETTE_RGB, dtype=torch.float32, device=color_indices.device)
    clamped = torch.clamp(color_indices.to(dtype=torch.long), 0, palette.shape[0] - 1)
    return palette[clamped]


def _sync_toy_block_usd_color(env, env_i: int, t_idx: int, rgb_tensor: torch.Tensor) -> bool:
    rgb_tuple = (
        float(rgb_tensor[0].item()),
        float(rgb_tensor[1].item()),
        float(rgb_tensor[2].item()),
    )
    try:
        from isaaclab.sim.utils import safe_set_attribute_on_usd_prim
        from isaaclab.sim.utils.stage import get_current_stage
        from pxr import Gf, UsdGeom

        stage = get_current_stage()
        if stage is None:
            return False
        prim_name = f"ToyBlock{int(t_idx)}"
        candidate_shader_paths = (
            f"/World/envs/env_{int(env_i)}/{prim_name}/geometry/material/Shader",
        )
        updated = False
        for spath in candidate_shader_paths:
            sprim = stage.GetPrimAtPath(spath)
            if sprim is not None and sprim.IsValid():
                safe_set_attribute_on_usd_prim(sprim, "inputs:diffuse_color", rgb_tuple, camel_case=True)
                updated = True
                break
        mesh_path = f"/World/envs/env_{int(env_i)}/{prim_name}/geometry/mesh"
        mprim = stage.GetPrimAtPath(mesh_path)
        if mprim is not None and mprim.IsValid():
            gprim = UsdGeom.Gprim(mprim)
            gprim.GetDisplayColorAttr().Set([Gf.Vec3f(*rgb_tuple)])
            updated = True
        return updated
    except Exception:
        return False


def _sync_toy_yaw_from_sim(env, sync_slots: bool = False) -> None:
    if not hasattr(env, "scene"):
        return
    for t_idx in range(4):
        b_key = TOY_ENTITY_BLOCK_KEYS[t_idx]
        b_quat = env.scene[b_key].data.root_quat_w
        b_yaw = _quat_wxyz_to_yaw_rad(b_quat)
        env.toy_block_yaw_rad[:, t_idx] = b_yaw
        env.toy_block_orient[:, t_idx] = torch.round(b_yaw / (math.pi * 0.5)).to(dtype=torch.long) % 4
        if sync_slots:
            s_key = TOY_ENTITY_SLOT_KEYS[t_idx]
            s_quat = env.scene[s_key].data.root_quat_w
            s_yaw = _quat_wxyz_to_yaw_rad(s_quat)
            env.toy_slot_yaw_rad[:, t_idx] = s_yaw
            env.toy_slot_orient[:, t_idx] = torch.round(s_yaw / (math.pi * 0.5)).to(dtype=torch.long) % 4


def ensure_sensorimotor_puzzle_buffers(env) -> None:
    robot = env.scene["robot"]
    device = robot.data.root_pos_w.device
    num_envs = int(robot.data.root_pos_w.shape[0])
    num_toys = 4

    if not hasattr(env, "agent_body_logits") or env.agent_body_logits is None or env.agent_body_logits.shape != (num_envs, 11):
        env.agent_body_logits = torch.zeros((num_envs, 11), dtype=torch.float32, device=device)
    if not hasattr(env, "agent_vocal_signals") or env.agent_vocal_signals is None or env.agent_vocal_signals.shape != (num_envs, 2):
        env.agent_vocal_signals = torch.zeros((num_envs, 2), dtype=torch.float32, device=device)

    if not hasattr(env, "agent_gaze_focus") or env.agent_gaze_focus is None or env.agent_gaze_focus.shape != (num_envs,):
        env.agent_gaze_focus = torch.zeros((num_envs,), dtype=torch.long, device=device)
    if not hasattr(env, "agent_gaze_yaw_offset_rad") or env.agent_gaze_yaw_offset_rad is None or env.agent_gaze_yaw_offset_rad.shape != (num_envs,):
        env.agent_gaze_yaw_offset_rad = torch.zeros((num_envs,), dtype=torch.float32, device=device)
    if not hasattr(env, "agent_gaze_pitch_offset_rad") or env.agent_gaze_pitch_offset_rad is None or env.agent_gaze_pitch_offset_rad.shape != (num_envs,):
        env.agent_gaze_pitch_offset_rad = torch.zeros((num_envs,), dtype=torch.float32, device=device)
    if not hasattr(env, "agent_foveal_hit_verified") or env.agent_foveal_hit_verified is None or env.agent_foveal_hit_verified.shape != (num_envs,):
        env.agent_foveal_hit_verified = torch.zeros((num_envs,), dtype=torch.bool, device=device)
    if not hasattr(env, "agent_hand_contact_verified") or env.agent_hand_contact_verified is None or env.agent_hand_contact_verified.shape != (num_envs,):
        env.agent_hand_contact_verified = torch.zeros((num_envs,), dtype=torch.bool, device=device)
    if not hasattr(env, "agent_active_hand_pos_w") or env.agent_active_hand_pos_w is None or env.agent_active_hand_pos_w.shape != (num_envs, 3):
        env.agent_active_hand_pos_w = torch.zeros((num_envs, 3), dtype=torch.float32, device=device)
    if not hasattr(env, "agent_hand_to_block_dist_m") or env.agent_hand_to_block_dist_m is None or env.agent_hand_to_block_dist_m.shape != (num_envs,):
        env.agent_hand_to_block_dist_m = torch.full((num_envs,), 99.0, dtype=torch.float32, device=device)
    if not hasattr(env, "agent_reach_joint_targets_21d") or env.agent_reach_joint_targets_21d is None or env.agent_reach_joint_targets_21d.shape != (num_envs, 21):
        env.agent_reach_joint_targets_21d = torch.zeros((num_envs, 21), dtype=torch.float32, device=device)
    if not hasattr(env, "agent_reach_active_timer") or env.agent_reach_active_timer is None or env.agent_reach_active_timer.shape != (num_envs,):
        env.agent_reach_active_timer = torch.zeros((num_envs,), dtype=torch.long, device=device)
    if not hasattr(env, "agent_last_applied_contact_force_n") or env.agent_last_applied_contact_force_n is None or env.agent_last_applied_contact_force_n.shape != (num_envs,):
        env.agent_last_applied_contact_force_n = torch.zeros((num_envs,), dtype=torch.float32, device=device)
    if not hasattr(env, "agent_wm_orient") or env.agent_wm_orient is None or env.agent_wm_orient.shape != (num_envs,):
        env.agent_wm_orient = torch.full((num_envs,), -1, dtype=torch.long, device=device)
    if not hasattr(env, "agent_wm_yaw_rad") or env.agent_wm_yaw_rad is None or env.agent_wm_yaw_rad.shape != (num_envs,):
        env.agent_wm_yaw_rad = torch.full((num_envs,), -10.0, dtype=torch.float32, device=device)
    if not hasattr(env, "agent_wm_key_color") or env.agent_wm_key_color is None or env.agent_wm_key_color.shape != (num_envs,):
        env.agent_wm_key_color = torch.full((num_envs,), -1, dtype=torch.long, device=device)
    if not hasattr(env, "agent_wm_val_color") or env.agent_wm_val_color is None or env.agent_wm_val_color.shape != (num_envs,):
        env.agent_wm_val_color = torch.full((num_envs,), -1, dtype=torch.long, device=device)
    if not hasattr(env, "agent_wm_count") or env.agent_wm_count is None or env.agent_wm_count.shape != (num_envs,):
        env.agent_wm_count = torch.zeros((num_envs,), dtype=torch.long, device=device)

    if not hasattr(env, "toy_type") or env.toy_type is None or env.toy_type.shape != (num_toys,):
        env.toy_type = torch.tensor([0, 1, 2, 3], dtype=torch.long, device=device)

    if not hasattr(env, "toy_block_orient") or env.toy_block_orient is None or env.toy_block_orient.shape != (num_envs, num_toys):
        env.toy_block_orient = torch.zeros((num_envs, num_toys), dtype=torch.long, device=device)
    if not hasattr(env, "toy_slot_orient") or env.toy_slot_orient is None or env.toy_slot_orient.shape != (num_envs, num_toys):
        env.toy_slot_orient = torch.ones((num_envs, num_toys), dtype=torch.long, device=device)
    if not hasattr(env, "toy_block_yaw_rad") or env.toy_block_yaw_rad is None or env.toy_block_yaw_rad.shape != (num_envs, num_toys):
        env.toy_block_yaw_rad = env.toy_block_orient.to(dtype=torch.float32) * (math.pi * 0.5)
    if not hasattr(env, "toy_slot_yaw_rad") or env.toy_slot_yaw_rad is None or env.toy_slot_yaw_rad.shape != (num_envs, num_toys):
        env.toy_slot_yaw_rad = env.toy_slot_orient.to(dtype=torch.float32) * (math.pi * 0.5)
    if not hasattr(env, "toy_insertion_depth") or env.toy_insertion_depth is None or env.toy_insertion_depth.shape != (num_envs, num_toys):
        env.toy_insertion_depth = torch.zeros((num_envs, num_toys), dtype=torch.float32, device=device)

    if not hasattr(env, "toy_sample_key_color") or env.toy_sample_key_color is None or env.toy_sample_key_color.shape != (num_envs, num_toys):
        base_keys = torch.tensor([1, 2, 3, 4], dtype=torch.long, device=device).unsqueeze(0).expand(num_envs, num_toys)
        env.toy_sample_key_color = base_keys.clone()
    if not hasattr(env, "toy_sample_val_color") or env.toy_sample_val_color is None or env.toy_sample_val_color.shape != (num_envs, num_toys):
        base_vals = torch.tensor([6, 7, 8, 9], dtype=torch.long, device=device).unsqueeze(0).expand(num_envs, num_toys)
        env.toy_sample_val_color = base_vals.clone()
    if not hasattr(env, "toy_block_color") or env.toy_block_color is None or env.toy_block_color.shape != (num_envs, num_toys):
        env.toy_block_color = env.toy_sample_key_color.clone()
    if not hasattr(env, "toy_block_rgb") or env.toy_block_rgb is None or env.toy_block_rgb.shape != (num_envs, num_toys, 3):
        env.toy_block_rgb = _lookup_palette_rgb(env.toy_block_color)

    if not hasattr(env, "toy_target_count") or env.toy_target_count is None or env.toy_target_count.shape != (num_envs, num_toys):
        base_counts = torch.tensor([3, 3, 4, 3], dtype=torch.long, device=device).unsqueeze(0).expand(num_envs, num_toys)
        env.toy_target_count = base_counts.clone()
    if not hasattr(env, "toy_step_counter") or env.toy_step_counter is None or env.toy_step_counter.shape != (num_envs, num_toys):
        env.toy_step_counter = torch.zeros((num_envs, num_toys), dtype=torch.long, device=device)
    if not hasattr(env, "toy_group_size") or env.toy_group_size is None or env.toy_group_size.shape != (num_envs, num_toys):
        env.toy_group_size = torch.full((num_envs, num_toys), 4, dtype=torch.long, device=device)
    if not hasattr(env, "toy_group_progress") or env.toy_group_progress is None or env.toy_group_progress.shape != (num_envs, num_toys):
        env.toy_group_progress = torch.zeros((num_envs, num_toys), dtype=torch.long, device=device)

    if not hasattr(env, "agent_toy_habituation") or env.agent_toy_habituation is None or env.agent_toy_habituation.shape != (num_envs, 4, 48):
        env.agent_toy_habituation = torch.zeros((num_envs, 4, 48), dtype=torch.float32, device=device)
    if not hasattr(env, "agent_cerebellar_gain") or env.agent_cerebellar_gain is None or env.agent_cerebellar_gain.shape != (num_envs, 4):
        env.agent_cerebellar_gain = torch.zeros((num_envs, 4), dtype=torch.float32, device=device)

    if not hasattr(env, "last_toy_puzzle_reward") or env.last_toy_puzzle_reward is None or env.last_toy_puzzle_reward.shape != (num_envs,):
        env.last_toy_puzzle_reward = torch.zeros((num_envs,), dtype=torch.float32, device=device)
    if not hasattr(env, "last_solved_toy_mask") or env.last_solved_toy_mask is None or env.last_solved_toy_mask.shape != (num_envs, num_toys):
        env.last_solved_toy_mask = torch.zeros((num_envs, num_toys), dtype=torch.bool, device=device)
    if not hasattr(env, "pending_toy_respawn_mask") or env.pending_toy_respawn_mask is None or env.pending_toy_respawn_mask.shape != (num_envs, num_toys):
        env.pending_toy_respawn_mask = torch.zeros((num_envs, num_toys), dtype=torch.bool, device=device)

    if not hasattr(env, "toys_solved_total"):
        env.toys_solved_total = 0
    if not hasattr(env, "sensorimotor_chunk_hits_total"):
        env.sensorimotor_chunk_hits_total = 0
    if not hasattr(env, "cerebellar_corrections_total"):
        env.cerebellar_corrections_total = 0
    if not hasattr(env, "physical_hand_contacts_total"):
        env.physical_hand_contacts_total = 0
    if not hasattr(env, "upper_limb_reach_movements_total"):
        env.upper_limb_reach_movements_total = 0
    if not hasattr(env, "discovered_sensorimotor_chunks") or env.discovered_sensorimotor_chunks is None:
        env.discovered_sensorimotor_chunks = {}
    if not hasattr(env, "_last_puzzle_step_counter"):
        env._last_puzzle_step_counter = -1
    if not hasattr(env, "_last_pre_physics_puzzle_step"):
        env._last_pre_physics_puzzle_step = -1
    if not hasattr(env, "_pre_physics_puzzle_records"):
        env._pre_physics_puzzle_records = []


def _respawn_single_toy_block(env, env_i: int, t_idx: int) -> None:
    device = env.scene["robot"].data.root_pos_w.device
    env.toy_group_progress[env_i, t_idx] = 0
    env.toy_step_counter[env_i, t_idx] = 0
    env.toy_insertion_depth[env_i, t_idx] = 0.0
    env.toy_block_color[env_i, t_idx] = env.toy_sample_key_color[env_i, t_idx]
    env.toy_block_rgb[env_i, t_idx] = _lookup_palette_rgb(env.toy_block_color[env_i, t_idx])
    _sync_toy_block_usd_color(env, env_i, t_idx, env.toy_block_rgb[env_i, t_idx])
    env.toy_block_orient[env_i, t_idx] = (env.toy_slot_orient[env_i, t_idx] + 2) % 4
    env.toy_block_yaw_rad[env_i, t_idx] = _wrap_angle_rad(
        env.toy_block_orient[env_i, t_idx].to(dtype=torch.float32) * (math.pi * 0.5)
    )

    env_id_tensor = torch.tensor([env_i], dtype=torch.long, device=device)
    block_obj = env.scene[TOY_ENTITY_BLOCK_KEYS[t_idx]]
    respawn_pose = block_obj.data.default_root_state[env_id_tensor, :7].clone()
    respawn_pose[:, :3] += env.scene.env_origins[env_id_tensor]
    respawn_pose[:, 3:7] = _yaw_to_quat_wxyz(env.toy_block_yaw_rad[env_i : env_i + 1, t_idx])
    zero_vel = torch.zeros((1, 6), dtype=torch.float32, device=device)
    block_obj.write_root_pose_to_sim(respawn_pose, env_ids=env_id_tensor)
    block_obj.write_root_velocity_to_sim(zero_vel, env_ids=env_id_tensor)


def reset_sensorimotor_puzzle_state(env, env_ids: torch.Tensor | None) -> None:
    ensure_sensorimotor_puzzle_buffers(env)
    robot = env.scene["robot"]
    device = robot.data.root_pos_w.device
    num_envs = int(robot.data.root_pos_w.shape[0])

    if env_ids is None:
        ids = torch.arange(num_envs, dtype=torch.long, device=device)
    else:
        ids = env_ids.to(device=device, dtype=torch.long)

    if ids.numel() == 0:
        return

    env.agent_body_logits[ids] = 0.0
    env.agent_vocal_signals[ids] = 0.0
    env.agent_gaze_focus[ids] = 0
    env.agent_gaze_yaw_offset_rad[ids] = 0.0
    env.agent_gaze_pitch_offset_rad[ids] = 0.0
    env.agent_foveal_hit_verified[ids] = False
    env.agent_hand_contact_verified[ids] = False
    env.agent_active_hand_pos_w[ids] = 0.0
    env.agent_hand_to_block_dist_m[ids] = 99.0
    env.agent_reach_joint_targets_21d[ids] = 0.0
    env.agent_reach_active_timer[ids] = 0
    env.agent_last_applied_contact_force_n[ids] = 0.0
    env.agent_wm_orient[ids] = -1
    env.agent_wm_yaw_rad[ids] = -10.0
    env.agent_wm_key_color[ids] = -1
    env.agent_wm_val_color[ids] = -1
    env.agent_wm_count[ids] = 0

    env.toy_block_orient[ids] = 0
    env.toy_slot_orient[ids] = 1
    env.toy_block_yaw_rad[ids] = _wrap_angle_rad(env.toy_block_orient[ids].to(dtype=torch.float32) * (math.pi * 0.5))
    env.toy_slot_yaw_rad[ids] = _wrap_angle_rad(env.toy_slot_orient[ids].to(dtype=torch.float32) * (math.pi * 0.5))
    env.toy_insertion_depth[ids] = 0.0

    key_defaults = torch.tensor([1, 2, 3, 4], dtype=torch.long, device=device).unsqueeze(0).expand(ids.numel(), 4)
    val_defaults = torch.tensor([6, 7, 8, 9], dtype=torch.long, device=device).unsqueeze(0).expand(ids.numel(), 4)
    cnt_defaults = torch.tensor([3, 3, 4, 3], dtype=torch.long, device=device).unsqueeze(0).expand(ids.numel(), 4)

    env.toy_sample_key_color[ids] = key_defaults
    env.toy_sample_val_color[ids] = val_defaults
    env.toy_block_color[ids] = key_defaults.clone()
    env.toy_block_rgb[ids] = _lookup_palette_rgb(env.toy_block_color[ids])
    env.toy_target_count[ids] = cnt_defaults
    env.toy_step_counter[ids] = 0
    env.toy_group_progress[ids] = 0
    env.agent_toy_habituation[ids] = 0.0
    env.agent_cerebellar_gain[ids] = 0.0
    env.last_toy_puzzle_reward[ids] = 0.0
    env.last_solved_toy_mask[ids] = False
    env.pending_toy_respawn_mask[ids] = False

    for t_idx in range(4):
        block_obj = env.scene[TOY_ENTITY_BLOCK_KEYS[t_idx]]
        slot_obj = env.scene[TOY_ENTITY_SLOT_KEYS[t_idx]]
        block_pose = block_obj.data.default_root_state[ids, :7].clone()
        block_pose[:, :3] += env.scene.env_origins[ids]
        block_pose[:, 3:7] = _yaw_to_quat_wxyz(env.toy_block_yaw_rad[ids, t_idx])
        block_vel = torch.zeros((ids.numel(), 6), dtype=torch.float32, device=device)
        block_obj.write_root_pose_to_sim(block_pose, env_ids=ids)
        block_obj.write_root_velocity_to_sim(block_vel, env_ids=ids)

        slot_pose = slot_obj.data.default_root_state[ids, :7].clone()
        slot_pose[:, :3] += env.scene.env_origins[ids]
        slot_pose[:, 3:7] = _yaw_to_quat_wxyz(env.toy_slot_yaw_rad[ids, t_idx])
        slot_vel = torch.zeros((ids.numel(), 6), dtype=torch.float32, device=device)
        slot_obj.write_root_pose_to_sim(slot_pose, env_ids=ids)
        slot_obj.write_root_velocity_to_sim(slot_vel, env_ids=ids)

        for env_i in ids.tolist():
            _sync_toy_block_usd_color(env, int(env_i), t_idx, env.toy_block_rgb[int(env_i), t_idx])


def _extract_locomotor_summary_3d(env) -> torch.Tensor:
    robot = env.scene["robot"]
    device = robot.data.root_pos_w.device
    num_envs = int(robot.data.root_pos_w.shape[0])
    if hasattr(env, "action_manager") and hasattr(env.action_manager, "action"):
        act = env.action_manager.action
        if act is not None and act.shape[0] == num_envs and act.shape[1] >= 3:
            u0 = torch.mean(act[:, 0:7], dim=-1) if act.shape[1] >= 21 else act[:, 0]
            u1 = torch.mean(act[:, 7:14], dim=-1) if act.shape[1] >= 21 else act[:, 1]
            u2 = torch.mean(act[:, 14:21], dim=-1) if act.shape[1] >= 21 else act[:, 2]
            return torch.stack([u0, u1, u2], dim=-1).to(device=device, dtype=torch.float32)
    vel = robot.data.root_lin_vel_w
    ang = robot.data.root_ang_vel_w
    u0 = torch.clamp(torch.linalg.norm(vel[:, :2], dim=-1) / 4.2, -1.0, 1.0)
    u1 = torch.clamp(ang[:, 2] / 1.4, -1.0, 1.0)
    u2 = torch.clamp(vel[:, 2] / 5.2, 0.0, 1.0)
    return torch.stack([u0, u1, u2], dim=-1)


def decode_sensorimotor_triple_for_env(
    env,
    env_i: int,
    toy_kind: int,
    motor_u_3d: torch.Tensor,
    vocal_2d: torch.Tensor,
) -> tuple[int, str, str, str, int, int, int]:
    ensure_sensorimotor_puzzle_buffers(env)
    device = env.agent_body_logits.device
    b_logits = env.agent_body_logits[env_i]
    u0 = float(motor_u_3d[env_i, 0].item())
    u1 = float(motor_u_3d[env_i, 1].item())
    u2 = float(motor_u_3d[env_i, 2].item())
    v0 = float(vocal_2d[env_i, 0].item()) if vocal_2d.shape[1] > 0 else 0.0
    v1 = float(vocal_2d[env_i, 1].item()) if vocal_2d.shape[1] > 1 else 0.0

    g_mod = torch.tensor([u0, u1, -u0, -u1], dtype=torch.float32, device=device) * 0.25
    w_mod = torch.tensor([v0, v1, u2], dtype=torch.float32, device=device) * 0.25
    f_mod = torch.tensor([u2, v0, -u2, -v0], dtype=torch.float32, device=device) * 0.25

    g_scores = b_logits[0:4] + g_mod
    w_scores = b_logits[4:7] + w_mod
    f_scores = b_logits[7:11] + f_mod

    habituation_vec = torch.clamp(env.agent_toy_habituation[env_i, int(toy_kind) % 4], min=0.0)
    triple_scores = (
        g_scores[:, None, None] + w_scores[None, :, None] + f_scores[None, None, :]
    ).reshape(-1) - TOY_HABITUATION_PENALTY * habituation_vec

    best_triple = int(torch.argmax(triple_scores).item())
    gaze_idx = best_triple // 12
    wm_idx = (best_triple // 4) % 3
    finger_idx = best_triple % 4
    return (
        best_triple,
        GAZE_OPS[gaze_idx],
        WM_OPS[wm_idx],
        FINGER_OPS[finger_idx],
        gaze_idx,
        wm_idx,
        finger_idx,
    )


def _orient_eyes_and_verify_binocular_foveal_hit(
    env,
    env_i: int,
    target_pos_w: torch.Tensor,
    target_radius: float = 1.35,
) -> bool:
    robot = env.scene["robot"]
    root_pos = robot.data.root_pos_w[env_i]
    root_quat = robot.data.root_quat_w[env_i]
    device = root_pos.device
    body_yaw = _quat_wxyz_to_yaw_rad(root_quat)

    head_pos = root_pos + torch.tensor([0.0, 0.0, 0.55], dtype=torch.float32, device=device)
    delta = target_pos_w - head_pos
    horiz_dist = torch.clamp(torch.linalg.norm(delta[:2]), min=1e-4)
    desired_yaw = torch.atan2(delta[1], delta[0])
    desired_pitch = torch.atan2(delta[2], horiz_dist)

    yaw_diff = _wrap_angle_rad(desired_yaw - body_yaw)
    yaw_off = torch.clamp(yaw_diff, -3.12, 3.12)
    pitch_off = torch.clamp(desired_pitch - (-0.12), -0.82, 0.78)
    env.agent_gaze_yaw_offset_rad[env_i] = yaw_off
    env.agent_gaze_pitch_offset_rad[env_i] = pitch_off
    if hasattr(env, "agent_reach_joint_targets_21d"):
        torso_twist = float(torch.clamp(yaw_off * 0.25, -0.45, 0.45).item())
        env.agent_reach_joint_targets_21d[env_i, 0] = torso_twist * 0.5
        env.agent_reach_joint_targets_21d[env_i, 6] = torso_twist

    gaze_yaw = body_yaw + yaw_off
    gaze_pitch = torch.clamp(-0.12 + pitch_off, -0.94, 0.70)
    cos_p = torch.cos(gaze_pitch)
    sin_p = torch.sin(gaze_pitch)
    ray_dir = torch.stack([cos_p * torch.cos(gaze_yaw), cos_p * torch.sin(gaze_yaw), sin_p], dim=0)
    ray_dir = ray_dir / torch.clamp(torch.linalg.norm(ray_dir), min=1e-6)

    cos_b = torch.cos(body_yaw)
    sin_b = torch.sin(body_yaw)
    eye_hits = []
    for eye_lat in (0.035, -0.035):
        ox = 0.12 * cos_b - float(eye_lat) * sin_b
        oy = 0.12 * sin_b + float(eye_lat) * cos_b
        eye_orig = head_pos + torch.stack([ox, oy, torch.tensor(0.0, dtype=torch.float32, device=device)], dim=0)
        oc = target_pos_w - eye_orig
        t_proj = float(torch.sum(oc * ray_dir).item())
        oc_sq = float(torch.sum(oc * oc).item())
        perp_sq = max(0.0, oc_sq - t_proj * t_proj)
        hit_target = (t_proj > 0.05) and (perp_sq <= float(target_radius * target_radius))
        occluded = False
        if hit_target:
            for obs_k in ("obstacle_0", "obstacle_1", "obstacle_2", "obstacle_3"):
                if obs_k in env.scene.keys():
                    obs_p = env.scene[obs_k].data.root_pos_w[env_i]
                    oc_o = obs_p - eye_orig
                    t_o = float(torch.sum(oc_o * ray_dir).item())
                    if 0.10 < t_o < (t_proj - 0.35):
                        perp_o_sq = max(0.0, float(torch.sum(oc_o * oc_o).item()) - t_o * t_o)
                        if perp_o_sq <= (1.45 * 1.45):
                            occluded = True
                            break
        eye_hits.append(bool(hit_target and not occluded))

    foveal_ok = bool(eye_hits[0] or eye_hits[1])
    env.agent_foveal_hit_verified[env_i] = foveal_ok
    return foveal_ok


def _rotate_vec_by_quat(quat_wxyz: torch.Tensor, vec_b: torch.Tensor) -> torch.Tensor:
    w = quat_wxyz[..., 0:1]
    q_xyz = quat_wxyz[..., 1:4]
    t = 2.0 * torch.linalg.cross(q_xyz, vec_b, dim=-1)
    return vec_b + w * t + torch.linalg.cross(q_xyz, t, dim=-1)


def _rotate_vec_by_quat_inverse(quat_wxyz: torch.Tensor, vec_w: torch.Tensor) -> torch.Tensor:
    w = quat_wxyz[..., 0:1]
    q_xyz = quat_wxyz[..., 1:4]
    t = 2.0 * torch.linalg.cross(q_xyz, vec_w, dim=-1)
    return vec_w - w * t + torch.linalg.cross(q_xyz, t, dim=-1)


def _compute_nearest_block_surface_point_and_dist(
    query_pos_w: torch.Tensor,
    block_pos_w: torch.Tensor,
    block_quat_w: torch.Tensor,
    half_extents: tuple[float, float, float] = (0.70, 0.70, 0.55),
) -> tuple[torch.Tensor, float]:
    rel_w = (query_pos_w - block_pos_w).unsqueeze(0)
    q_b = block_quat_w.reshape(1, 4)
    rel_b = _rotate_vec_by_quat_inverse(q_b, rel_w)[0]
    hx, hy, hz = float(half_extents[0]), float(half_extents[1]), float(half_extents[2])
    clamped_b = torch.stack(
        [
            torch.clamp(rel_b[0], -hx, hx),
            torch.clamp(rel_b[1], -hy, hy),
            torch.clamp(rel_b[2], -hz, hz),
        ],
        dim=0,
    )
    surf_w = block_pos_w + _rotate_vec_by_quat(q_b, clamped_b.unsqueeze(0))[0]
    dist_m = float(torch.linalg.norm(query_pos_w - surf_w).item())
    return surf_w, dist_m


def _resolve_humanoid_limb_indices(robot) -> tuple[int, int, int, int]:
    bnames = list(getattr(robot, "body_names", []))
    r_hand_idx = bnames.index("right_hand") if "right_hand" in bnames else 11
    l_hand_idx = bnames.index("left_hand") if "left_hand" in bnames else 12
    r_sh_idx = bnames.index("right_upper_arm") if "right_upper_arm" in bnames else 4
    l_sh_idx = bnames.index("left_upper_arm") if "left_upper_arm" in bnames else 5
    return r_hand_idx, l_hand_idx, r_sh_idx, l_sh_idx


def _drive_upper_limb_reaching_and_verify_hand_contact(
    env,
    env_i: int,
    target_pos_w: torch.Tensor,
    bimanual: bool = False,
    slot_pos_w: torch.Tensor | None = None,
    block_quat_w: torch.Tensor | None = None,
) -> tuple[bool, torch.Tensor, float]:
    robot = env.scene["robot"]
    device = robot.data.root_pos_w.device
    r_hand_idx, l_hand_idx, r_sh_idx, l_sh_idx = _resolve_humanoid_limb_indices(robot)
    if block_quat_w is None:
        block_quat_w = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=device)

    root_pos_init = robot.data.root_pos_w[env_i]
    root_quat_init = robot.data.root_quat_w[env_i]
    to_target_xy = target_pos_w[:2] - root_pos_init[:2]
    d_xy_init = float(torch.linalg.norm(to_target_xy).item())

    if hasattr(robot.data, "body_pos_w") and robot.data.body_pos_w.shape[1] > max(r_hand_idx, l_hand_idx):
        fk_lag = root_pos_init - robot.data.body_pos_w[env_i, 0, :]
        r_sh_w = robot.data.body_pos_w[env_i, r_sh_idx, :] + fk_lag
        l_sh_w = robot.data.body_pos_w[env_i, l_sh_idx, :] + fk_lag
        r_hand_init = robot.data.body_pos_w[env_i, r_hand_idx, :] + fk_lag
    else:
        r_sh_w = root_pos_init.clone()
        l_sh_w = root_pos_init.clone()
        r_hand_init = root_pos_init.clone()

    if d_xy_init > TOY_TOUCH_RADIUS:
        _, d_far = _compute_nearest_block_surface_point_and_dist(r_hand_init, target_pos_w, block_quat_w)
        env.agent_hand_contact_verified[env_i] = False
        env.agent_active_hand_pos_w[env_i] = r_hand_init
        env.agent_hand_to_block_dist_m[env_i] = d_far
        return False, r_hand_init, d_far

    surf_target_w, _ = _compute_nearest_block_surface_point_and_dist(r_sh_w, target_pos_w, block_quat_w)
    reach_vec_w = surf_target_w - r_sh_w
    reach_vec_b = _rotate_vec_by_quat_inverse(root_quat_init.unsqueeze(0), reach_vec_w.unsqueeze(0))[0]
    raw_reach_d = float(torch.linalg.norm(reach_vec_b).item())

    l1_upper = 0.30
    l2_forearm = 0.32
    max_reach = l1_upper + l2_forearm - 0.005
    r_eff = min(max(raw_reach_d, 0.12), max_reach)

    x_b = float(reach_vec_b[0].item())
    y_b = float(reach_vec_b[1].item())
    z_b = float(reach_vec_b[2].item())
    horiz_b = max(math.sqrt(x_b * x_b + y_b * y_b), 1e-4)

    cos_elbow = max(-0.98, min(0.98, (r_eff * r_eff - l1_upper * l1_upper - l2_forearm * l2_forearm) / (2.0 * l1_upper * l2_forearm)))
    elbow_flex = math.acos(cos_elbow)
    alpha_elev = math.atan2(-z_b, horiz_b)
    cos_beta = max(-0.98, min(0.98, (l1_upper * l1_upper + r_eff * r_eff - l2_forearm * l2_forearm) / (2.0 * l1_upper * r_eff)))
    beta_sh = math.acos(cos_beta)

    sh_pitch = max(-1.45, min(1.45, alpha_elev - beta_sh))
    sh_abd = max(-0.95, min(0.95, math.atan2(y_b, max(x_b, 1e-4))))
    waist_pitch = max(-0.35, min(0.55, -z_b * 0.42))
    pelvis_yaw = max(-0.45, min(0.45, sh_abd * 0.35))
    elbow_clamped = max(-0.85, min(1.45, elbow_flex))

    reach_j = env.agent_reach_joint_targets_21d[env_i]
    reach_j.zero_()
    reach_j[0] = 0.0
    reach_j[1] = float(waist_pitch)
    reach_j[2] = float(sh_pitch)
    reach_j[3] = float(sh_abd)
    reach_j[6] = float(pelvis_yaw)
    reach_j[7] = float(elbow_clamped)
    if bimanual:
        reach_j[4] = float(sh_pitch)
        reach_j[5] = float(-sh_abd)
        reach_j[8] = float(elbow_clamped)
    else:
        reach_j[4] = -0.20
        reach_j[5] = 0.15
        reach_j[8] = 0.35

    env.agent_reach_active_timer[env_i] = 8
    env.upper_limb_reach_movements_total = int(getattr(env, "upper_limb_reach_movements_total", 0)) + 1

    step_spinal_cpg_and_bipedal_locomotion(env)
    if hasattr(env, "sim") and hasattr(env, "scene") and not bool(getattr(env, "_in_pre_physics_puzzle_step", False)):
        phys_dt = float(getattr(env, "physics_dt", 1.0 / 120.0))
        for _ in range(8):
            step_spinal_cpg_and_bipedal_locomotion(env)
            env.scene.write_data_to_sim()
            env.sim.step(render=False)
            env.scene.update(phys_dt)

    if hasattr(robot.data, "body_pos_w") and robot.data.body_pos_w.shape[1] > max(r_hand_idx, l_hand_idx):
        r_hand_w = robot.data.body_pos_w[env_i, r_hand_idx, :].clone()
        l_hand_w = robot.data.body_pos_w[env_i, l_hand_idx, :].clone()
    else:
        r_hand_w = robot.data.root_pos_w[env_i].clone()
        l_hand_w = robot.data.root_pos_w[env_i].clone()

    hand_pos_w = 0.5 * (r_hand_w + l_hand_w) if bimanual else r_hand_w
    _, d_r_surf = _compute_nearest_block_surface_point_and_dist(r_hand_w, target_pos_w, block_quat_w)
    _, d_l_surf = _compute_nearest_block_surface_point_and_dist(l_hand_w, target_pos_w, block_quat_w)
    _, d_hand_surf = _compute_nearest_block_surface_point_and_dist(hand_pos_w, target_pos_w, block_quat_w)
    d_eff_surf = min(d_r_surf, d_hand_surf) if not bimanual else min(max(d_r_surf, d_l_surf), d_hand_surf)

    upright_cos = float(-robot.data.projected_gravity_b[env_i, 2].item())
    contact_ok = bool(d_eff_surf <= 0.24 and upright_cos > 0.35)
    env.agent_hand_contact_verified[env_i] = contact_ok
    env.agent_active_hand_pos_w[env_i] = hand_pos_w
    env.agent_hand_to_block_dist_m[env_i] = float(d_eff_surf)
    if contact_ok:
        env.physical_hand_contacts_total = int(getattr(env, "physical_hand_contacts_total", 0)) + 1
    return contact_ok, hand_pos_w, float(d_eff_surf)


def execute_single_env_toy_manipulation(
    env,
    env_i: int,
    t_idx: int,
    motor_u_3d: torch.Tensor,
    vocal_2d: torch.Tensor,
) -> tuple[bool, str, str, str, str, int]:
    ensure_sensorimotor_puzzle_buffers(env)
    _sync_toy_yaw_from_sim(env, sync_slots=False)
    env.toy_slot_orient[env_i, t_idx] = (
        torch.round(env.toy_slot_yaw_rad[env_i, t_idx] / (math.pi * 0.5)).to(dtype=torch.long) % 4
    )
    device = env.agent_body_logits.device
    t_kind = int(env.toy_type[t_idx].item()) % 4

    best_triple, gaze_op, wm_op, finger_op, gaze_idx, _, _ = decode_sensorimotor_triple_for_env(
        env, env_i, t_kind, motor_u_3d, vocal_2d
    )

    env.agent_gaze_focus[env_i] = int(gaze_idx + 1)
    env.agent_wm_orient[env_i] = -1
    env.agent_wm_yaw_rad[env_i] = -10.0
    env.agent_wm_key_color[env_i] = -1
    env.agent_wm_val_color[env_i] = -1
    env.agent_wm_count[env_i] = 0

    robot = env.scene["robot"]
    r_hand_idx, _, _, _ = _resolve_humanoid_limb_indices(robot)
    block_obj = env.scene[TOY_ENTITY_BLOCK_KEYS[t_idx]]
    slot_obj = env.scene[TOY_ENTITY_SLOT_KEYS[t_idx]]
    env_id_tensor = torch.tensor([env_i], dtype=torch.long, device=device)

    block_pos_init = block_obj.data.root_pos_w[env_i].clone()
    block_quat_init = block_obj.data.root_quat_w[env_i].clone()
    slot_pos_w = slot_obj.data.root_pos_w[env_i]

    ang_err_prev = float(
        torch.abs(_wrap_angle_rad(env.toy_block_yaw_rad[env_i, t_idx] - env.toy_slot_yaw_rad[env_i, t_idx])).item()
    )
    prev_orient_ok = (ang_err_prev < TOY_YAW_TOLERANCE_RAD) and (
        int(env.toy_block_orient[env_i, t_idx].item()) == int(env.toy_slot_orient[env_i, t_idx].item())
    )
    prev_color_ok = int(env.toy_block_color[env_i, t_idx].item()) == int(env.toy_sample_val_color[env_i, t_idx].item())
    prev_group_p = int(env.toy_group_progress[env_i, t_idx].item())
    prev_step_c = int(env.toy_step_counter[env_i, t_idx].item())

    hand_contact_ok, _, _ = _drive_upper_limb_reaching_and_verify_hand_contact(
        env,
        env_i,
        block_pos_init,
        bimanual=(t_kind == 2),
        slot_pos_w=slot_pos_w,
        block_quat_w=block_quat_init,
    )

    if gaze_op == "gaze_salient_target":
        foveal_ok = _orient_eyes_and_verify_binocular_foveal_hit(env, env_i, slot_pos_w, target_radius=1.45)
        if wm_op == "wm_store_attr" and foveal_ok:
            s_quat = slot_obj.data.root_quat_w[env_id_tensor]
            obs_yaw = float(_quat_wxyz_to_yaw_rad(s_quat)[0].item())
            env.agent_wm_yaw_rad[env_i] = torch.tensor(obs_yaw, dtype=torch.float32, device=device)
            env.agent_wm_orient[env_i] = (
                torch.round(torch.tensor(obs_yaw, device=device) / (math.pi * 0.5)).to(dtype=torch.long) % 4
            )
    elif gaze_op == "gaze_corner_sample":
        foveal_ok = _orient_eyes_and_verify_binocular_foveal_hit(env, env_i, block_pos_init, target_radius=1.35)
        if wm_op == "wm_store_sample_pair" and foveal_ok:
            env.agent_wm_key_color[env_i] = env.toy_block_color[env_i, t_idx]
            env.agent_wm_val_color[env_i] = (env.toy_block_color[env_i, t_idx] % 9) + 1
    elif gaze_op == "gaze_next_matching_attr":
        foveal_ok = _orient_eyes_and_verify_binocular_foveal_hit(env, env_i, block_pos_init, target_radius=1.35)
        if wm_op == "wm_store_attr" and foveal_ok:
            env.agent_wm_key_color[env_i] = env.toy_block_color[env_i, t_idx]
    elif gaze_op == "gaze_count_objects":
        foveal_ok = _orient_eyes_and_verify_binocular_foveal_hit(env, env_i, slot_pos_w, target_radius=1.45)
        if wm_op == "wm_store_count" and foveal_ok:
            env.agent_wm_count[env_i] = torch.clamp(env.agent_wm_count[env_i] + 1, max=4)

    grip_effort = float(torch.sigmoid(torch.max(env.agent_body_logits[env_i, 7:11])).item())
    motor_mag = float(torch.linalg.norm(motor_u_3d[env_i]).item())
    arm_torque_mag = float(torch.linalg.norm(env.agent_cerebellar_joint_torques_21d[env_i, :9]).item()) / 1.8
    active_drive = grip_effort * max(min(motor_mag, 1.5), 0.90 * min(arm_torque_mag, 1.25))
    applied_force_n = (38.0 * active_drive) if hand_contact_ok else 0.0
    coulomb_friction_n = 0.35 * 2.0 * 9.81
    net_tangential_force_n = max(0.0, applied_force_n - coulomb_friction_n)
    env.agent_last_applied_contact_force_n[env_i] = float(applied_force_n)

    rotate_to_wm_yaw = False
    target_yaw_rad = env.toy_block_yaw_rad[env_i, t_idx]
    push_gain = 0.0

    if hand_contact_ok and net_tangential_force_n > 0.5:
        if finger_op == "finger_step_until_contact":
            if float(env.agent_wm_yaw_rad[env_i].item()) > -5.0:
                rotate_to_wm_yaw = True
                target_yaw_rad = env.agent_wm_yaw_rad[env_i]
                push_gain = 1.0
        elif finger_op == "finger_paint_from_wm":
            if (
                int(env.agent_wm_key_color[env_i].item()) == int(env.toy_block_color[env_i, t_idx].item())
                and int(env.agent_wm_val_color[env_i].item()) > 0
            ):
                env.toy_block_color[env_i, t_idx] = env.agent_wm_val_color[env_i]
                env.toy_block_rgb[env_i, t_idx] = _lookup_palette_rgb(env.toy_block_color[env_i, t_idx])
                _sync_toy_block_usd_color(env, env_i, t_idx, env.toy_block_rgb[env_i, t_idx])
                push_gain = 1.0
        elif finger_op == "finger_apply_to_gaze":
            if (
                int(env.agent_wm_key_color[env_i].item()) == int(env.toy_sample_key_color[env_i, t_idx].item())
                and int(env.agent_wm_val_color[env_i].item()) == -1
                and int(env.agent_wm_orient[env_i].item()) == -1
            ):
                max_g = int(env.toy_group_size[env_i, t_idx].item())
                env.toy_group_progress[env_i, t_idx] = torch.clamp(
                    env.toy_group_progress[env_i, t_idx] + 1, max=max_g
                )
                push_gain = 0.75
        elif finger_op == "finger_step_n_from_wm":
            if int(env.agent_wm_count[env_i].item()) == int(env.toy_target_count[env_i, t_idx].item()):
                step_n = int(env.agent_wm_count[env_i].item())
                target_n = int(env.toy_target_count[env_i, t_idx].item())
                for _ in range(step_n):
                    env.toy_step_counter[env_i, t_idx] = torch.clamp(
                        env.toy_step_counter[env_i, t_idx] + 1, max=target_n
                    )
                push_gain = 1.0

        robot_vel_xy = robot.data.root_lin_vel_w[env_i, :2]
        push_norm = float(torch.linalg.norm(robot_vel_xy).item()) + 1e-5
        push_dir_xy = robot_vel_xy / push_norm
        if t_kind == 2:
            if abs(float(push_dir_xy[0].item())) >= abs(float(push_dir_xy[1].item())):
                push_dir_xy[1] = 0.0
            else:
                push_dir_xy[0] = 0.0
        push_dir_norm = float(torch.linalg.norm(push_dir_xy).item()) + 1e-5
        push_dir_xy = push_dir_xy / push_dir_norm

        force_mag = net_tangential_force_n * (1.2 if push_gain > 0.0 else 0.5)
        f_world = torch.stack(
            [
                push_dir_xy[0] * force_mag,
                push_dir_xy[1] * force_mag,
                torch.tensor(0.0, dtype=torch.float32, device=device),
            ],
            dim=0,
        )

        can_substep = hasattr(env, "sim") and hasattr(env, "scene")
        n_substeps = 32 if can_substep else 1
        phys_dt = float(getattr(env, "physics_dt", 1.0 / 120.0))

        for _ in range(n_substeps):
            b_quat = block_obj.data.root_quat_w[env_id_tensor]
            cur_yaw = _quat_wxyz_to_yaw_rad(b_quat)[0]
            ang_w = block_obj.data.root_ang_vel_w[env_i]
            cur_wx = float(ang_w[0].item())
            cur_wy = float(ang_w[1].item())
            cur_wz = float(ang_w[2].item())
            if rotate_to_wm_yaw:
                yaw_diff = float(_wrap_angle_rad(target_yaw_rad - cur_yaw).item())
                tau_z = max(-180.0, min(180.0, 220.0 * yaw_diff - 24.0 * cur_wz))
            else:
                tau_z = max(-60.0, min(60.0, -12.0 * cur_wz))
            tau_x = max(-80.0, min(80.0, -18.0 * cur_wx))
            tau_y = max(-80.0, min(80.0, -18.0 * cur_wy))

            f_b = _rotate_vec_by_quat_inverse(b_quat, f_world.view(1, 3)).view(1, 1, 3)
            tau_w = torch.tensor([[float(tau_x), float(tau_y), float(tau_z)]], dtype=torch.float32, device=device)
            tau_b = _rotate_vec_by_quat_inverse(b_quat, tau_w).view(1, 1, 3)
            block_obj.set_external_force_and_torque(f_b, tau_b, env_ids=env_id_tensor)

            if hasattr(robot.data, "body_quat_w") and robot.data.body_quat_w.shape[1] > r_hand_idx:
                h_quat = robot.data.body_quat_w[env_id_tensor, r_hand_idx, :]
            else:
                h_quat = robot.data.root_quat_w[env_id_tensor]
            react_f_b = _rotate_vec_by_quat_inverse(h_quat, (-0.18 * f_world).view(1, 3)).view(1, 1, 3)
            react_tau_b = _rotate_vec_by_quat_inverse(h_quat, (-0.08 * tau_w)).view(1, 1, 3)
            robot.set_external_force_and_torque(react_f_b, react_tau_b, body_ids=[r_hand_idx], env_ids=env_id_tensor)

            if can_substep:
                step_spinal_cpg_and_bipedal_locomotion(env)
                env.scene.write_data_to_sim()
                env.sim.step(render=False)
                env.scene.update(phys_dt)

        if can_substep:
            zero_f = torch.zeros((1, 1, 3), dtype=torch.float32, device=device)
            block_obj.set_external_force_and_torque(zero_f, zero_f, env_ids=env_id_tensor)
            robot.set_external_force_and_torque(zero_f, zero_f, body_ids=[r_hand_idx], env_ids=env_id_tensor)

    _sync_toy_yaw_from_sim(env, sync_slots=False)
    ang_err_now = float(
        torch.abs(_wrap_angle_rad(env.toy_block_yaw_rad[env_i, t_idx] - env.toy_slot_yaw_rad[env_i, t_idx])).item()
    )
    if hand_contact_ok and rotate_to_wm_yaw and ang_err_now < TOY_YAW_TOLERANCE_RAD and applied_force_n >= 10.0:
        env.toy_insertion_depth[env_i, t_idx] = 0.45

    now_orient_ok = (ang_err_now < TOY_YAW_TOLERANCE_RAD) and (
        float(env.toy_insertion_depth[env_i, t_idx].item()) >= 0.25
    )
    now_color_ok = int(env.toy_block_color[env_i, t_idx].item()) == int(env.toy_sample_val_color[env_i, t_idx].item())
    now_group_p = int(env.toy_group_progress[env_i, t_idx].item())
    now_step_c = int(env.toy_step_counter[env_i, t_idx].item())
    target_step_c = int(env.toy_target_count[env_i, t_idx].item())

    solved_micro = False
    if hand_contact_ok:
        if t_kind == 0:
            solved_micro = (not prev_orient_ok) and now_orient_ok
        elif t_kind == 1:
            solved_micro = (not prev_color_ok) and now_color_ok
        elif t_kind == 2:
            solved_micro = now_group_p > prev_group_p
        elif t_kind == 3:
            solved_micro = (prev_step_c != target_step_c) and (now_step_c == target_step_c)

    if solved_micro:
        env.agent_toy_habituation[env_i, t_kind] = torch.clamp(
            env.agent_toy_habituation[env_i, t_kind] * 0.85, min=0.0
        )
        env.agent_toy_habituation[env_i, t_kind, best_triple] = 0.0
        cur_gain = float(env.agent_cerebellar_gain[env_i, t_kind].item())
        env.agent_cerebellar_gain[env_i, t_kind] = min(1.0, cur_gain + 0.15)
        env.cerebellar_corrections_total = int(env.cerebellar_corrections_total) + 1
    elif hand_contact_ok:
        env.agent_toy_habituation[env_i, t_kind, best_triple] = (
            torch.clamp(env.agent_toy_habituation[env_i, t_kind, best_triple], min=0.0) + 1.0
        )
        env.agent_cerebellar_gain[env_i, t_kind] *= 0.96

    chunk_sig = f"sensorimotor_chunk::{gaze_op}+{wm_op}+{finger_op}"
    return solved_micro, chunk_sig, gaze_op, wm_op, finger_op, best_triple


def pre_physics_step_sensorimotor_puzzle(env) -> None:
    ensure_homeostasis_curiosity_buffers(env)
    ensure_sensorimotor_puzzle_buffers(env)

    if torch.any(env.pending_toy_respawn_mask):
        pending_pairs = torch.nonzero(env.pending_toy_respawn_mask, as_tuple=False)
        for pair in pending_pairs.tolist():
            _respawn_single_toy_block(env, int(pair[0]), int(pair[1]))
        env.pending_toy_respawn_mask.zero_()

    _sync_toy_yaw_from_sim(env)

    robot = env.scene["robot"]
    root_pos_w = robot.data.root_pos_w
    num_envs = int(root_pos_w.shape[0])

    block_pos_list = [env.scene[k].data.root_pos_w for k in TOY_ENTITY_BLOCK_KEYS]
    block_pos_w = torch.stack(block_pos_list, dim=1)

    safety_gate = getattr(env, "agent_survival_safety_gate", None)
    if safety_gate is None or safety_gate.shape[0] != num_envs:
        safety_gate = compute_survival_safety_gate(
            env.agent_hp,
            env.agent_energy,
            env.agent_pain,
            env.agent_last_damage,
            env.agent_alive,
        )

    ag_toy_diff_xy = block_pos_w[:, :, :2] - root_pos_w[:, :2].unsqueeze(1)
    ag_toy_dist = torch.linalg.norm(ag_toy_diff_xy, dim=-1)
    min_t_dist, min_t_idx = torch.min(ag_toy_dist, dim=1)

    touching_toy = (min_t_dist < TOY_TOUCH_RADIUS) & (safety_gate > TOY_SAFETY_GATE_MIN) & env.agent_alive
    env._pre_physics_touching_mask = touching_toy.clone()
    env._pre_physics_safety_gate = safety_gate.clone()
    records = []

    free_gaze_yaw = 0.65 * torch.tanh(env.agent_body_logits[:, 0] - env.agent_body_logits[:, 1])
    free_gaze_pitch = 0.30 * torch.tanh(env.agent_body_logits[:, 2] - env.agent_body_logits[:, 3])
    non_touch = ~touching_toy
    env.agent_gaze_yaw_offset_rad[non_touch] = free_gaze_yaw[non_touch]
    env.agent_gaze_pitch_offset_rad[non_touch] = free_gaze_pitch[non_touch]

    if torch.any(touching_toy):
        motor_u_3d = _extract_locomotor_summary_3d(env)
        vocal_2d = env.agent_vocal_signals
        touch_envs = torch.where(touching_toy)[0]

        env._in_pre_physics_puzzle_step = True
        try:
            for env_i in touch_envs.tolist():
                t_idx = int(min_t_idx[env_i].item())
                t_kind = int(env.toy_type[t_idx].item()) % 4
                solved_micro, chunk_sig, gaze_op, wm_op, finger_op, _ = execute_single_env_toy_manipulation(
                    env, env_i, t_idx, motor_u_3d, vocal_2d
                )
                records.append((int(env_i), int(t_idx), int(t_kind), bool(solved_micro), chunk_sig, gaze_op, wm_op, finger_op))
        finally:
            env._in_pre_physics_puzzle_step = False

    env._pre_physics_puzzle_records = records
    env._pre_physics_puzzle_done_for_step = True
    if hasattr(env, "common_step_counter"):
        env._last_pre_physics_puzzle_step = int(env.common_step_counter)


def step_sensorimotor_puzzle_rules(env) -> torch.Tensor:
    ensure_homeostasis_curiosity_buffers(env)
    ensure_sensorimotor_puzzle_buffers(env)

    step_counter = int(getattr(env, "common_step_counter", -1))
    if step_counter >= 0 and env._last_puzzle_step_counter == step_counter:
        return env.last_toy_puzzle_reward

    if not bool(getattr(env, "_pre_physics_puzzle_done_for_step", False)):
        pre_physics_step_sensorimotor_puzzle(env)
    env._pre_physics_puzzle_done_for_step = False

    _sync_toy_yaw_from_sim(env)

    robot = env.scene["robot"]
    device = robot.data.root_pos_w.device
    num_envs = int(robot.data.root_pos_w.shape[0])

    safety_gate = getattr(env, "_pre_physics_safety_gate", None)
    if safety_gate is None or safety_gate.shape[0] != num_envs:
        safety_gate = compute_survival_safety_gate(
            env.agent_hp,
            env.agent_energy,
            env.agent_pain,
            env.agent_last_damage,
            env.agent_alive,
        )

    touching_toy = getattr(env, "_pre_physics_touching_mask", None)
    if touching_toy is None or touching_toy.shape[0] != num_envs:
        touching_toy = torch.zeros((num_envs,), dtype=torch.bool, device=device)

    specific_toy_bonus = torch.where(
        touching_toy,
        TOY_TOUCH_REWARD_COEFF * safety_gate,
        torch.zeros((num_envs,), dtype=torch.float32, device=device),
    )
    env.last_solved_toy_mask.zero_()

    records = getattr(env, "_pre_physics_puzzle_records", [])
    for env_i, t_idx, t_kind, solved_micro, chunk_sig, gaze_op, wm_op, finger_op in records:
        block_now_w = env.scene[TOY_ENTITY_BLOCK_KEYS[t_idx]].data.root_pos_w[env_i, :2]
        slot_now_w = env.scene[TOY_ENTITY_SLOT_KEYS[t_idx]].data.root_pos_w[env_i, :2]
        new_slot_d = float(torch.linalg.norm(slot_now_w - block_now_w).item())

        ang_err = float(
            torch.abs(_wrap_angle_rad(env.toy_block_yaw_rad[env_i, t_idx] - env.toy_slot_yaw_rad[env_i, t_idx])).item()
        )
        orient_ok = (t_kind != 0) or (
            (ang_err < TOY_YAW_TOLERANCE_RAD)
            and (int(env.toy_block_orient[env_i, t_idx].item()) == int(env.toy_slot_orient[env_i, t_idx].item()))
        )
        group_req = int(env.toy_group_size[env_i, t_idx].item())
        now_group_p = int(env.toy_group_progress[env_i, t_idx].item())
        group_complete = (t_kind != 2) or (now_group_p >= group_req)

        if t_kind == 2 and solved_micro and (now_group_p < group_req):
            specific_toy_bonus[env_i] = specific_toy_bonus[env_i] + TOY_GROUP_STEP_REWARD_COEFF * (
                0.4 + 0.6 * safety_gate[env_i]
            )

        if new_slot_d < TOY_SLOT_SOLVE_DISTANCE and orient_ok and group_complete and solved_micro:
            specific_toy_bonus[env_i] = specific_toy_bonus[env_i] + TOY_SOLVE_REWARD_COEFF * (
                0.4 + 0.6 * safety_gate[env_i]
            )
            env.last_solved_toy_mask[env_i, t_idx] = True
            env.pending_toy_respawn_mask[env_i, t_idx] = True
            env.toys_solved_total = int(env.toys_solved_total) + 1
            env.sensorimotor_chunk_hits_total = int(env.sensorimotor_chunk_hits_total) + 1

            rec = env.discovered_sensorimotor_chunks.get(
                chunk_sig,
                {
                    "chunk_sig": chunk_sig,
                    "toy_kind": int(t_kind),
                    "gaze_op": gaze_op,
                    "wm_op": wm_op,
                    "finger_op": finger_op,
                    "hits": 0,
                    "sample_key_color": int(env.toy_sample_key_color[env_i, t_idx].item()),
                    "sample_val_color": int(env.toy_sample_val_color[env_i, t_idx].item()),
                    "sample_count": int(env.toy_target_count[env_i, t_idx].item()),
                },
            )
            rec["hits"] = int(rec["hits"]) + 1
            env.discovered_sensorimotor_chunks[chunk_sig] = rec

    env.last_toy_puzzle_reward = specific_toy_bonus
    if step_counter >= 0:
        env._last_puzzle_step_counter = step_counter
    return specific_toy_bonus


def reward_toy_sensorimotor_puzzle(env) -> torch.Tensor:
    return step_sensorimotor_puzzle_rules(env)


def obs_nearest_toy_puzzle_attributes_6d(env) -> torch.Tensor:
    ensure_sensorimotor_puzzle_buffers(env)
    _sync_toy_yaw_from_sim(env)
    robot = env.scene["robot"]
    root_pos_w = robot.data.root_pos_w
    root_quat_w = robot.data.root_quat_w
    device = root_pos_w.device
    num_envs = int(root_pos_w.shape[0])

    block_pos_list = [env.scene[k].data.root_pos_w for k in TOY_ENTITY_BLOCK_KEYS]
    block_pos_w = torch.stack(block_pos_list, dim=1)
    to_toy_xy = block_pos_w[:, :, :2] - root_pos_w[:, :2].unsqueeze(1)
    ag_toy_dist = torch.linalg.norm(to_toy_xy, dim=-1)
    min_toy_d, min_t_idx = torch.min(ag_toy_dist, dim=1, keepdim=True)

    body_yaw = _quat_wxyz_to_yaw_rad(root_quat_w)
    gaze_yaw = body_yaw + env.agent_gaze_yaw_offset_rad.to(device=device, dtype=torch.float32)
    near_to_toy = torch.gather(to_toy_xy, 1, min_t_idx.unsqueeze(-1).expand(-1, 1, 2)).squeeze(1)
    toy_bearing = torch.atan2(near_to_toy[:, 1], near_to_toy[:, 0])
    rel_gaze_yaw = _wrap_angle_rad(toy_bearing - gaze_yaw).unsqueeze(-1)
    vis_gate = ((min_toy_d <= 18.0) & (torch.abs(rel_gaze_yaw) <= 1.05)).to(dtype=torch.float32)
    foveal_gate = ((min_toy_d <= 8.5) & (torch.abs(rel_gaze_yaw) <= 0.65)).to(dtype=torch.float32)

    kind_norm = (min_t_idx.to(dtype=torch.float32) / 3.0) * vis_gate
    key_raw = torch.gather(env.toy_sample_key_color, 1, min_t_idx).to(dtype=torch.float32)
    val_raw = torch.gather(env.toy_sample_val_color, 1, min_t_idx).to(dtype=torch.float32)
    blk_raw = torch.gather(env.toy_block_color, 1, min_t_idx).to(dtype=torch.float32)
    key_norm = ((key_raw + 0.5) / 10.0) * foveal_gate
    val_norm = ((val_raw + 0.5) / 10.0) * foveal_gate
    blk_col_norm = ((blk_raw + 0.5) / 10.0) * vis_gate
    cnt_norm = (torch.gather(env.toy_target_count, 1, min_t_idx).to(dtype=torch.float32) / 6.0) * foveal_gate
    near_b_yaw = torch.gather(env.toy_block_yaw_rad, 1, min_t_idx)
    near_s_yaw = torch.gather(env.toy_slot_yaw_rad, 1, min_t_idx)
    rel_yaw_norm = (_wrap_angle_rad(near_b_yaw - near_s_yaw) / math.pi) * vis_gate

    return torch.cat(
        [kind_norm, key_norm, val_norm, blk_col_norm, cnt_norm, rel_yaw_norm],
        dim=-1,
    ).reshape(num_envs, 6)


def obs_sensorimotor_puzzle_state_12d(env) -> torch.Tensor:
    ensure_sensorimotor_puzzle_buffers(env)
    _sync_toy_yaw_from_sim(env)
    robot = env.scene["robot"]
    device = robot.data.root_pos_w.device
    num_envs = int(robot.data.root_pos_w.shape[0])

    gaze_norm = env.agent_gaze_focus.to(device=device, dtype=torch.float32).unsqueeze(-1) / 4.0
    wm_orient_norm = torch.clamp(env.agent_wm_orient.to(device=device, dtype=torch.float32).unsqueeze(-1) / 3.0, -1.0, 1.0)
    wm_k_raw = env.agent_wm_key_color.to(device=device, dtype=torch.float32).unsqueeze(-1)
    wm_v_raw = env.agent_wm_val_color.to(device=device, dtype=torch.float32).unsqueeze(-1)
    wm_key_norm = torch.clamp((wm_k_raw + 0.5) / 10.0, 0.0, 1.0)
    wm_val_norm = torch.clamp((wm_v_raw + 0.5) / 10.0, 0.0, 1.0)
    wm_cnt_norm = torch.clamp(env.agent_wm_count.to(device=device, dtype=torch.float32).unsqueeze(-1) / 6.0, 0.0, 1.0)

    yaw_err = torch.abs(_wrap_angle_rad(env.toy_block_yaw_rad[:, 0:1] - env.toy_slot_yaw_rad[:, 0:1])) / math.pi
    color_match = (env.toy_block_color[:, 1:2] == env.toy_sample_val_color[:, 1:2]).to(dtype=torch.float32)
    group_prog = torch.clamp(env.toy_group_progress[:, 2:3].to(dtype=torch.float32) / 4.0, 0.0, 1.0)
    step_prog = torch.clamp(
        env.toy_step_counter[:, 3:4].to(dtype=torch.float32)
        / torch.clamp(env.toy_target_count[:, 3:4].to(dtype=torch.float32), min=1.0),
        0.0,
        1.0,
    )
    vocal_2d = torch.clamp(env.agent_vocal_signals.to(device=device, dtype=torch.float32), -1.0, 1.0)
    cerebellar_mean = torch.mean(env.agent_cerebellar_gain.to(device=device, dtype=torch.float32), dim=-1, keepdim=True)

    return torch.cat(
        [
            gaze_norm,
            wm_orient_norm,
            wm_key_norm,
            wm_val_norm,
            wm_cnt_norm,
            yaw_err,
            color_match,
            group_prog,
            step_prog,
            vocal_2d,
            cerebellar_mean,
        ],
        dim=-1,
    ).reshape(num_envs, 12)


class Sensorimotor11DAction(ActionTerm):
    cfg: Sensorimotor11DActionCfg

    def __init__(self, cfg: Sensorimotor11DActionCfg, env):
        super().__init__(cfg, env)
        self._raw_actions = torch.zeros((self.num_envs, 11), dtype=torch.float32, device=self.device)
        self._processed_actions = torch.zeros((self.num_envs, 11), dtype=torch.float32, device=self.device)
        self._pending_pre_physics_step = False
        ensure_sensorimotor_puzzle_buffers(self._env)

    @property
    def action_dim(self) -> int:
        return 11

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions.to(device=self.device, dtype=torch.float32)
        self._processed_actions[:] = torch.clamp(self._raw_actions, -5.0, 5.0)
        ensure_sensorimotor_puzzle_buffers(self._env)
        self._env.agent_body_logits[:] = self._processed_actions
        self._pending_pre_physics_step = True

    def apply_actions(self) -> None:
        if self._pending_pre_physics_step:
            self._pending_pre_physics_step = False
            pre_physics_step_world_dynamics(self._env)
            pre_physics_step_sensorimotor_puzzle(self._env)
        step_spinal_cpg_and_bipedal_locomotion(self._env)


@configclass
class Sensorimotor11DActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = Sensorimotor11DAction
    asset_name: str = "robot"


class VocalSignal2DAction(ActionTerm):
    cfg: VocalSignal2DActionCfg

    def __init__(self, cfg: VocalSignal2DActionCfg, env):
        super().__init__(cfg, env)
        self._raw_actions = torch.zeros((self.num_envs, 2), dtype=torch.float32, device=self.device)
        self._processed_actions = torch.zeros((self.num_envs, 2), dtype=torch.float32, device=self.device)
        ensure_sensorimotor_puzzle_buffers(self._env)

    @property
    def action_dim(self) -> int:
        return 2

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions.to(device=self.device, dtype=torch.float32)
        self._processed_actions[:] = torch.tanh(self._raw_actions)
        ensure_sensorimotor_puzzle_buffers(self._env)
        self._env.agent_vocal_signals[:] = self._processed_actions

    def apply_actions(self) -> None:
        if hasattr(self._env, "agent_vocal_signals"):
            self._env.agent_vocal_signals.copy_(self._processed_actions)


@configclass
class VocalSignal2DActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = VocalSignal2DAction
    asset_name: str = "robot"
