import ctypes
import json
import math
import os
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from unified_biological_brain import UnifiedBiologicalBrainSystem
from isaac_warp_3d_engine import get_global_isaac_warp_engine


def elevate_process_priority_high():
    if os.name == "nt":
        try:
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.GetCurrentProcess()
            if not kernel32.SetPriorityClass(handle, 0x00000080):
                kernel32.SetPriorityClass(handle, 0x00008000)
        except Exception:
            pass


elevate_process_priority_high()

ROOT_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class IsaacStyle3DPhysicsArena:
    def __init__(self, pop_size=64, arena_size=100.0, max_height=30.0, n_food=28, n_predators=8, n_obstacles=10, seed=42):
        self.pop_size = pop_size
        self.L = float(arena_size)
        self.Hz = float(max_height)
        self.n_food = n_food
        self.n_predators = n_predators
        self.n_obstacles = n_obstacles
        self.rng = np.random.default_rng(seed)
        self.dt = 0.20
        self.gravity = 9.8
        self.friction = 0.86
        self.max_speed = 4.2
        self.max_omega = 1.4
        self.ray_Range = 28.0

        az_deg = [-50.0, -25.0, 0.0, 25.0, 50.0, -35.0, 0.0, 35.0]
        el_deg = [0.0, 0.0, 0.0, 0.0, 0.0, 22.0, 28.0, -18.0]
        self.ray_az = torch.tensor([math.radians(a) for a in az_deg], dtype=torch.float32, device=DEVICE)
        self.ray_el = torch.tensor([math.radians(e) for e in el_deg], dtype=torch.float32, device=DEVICE)
        self.n_rays = len(az_deg)
        self.procedural_cycle_count = 0
        self.hazard_dna = {}
        self.discovered_sensorimotor_chunks = {}
        self.sensorimotor_chunk_hits_total = 0
        self.cerebellar_corrections_total = 0
        self.sample_procedural_hazard_dna(0)
        self.reset_world(unpredictable_mode="standard")

    def sample_procedural_hazard_dna(self, cycle_idx=0):
        self.procedural_cycle_count += 1
        dna = {
            "cycle_idx": int(cycle_idx),
            "terrain_amp_1": float(self.rng.uniform(2.2, 5.2)),
            "terrain_amp_2": float(self.rng.uniform(2.5, 5.6)),
            "terrain_amp_wave": float(self.rng.uniform(1.8, 4.2)),
            "terrain_freq_x": float(self.rng.uniform(0.05, 0.14)),
            "terrain_freq_y": float(self.rng.uniform(0.05, 0.14)),
            "terrain_phase_x": float(self.rng.uniform(0.0, 2.0 * math.pi)),
            "terrain_phase_y": float(self.rng.uniform(0.0, 2.0 * math.pi)),
            "peak1_x": float(self.rng.uniform(22.0, 45.0)),
            "peak1_y": float(self.rng.uniform(22.0, 45.0)),
            "peak2_x": float(self.rng.uniform(55.0, 80.0)),
            "peak2_y": float(self.rng.uniform(55.0, 80.0)),
            "meteor_fall_speed": float(self.rng.uniform(4.0, 6.2)),
            "meteor_spiral_omega": float(self.rng.uniform(0.06, 0.22)),
            "meteor_lateral_drift": float(self.rng.uniform(0.8, 2.6)),
            "meteor_blast_radius": float(self.rng.uniform(5.2, 7.2)),
            "pred_speed": float(self.rng.uniform(2.35, 3.15)),
            "pred_flank_angle": float(self.rng.uniform(-0.65, 0.65)),
            "stealth_period": int(self.rng.integers(28, 52)),
            "stealth_duty": int(self.rng.integers(14, 26)),
            "sinkhole_period": int(self.rng.integers(22, 45)),
            "sinkhole_radius": float(self.rng.uniform(4.2, 6.0)),
            "polarity_shift_period": int(self.rng.integers(32, 64)),
            "panel_gravity_mod": float(self.rng.uniform(0.25, 0.65)),
            "vortex_pull": float(self.rng.uniform(1.3, 2.5)),
            "vortex_swirl": float(self.rng.uniform(2.4, 4.0)),
            "panel_rules": [float(self.rng.choice([-1.0, 1.0])) for _ in range(6)],
        }
        self.hazard_dna = dna
        return dna

    def terrain_height(self, x, y):
        sx = x - self.terrain_shift_x
        if getattr(self, "unpredictable_mode", "standard") == "infinite_procedural_chaos" and self.hazard_dna:
            d = self.hazard_dna
            h1 = d["terrain_amp_1"] * torch.exp(-((sx - d["peak1_x"]) ** 2 + (y - d["peak1_y"]) ** 2) / 320.0)
            h2 = d["terrain_amp_2"] * torch.exp(-((sx - d["peak2_x"]) ** 2 + (y - d["peak2_y"]) ** 2) / 380.0)
            h3 = d["terrain_amp_wave"] * torch.sin(sx * d["terrain_freq_x"] + d["terrain_phase_x"]) * torch.cos(y * d["terrain_freq_y"] + d["terrain_phase_y"])
            return torch.clamp(h1 + h2 + h3, min=0.0, max=12.0)
        h1 = 3.5 * torch.exp(-((sx - 30.0) ** 2 + (y - 30.0) ** 2) / 320.0)
        h2 = 4.2 * torch.exp(-((sx - 70.0) ** 2 + (y - 68.0) ** 2) / 380.0)
        h3 = 2.8 * torch.sin(sx * 0.08) * torch.cos(y * 0.08)
        return torch.clamp(h1 + h2 + h3, min=0.0, max=12.0)

    def reset_world(self, unpredictable_mode="standard"):
        self.unpredictable_mode = unpredictable_mode
        if unpredictable_mode == "infinite_procedural_chaos":
            self.sample_procedural_hazard_dna(self.procedural_cycle_count)
        self.step_idx = 0
        self.food_polarity = torch.full((self.n_food,), 1.0, dtype=torch.float32, device=DEVICE)
        self.food_polarity[self.n_food // 2 :] = -1.0
        self.toxic_polarity = 1.0 if unpredictable_mode != "standard" else -1.0
        self.polarity_shift_period = (
            int(self.hazard_dna.get("polarity_shift_period", 55))
            if unpredictable_mode == "infinite_procedural_chaos"
            else 55
        )
        self.ambush_stealth_phase = False
        self.terrain_shift_x = (
            float(self.rng.uniform(-20.0, 20.0))
            if unpredictable_mode == "infinite_procedural_chaos"
            else (18.0 if unpredictable_mode in ("abyss_collapse", "chaos_all") else 0.0)
        )
        self.yaw_bias_drift = (
            float(self.rng.uniform(-0.60, 0.60))
            if unpredictable_mode == "infinite_procedural_chaos"
            else (
                float(self.rng.choice([-0.55, 0.55]))
                if unpredictable_mode in ("meteor_3d", "vortex_storm", "chaos_all")
                else 0.0
            )
        )

        pos_xy = self.rng.uniform(12.0, self.L - 12.0, (self.pop_size, 2)).astype(np.float32)
        self.pos = torch.zeros((self.pop_size, 3), dtype=torch.float32, device=DEVICE)
        self.pos[:, 0] = torch.tensor(pos_xy[:, 0], device=DEVICE)
        self.pos[:, 1] = torch.tensor(pos_xy[:, 1], device=DEVICE)
        self.pos[:, 2] = self.terrain_height(self.pos[:, 0], self.pos[:, 1])

        self.vel = torch.zeros((self.pop_size, 3), dtype=torch.float32, device=DEVICE)
        self.yaw = torch.tensor(self.rng.uniform(-math.pi, math.pi, (self.pop_size,)).astype(np.float32), device=DEVICE)
        self.signals = torch.zeros((self.pop_size, 2), dtype=torch.float32, device=DEVICE)

        self.hp = torch.ones((self.pop_size,), dtype=torch.float32, device=DEVICE)
        self.energy = torch.full((self.pop_size,), 0.56, dtype=torch.float32, device=DEVICE)
        self.pain = torch.zeros((self.pop_size,), dtype=torch.float32, device=DEVICE)
        self.surprise = torch.zeros((self.pop_size,), dtype=torch.float32, device=DEVICE)
        self.alive = torch.ones((self.pop_size,), dtype=torch.bool, device=DEVICE)

        f_xy = self.rng.uniform(10.0, self.L - 10.0, (self.n_food, 2)).astype(np.float32)
        self.food_pos = torch.zeros((self.n_food, 3), dtype=torch.float32, device=DEVICE)
        self.food_pos[:, 0] = torch.tensor(f_xy[:, 0], device=DEVICE)
        self.food_pos[:, 1] = torch.tensor(f_xy[:, 1], device=DEVICE)
        self.food_pos[:, 2] = self.terrain_height(self.food_pos[:, 0], self.food_pos[:, 1]) + 0.5
        self.food_is_coop = torch.zeros((self.n_food,), dtype=torch.bool, device=DEVICE)
        self.food_is_coop[: self.n_food // 4] = True
        self.food_is_mimic = torch.zeros((self.n_food,), dtype=torch.bool, device=DEVICE)
        if unpredictable_mode in ("mimic_trap", "chaos_all", "infinite_procedural_chaos"):
            self.food_is_mimic[self.n_food // 4 : self.n_food // 2] = True

        p_xy = self.rng.uniform(8.0, self.L - 8.0, (self.n_predators, 2)).astype(np.float32)
        self.pred_pos = torch.zeros((self.n_predators, 3), dtype=torch.float32, device=DEVICE)
        self.pred_pos[:, 0] = torch.tensor(p_xy[:, 0], device=DEVICE)
        self.pred_pos[:, 1] = torch.tensor(p_xy[:, 1], device=DEVICE)
        self.pred_pos[:, 2] = self.terrain_height(self.pred_pos[:, 0], self.pred_pos[:, 1]) + 0.5
        p_ang = self.rng.uniform(-math.pi, math.pi, (self.n_predators,)).astype(np.float32)
        self.pred_vel = torch.zeros((self.n_predators, 3), dtype=torch.float32, device=DEVICE)
        self.pred_vel[:, 0] = torch.tensor(np.cos(p_ang) * 2.4, device=DEVICE)
        self.pred_vel[:, 1] = torch.tensor(np.sin(p_ang) * 2.4, device=DEVICE)

        o_xy = self.rng.uniform(18.0, self.L - 18.0, (self.n_obstacles, 2)).astype(np.float32)
        self.obs_pos = torch.zeros((self.n_obstacles, 3), dtype=torch.float32, device=DEVICE)
        self.obs_pos[:, 0] = torch.tensor(o_xy[:, 0], device=DEVICE)
        self.obs_pos[:, 1] = torch.tensor(o_xy[:, 1], device=DEVICE)
        self.obs_pos[:, 2] = self.terrain_height(self.obs_pos[:, 0], self.obs_pos[:, 1]) + 2.0
        self.obs_radius = torch.full((self.n_obstacles,), 3.8, dtype=torch.float32, device=DEVICE)

        self.meteors = torch.zeros((6, 3), dtype=torch.float32, device=DEVICE)
        self.meteors[:, 0] = torch.tensor(self.rng.uniform(10.0, 46.0, (6,)).astype(np.float32), device=DEVICE)
        self.meteors[:, 1] = torch.tensor(self.rng.uniform(10.0, 46.0, (6,)).astype(np.float32), device=DEVICE)
        self.meteors[:, 2] = torch.tensor(self.rng.uniform(14.0, 28.0, (6,)).astype(np.float32), device=DEVICE)

        ab_xy = np.zeros((4, 2), dtype=np.float32)
        ab_xy[:, 0] = self.rng.uniform(54.0, self.L - 12.0, (4,))
        ab_xy[:, 1] = self.rng.uniform(12.0, 46.0, (4,))
        self.abyss_centers = torch.tensor(ab_xy, device=DEVICE)

        vx_xy = np.zeros((3, 2), dtype=np.float32)
        vx_xy[:, 0] = self.rng.uniform(12.0, 46.0, (3,))
        vx_xy[:, 1] = self.rng.uniform(54.0, self.L - 12.0, (3,))
        self.vortex_centers = torch.tensor(vx_xy, device=DEVICE)
        self.vortex_vel = torch.tensor(
            self.rng.normal(0.0, 1.2, (3, 2)).astype(np.float32), device=DEVICE
        )
        self.panel_centers = torch.tensor(
            self.rng.uniform(18.0, self.L - 18.0, (6, 2)).astype(np.float32), device=DEVICE
        )
        panel_r = (
            self.hazard_dna.get("panel_rules", [1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
            if unpredictable_mode == "infinite_procedural_chaos"
            else [1.0, -1.0, 1.0, -1.0, 1.0, -1.0]
        )
        self.panel_color_rule = torch.tensor(panel_r, dtype=torch.float32, device=DEVICE)
        self.red_leader_traj = torch.tensor([1.0, 0.0], dtype=torch.float32, device=DEVICE)

        self.quadrant_resource_level = torch.ones((4,), dtype=torch.float32, device=DEVICE)
        self.agent_visited_quadrants = torch.zeros((self.pop_size, 4), dtype=torch.float32, device=DEVICE)
        init_q = (self.pos[:, 0] >= 50.0).long() + 2 * (self.pos[:, 1] >= 50.0).long()
        self.agent_visited_quadrants[torch.arange(self.pop_size, device=DEVICE), init_q] = 1.0

        self.n_toys = 8
        self.toy_type = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], dtype=torch.long, device=DEVICE)
        quad_bounds = [
            (14.0, 42.0, 14.0, 42.0),
            (58.0, 86.0, 14.0, 42.0),
            (14.0, 42.0, 58.0, 86.0),
            (58.0, 86.0, 58.0, 86.0),
        ]
        tb_pos = np.zeros((self.n_toys, 3), dtype=np.float32)
        ts_pos = np.zeros((self.n_toys, 3), dtype=np.float32)
        for t_idx in range(self.n_toys):
            q_id = t_idx % 4
            x_lo, x_hi, y_lo, y_hi = quad_bounds[q_id]
            sx = float(self.rng.uniform(x_lo + 4.0, x_hi - 4.0))
            sy = float(self.rng.uniform(y_lo + 4.0, y_hi - 4.0))
            ang = float(self.rng.uniform(-math.pi, math.pi))
            dist_t = float(self.rng.uniform(3.8, 5.8))
            bx = float(np.clip(sx + math.cos(ang) * dist_t, x_lo, x_hi))
            by = float(np.clip(sy + math.sin(ang) * dist_t, y_lo, y_hi))
            tb_pos[t_idx, 0] = bx
            tb_pos[t_idx, 1] = by
            ts_pos[t_idx, 0] = sx
            ts_pos[t_idx, 1] = sy
        self.toy_block_pos = torch.tensor(tb_pos, dtype=torch.float32, device=DEVICE)
        self.toy_block_pos[:, 2] = self.terrain_height(self.toy_block_pos[:, 0], self.toy_block_pos[:, 1]) + 0.6
        self.toy_slot_pos = torch.tensor(ts_pos, dtype=torch.float32, device=DEVICE)
        self.toy_slot_pos[:, 2] = self.terrain_height(self.toy_slot_pos[:, 0], self.toy_slot_pos[:, 1]) + 0.4
        self.toy_block_orient = torch.tensor(self.rng.integers(0, 4, (self.n_toys,)), dtype=torch.long, device=DEVICE)
        self.toy_slot_orient = torch.tensor(self.rng.integers(0, 4, (self.n_toys,)), dtype=torch.long, device=DEVICE)
        self.toy_sample_key_color = torch.tensor(self.rng.integers(1, 5, (self.n_toys,)), dtype=torch.long, device=DEVICE)
        self.toy_sample_val_color = torch.tensor(self.rng.integers(5, 10, (self.n_toys,)), dtype=torch.long, device=DEVICE)
        self.toy_block_color = self.toy_sample_key_color.clone()
        self.toy_target_count = torch.tensor(self.rng.integers(2, 6, (self.n_toys,)), dtype=torch.long, device=DEVICE)
        self.toy_step_counter = torch.zeros((self.n_toys,), dtype=torch.long, device=DEVICE)
        self.toy_group_size = torch.tensor(self.rng.integers(3, 6, (self.n_toys,)), dtype=torch.long, device=DEVICE)
        self.toy_group_progress = torch.zeros((self.n_toys,), dtype=torch.long, device=DEVICE)
        self.agent_gaze_focus = torch.zeros((self.pop_size,), dtype=torch.long, device=DEVICE)
        self.agent_wm_key_color = torch.zeros((self.pop_size,), dtype=torch.long, device=DEVICE)
        self.agent_wm_val_color = torch.zeros((self.pop_size,), dtype=torch.long, device=DEVICE)
        self.agent_wm_count = torch.zeros((self.pop_size,), dtype=torch.long, device=DEVICE)
        self.agent_wm_orient = torch.full((self.pop_size,), -1, dtype=torch.long, device=DEVICE)
        self.toy_block_yaw_rad = self.toy_block_orient.to(torch.float32) * (math.pi * 0.5)
        self.toy_slot_yaw_rad = self.toy_slot_orient.to(torch.float32) * (math.pi * 0.5)
        self.toy_insertion_depth = torch.zeros((self.n_toys,), dtype=torch.float32, device=DEVICE)
        self.agent_wm_yaw_rad = torch.full((self.pop_size,), -10.0, dtype=torch.float32, device=DEVICE)
        self.prev_left_retinal_lum = None
        self.last_binocular_retinal_bundle = None
        self.agent_body_logits = torch.zeros((self.pop_size, 11), dtype=torch.float32, device=DEVICE)
        self.agent_toy_habituation = torch.zeros((self.pop_size, 4, 48), dtype=torch.float32, device=DEVICE)
        self.agent_cerebellar_gain = torch.zeros((self.pop_size, 4), dtype=torch.float32, device=DEVICE)
        self.toys_solved_total = 0
        self.curiosity_explorations_total = 0
        self.threat_observations_total = 0
        self.biome_dispersion_entropy = 1.0

    def step_toy_sensorimotor_body(self, a_i, t_i, t_kind, motor_u, vocal_s):
        gaze_ops = (
            "gaze_salient_target",
            "gaze_corner_sample",
            "gaze_next_matching_attr",
            "gaze_count_objects",
        )
        wm_ops = (
            "wm_store_attr",
            "wm_store_sample_pair",
            "wm_store_count",
        )
        finger_ops = (
            "finger_step_until_contact",
            "finger_paint_from_wm",
            "finger_apply_to_gaze",
            "finger_step_n_from_wm",
        )
        b_logits = self.agent_body_logits[a_i]
        u0 = float(motor_u[a_i, 0].item())
        u1 = float(motor_u[a_i, 1].item())
        u2 = float(motor_u[a_i, 2].item())
        v0 = float(vocal_s[a_i, 0].item()) if vocal_s is not None and vocal_s.shape[1] > 0 else 0.0
        v1 = float(vocal_s[a_i, 1].item()) if vocal_s is not None and vocal_s.shape[1] > 1 else 0.0
        g_scores = b_logits[0:4] + torch.tensor([u0, u1, -u0, -u1], dtype=torch.float32, device=DEVICE) * 0.25
        w_scores = b_logits[4:7] + torch.tensor([v0, v1, u2], dtype=torch.float32, device=DEVICE) * 0.25
        f_scores = b_logits[7:11] + torch.tensor([u2, v0, -u2, -v0], dtype=torch.float32, device=DEVICE) * 0.25
        triple_scores = (
            g_scores[:, None, None] + w_scores[None, :, None] + f_scores[None, None, :]
        ).reshape(-1) - 1.85 * self.agent_toy_habituation[a_i, int(t_kind) % 4]
        best_triple = int(torch.argmax(triple_scores).item())
        gaze_idx = best_triple // 12
        wm_idx = (best_triple // 4) % 3
        finger_idx = best_triple % 4
        gaze_op = gaze_ops[gaze_idx]
        wm_op = wm_ops[wm_idx]
        finger_op = finger_ops[finger_idx]
        self.agent_gaze_focus[a_i] = int(gaze_idx + 1)
        self.agent_wm_orient[a_i] = -1
        self.agent_wm_yaw_rad[a_i] = -10.0
        self.agent_wm_key_color[a_i] = -1
        self.agent_wm_val_color[a_i] = -1
        self.agent_wm_count[a_i] = 0
        ang_err_prev = float(
            torch.abs(
                torch.atan2(
                    torch.sin(self.toy_block_yaw_rad[t_i] - self.toy_slot_yaw_rad[t_i]),
                    torch.cos(self.toy_block_yaw_rad[t_i] - self.toy_slot_yaw_rad[t_i]),
                )
            ).item()
        )
        prev_orient_ok = (ang_err_prev < 0.22) and (
            int(self.toy_block_orient[t_i].item()) == int(self.toy_slot_orient[t_i].item())
        )
        prev_color_ok = int(self.toy_block_color[t_i].item()) == int(self.toy_sample_val_color[t_i].item())
        prev_group_p = int(self.toy_group_progress[t_i].item())
        prev_step_c = int(self.toy_step_counter[t_i].item())
        if gaze_op == "gaze_salient_target":
            if wm_op == "wm_store_attr":
                self.agent_wm_yaw_rad[a_i] = self.toy_slot_yaw_rad[t_i]
                self.agent_wm_orient[a_i] = torch.round(self.toy_slot_yaw_rad[t_i] / (math.pi * 0.5)).long() % 4
        elif gaze_op == "gaze_corner_sample":
            if wm_op == "wm_store_sample_pair":
                self.agent_wm_key_color[a_i] = self.toy_sample_key_color[t_i]
                self.agent_wm_val_color[a_i] = self.toy_sample_val_color[t_i]
        elif gaze_op == "gaze_next_matching_attr":
            if wm_op == "wm_store_attr":
                self.agent_wm_key_color[a_i] = self.toy_sample_key_color[t_i]
        elif gaze_op == "gaze_count_objects":
            if wm_op == "wm_store_count":
                self.agent_wm_count[a_i] = self.toy_target_count[t_i]
        if finger_op == "finger_step_until_contact":
            if float(self.agent_wm_yaw_rad[a_i].item()) > -5.0:
                target_yaw = self.agent_wm_yaw_rad[a_i]
                delta_yaw = torch.atan2(
                    torch.sin(target_yaw - self.toy_block_yaw_rad[t_i]),
                    torch.cos(target_yaw - self.toy_block_yaw_rad[t_i]),
                )
                self.toy_block_yaw_rad[t_i] = self.toy_block_yaw_rad[t_i] + delta_yaw
                self.toy_block_pos[t_i, :2] = 0.35 * self.toy_block_pos[t_i, :2] + 0.65 * self.toy_slot_pos[t_i, :2]
                new_err = float(
                    torch.abs(
                        torch.atan2(
                            torch.sin(self.toy_block_yaw_rad[t_i] - self.toy_slot_yaw_rad[t_i]),
                            torch.cos(self.toy_block_yaw_rad[t_i] - self.toy_slot_yaw_rad[t_i]),
                        )
                    ).item()
                )
                if new_err < 0.22:
                    self.toy_insertion_depth[t_i] = 0.45
                    self.toy_block_pos[t_i, 2] = self.toy_slot_pos[t_i, 2] + 0.05
                    self.toy_block_orient[t_i] = torch.round(self.toy_block_yaw_rad[t_i] / (math.pi * 0.5)).long() % 4
        elif finger_op == "finger_paint_from_wm":
            if (
                int(self.agent_wm_key_color[a_i].item()) == int(self.toy_block_color[t_i].item())
                and int(self.agent_wm_val_color[a_i].item()) > 0
            ):
                self.toy_block_color[t_i] = self.agent_wm_val_color[t_i]
        elif finger_op == "finger_apply_to_gaze":
            if (
                int(self.agent_wm_key_color[a_i].item()) == int(self.toy_sample_key_color[t_i].item())
                and int(self.agent_wm_val_color[a_i].item()) == -1
                and int(self.agent_wm_orient[a_i].item()) == -1
            ):
                self.toy_block_pos[t_i, :2] = 0.5 * self.toy_block_pos[t_i, :2] + 0.5 * self.toy_slot_pos[t_i, :2]
                self.toy_group_progress[t_i] = self.toy_group_progress[t_i] + 1
        elif finger_op == "finger_step_n_from_wm":
            if int(self.agent_wm_count[a_i].item()) == int(self.toy_target_count[t_i].item()):
                step_n = int(self.agent_wm_count[a_i].item())
                for _ in range(step_n):
                    self.toy_step_counter[t_i] = torch.clamp(
                        self.toy_step_counter[t_i] + 1, max=int(self.toy_target_count[t_i].item())
                    )
        ang_err_now = float(
            torch.abs(
                torch.atan2(
                    torch.sin(self.toy_block_yaw_rad[t_i] - self.toy_slot_yaw_rad[t_i]),
                    torch.cos(self.toy_block_yaw_rad[t_i] - self.toy_slot_yaw_rad[t_i]),
                )
            ).item()
        )
        now_orient_ok = (ang_err_now < 0.22) and (float(self.toy_insertion_depth[t_i].item()) >= 0.25)
        now_color_ok = int(self.toy_block_color[t_i].item()) == int(self.toy_sample_val_color[t_i].item())
        now_group_p = int(self.toy_group_progress[t_i].item())
        now_step_c = int(self.toy_step_counter[t_i].item())
        solved_micro = False
        if t_kind == 0:
            solved_micro = (not prev_orient_ok) and now_orient_ok
        elif t_kind == 1:
            solved_micro = (not prev_color_ok) and now_color_ok
        elif t_kind == 2:
            solved_micro = now_group_p > prev_group_p
        elif t_kind == 3:
            solved_micro = (prev_step_c != int(self.toy_target_count[t_i].item())) and (
                now_step_c == int(self.toy_target_count[t_i].item())
            )
        if solved_micro:
            self.agent_toy_habituation[a_i, int(t_kind) % 4] *= 0.85
            self.agent_toy_habituation[a_i, int(t_kind) % 4, best_triple] = -0.65
            self.agent_cerebellar_gain[a_i, int(t_kind) % 4] = float(
                min(1.0, float(self.agent_cerebellar_gain[a_i, int(t_kind) % 4].item()) + 0.15)
            )
            self.cerebellar_corrections_total += 1
        else:
            self.agent_toy_habituation[a_i, int(t_kind) % 4, best_triple] += 1.0
            self.agent_cerebellar_gain[a_i, int(t_kind) % 4] *= 0.96
        chunk_sig = f"sensorimotor_chunk::{gaze_op}+{wm_op}+{finger_op}"
        return solved_micro, chunk_sig, gaze_op, wm_op, finger_op, best_triple

    def render_first_person_retinal_grids(self, retina_h=9, retina_w=9):
        N = self.pop_size
        is_toxic_now = self.food_is_mimic | (
            (self.food_polarity * self.toxic_polarity > 0.0)
            if self.unpredictable_mode != "standard"
            else torch.zeros_like(self.food_is_mimic)
        )
        food_cols = torch.where(
            is_toxic_now,
            torch.full((self.n_food,), 8, dtype=torch.int32, device=DEVICE),
            torch.full((self.n_food,), 3, dtype=torch.int32, device=DEVICE),
        )
        box_groups = [
            (self.obs_pos, (3.2, 3.2, 2.5), 0.0, 4),
            (self.food_pos, (1.6, 1.6, 1.2), 0.0, food_cols),
        ]
        if not self.ambush_stealth_phase:
            box_groups.append((self.pred_pos, (2.2, 2.2, 1.8), 0.0, 2))
        if self.unpredictable_mode in ("meteor_3d", "chaos_all", "infinite_procedural_chaos"):
            box_groups.append((self.meteors, (2.4, 2.4, 2.0), 0.0, 6))
        if self.unpredictable_mode in ("abyss_chasm", "abyss_collapse", "chaos_all", "infinite_procedural_chaos"):
            ab_z = self.terrain_height(self.abyss_centers[:, 0], self.abyss_centers[:, 1])
            ab_3d = torch.stack([self.abyss_centers[:, 0], self.abyss_centers[:, 1], ab_z], dim=-1)
            box_groups.append((ab_3d, (3.0, 3.0, 1.2), 0.0, 6))
        if self.unpredictable_mode in ("vortex_field", "vortex_storm", "chaos_all", "infinite_procedural_chaos"):
            vx_z = self.terrain_height(self.vortex_centers[:, 0], self.vortex_centers[:, 1]) + 2.0
            vx_3d = torch.stack([self.vortex_centers[:, 0], self.vortex_centers[:, 1], vx_z], dim=-1)
            box_groups.append((vx_3d, (2.5, 2.5, 1.5), 0.0, 6))
        if hasattr(self, "toy_block_pos") and hasattr(self, "toy_slot_pos"):
            box_groups.append((self.toy_slot_pos, (1.8, 1.8, 0.9), self.toy_slot_yaw_rad, 9))
            box_groups.append((self.toy_block_pos, (1.4, 1.4, 1.1), self.toy_block_yaw_rad, 7))
        if self.unpredictable_mode in ("signpost_indicator", "chaos_all", "infinite_procedural_chaos"):
            pn_z = self.terrain_height(self.panel_centers[:, 0], self.panel_centers[:, 1]) + 1.0
            pn_3d = torch.stack([self.panel_centers[:, 0], self.panel_centers[:, 1], pn_z], dim=-1)
            pn_cols = torch.where(
                self.panel_color_rule >= 0.0,
                torch.full((self.panel_centers.shape[0],), 7, dtype=torch.int32, device=DEVICE),
                torch.full((self.panel_centers.shape[0],), 9, dtype=torch.int32, device=DEVICE),
            )
            box_groups.append((pn_3d, (1.8, 1.8, 1.4), 0.0, pn_cols))

        warp_engine = get_global_isaac_warp_engine()
        bino_out = warp_engine.render_arena_population_binocular_gpu(
            self.pos,
            self.yaw,
            box_groups,
            retina_h=retina_h,
            retina_w=retina_w,
            prev_left_lum_t=self.prev_left_retinal_lum,
        )
        self.prev_left_retinal_lum = bino_out["left_lum"].detach().clone()
        self.last_binocular_retinal_bundle = bino_out

        retinas = bino_out["left_color"].clone()
        z_buffers = bino_out["reconstructed_depth"].clone()
        horizon_r = retina_h // 2
        center_c = retina_w // 2
        ground_z = self.terrain_height(self.pos[:, 0], self.pos[:, 1])
        high_ground = (ground_z > 5.0).unsqueeze(-1).unsqueeze(-1)
        ground_fill = torch.where(
            high_ground.expand(N, retina_h - horizon_r - 1, retina_w),
            torch.full((N, retina_h - horizon_r - 1, retina_w), 5, dtype=torch.int32, device=DEVICE),
            torch.full((N, retina_h - horizon_r - 1, retina_w), 1, dtype=torch.int32, device=DEVICE),
        )
        retinas[:, horizon_r + 1 :, :] = torch.where(
            retinas[:, horizon_r + 1 :, :] == 0,
            ground_fill,
            retinas[:, horizon_r + 1 :, :],
        )

        prox_map = torch.clamp(1.0 - z_buffers / self.ray_Range, 0.0, 1.0)
        is_food_mask = ((retinas == 3) | (retinas == 7) | (retinas == 8)).to(torch.float32)
        is_threat_mask = ((retinas == 2) | (retinas == 6)).to(torch.float32)
        is_obs_mask = ((retinas == 4) | (retinas == 9)).to(torch.float32)

        r_grid = torch.arange(retina_h, device=DEVICE).view(1, retina_h, 1)
        c_grid = torch.arange(retina_w, device=DEVICE).view(1, 1, retina_w)
        fov_mask = (((r_grid - horizon_r).abs() <= 1) & ((c_grid - center_c).abs() <= 1)).to(torch.float32)
        per_mask = 1.0 - fov_mask

        foveal_salience = torch.zeros((N, 4), dtype=torch.float32, device=DEVICE)
        peripheral_salience = torch.zeros((N, 4), dtype=torch.float32, device=DEVICE)

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
    def extract_retinal_16d_observations(self, retinas=None, foveal_salience=None, peripheral_salience=None):
        if retinas is None or foveal_salience is None or peripheral_salience is None:
            retinas, foveal_salience, peripheral_salience = self.render_first_person_retinal_grids()
        N, rh, rw = retinas.shape
        g_scale = float(max(rh, rw, 1))
        r_coords = torch.arange(rh, dtype=torch.float32, device=DEVICE).view(1, rh, 1).expand(N, rh, rw)
        c_coords = torch.arange(rw, dtype=torch.float32, device=DEVICE).view(1, 1, rw).expand(N, rh, rw)
        food_mask = ((retinas == 3) | (retinas == 7)).float()
        threat_mask = ((retinas == 2) | (retinas == 6) | (retinas == 8)).float()
        obs_mask = ((retinas == 4) | (retinas == 9)).float()
        if self.last_binocular_retinal_bundle is not None and self.last_binocular_retinal_bundle["reconstructed_depth"].shape == retinas.shape:
            rec_z = self.last_binocular_retinal_bundle["reconstructed_depth"]
            prox_map = torch.clamp(1.0 - rec_z / self.ray_Range, 0.0, 1.0)
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
        gravity_cue = (((self.pos[:, 2] - ground_z) > 0.35) | ((retinas[:, : rh // 2, :] == 6).any(dim=(1, 2)))).float()
        enclosure_cue = (peripheral_salience[:, 2] > 0.25).float()
        return torch.stack([
            dx_t, dy_t, dz_t,
            dx_h, dy_h, dz_h,
            ratio_w, ratio_h,
            contact_cue, depth_occlude_cue, aspect_mismatch, gravity_cue, enclosure_cue,
            self.hp, self.energy, self.pain + 0.3 * self.surprise,
        ], dim=-1)
    def get_observations(self):
        N = self.pop_size
        K = self.n_rays

        retinas, _, _ = self.render_first_person_retinal_grids(retina_h=9, retina_w=9)
        z_buffers = self.last_binocular_retinal_bundle["reconstructed_depth"]
        prox_map = torch.clamp(1.0 - z_buffers / self.ray_Range, 0.0, 1.0)

        r_obs = torch.zeros((N, K), dtype=torch.float32, device=DEVICE)
        r_food = torch.zeros((N, K), dtype=torch.float32, device=DEVICE)
        r_haz = torch.zeros((N, K), dtype=torch.float32, device=DEVICE)

        for k in range(K):
            col_ret = retinas[:, :, k : k + 2]
            col_prox = prox_map[:, :, k : k + 2]
            obs_k = torch.where((col_ret == 4) | (col_ret == 9), col_prox, torch.zeros_like(col_prox)).amax(dim=(1, 2))
            panel_sign = torch.where((col_ret == 9).any(dim=(1, 2)), -1.0, 1.0)
            r_obs[:, k] = obs_k * panel_sign

            food_pos_k = torch.where((col_ret == 3) | (col_ret == 7), col_prox, torch.zeros_like(col_prox)).amax(dim=(1, 2))
            food_neg_k = torch.where(col_ret == 8, col_prox, torch.zeros_like(col_prox)).amax(dim=(1, 2))
            r_food[:, k] = torch.where(food_pos_k >= food_neg_k, food_pos_k, -food_neg_k)

            haz_k = torch.where((col_ret == 2) | (col_ret == 6), col_prox, torch.zeros_like(col_prox)).amax(dim=(1, 2))
            r_haz[:, k] = haz_k

        ag_diff = self.pos[None, :, :2] - self.pos[:, None, :2]
        ag_dist = torch.norm(ag_diff, dim=-1) + 1e-5
        mask_other = (ag_dist < 26.0) & (~torch.eye(N, dtype=torch.bool, device=DEVICE)) & self.alive[None, :]
        weights = torch.where(mask_other, 1.0 / (1.0 + 0.15 * ag_dist), torch.zeros_like(ag_dist))
        w_sum = weights.sum(dim=1, keepdim=True) + 1e-6
        audition = (weights @ self.signals) / w_sum
        if self.unpredictable_mode in ("signpost_indicator", "chaos_all", "infinite_procedural_chaos"):
            audition = 0.65 * audition + 0.35 * self.red_leader_traj[None, :].expand(N, 2)

        ground_z = self.terrain_height(self.pos[:, 0], self.pos[:, 1])
        alt_norm = torch.clamp((self.pos[:, 2] - ground_z) / 10.0, 0.0, 1.0)
        speed_norm = torch.clamp(torch.norm(self.vel[:, :2], dim=-1) / self.max_speed, 0.0, 1.0)
        vz_norm = torch.clamp(self.vel[:, 2] / 5.0, -1.0, 1.0)

        somatic = torch.stack([
            self.hp,
            self.energy,
            self.pain,
            self.surprise,
            speed_norm,
            vz_norm,
            alt_norm,
        ], dim=-1)

        obs = torch.cat([r_haz, r_food, r_obs, audition, somatic], dim=-1)
        return obs

    def step_physics(self, motor_u, vocal_s, thought_metabolic_tax, comm_enabled=True):
        self.step_idx += 1
        N = self.pop_size
        self.signals = vocal_s if comm_enabled else torch.zeros_like(vocal_s)
        is_proc = self.unpredictable_mode == "infinite_procedural_chaos"
        dna = self.hazard_dna if is_proc else {}

        if self.unpredictable_mode == "standard":
            self.toxic_polarity = -1.0
        elif self.unpredictable_mode in ("mimic_trap", "signpost_indicator", "chaos_all", "infinite_procedural_chaos"):
            if self.step_idx % self.polarity_shift_period == 0:
                self.toxic_polarity *= -1.0
                self.panel_color_rule *= -1.0
                if is_proc:
                    self.terrain_shift_x += float(self.rng.uniform(-6.0, 6.0))
        else:
            self.toxic_polarity = 1.0

        if self.unpredictable_mode in ("ambush_stalker", "chaos_all", "infinite_procedural_chaos"):
            st_per = int(dna.get("stealth_period", 40))
            st_duty = int(dna.get("stealth_duty", 22))
            self.ambush_stealth_phase = (self.step_idx % st_per) >= st_duty
        else:
            self.ambush_stealth_phase = False

        panel_yaw_sign = torch.ones((N,), dtype=torch.float32, device=DEVICE)
        panel_gravity_scale = torch.ones((N,), dtype=torch.float32, device=DEVICE)
        if self.unpredictable_mode in ("signpost_indicator", "chaos_all", "infinite_procedural_chaos"):
            p_diff = self.pos[:, None, :2] - self.panel_centers[None, :, :]
            p_dist = torch.norm(p_diff, dim=-1)
            min_p_d, min_p_idx = p_dist.min(dim=1)
            on_panel = min_p_d < 9.5
            rule_val = self.panel_color_rule[min_p_idx]
            panel_yaw_sign = torch.where(on_panel, rule_val, panel_yaw_sign)
            g_mod = float(dna.get("panel_gravity_mod", 0.35))
            panel_gravity_scale = torch.where(on_panel & (rule_val < 0.0), torch.full_like(panel_gravity_scale, g_mod), panel_gravity_scale)

        if self.unpredictable_mode in ("vortex_storm", "chaos_all", "infinite_procedural_chaos") and (self.step_idx % 50) >= 25:
            motor_yaw_sign = -1.0 * panel_yaw_sign
        else:
            motor_yaw_sign = panel_yaw_sign

        thrust = motor_u[:, 0] * 3.9
        torque_yaw = (motor_u[:, 1] * self.max_omega * motor_yaw_sign) + self.yaw_bias_drift
        jump_impulse = torch.clamp(motor_u[:, 2], 0.0, 1.0)

        self.yaw = torch.atan2(torch.sin(self.yaw + torque_yaw * self.dt), torch.cos(self.yaw + torque_yaw * self.dt))
        forward_dir = torch.stack([torch.cos(self.yaw), torch.sin(self.yaw)], dim=-1)

        self.vel[:, :2] = self.friction * self.vel[:, :2] + thrust[:, None] * forward_dir * self.dt
        speed_xy = torch.norm(self.vel[:, :2], dim=-1, keepdim=True) + 1e-6
        self.vel[:, :2] = torch.where(
            speed_xy > self.max_speed,
            self.vel[:, :2] / speed_xy * self.max_speed,
            self.vel[:, :2],
        )

        ground_z = self.terrain_height(self.pos[:, 0], self.pos[:, 1])
        on_ground = (self.pos[:, 2] - ground_z) <= 0.35
        self.vel[:, 2] = torch.where(
            on_ground & (jump_impulse > 0.45),
            jump_impulse * 5.2,
            self.vel[:, 2] - (self.gravity * panel_gravity_scale) * self.dt,
        )

        if self.unpredictable_mode in ("vortex_storm", "chaos_all", "infinite_procedural_chaos"):
            self.vortex_centers = torch.clamp(self.vortex_centers + self.vortex_vel * self.dt, 15.0, self.L - 15.0)
            v_diff = self.vortex_centers[None, :, :] - self.pos[:, None, :2]
            v_dist = torch.norm(v_diff, dim=-1) + 1e-4
            in_vortex = v_dist < 16.0
            if torch.any(in_vortex):
                tangential = torch.stack([-v_diff[..., 1], v_diff[..., 0]], dim=-1) / v_dist[..., None]
                vp_s = float(dna.get("vortex_pull", 1.8))
                vt_s = float(dna.get("vortex_swirl", 3.2))
                pull = (v_diff / v_dist[..., None]) * vp_s + tangential * vt_s
                strength = torch.where(in_vortex, (16.0 - v_dist) / 16.0, torch.zeros_like(v_dist))
                net_force = (strength[..., None] * pull).sum(dim=1)
                self.vel[:, :2] = self.vel[:, :2] + net_force * self.dt

        new_pos = self.pos + self.vel * self.dt

        diff_ob = new_pos[:, None, :2] - self.obs_pos[None, :, :2]
        dist_ob = torch.norm(diff_ob, dim=-1) + 1e-5
        pen_ob = self.obs_radius[None, :] + 0.8 - dist_ob
        coll_ob = (pen_ob > 0.0) & ((new_pos[:, None, 2] - self.obs_pos[None, :, 2]).abs() < 3.5)
        if torch.any(coll_ob):
            push_dir = diff_ob / dist_ob[..., None]
            push_vec = (torch.where(coll_ob, pen_ob, torch.zeros_like(pen_ob))[..., None] * push_dir).sum(dim=1)
            new_pos[:, :2] = new_pos[:, :2] + push_vec

        wall_dmg = torch.zeros((N,), dtype=torch.float32, device=DEVICE)
        out_bound = (new_pos[:, 0] < 1.5) | (new_pos[:, 0] > self.L - 1.5) | (new_pos[:, 1] < 1.5) | (new_pos[:, 1] > self.L - 1.5)
        wall_dmg = torch.where(out_bound, torch.full_like(wall_dmg, 0.07), wall_dmg)
        new_pos[:, 0] = torch.clamp(new_pos[:, 0], 1.5, self.L - 1.5)
        new_pos[:, 1] = torch.clamp(new_pos[:, 1], 1.5, self.L - 1.5)

        new_ground_z = self.terrain_height(new_pos[:, 0], new_pos[:, 1])
        below_ground = new_pos[:, 2] < new_ground_z
        new_pos[:, 2] = torch.where(below_ground, new_ground_z, new_pos[:, 2])
        self.vel[:, 2] = torch.where(below_ground, torch.zeros_like(self.vel[:, 2]), self.vel[:, 2])
        new_pos[:, 2] = torch.clamp(new_pos[:, 2], 0.0, self.Hz)
        self.pos = new_pos

        p_to_ag = self.pos[None, :, :2] - self.pred_pos[:, None, :2]
        p_dist = torch.norm(p_to_ag, dim=-1) + 1e-5
        p_dist_alive = torch.where(self.alive[None, :], p_dist, torch.full_like(p_dist, 1e6))
        nearest_ag = torch.argmin(p_dist_alive, dim=1)
        chase_vec = p_to_ag[torch.arange(self.n_predators, device=DEVICE), nearest_ag]
        chase_norm = torch.norm(chase_vec, dim=-1, keepdim=True) + 1e-5
        chase_dir = chase_vec / chase_norm
        if is_proc:
            flank_ang = float(dna.get("pred_flank_angle", 0.35)) * math.sin(0.09 * float(self.step_idx))
            ca = math.cos(flank_ang)
            sa = math.sin(flank_ang)
            rot_x = chase_dir[:, 0] * ca - chase_dir[:, 1] * sa
            rot_y = chase_dir[:, 0] * sa + chase_dir[:, 1] * ca
            chase_dir = torch.stack([rot_x, rot_y], dim=-1)
        pred_speed = (
            float(dna.get("pred_speed", 2.85))
            if is_proc
            else (2.95 if self.unpredictable_mode in ("ambush_stalker", "chaos_all") else 2.25)
        )
        self.pred_vel[:, :2] = 0.80 * self.pred_vel[:, :2] + 0.20 * chase_dir * pred_speed
        self.pred_pos[:, :2] = torch.clamp(self.pred_pos[:, :2] + self.pred_vel[:, :2] * self.dt, 2.0, self.L - 2.0)
        self.pred_pos[:, 2] = self.terrain_height(self.pred_pos[:, 0], self.pred_pos[:, 1]) + 0.5
        red_v = self.pred_vel[0, :2]
        self.red_leader_traj = red_v / (torch.norm(red_v) + 1e-6)

        dmg = wall_dmg.clone()

        ag_pred_dist = torch.norm(self.pos[:, None, :] - self.pred_pos[None, :, :], dim=-1)
        min_pred_d = ag_pred_dist.min(dim=1).values
        pred_hit = min_pred_d < 3.0
        dmg = dmg + torch.where(pred_hit, 0.24 * (3.0 - min_pred_d) / 3.0, torch.zeros_like(dmg))

        traj_bonus_e = torch.zeros((N,), dtype=torch.float32, device=DEVICE)
        if self.unpredictable_mode in ("signpost_indicator", "chaos_all", "infinite_procedural_chaos"):
            red_d = ag_pred_dist[:, 0]
            in_red_field = (red_d < 22.0) & self.alive
            ag_v_dir = self.vel[:, :2] / (torch.norm(self.vel[:, :2], dim=-1, keepdim=True) + 1e-6)
            traj_align = (ag_v_dir * self.red_leader_traj[None, :]).sum(dim=-1)
            violated_traj = in_red_field & (traj_align < -0.15)
            followed_traj = in_red_field & (traj_align > 0.45)
            dmg = dmg + torch.where(violated_traj, torch.full_like(dmg, 0.22), torch.zeros_like(dmg))
            traj_bonus_e = torch.where(followed_traj, torch.full_like(traj_bonus_e, 0.08), traj_bonus_e)

        if self.unpredictable_mode in ("meteor_3d", "chaos_all", "infinite_procedural_chaos"):
            m_fall = float(dna.get("meteor_fall_speed", 4.8))
            self.meteors[:, 2] -= m_fall * self.dt
            if is_proc:
                m_om = float(dna.get("meteor_spiral_omega", 0.12))
                m_dr = float(dna.get("meteor_lateral_drift", 1.5))
                self.meteors[:, 0] = torch.clamp(self.meteors[:, 0] + m_dr * math.sin(m_om * float(self.step_idx)) * self.dt, 8.0, 48.0)
                self.meteors[:, 1] = torch.clamp(self.meteors[:, 1] + m_dr * math.cos(m_om * float(self.step_idx)) * self.dt, 8.0, 48.0)
            impacted = self.meteors[:, 2] <= self.terrain_height(self.meteors[:, 0], self.meteors[:, 1]) + 0.8
            if torch.any(impacted):
                imp_pos = self.meteors[impacted]
                d_imp = torch.norm(self.pos[:, None, :2] - imp_pos[None, :, :2], dim=-1).min(dim=1).values
                b_rad = float(dna.get("meteor_blast_radius", 6.2))
                hit_m = d_imp < b_rad
                dmg = dmg + torch.where(hit_m, 0.30 * (b_rad - d_imp) / b_rad, torch.zeros_like(dmg))
                n_imp = int(impacted.sum().item())
                self.meteors[impacted, 0] = torch.tensor(self.rng.uniform(10.0, 46.0, (n_imp,)).astype(np.float32), device=DEVICE)
                self.meteors[impacted, 1] = torch.tensor(self.rng.uniform(10.0, 46.0, (n_imp,)).astype(np.float32), device=DEVICE)
                self.meteors[impacted, 2] = torch.tensor(self.rng.uniform(22.0, 30.0, (n_imp,)).astype(np.float32), device=DEVICE)

        if self.unpredictable_mode in ("abyss_collapse", "chaos_all", "infinite_procedural_chaos"):
            sk_per = int(dna.get("sinkhole_period", 35))
            sk_rad = float(dna.get("sinkhole_radius", 5.0))
            if self.step_idx % sk_per == 0:
                ab_xy = np.zeros((4, 2), dtype=np.float32)
                ab_xy[:, 0] = self.rng.uniform(54.0, self.L - 12.0, (4,))
                ab_xy[:, 1] = self.rng.uniform(12.0, 46.0, (4,))
                self.abyss_centers = torch.tensor(ab_xy, device=DEVICE)
            d_ab = torch.norm(self.pos[:, None, :2] - self.abyss_centers[None, :, :], dim=-1).min(dim=1).values
            in_abyss = (d_ab < sk_rad) & ((self.pos[:, 2] - ground_z) < 1.2)
            dmg = dmg + torch.where(in_abyss, torch.full_like(dmg, 0.25), torch.zeros_like(dmg))

        ag_quad = (self.pos[:, 0] >= 50.0).long() + 2 * (self.pos[:, 1] >= 50.0).long()
        food_quad = (self.food_pos[:, 0] >= 50.0).long() + 2 * (self.food_pos[:, 1] >= 50.0).long()

        ag_food_dist = torch.norm(self.pos[:, None, :] - self.food_pos[None, :, :], dim=-1)
        near_food = (ag_food_dist < 3.6) & self.alive[:, None]
        agents_at_food = near_food.sum(dim=0)

        is_toxic_now = self.food_is_mimic | (
            (self.food_polarity * self.toxic_polarity > 0.0)
            if self.unpredictable_mode != "standard"
            else torch.zeros_like(self.food_is_mimic)
        )
        toxic_visits = near_food & is_toxic_now[None, :]
        coop_active = (~is_toxic_now) & self.food_is_coop & (agents_at_food >= 2)
        solo_active = (~is_toxic_now) & (~self.food_is_coop) & (agents_at_food >= 1)

        coop_visits = near_food & coop_active[None, :]
        solo_visits = near_food & solo_active[None, :]

        food_res_scale = self.quadrant_resource_level[food_quad][None, :]
        dmg = dmg + toxic_visits.float().sum(dim=1) * 0.22
        delta_e = (coop_visits.float() * food_res_scale).sum(dim=1) * 0.44 + (solo_visits.float() * food_res_scale).sum(dim=1) * 0.30 + traj_bonus_e
        respawn_mask = toxic_visits.any(dim=0) | coop_active | solo_active

        self.quadrant_resource_level = torch.clamp(self.quadrant_resource_level + 0.008, 0.25, 1.0)
        if torch.any(respawn_mask):
            for q_idx in range(4):
                harvested_in_q = ((coop_active | solo_active) & (food_quad == q_idx)).sum().float()
                if harvested_in_q > 0:
                    self.quadrant_resource_level[q_idx] = torch.clamp(
                        self.quadrant_resource_level[q_idx] - 0.045 * harvested_in_q,
                        0.25,
                        1.0,
                    )
            n_resp = int(respawn_mask.sum().item())
            new_fxy = self.rng.uniform(10.0, self.L - 10.0, (n_resp, 2)).astype(np.float32)
            self.food_pos[respawn_mask, 0] = torch.tensor(new_fxy[:, 0], device=DEVICE)
            self.food_pos[respawn_mask, 1] = torch.tensor(new_fxy[:, 1], device=DEVICE)
            self.food_pos[respawn_mask, 2] = self.terrain_height(self.food_pos[respawn_mask, 0], self.food_pos[respawn_mask, 1]) + 0.5

        survival_safety_gate = (
            torch.clamp((self.hp - 0.35) / 0.65, 0.0, 1.0)
            * torch.clamp((self.energy - 0.22) / 0.78, 0.0, 1.0)
            * torch.exp(-4.5 * (self.pain + dmg))
            * self.alive.float()
        )

        self.agent_visited_quadrants = self.agent_visited_quadrants * 0.992
        cur_visit_val = self.agent_visited_quadrants[torch.arange(N, device=DEVICE), ag_quad]
        entered_novel_biome = (cur_visit_val < 0.35) & (survival_safety_gate > 0.25)
        diversive_cur_bonus = torch.where(
            entered_novel_biome,
            0.12 * survival_safety_gate * self.quadrant_resource_level[ag_quad],
            torch.zeros_like(delta_e),
        )
        self.agent_visited_quadrants[torch.arange(N, device=DEVICE), ag_quad] = 1.0
        self.curiosity_explorations_total += int(entered_novel_biome.sum().item())

        specific_toy_bonus = torch.zeros_like(delta_e)
        if hasattr(self, "toy_block_pos") and hasattr(self, "toy_slot_pos"):
            ag_toy_diff = self.toy_block_pos[None, :, :2] - self.pos[:, None, :2]
            ag_toy_dist = torch.norm(ag_toy_diff, dim=-1) + 1e-5
            min_t_dist, min_t_idx = ag_toy_dist.min(dim=1)
            touching_toy = (min_t_dist < 4.2) & (survival_safety_gate > 0.20)
            if torch.any(touching_toy):
                specific_toy_bonus = torch.where(
                    touching_toy,
                    0.04 * survival_safety_gate,
                    specific_toy_bonus,
                )
                touch_ag = torch.where(touching_toy)[0]
                for a_i in touch_ag.tolist():
                    t_i = int(min_t_idx[a_i].item())
                    t_kind = int(self.toy_type[t_i].item())
                    solved_micro, chunk_sig, gaze_op, wm_op, finger_op, best_triple = self.step_toy_sensorimotor_body(
                        a_i, t_i, t_kind, motor_u, vocal_s
                    )
                    to_slot = self.toy_slot_pos[t_i, :2] - self.toy_block_pos[t_i, :2]
                    slot_d = float(torch.norm(to_slot).item()) + 1e-5
                    ag_push = self.vel[a_i, :2]
                    push_norm = float(torch.norm(ag_push).item()) + 1e-5
                    gaze_locked = solved_micro or (int(self.agent_wm_orient[a_i].item()) == int(self.toy_slot_orient[t_i].item()))
                    slot_pull = 0.65 if gaze_locked else 0.08
                    push_dir = slot_pull * (to_slot / slot_d) + (1.0 - slot_pull) * (ag_push / push_norm)
                    if t_kind == 2:
                        if abs(float(to_slot[0].item())) >= abs(float(to_slot[1].item())):
                            push_dir[1] = 0.0
                        else:
                            push_dir[0] = 0.0
                    self.toy_block_pos[t_i, :2] = self.toy_block_pos[t_i, :2] + push_dir * 1.45 * self.dt * 4.0
                    self.toy_block_pos[t_i, 2] = self.terrain_height(self.toy_block_pos[t_i, 0], self.toy_block_pos[t_i, 1]) + 0.6
                    new_slot_d = float(torch.norm(self.toy_slot_pos[t_i, :2] - self.toy_block_pos[t_i, :2]).item())
                    orient_ok = (t_kind != 0) or (int(self.toy_block_orient[t_i].item()) == int(self.toy_slot_orient[t_i].item()))
                    if (new_slot_d < 2.4 or solved_micro) and orient_ok and solved_micro:
                        specific_toy_bonus[a_i] = specific_toy_bonus[a_i] + 0.36 * (0.4 + 0.6 * survival_safety_gate[a_i])
                        self.toys_solved_total += 1
                        self.sensorimotor_chunk_hits_total += 1
                        rec = self.discovered_sensorimotor_chunks.get(
                            chunk_sig,
                            {
                                "chunk_sig": chunk_sig,
                                "toy_kind": int(t_kind),
                                "gaze_op": gaze_op,
                                "wm_op": wm_op,
                                "finger_op": finger_op,
                                "hits": 0,
                                "sample_key_color": int(self.toy_sample_key_color[t_i].item()),
                                "sample_val_color": int(self.toy_sample_val_color[t_i].item()),
                                "sample_count": int(self.toy_target_count[t_i].item()),
                            },
                        )
                        rec["hits"] = int(rec["hits"]) + 1
                        self.discovered_sensorimotor_chunks[chunk_sig] = rec
                        self.toy_group_progress[t_i] = 0
                        self.toy_step_counter[t_i] = 0
                        self.toy_block_color[t_i] = self.toy_sample_key_color[t_i]
                        q_id = t_i % 4
                        quad_bounds = [
                            (14.0, 42.0, 14.0, 42.0),
                            (58.0, 86.0, 14.0, 42.0),
                            (14.0, 42.0, 58.0, 86.0),
                            (58.0, 86.0, 58.0, 86.0),
                        ]
                        x_lo, x_hi, y_lo, y_hi = quad_bounds[q_id]
                        ang = float(self.rng.uniform(-math.pi, math.pi))
                        dist_t = float(self.rng.uniform(3.8, 5.8))
                        sx = float(self.toy_slot_pos[t_i, 0].item())
                        sy = float(self.toy_slot_pos[t_i, 1].item())
                        self.toy_block_pos[t_i, 0] = float(np.clip(sx + math.cos(ang) * dist_t, x_lo, x_hi))
                        self.toy_block_pos[t_i, 1] = float(np.clip(sy + math.sin(ang) * dist_t, y_lo, y_hi))
                        self.toy_block_pos[t_i, 2] = self.terrain_height(self.toy_block_pos[t_i, 0], self.toy_block_pos[t_i, 1]) + 0.6
                        self.toy_block_orient[t_i] = int(self.rng.integers(0, 4))

        safe_threat_watch = (min_pred_d > 4.2) & (min_pred_d < 13.5) & (dmg < 0.01) & (survival_safety_gate > 0.25)
        threat_obs_bonus = torch.where(safe_threat_watch, 0.035 * survival_safety_gate, torch.zeros_like(delta_e))
        self.threat_observations_total += int(safe_threat_watch.sum().item())

        quad_counts = torch.bincount(ag_quad[self.alive], minlength=4).float()
        alive_total = quad_counts.sum().clamp(min=1.0)
        quad_probs = quad_counts / alive_total
        nz_probs = quad_probs[quad_probs > 0]
        self.biome_dispersion_entropy = float(-(nz_probs * torch.log(nz_probs + 1e-8)).sum().item() / math.log(4.0))
        under_populated_biome = (quad_probs[ag_quad] < 0.28) & (survival_safety_gate > 0.25)
        dispersion_bonus = torch.where(under_populated_biome, 0.025 * survival_safety_gate, torch.zeros_like(delta_e))

        delta_e = delta_e + diversive_cur_bonus + specific_toy_bonus + threat_obs_bonus + dispersion_bonus

        physical_metabolic_cost = 0.0028 + 0.0007 * (motor_u[:, 0].abs() + jump_impulse)
        total_metabolic_cost = physical_metabolic_cost + thought_metabolic_tax
        self.energy = torch.clamp(self.energy + delta_e - total_metabolic_cost, 0.0, 1.0)
        starvation_dmg = torch.where(self.energy <= 0.03, torch.full_like(dmg, 0.022), torch.zeros_like(dmg))
        total_dmg = dmg + starvation_dmg

        self.pain = 1.0 - torch.exp(-2.5 * total_dmg)
        self.hp = torch.clamp(self.hp - total_dmg - 0.0022 * (1.0 - self.energy), min=0.0, max=1.0)
        return total_dmg, delta_e, total_metabolic_cost

