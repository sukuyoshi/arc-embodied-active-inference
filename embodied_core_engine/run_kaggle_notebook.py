import os
import sys
import json
import time
import zipfile
import types
from pathlib import Path
import numpy as np
import torch

class MockWarpEngine:
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.effector_hand_world_xyz = [112.0, 50.16, 1.49]
        self.last_seated_joint_targets_21d = np.zeros(21, dtype=np.float32)
        self.safe_room_bounds_xyz = (100.0, 124.0, 38.0, 62.0, 0.70, 4.70)
        self.safe_room_doorway_xyz = (100.0, 50.0, 0.79)
        self.safe_room_desk_world_xyz = (112.0, 50.16, 1.46)
        self.safe_room_desk_surface_world_xyz = (112.0, 50.16, 1.49)

    def perceive_3d_tabletop_via_human_vision(
        self,
        grid,
        head_center_xyz=(0.0, 0.0, 0.52),
        head_shift_delta_xyz=(0.045, 0.0, 0.0),
        gaze_uv_deg=(0.0, 0.0),
        block_heights=None,
        step_physx=False,
        skip_spatial_3d=True,
    ):
        g_arr = np.asarray(grid, dtype=np.int32)
        H_g, W_g = int(g_arr.shape[0]), int(g_arr.shape[1])
        W_s, H_s = 64, 64
        grid_dev = self.device
        palette_t = torch.tensor(
            (
                (0.05, 0.05, 0.07),
                (0.00, 0.45, 0.85),
                (0.90, 0.15, 0.15),
                (0.18, 0.80, 0.25),
                (0.95, 0.85, 0.10),
                (0.65, 0.65, 0.68),
                (0.90, 0.20, 0.75),
                (0.95, 0.50, 0.10),
                (0.35, 0.80, 0.95),
                (0.55, 0.10, 0.20),
            ),
            dtype=torch.float32,
            device=grid_dev,
        )
        grid_t = torch.as_tensor(g_arr, dtype=torch.long, device=grid_dev)
        vals_p, cnts_p = torch.unique(grid_t, return_counts=True)
        bg_perc = int(vals_p[int(torch.argmax(cnts_p).item())].item()) if vals_p.numel() > 0 else 0
        fg_t = (grid_t != bg_perc).to(torch.float32)
        height_map_t = fg_t * 0.36
        if H_g > 1:
            down_free = float(torch.mean(fg_t[:-1, :] * (1.0 - fg_t[1:, :])).item())
            up_free = float(torch.mean(fg_t[1:, :] * (1.0 - fg_t[:-1, :])).item())
        else:
            down_free, up_free = 0.0, 0.0
        if W_g > 1:
            right_free = float(torch.mean(fg_t[:, :-1] * (1.0 - fg_t[:, 1:])).item())
            left_free = float(torch.mean(fg_t[:, 1:] * (1.0 - fg_t[:, :-1])).item())
        else:
            right_free, left_free = 0.0, 0.0
        c_vec = np.array([down_free, up_free, right_free, left_free], dtype=np.float32)
        dom_grav = int(np.argmax(c_vec))
        mean_h = float(torch.mean(height_map_t).item())
        max_h = float(torch.max(height_map_t).item())
        std_h = float(torch.std(height_map_t).item()) if height_map_t.numel() > 1 else 0.0
        affordance_16 = np.array(
            [
                c_vec[0], c_vec[1], c_vec[2], c_vec[3],
                mean_h, max_h, std_h,
                0.0, 0.0, 0.0,
                1.0, 0.0, 0.0,
                float(dom_grav == 0), float(dom_grav == 1), float(dom_grav >= 2),
            ],
            dtype=np.float32,
        )
        color_t = torch.full((H_s, W_s), bg_perc, dtype=torch.int32, device=grid_dev)
        scale_r = max(1, H_s // max(1, H_g))
        scale_c = max(1, W_s // max(1, W_g))
        sub_h = min(H_s, H_g * scale_r)
        sub_w = min(W_s, W_g * scale_c)
        g_scaled = torch.nn.functional.interpolate(
            grid_t.unsqueeze(0).unsqueeze(0).float(),
            size=(sub_h, sub_w),
            mode="nearest",
        ).squeeze(0).squeeze(0).to(torch.int32)
        r_start = (H_s - sub_h) // 2
        c_start = (W_s - sub_w) // 2
        color_t[r_start:r_start + sub_h, c_start:c_start + sub_w] = g_scaled
        rgb_3ch = palette_t[torch.clamp(color_t.long(), 0, 9)]
        lum = 0.299 * rgb_3ch[..., 0] + 0.587 * rgb_3ch[..., 1] + 0.114 * rgb_3ch[..., 2]
        rgb_lum_t = torch.cat([rgb_3ch, lum.unsqueeze(-1)], dim=-1)
        tier_t = torch.zeros((H_s, W_s), dtype=torch.int32, device=grid_dev)
        left_ret = {
            "rgb_lum": rgb_lum_t,
            "color_idx": color_t,
            "tier_id": tier_t,
            "angular_pitch_deg": torch.zeros((H_s, W_s), dtype=torch.float32, device=grid_dev),
            "diagnostic_true_depth": torch.full((H_s, W_s), 0.52, dtype=torch.float32, device=grid_dev),
        }
        recon_mock = {
            "disparity_map": torch.zeros((H_s, W_s), dtype=torch.float32, device=grid_dev),
            "optical_flow_map": torch.zeros((H_s, W_s), dtype=torch.float32, device=grid_dev),
            "t_junction_map": torch.zeros((H_s, W_s), dtype=torch.float32, device=grid_dev),
            "z_fused": torch.full((H_s, W_s), 0.52, dtype=torch.float32, device=grid_dev),
            "perceived_desk_depth_m": torch.full((H_s, W_s), 0.52, dtype=torch.float32, device=grid_dev),
            "physical_elevation_m": torch.zeros((H_s, W_s), dtype=torch.float32, device=grid_dev),
        }
        return {
            "height_map": height_map_t.detach().cpu().numpy().astype(np.float32),
            "height_map_cuda": height_map_t,
            "contact_support_vec": c_vec,
            "tabletop_affordance_vec": affordance_16,
            "dominant_gravity_axis": dom_grav,
            "left_retinal_screen": left_ret,
            "right_retinal_screen": left_ret,
            "moved_retinal_screen": left_ret,
            "reconstruction_3d": recon_mock,
            "perceived_color_grid_cuda": color_t,
            "reconstructed_grid": g_arr.copy(),
            "num_mesh_triangles": 0,
            "desk_voxel_blocks_count": int(torch.sum(fg_t).item()),
            "fixed_eye_positions_xyz": {
                "left_eye": (-0.032, 0.0, 0.52),
                "right_eye": (0.032, 0.0, 0.52),
                "moved_eye": (0.010, 0.0, 0.52),
            },
            "isaaclab_seated_sync": True,
            "isaaclab_gaze_pitch_rad": -0.45,
        }

    def execute_physical_tabletop_stroke_on_desk(self, canvas_grid, stroke_cells, bg_color=0, step_physx=True):
        return {
            "strokes_executed": len(stroke_cells) if stroke_cells else 0,
            "contacted_cells": list(stroke_cells) if stroke_cells else [],
            "hand_world_xyz": self.effector_hand_world_xyz,
            "joint_targets_21d": self.last_seated_joint_targets_21d.tolist(),
        }

    def enter_safe_study_room_and_sit_at_desk(self, start_world_xyz=(101.0, 50.0, 0.0), n_walk_steps=6):
        return {
            "in_safe_study_room": True,
            "walked_into_room": True,
            "isaaclab_humanoid_seated_at_desk": True,
            "seated_joint_targets_21d": self.last_seated_joint_targets_21d.tolist(),
            "room_bounds_xyz": self.safe_room_bounds_xyz,
            "doorway_xyz": self.safe_room_doorway_xyz,
            "desk_world_xyz": self.safe_room_desk_world_xyz,
            "desk_surface_world_xyz": self.safe_room_desk_surface_world_xyz,
            "seated_world_xyz": (112.0, 49.70, 1.16),
            "seated_eye_world_xyz": (112.0, 49.82, 1.34),
            "room_hazard_level": 0.0,
            "predator_count_in_room": 0,
            "walk_trajectory_xyz": [[101.0, 50.0, 0.0], [112.0, 50.16, 1.49]],
        }

    def reset_humanoid_at_study_room_desk(self):
        return True

_mock_instance = MockWarpEngine()
mock_module = types.ModuleType("isaac_warp_3d_engine")
mock_module.get_global_isaac_warp_engine = lambda: _mock_instance
mock_module.IsaacWarp3DTabletopEngine = MockWarpEngine
mock_module.ARC_PALETTE_RGB = (
    (0.05, 0.05, 0.07),
    (0.00, 0.45, 0.85),
    (0.90, 0.15, 0.15),
    (0.18, 0.80, 0.25),
    (0.95, 0.85, 0.10),
    (0.65, 0.65, 0.68),
    (0.90, 0.20, 0.75),
    (0.95, 0.50, 0.10),
    (0.35, 0.80, 0.95),
    (0.55, 0.10, 0.20),
)
sys.modules["isaac_warp_3d_engine"] = mock_module

def setup_kaggle_environment():
    cwd = Path.cwd()
    search_roots = [Path("/kaggle/input"), Path("/kaggle/working"), cwd, cwd.parent]
    found_zips = []
    for root_dir in search_roots:
        if root_dir.exists():
            for root, dirs, files in os.walk(root_dir):
                for f in files:
                    if f.lower().endswith(".zip") and "submission" in f.lower():
                        found_zips.append(Path(root) / f)
                    elif f.lower().endswith(".zip") and not found_zips:
                        found_zips.append(Path(root) / f)
    for zpath in found_zips:
        try:
            with zipfile.ZipFile(zpath, "r") as zf:
                zf.extractall(cwd)
                print(f"Extracted: {zpath} -> {cwd}")
        except Exception as e:
            print(f"Zip extraction error for {zpath}: {e}")
    target_filename = "stage4_kaggle_submission_inference.py"
    target_dirs = set()
    for root_dir in search_roots:
        if root_dir.exists():
            for root, dirs, files in os.walk(root_dir):
                if target_filename in files:
                    target_dirs.add(Path(root).resolve())
    for td in sorted(target_dirs, key=lambda p: len(str(p))):
        p_str = str(td)
        if p_str not in sys.path:
            sys.path.insert(0, p_str)
            print(f"Added to sys.path: {p_str}")

setup_kaggle_environment()

from stage2_embodied_puzzle_solver import EmbodiedPuzzleSolver
_orig_shift = EmbodiedPuzzleSolver.shift_3d_head_tilt_viewpoint
def _safe_shift_3d_head_tilt_viewpoint(self, grid, surprise_val=0.5, fatigue_val=0.5):
    res = _orig_shift(self, grid, surprise_val=surprise_val, fatigue_val=fatigue_val)
    if isinstance(res, dict) and "tabletop_reprojection" in res:
        tr = res["tabletop_reprojection"]
        if isinstance(tr, dict):
            res["tabletop_reprojection"] = tr.get("reconstructed_grid", res.get("warped_grid", grid))
        elif not isinstance(tr, np.ndarray):
            res["tabletop_reprojection"] = res.get("warped_grid", grid)
    return res
EmbodiedPuzzleSolver.shift_3d_head_tilt_viewpoint = _safe_shift_3d_head_tilt_viewpoint

from stage4_kaggle_submission_inference import (
    LightweightSubmissionEngine,
    DEFAULT_ARC2_EVAL_DIR,
    DEFAULT_ARC1_EVAL_DIR,
    ARTIFACTS_DIR,
)

def find_kaggle_test_inputs():
    search_dir = Path("/kaggle/input")
    found_challenges = []
    found_sample_sub = None
    all_json_files = []
    if search_dir.exists():
        for root, dirs, files in os.walk(search_dir):
            for f in files:
                fp = Path(root) / f
                fl = f.lower()
                if fl.endswith(".json"):
                    all_json_files.append(fp)
                    if fl == "sample_submission.json":
                        found_sample_sub = fp
                    elif "test_challenges" in fl:
                        found_challenges.insert(0, fp)
                    elif "evaluation_challenges" in fl:
                        found_challenges.append(fp)
                    elif "challenges" in fl and "train" not in fl:
                        found_challenges.append(fp)
    print("Discovered JSON files under /kaggle/input:")
    for jf in all_json_files:
        print(f"  {jf}")
    selected_ch = found_challenges[0] if found_challenges else None
    return selected_ch, found_sample_sub

def main():
    t0 = time.time()
    ch_path, sample_sub_path = find_kaggle_test_inputs()
    print(f"Selected challenges path: {ch_path}")
    print(f"Sample submission path: {sample_sub_path}")
    
    engine = LightweightSubmissionEngine()
    out_file = Path("submission.json")
    submission_dict = {}

    challenges_data = {}
    if ch_path and ch_path.exists():
        with open(ch_path, "r", encoding="utf-8") as f:
            challenges_data = json.load(f)
        print(f"Loaded {len(challenges_data)} tasks from challenges file.")
    elif DEFAULT_ARC2_EVAL_DIR.exists():
        for jf in sorted(DEFAULT_ARC2_EVAL_DIR.glob("*.json")):
            with open(jf, "r", encoding="utf-8") as f:
                challenges_data[jf.stem] = json.load(f)
        print(f"Loaded {len(challenges_data)} tasks from DEFAULT_ARC2_EVAL_DIR.")
    elif DEFAULT_ARC1_EVAL_DIR.exists():
        for jf in sorted(DEFAULT_ARC1_EVAL_DIR.glob("*.json")):
            with open(jf, "r", encoding="utf-8") as f:
                challenges_data[jf.stem] = json.load(f)
        print(f"Loaded {len(challenges_data)} tasks from DEFAULT_ARC1_EVAL_DIR.")

    required_keys = set()
    sample_sub_data = {}
    if sample_sub_path and sample_sub_path.exists():
        try:
            with open(sample_sub_path, "r", encoding="utf-8") as f:
                sample_sub_data = json.load(f)
            required_keys = set(sample_sub_data.keys())
            print(f"Required task keys from sample submission: {len(required_keys)}")
        except Exception as e:
            print(f"Could not parse sample_submission: {e}")

    all_target_keys = sorted(set(challenges_data.keys()) | required_keys)
    print(f"Total target keys to process: {len(all_target_keys)}")

    for idx, tid in enumerate(all_target_keys):
        try:
            task = challenges_data.get(tid)
            if task is not None:
                attempts, info = engine.predict_task(task, max_steps=8)
                submission_dict[str(tid)] = attempts
                exact_flag = info.get("solved_train_exact", False)
                steps = info.get("steps_taken", 0)
                if idx < 10 or (idx + 1) % 10 == 0:
                    print(f"[{idx+1}/{len(all_target_keys)}] Task {tid}: steps={steps}, train_exact={exact_flag}")
            elif tid in sample_sub_data:
                submission_dict[str(tid)] = sample_sub_data[tid]
            else:
                submission_dict[str(tid)] = [{"attempt_1": [[0, 0], [0, 0]], "attempt_2": [[0, 0], [0, 0]]}]
        except Exception as e:
            print(f"[{idx+1}/{len(all_target_keys)}] Task {tid} fallback due to: {e}")
            if tid in sample_sub_data:
                submission_dict[str(tid)] = sample_sub_data[tid]
            else:
                submission_dict[str(tid)] = [{"attempt_1": [[0, 0], [0, 0]], "attempt_2": [[0, 0], [0, 0]]}]

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(submission_dict, f)

    elapsed = round(time.time() - t0, 2)
    sub_size_kb = round(out_file.stat().st_size / 1024, 2) if out_file.exists() else 0.0
    print(f"DONE: Generated submission.json with {len(submission_dict)} tasks in {elapsed}s ({sub_size_kb} KB)")

if __name__ == "__main__":
    main()
