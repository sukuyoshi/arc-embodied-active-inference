import argparse
import concurrent.futures
import ctypes
import gc
import json
import os
import shutil
import time
import urllib.request
from pathlib import Path
import numpy as np


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

try:
    import torch
except ImportError:
    torch = None

from stage1_isaac3d_embodied_arena import evolve_infinite_procedural_hazard_manifold
from stage2_embodied_puzzle_solver import (
    ARTIFACTS_DIR,
    EmbodiedPuzzleSolver,
)
from million_scale_arc_harvester import MillionScaleARCMultiverseHarvester
from isaac_warp_3d_engine import get_global_isaac_warp_engine

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_TRAIN_DIR = REPO_ROOT / "arc_repo" / "data" / "training"
DATA_EVAL_DIR = REPO_ROOT / "arc_repo" / "data" / "evaluation"
EXTERNAL_CACHE_DIR = ARTIFACTS_DIR / "external_arc_databank"
CHECKPOINTS_DIR = ARTIFACTS_DIR / "checkpoints"


def cleanup_neural_memory():
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def make_blind_test_task(task):
    raw_test = task.get("test", [])
    if isinstance(raw_test, dict):
        blind_test = {"input": raw_test["input"]}
    else:
        blind_test = [{"input": tp["input"]} for tp in raw_test]
    blind = dict(task)
    blind["test"] = blind_test
    return blind


class Physical3DSimulator:
    def __init__(self, grid_size=10):
        self.grid_size = grid_size
        self.last_reconstruction_3d = None

    def project_3d_to_2d(self, objects):
        engine = get_global_isaac_warp_engine()
        grid_gpu, _, recon = engine.project_entities_3d_via_human_vision(
            objects, grid_size=self.grid_size, color_key="color"
        )
        self.last_reconstruction_3d = recon
        return grid_gpu.copy()

    def simulate_tabletop_rollout(self, grid, sim_type="gravity", steps=15):
        engine = get_global_isaac_warp_engine()
        out_grid, perc = engine.simulate_3d_rigid_body_tabletop_Rollout(grid, sim_type=sim_type, steps=steps)
        return out_grid

    def step_physical_interaction(self, objects, action_type, delta):
        engine = get_global_isaac_warp_engine()
        return engine.simulate_entities_3d_rigid_body_warp(
            objects,
            action_type=action_type,
            delta=delta,
        )


def build_physical_core_knowledge_tasks():
    sim = Physical3DSimulator(grid_size=10)
    tasks = []

    t1_train = []
    for c_obj in [1, 2, 3]:
        objs = [
            {"color": c_obj, "pos": [-3.5, 0.0, 7.0], "size": [1.5, 1.5, 1.5]},
            {"color": 4, "pos": [0.0, 0.0, 4.0], "size": [3.0, 3.0, 1.0]},
        ]
        g_in = sim.project_3d_to_2d(objs)
        g_out = sim.project_3d_to_2d(sim.step_physical_interaction(objs, "occlude_pass", [3.5, 0.0]))
        t1_train.append({"input": g_in.tolist(), "output": g_out.tolist()})
    t1_test_objs = [
        {"color": 5, "pos": [-3.5, 0.0, 7.0], "size": [1.5, 1.5, 1.5]},
        {"color": 6, "pos": [0.0, 0.0, 4.0], "size": [3.0, 3.0, 1.0]},
    ]
    tasks.append({
        "id": "Phys_T1",
        "name": "Occlusion & Object Persistence (Behind Wall)",
        "train": t1_train,
        "test": {
            "input": sim.project_3d_to_2d(t1_test_objs).tolist(),
            "output": sim.project_3d_to_2d(sim.step_physical_interaction(t1_test_objs, "occlude_pass", [3.5, 0.0])).tolist(),
        },
    })

    t2_train = []
    for c1, c2 in [(1, 3), (2, 4), (7, 8)]:
        objs = [
            {"color": c1, "pos": [-3.0, -1.0, 5.0], "size": [1.5, 1.5, 1.5]},
            {"color": c2, "pos": [-0.5, -1.0, 5.0], "size": [1.5, 1.5, 1.5]},
        ]
        g_in = sim.project_3d_to_2d(objs)
        g_out = sim.project_3d_to_2d(sim.step_physical_interaction(objs, "push", [3.0, 0.0]))
        t2_train.append({"input": g_in.tolist(), "output": g_out.tolist()})
    t2_test_objs = [
        {"color": 5, "pos": [-3.0, -1.0, 5.0], "size": [1.5, 1.5, 1.5]},
        {"color": 2, "pos": [-0.5, -1.0, 5.0], "size": [1.5, 1.5, 1.5]},
    ]
    tasks.append({
        "id": "Phys_T2",
        "name": "Contact Push Cascade Dynamics",
        "train": t2_train,
        "test": {
            "input": sim.project_3d_to_2d(t2_test_objs).tolist(),
            "output": sim.project_3d_to_2d(sim.step_physical_interaction(t2_test_objs, "push", [3.0, 0.0])).tolist(),
        },
    })

    t3_train = []
    for c in [1, 2, 3]:
        objs = [{"color": c, "pos": [0.0, 0.0, 5.0], "size": [3.5, 1.5, 1.0]}]
        g_in = sim.project_3d_to_2d(objs)
        g_out = sim.project_3d_to_2d(sim.step_physical_interaction(objs, "rotate_3d", np.pi * 0.5))
        t3_train.append({"input": g_in.tolist(), "output": g_out.tolist()})
    t3_test_objs = [{"color": 6, "pos": [0.0, 0.0, 5.0], "size": [3.5, 1.5, 1.0]}]
    tasks.append({
        "id": "Phys_T3",
        "name": "3D Rigid Rotation & 2D Projection",
        "train": t3_train,
        "test": {
            "input": sim.project_3d_to_2d(t3_test_objs).tolist(),
            "output": sim.project_3d_to_2d(sim.step_physical_interaction(t3_test_objs, "rotate_3d", np.pi * 0.5)).tolist(),
        },
    })

    t4_train = []
    for c1, c2 in [(1, 4), (2, 5), (3, 7)]:
        objs = [
            {"color": c1, "pos": [0.0, -3.5, 5.0], "size": [2.0, 1.0, 1.0]},
            {"color": c2, "pos": [0.0, 2.5, 5.0], "size": [1.5, 1.5, 1.5]},
        ]
        g_in = sim.project_3d_to_2d(objs)
        g_out = sim.project_3d_to_2d(sim.step_physical_interaction(objs, "gravity_fall", None))
        t4_train.append({"input": g_in.tolist(), "output": g_out.tolist()})
    t4_test_objs = [
        {"color": 8, "pos": [0.0, -3.5, 5.0], "size": [2.0, 1.0, 1.0]},
        {"color": 3, "pos": [0.0, 2.5, 5.0], "size": [1.5, 1.5, 1.5]},
    ]
    tasks.append({
        "id": "Phys_T4",
        "name": "Gravitational Stacking & Physical Support",
        "train": t4_train,
        "test": {
            "input": sim.project_3d_to_2d(t4_test_objs).tolist(),
            "output": sim.project_3d_to_2d(sim.step_physical_interaction(t4_test_objs, "gravity_fall", None)).tolist(),
        },
    })

    t5_train = []
    for c_base, c_top in [(1, 3), (2, 4), (5, 6)]:
        objs = [
            {"color": c_base, "pos": [0.0, 0.0, 1.0], "size": [4.0, 1.0, 1.0]},
            {"color": c_top, "pos": [1.5, 0.0, 3.0], "size": [1.5, 1.0, 2.5]},
        ]
        g_in = sim.project_3d_to_2d(objs)
        g_out = sim.project_3d_to_2d(sim.step_physical_interaction(objs, "gravity_fall", None))
        t5_train.append({"input": g_in.tolist(), "output": g_out.tolist()})
    t5_test_objs = [
        {"color": 7, "pos": [0.0, 0.0, 1.0], "size": [4.0, 1.0, 1.0]},
        {"color": 8, "pos": [1.5, 0.0, 3.0], "size": [1.5, 1.0, 2.5]},
    ]
    tasks.append({
        "id": "Phys_T5",
        "name": "Rigid Structure Collapse & Equilibrium",
        "train": t5_train,
        "test": {
            "input": sim.project_3d_to_2d(t5_test_objs).tolist(),
            "output": sim.project_3d_to_2d(sim.step_physical_interaction(t5_test_objs, "gravity_fall", None)).tolist(),
        },
    })

    return tasks


def evaluate_physical_suite(solver=None):
    phys_tasks = build_physical_core_knowledge_tasks()
    embodied_solver = (
        solver
        if solver is not None
        else EmbodiedPuzzleSolver(use_pain=True, use_curiosity=True, use_mortality=True, plastic=False)
    )
    results = []

    for t in phys_tasks:
        blind_t = make_blind_test_task(t)
        res = embodied_solver.study_and_solve_tabletop_task(blind_t, max_steps=6)
        test_out = np.array(t["test"]["output"], dtype=np.int32)
        pred = res["prediction"]
        if pred.shape == test_out.shape:
            pix_acc = float(np.mean(pred == test_out) * 100.0)
            exact = bool(np.array_equal(pred, test_out))
        else:
            pix_acc = 0.0
            exact = False
        results.append({
            "task_id": t["id"],
            "name": t["name"],
            "pixel_acc_pct": round(pix_acc, 2),
            "exact_match": exact,
            "program": res["program"],
            "winning_axis": res["winning_axis"],
            "steps_taken": res["steps_taken"],
        })
    return results


def evaluate_official_arc_dataset(dataset_dir, limit=400, max_steps=6, full_solver=None, nopain_solver=None):
    json_files = sorted(dataset_dir.glob("*.json"))[:limit]
    if full_solver is None:
        full_solver = EmbodiedPuzzleSolver(use_pain=True, use_curiosity=True, use_mortality=True, plastic=False)
    if nopain_solver is None:
        nopain_solver = EmbodiedPuzzleSolver(use_pain=False, use_curiosity=False, use_mortality=False, plastic=False)

    full_solved_test = []
    full_solved_train = []
    nopain_solved_test = []
    axis_distribution = {i: 0 for i in range(8)}
    step_counts_full = []
    step_counts_nopain = []
    spike_rates_solved = []
    thought_energy_solved = []
    spike_rates_unsolved = []
    thought_energy_unsolved = []
    hippocampal_replay_solves = []
    inner_monologue_solves = []
    hierarchical_macro_solves = []
    dream_mutation_solves = []
    social_empathy_solves = []
    bred_macro_tool_solves = []
    nested_tool_solves = []
    conditional_monologue_solves = []
    nested_conditional_solves = []
    specialist_leader_solves = []
    deep_macro_brake_solves = []

    for jf in json_files:
        with open(jf, "r", encoding="utf-8") as f:
            task = json.load(f)

        test_pair = task["test"][0] if isinstance(task["test"], list) else task["test"]
        test_gt = np.array(test_pair["output"], dtype=np.int32)
        blind_task = make_blind_test_task(task)

        res_full = full_solver.study_and_solve_tabletop_task(blind_task, max_steps=max_steps)
        if res_full["solved_train_exact"]:
            full_solved_train.append(jf.stem)
            step_counts_full.append(res_full["steps_taken"])
            spike_rates_solved.append(res_full.get("mean_spike_firing_rate", 0.0))
            thought_energy_solved.append(res_full.get("thought_energy_consumed", 0.0))
            pred_f = res_full["prediction"]
            if pred_f.shape == test_gt.shape and np.array_equal(pred_f, test_gt):
                if res_full.get("hippocampal_replay_used", False):
                    hippocampal_replay_solves.append(jf.stem)
                if res_full.get("inner_monologue_active", False):
                    inner_monologue_solves.append(jf.stem)
                if res_full.get("hierarchical_macro_active", False):
                    hierarchical_macro_solves.append(jf.stem)
                if res_full.get("dream_mutation_used", False):
                    dream_mutation_solves.append(jf.stem)
                if res_full.get("social_empathy_used", False):
                    social_empathy_solves.append(jf.stem)
                if res_full.get("bred_macro_tool_used", False):
                    bred_macro_tool_solves.append(jf.stem)
                if res_full.get("nested_tool_used", False):
                    nested_tool_solves.append(jf.stem)
                if res_full.get("conditional_monologue_used", False):
                    conditional_monologue_solves.append(jf.stem)
                if res_full.get("nested_conditional_used", False):
                    nested_conditional_solves.append(jf.stem)
                if res_full.get("specialist_leader_used", False):
                    specialist_leader_solves.append(jf.stem)
                if res_full.get("deep_macro_brake_used", False):
                    deep_macro_brake_solves.append(jf.stem)
                full_solved_test.append({
                    "task_id": jf.stem,
                    "program": res_full["program"],
                    "axis": res_full["winning_axis"],
                    "steps": res_full["steps_taken"],
                    "spike_rate": round(res_full.get("mean_spike_firing_rate", 0.0), 4),
                    "thought_energy": round(res_full.get("thought_energy_consumed", 0.0), 4),
                    "hippocampal_replay": bool(res_full.get("hippocampal_replay_used", False)),
                    "inner_monologue": bool(res_full.get("inner_monologue_active", False)),
                    "hierarchical_macro": bool(res_full.get("hierarchical_macro_active", False)),
                    "dream_mutation": bool(res_full.get("dream_mutation_used", False)),
                    "social_empathy": bool(res_full.get("social_empathy_used", False)),
                    "bred_macro_tool": bool(res_full.get("bred_macro_tool_used", False)),
                    "nested_tool": bool(res_full.get("nested_tool_used", False)),
                    "conditional_monologue": bool(res_full.get("conditional_monologue_used", False)),
                    "nested_conditional": bool(res_full.get("nested_conditional_used", False)),
                    "specialist_leader": bool(res_full.get("specialist_leader_used", False)),
                    "deep_macro_brake": bool(res_full.get("deep_macro_brake_used", False)),
                })
                if res_full["winning_axis"] in axis_distribution:
                    axis_distribution[res_full["winning_axis"]] += 1
        else:
            spike_rates_unsolved.append(res_full.get("mean_spike_firing_rate", 0.0))
            thought_energy_unsolved.append(res_full.get("thought_energy_consumed", 0.0))

        res_np = nopain_solver.study_and_solve_tabletop_task(blind_task, max_steps=max_steps)
        if res_np["solved_train_exact"]:
            step_counts_nopain.append(res_np["steps_taken"])
            pred_np = res_np["prediction"]
            if pred_np.shape == test_gt.shape and np.array_equal(pred_np, test_gt):
                nopain_solved_test.append(jf.stem)

    mean_tool_efe = (
        float(np.mean([t.get("efe_efficiency", 0.5) for t in full_solver.cultural_tool_library]))
        if full_solver.cultural_tool_library
        else 0.0
    )

    return {
        "total_tasks_evaluated": len(json_files),
        "Embodied_Mortal_Pain_Curiosity": {
            "train_exact_solved_count": len(full_solved_train),
            "test_exact_solved_count": len(full_solved_test),
            "test_solve_rate_pct": round(len(full_solved_test) / max(1, len(json_files)) * 100.0, 2),
            "mean_steps_to_solve": round(float(np.mean(step_counts_full)) if step_counts_full else 0.0, 2),
            "mean_spike_firing_rate_solved": round(float(np.mean(spike_rates_solved)) if spike_rates_solved else 0.0, 4),
            "mean_thought_energy_solved": round(float(np.mean(thought_energy_solved)) if thought_energy_solved else 0.0, 4),
            "mean_spike_firing_rate_unsolved": round(float(np.mean(spike_rates_unsolved)) if spike_rates_unsolved else 0.0, 4),
            "mean_thought_energy_unsolved": round(float(np.mean(thought_energy_unsolved)) if thought_energy_unsolved else 0.0, 4),
            "hippocampal_replay_solved_count": len(hippocampal_replay_solves),
            "hippocampal_replay_solved_ids": hippocampal_replay_solves,
            "inner_monologue_solved_count": len(inner_monologue_solves),
            "inner_monologue_solved_ids": inner_monologue_solves,
            "hierarchical_macro_solved_count": len(hierarchical_macro_solves),
            "hierarchical_macro_solved_ids": hierarchical_macro_solves,
            "dream_mutation_solved_count": len(dream_mutation_solves),
            "dream_mutation_solved_ids": dream_mutation_solves,
            "social_empathy_solved_count": len(social_empathy_solves),
            "social_empathy_solved_ids": social_empathy_solves,
            "bred_macro_tool_solved_count": len(bred_macro_tool_solves),
            "bred_macro_tool_solved_ids": bred_macro_tool_solves,
            "nested_tool_solved_count": len(nested_tool_solves),
            "nested_tool_solved_ids": nested_tool_solves,
            "conditional_monologue_solved_count": len(conditional_monologue_solves),
            "conditional_monologue_solved_ids": conditional_monologue_solves,
            "nested_conditional_solved_count": len(nested_conditional_solves),
            "nested_conditional_solved_ids": nested_conditional_solves,
            "specialist_leader_solved_count": len(specialist_leader_solves),
            "specialist_leader_solved_ids": specialist_leader_solves,
            "deep_macro_brake_solved_count": len(deep_macro_brake_solves),
            "deep_macro_brake_solved_ids": deep_macro_brake_solves,
            "social_empathy_events_total": int(full_solver.social_empathy_events_total),
            "specialist_leader_elections_total": int(full_solver.specialist_leader_elections_total),
            "if_then_else_switches_total": int(full_solver.if_then_else_switches_total),
            "nested_if_then_switches_total": int(full_solver.nested_if_then_switches_total),
            "cultural_tools_accumulated": int(len(full_solver.cultural_tool_library)),
            "tools_pruned_total": int(full_solver.tools_pruned_total),
            "pruning_cycles_run": int(full_solver.pruning_cycles_run),
            "mean_tool_efe_efficiency": round(mean_tool_efe, 4),
            "bred_macro_recipes_count": int(len(full_solver.bred_macro_recipes)),
            "nested_macro_recipes_count": int(len(full_solver.nested_macro_recipes)),
            "axis_solve_distribution": axis_distribution,
            "solved_tasks_sample": full_solved_test[:25],
        },
        "Disembodied_NoPain_NoCuriosity_Control": {
            "test_exact_solved_count": len(nopain_solved_test),
            "test_solve_rate_pct": round(len(nopain_solved_test) / max(1, len(json_files)) * 100.0, 2),
            "mean_steps_to_solve": round(float(np.mean(step_counts_nopain)) if step_counts_nopain else 0.0, 2),
        },
    }


class ExternalARCDatabankConnector:
    def __init__(self):
        self.cache_dir = EXTERNAL_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.arc2_eval_dir = self.cache_dir / "arc_agi_2_eval"
        self.arc2_train_dir = self.cache_dir / "arc_agi_2_train"
        self.arc2_eval_dir.mkdir(parents=True, exist_ok=True)
        self.arc2_train_dir.mkdir(parents=True, exist_ok=True)
        self.multiverse_harvester = MillionScaleARCMultiverseHarvester()

    def fetch_github_arc_agi_2_split(self, split="evaluation", limit=120):
        return self.multiverse_harvester.fetch_github_arc_agi_2_split(split=split, limit=limit)

    def load_local_external_bank(self, folder_path, source_label, limit=150):
        if not folder_path.exists():
            return []
        loaded = []
        for jf in sorted(folder_path.glob("*.json"))[:limit]:
            try:
                with open(jf, "r", encoding="utf-8") as f:
                    loaded.append({
                        "source_bank": source_label,
                        "task_id": jf.stem,
                        "task": json.load(f),
                    })
            except Exception:
                continue
        return loaded

    def stream_hunting_grounds(self, generation_v2=1):
        return self.multiverse_harvester.stream_generational_hunting_grounds(
            generation_v2=generation_v2,
            world_static_slice_size=220,
            multiverse_stream_size=140,
        )


def run_procedural_tabletop_practice(solver, harvester, n_tasks=48, max_steps=6, generation_seed=1, max_depth=6):
    batch = harvester.materialize_million_stream_batch(batch_id=int(generation_seed), batch_size=int(n_tasks))
    solved_cnt = 0
    chaos_cnt = 0
    chaos_solved = 0
    orig_max_depth = getattr(solver, "max_program_depth", 4)
    if hasattr(solver, "max_program_depth"):
        solver.max_program_depth = int(max_depth)
    for idx_b, item in enumerate(batch[:n_tasks]):
        task_obj = item.get("task", {})
        if not harvester._is_valid_arc_task(task_obj):
            continue
        is_chaos = (idx_b % 3 == 0)
        blind_task = make_blind_test_task(task_obj)
        if is_chaos:
            chaos_cnt += 1
            for tp in blind_task.get("train", []):
                inp = np.array(tp.get("input", []), dtype=np.int32)
                if inp.size > 0:
                    rr = int(idx_b % inp.shape[0])
                    cc = int((idx_b * 3) % inp.shape[1])
                    inp[rr, cc] = int((inp[rr, cc] + 1) % 10)
                    tp["input"] = inp.tolist()
        res = solver.study_and_solve_tabletop_task(blind_task, max_steps=max_steps)
        test_pair = task_obj["test"][0] if isinstance(task_obj["test"], list) else task_obj["test"]
        gt = np.array(test_pair["output"], dtype=np.int32)
        pred = res.get("prediction")
        if pred is not None and pred.shape == gt.shape and np.array_equal(pred, gt):
            solved_cnt += 1
            if is_chaos:
                chaos_solved += 1
    if hasattr(solver, "max_program_depth"):
        solver.max_program_depth = orig_max_depth
    total_eval = max(1, len(batch[:n_tasks]))
    return {
        "self_play_tasks_evaluated": int(len(batch[:n_tasks])),
        "self_play_solved_count": int(solved_cnt),
        "self_play_solve_rate_pct": round(float(solved_cnt) / float(total_eval) * 100.0, 2),
        "self_play_max_depth": int(max_depth),
        "glaciation_chaos_tasks_count": int(chaos_cnt),
        "glaciation_chaos_solved_count": int(chaos_solved),
    }


def run_tabletop_databank_expedition(solver, task_stream, max_steps=6, freeze_plastic=False):
    prev_plastic = bool(solver.plastic)
    if freeze_plastic:
        solver.plastic = False
    breakdown = {}
    total_tr = 0
    total_te = 0
    sample_solves = []
    streamed = 0
    try:
        for item in task_stream:
            src_bank = str(item.get("source_bank", "ExternalBank"))
            tid = str(item.get("task_id", "task"))
            task_obj = item.get("task", {})
            if not isinstance(task_obj, dict) or not task_obj.get("train") or task_obj.get("test") is None:
                continue
            streamed += 1
            if src_bank not in breakdown:
                breakdown[src_bank] = {
                    "tasks_evaluated": 0,
                    "train_exact_solved": 0,
                    "test_exact_solved": 0,
                    "solve_rate_pct": 0.0,
                }
            breakdown[src_bank]["tasks_evaluated"] += 1
            is_eval_bank = ("evaluation" in src_bank.lower())
            saved_p = bool(solver.plastic)
            if is_eval_bank:
                solver.plastic = False
            try:
                blind_task = make_blind_test_task(task_obj)
                res = solver.study_and_solve_tabletop_task(blind_task, max_steps=max_steps)
            finally:
                solver.plastic = saved_p
            if res.get("solved_train_exact", False):
                total_tr += 1
                breakdown[src_bank]["train_exact_solved"] += 1
            test_pair = task_obj["test"][0] if isinstance(task_obj["test"], list) else task_obj["test"]
            if isinstance(test_pair, dict) and "output" in test_pair:
                gt = np.array(test_pair["output"], dtype=np.int32)
                pred = res.get("prediction")
                if pred is not None and pred.shape == gt.shape and np.array_equal(pred, gt):
                    total_te += 1
                    breakdown[src_bank]["test_exact_solved"] += 1
                    if len(sample_solves) < 25:
                        sample_solves.append({
                            "source_bank": src_bank,
                            "task_id": tid,
                            "program": res.get("program", []),
                        })
        for bk in breakdown.values():
            bk["solve_rate_pct"] = round(
                float(bk["test_exact_solved"]) / float(max(1, bk["tasks_evaluated"])) * 100.0,
                2,
            )
    finally:
        solver.plastic = prev_plastic
    return {
        "total_external_tasks_streamed": int(streamed),
        "total_train_exact_solved": int(total_tr),
        "total_test_exact_solved": int(total_te),
        "overall_external_solve_rate_pct": round(float(total_te) / float(max(1, streamed)) * 100.0, 2),
        "databank_breakdown": breakdown,
        "sample_external_solves": sample_solves,
    }


def snapshot_generation_manifold(solver, generation_v2, is_new_best=False, extra_meta=None):
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    latest_path = CHECKPOINTS_DIR / "checkpoint_latest.npz"
    solver.save_evolution_checkpoint(latest_path, generation_v2=generation_v2, extra_meta=extra_meta)

    if is_new_best:
        best_path = CHECKPOINTS_DIR / "checkpoint_best.npz"
        solver.save_evolution_checkpoint(best_path, generation_v2=generation_v2, extra_meta=extra_meta)

    if generation_v2 == 1 or generation_v2 % 10 == 0 or generation_v2 % 100 == 0 or generation_v2 % 1000 == 0:
        gen_tag = f"checkpoint_gen{generation_v2}" if generation_v2 >= 1000 else f"checkpoint_gen{generation_v2:04d}"
        gen_path = CHECKPOINTS_DIR / f"{gen_tag}.npz"
        solver.save_evolution_checkpoint(gen_path, generation_v2=generation_v2, extra_meta=extra_meta)
        isaac_src = ARTIFACTS_DIR / "stage1_isaac3d_core_manifold.npz"
        if isaac_src.exists():
            shutil.copy2(isaac_src, CHECKPOINTS_DIR / f"stage1_isaac3d_gen{generation_v2}.npz")

    return str(latest_path)


def run_full_benchmark_once():
    t0 = time.time()
    route1_hazard_report = evolve_infinite_procedural_hazard_manifold(n_cycles=12, steps_per_cycle=140, seed=2026)
    cleanup_neural_memory()

    shared_full_solver = EmbodiedPuzzleSolver(use_pain=True, use_curiosity=True, use_mortality=True, plastic=True)
    shared_nopain_solver = EmbodiedPuzzleSolver(use_pain=False, use_curiosity=False, use_mortality=False, plastic=True)
    latest_ckpt = CHECKPOINTS_DIR / "checkpoint_latest.npz"
    if latest_ckpt.exists():
        shared_full_solver.load_evolution_checkpoint(latest_ckpt)
        shared_nopain_solver.load_evolution_checkpoint(latest_ckpt)
    shared_full_solver.absorb_stage1_3d_embodied_manifold()
    shared_nopain_solver.absorb_stage1_3d_embodied_manifold()

    phys_results = evaluate_physical_suite(solver=shared_full_solver)
    phys_exact_rate = float(np.mean([1.0 if r["exact_match"] else 0.0 for r in phys_results]) * 100.0)
    phys_pix_acc = float(np.mean([r["pixel_acc_pct"] for r in phys_results]))
    cleanup_neural_memory()

    connector = ExternalARCDatabankConnector()
    route2_self_play_report = run_procedural_tabletop_practice(
        shared_full_solver, connector.multiverse_harvester, n_tasks=72, max_steps=6, generation_seed=1
    )
    _ = run_procedural_tabletop_practice(
        shared_nopain_solver, connector.multiverse_harvester, n_tasks=72, max_steps=6, generation_seed=1
    )
    cleanup_neural_memory()

    arc_train_eval = evaluate_official_arc_dataset(
        DATA_TRAIN_DIR, limit=400, max_steps=6, full_solver=shared_full_solver, nopain_solver=shared_nopain_solver
    )
    cleanup_neural_memory()

    dream_replay_stats = shared_full_solver.offline_dream_replay(n_dream_cycles=15, dream_lr=0.012)
    _ = shared_nopain_solver.offline_dream_replay(n_dream_cycles=15, dream_lr=0.012)
    cleanup_neural_memory()

    external_train_stream = connector.stream_hunting_grounds()
    route3_train_rep = run_tabletop_databank_expedition(shared_full_solver, external_train_stream, max_steps=6)
    external_train_stream_np = connector.stream_hunting_grounds()
    _ = run_tabletop_databank_expedition(shared_nopain_solver, external_train_stream_np, max_steps=6)
    cleanup_neural_memory()

    prev_p_eval = bool(shared_full_solver.plastic)
    prev_np_p_eval = bool(shared_nopain_solver.plastic)
    shared_full_solver.plastic = False
    shared_nopain_solver.plastic = False
    try:
        arc_eval_eval = evaluate_official_arc_dataset(
            DATA_EVAL_DIR, limit=400, max_steps=6, full_solver=shared_full_solver, nopain_solver=shared_nopain_solver
        )
        arc2_eval_stream = connector.fetch_github_arc_agi_2_split(split="evaluation", limit=120)
        route3_eval_rep = run_tabletop_databank_expedition(
            shared_full_solver, arc2_eval_stream, max_steps=6, freeze_plastic=True
        )
    finally:
        shared_full_solver.plastic = prev_p_eval
        shared_nopain_solver.plastic = prev_np_p_eval
    cleanup_neural_memory()

    merged_bd_once = dict(route3_train_rep.get("databank_breakdown", {}))
    merged_bd_once.update(route3_eval_rep.get("databank_breakdown", {}))
    route3_external_report = dict(route3_train_rep)
    route3_external_report["databank_breakdown"] = merged_bd_once

    arc2_bd = route3_external_report.get("databank_breakdown", {}).get("GitHub_arcprize_ARC_AGI_2_evaluation", {})
    snapshot_generation_manifold(
        shared_full_solver,
        generation_v2=1,
        is_new_best=True,
        extra_meta={
            "arc2_eval_solved": arc2_bd.get("test_exact_solved", 0),
            "arc2_eval_solve_rate_pct": arc2_bd.get("solve_rate_pct", 0.0),
            "arc1_eval_solved": arc_eval_eval["Embodied_Mortal_Pain_Curiosity"]["test_exact_solved_count"],
        },
    )

    report = {
        "elapsed_sec": round(time.time() - t0, 2),
        "World_Expansion_Routes": {
            "Route1_Stage1_Infinite_Procedural_Hazards": route1_hazard_report,
            "Route2_Procedural_ARC_Self_Play_Breeding": route2_self_play_report,
            "Route3_External_Infinite_ARC_Databank_Expedition": route3_external_report,
        },
        "Physical_3D_to_2D_Benchmark": {
            "embodied_stage2_exact_match_pct": round(phys_exact_rate, 2),
            "embodied_stage2_mean_pixel_acc_pct": round(phys_pix_acc, 2),
            "tasks": phys_results,
        },
        "Offline_Sleep_Dream_Replay_Consolidation": dream_replay_stats,
        "Official_ARC_Training_Set_400": arc_train_eval,
        "Official_ARC_Evaluation_Set_400": arc_eval_eval,
    }

    out_path = ARTIFACTS_DIR / "stage3_arc_benchmark_metrics.json"
    tmp_out = ARTIFACTS_DIR / "stage3_arc_benchmark_metrics_tmp.json"
    with open(tmp_out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    tmp_out.replace(out_path)

    print(json.dumps(report, indent=2))
    return report


def run_infinite_evolution_loop(max_generations=None, hazard_cycles=6, self_play_tasks=48, dream_cycles=15):
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    status_path = ARTIFACTS_DIR / "infinite_evolution_status.json"
    metrics_path = ARTIFACTS_DIR / "stage3_arc_benchmark_metrics.json"

    solver = EmbodiedPuzzleSolver(use_pain=True, use_curiosity=True, use_mortality=True, plastic=True)
    connector = ExternalARCDatabankConnector()

    generation_v2 = 0
    best_arc2_solved = 0
    best_train_solved = 0
    best_efe_efficiency = 0.0
    history_tail = []

    latest_ckpt = CHECKPOINTS_DIR / "checkpoint_latest.npz"
    if latest_ckpt.exists():
        solver.load_evolution_checkpoint(latest_ckpt)
        latest_json = latest_ckpt.with_suffix(".json")
        if latest_json.exists():
            try:
                with open(latest_json, "r", encoding="utf-8") as f:
                    meta_prev = json.load(f)
                generation_v2 = int(meta_prev.get("generation_v2", 0))
                best_arc2_solved = int(meta_prev.get("extra_meta", {}).get("best_arc2_solved", 0))
                best_train_solved = int(meta_prev.get("extra_meta", {}).get("best_train_solved", 0))
                best_efe_efficiency = float(meta_prev.get("extra_meta", {}).get("best_efe_efficiency", 0.0))
            except Exception:
                pass

    if status_path.exists():
        try:
            with open(status_path, "r", encoding="utf-8") as f:
                st_prev = json.load(f)
            last_completed_gen = int(st_prev.get("latest_generation_summary", {}).get("generation_v2", 0))
            generation_v2 = max(generation_v2, last_completed_gen)
            best_arc2_solved = max(best_arc2_solved, int(st_prev.get("best_arc2_eval_solved", 0)))
            best_train_solved = max(best_train_solved, int(st_prev.get("best_train_solved", 0)))
            best_efe_efficiency = max(best_efe_efficiency, float(st_prev.get("best_mean_tool_efe_efficiency", 0.0)))
            history_tail = list(st_prev.get("recent_generations", []))[-24:]
        except Exception:
            pass

    last_route1_rep = {}
    gens_completed = 0
    while max_generations is None or max_generations <= 0 or gens_completed < max_generations:
        gen_t0 = time.time()
        target_gen = generation_v2 + 1

        def update_heartbeat(phase_name):
            hb = {
                "daemon_state": "RUNNING_IMMORTAL_EVOLUTION",
                "pid": int(os.getpid()),
                "active_generation_v2": int(target_gen),
                "active_phase": str(phase_name),
                "heartbeat_timestamp": round(time.time(), 2),
                "generation_v2": int(generation_v2 if phase_name != "GENERATION_COMPLETED" else target_gen),
                "best_arc2_eval_solved": int(best_arc2_solved),
                "best_arc2_eval_solve_rate_pct": round(best_arc2_solved / 120.0 * 100.0, 2),
                "best_mean_tool_efe_efficiency": round(best_efe_efficiency, 4),
                "current_cultural_tools_count": int(len(solver.cultural_tool_library)),
                "current_bred_recipes_count": int(len(solver.bred_macro_recipes)),
                "current_nested_recipes_count": int(len(solver.nested_macro_recipes)),
                "tools_pruned_cumulative": int(solver.tools_pruned_total),
                "relational_bindings_cumulative": int(getattr(solver, "relational_bindings_total", 0)),
                "recursive_chunks_compiled_total": int(getattr(solver, "recursive_chunks_compiled_total", 0)),
                "canvas_stitch_solves_total": int(getattr(solver, "canvas_stitch_solves_total", 0)),
                "micro_primitives_synthesized_total": int(getattr(solver, "micro_primitives_synthesized_total", 0)),
                "residual_subgoals_harvested_total": int(getattr(solver, "residual_subgoals_harvested_total", 0)),
                "residual_bridges_spliced_total": int(getattr(solver, "residual_bridges_spliced_total", 0)),
                "evolved_ast_organs_count": int(len(getattr(solver, "evolved_ast_organs", []))),
                "de_novo_codons_synthesized_count": int(len(getattr(solver, "de_novo_codon_pool", []))),
                "max_organ_codon_depth": int(max([len(o.get("codons", [])) for o in getattr(solver, "evolved_ast_organs", [])] + [0])),
                "evolving_stepping_stone_tasks_count": int(len(getattr(solver, "evolving_genome_archive", {}))),
                "ast_mutations_total": int(getattr(solver, "ast_mutations_total", 0)),
                "ast_crossovers_total": int(getattr(solver, "ast_crossovers_total", 0)),
                "adaptive_glaciation_depth": int(getattr(solver, "adaptive_glaciation_depth", 4)),
                "active_cell_ratio": round(float(getattr(solver, "last_active_cell_ratio", 0.35)), 4),
                "dormant_cell_ratio": round(float(getattr(solver, "last_dormant_cell_ratio", 0.65)), 4),
                "basal_ganglia_reflex_chunks_count": int(len(getattr(solver, "basal_ganglia_reflex_chunks", []))),
                "basal_ganglia_reflex_hits_total": int(getattr(solver, "basal_ganglia_reflex_hits_total", 0)),
                "basal_ganglia_energy_saved_total": round(float(getattr(solver, "basal_ganglia_energy_saved_total", 0.0)), 4),
                "foveal_saccades_executed_total": int(getattr(solver, "foveal_saccades_executed_total", 0)),
                "foveal_objects_perceived_total": int(getattr(solver, "foveal_objects_perceived_total", 0)),
                "one_step_subgoals_formulated_total": int(getattr(solver, "one_step_subgoals_formulated_total", 0)),
                "basal_ganglia_drawer_pulls_total": int(getattr(solver, "basal_ganglia_drawer_pulls_total", 0)),
                "dynamic_board_reconsiderations_total": int(getattr(solver, "dynamic_board_reconsiderations_total", 0)),
                "selective_object_grasps_total": int(getattr(solver, "selective_object_grasps_total", 0)),
                "intrinsic_gestalt_evaluations_total": int(getattr(solver, "intrinsic_gestalt_evaluations_total", 0)),
                "cross_example_diff_comparisons_total": int(getattr(solver, "cross_example_diff_comparisons_total", 0)),
                "in_grid_legend_reads_total": int(getattr(solver, "in_grid_legend_reads_total", 0)),
                "predicate_group_broadcasts_total": int(getattr(solver, "predicate_group_broadcasts_total", 0)),
                "surgical_guard_debugs_total": int(getattr(solver, "surgical_guard_debugs_total", 0)),
                "stage1_sensorimotor_chunks_transferred": int(getattr(solver, "stage1_sensorimotor_chunks_transferred", 0)),
                "sensorimotor_body_executions_total": int(getattr(solver, "sensorimotor_body_executions_total", 0)),
                "sensorimotor_chunks_bred_total": int(getattr(solver, "sensorimotor_chunks_bred_total", 0)),
                "cerebellar_error_corrections_total": int(getattr(solver, "cerebellar_error_corrections_total", 0)),
                "prefrontal_dynamic_chains_total": int(getattr(solver, "prefrontal_dynamic_chains_total", 0)),
                "attractor_snaps_total": int(getattr(solver, "attractor_snaps_total", 0)),
                "metacognitive_corrections_total": int(getattr(solver, "metacognitive_corrections_total", 0)),
                "verified_symbol_utterances_total": int(getattr(solver, "verified_symbol_utterances_total", 0)),
                "symbol_silence_choices_total": int(getattr(solver, "symbol_silence_choices_total", 0)),
                "shared_intentionality_groundings_total": int(getattr(solver, "shared_intentionality_groundings_total", 0)),
                "epistemic_memo_writes_total": int(getattr(solver, "epistemic_memo_writes_total", 0)),
                "wm_offload_flushes_total": int(getattr(solver, "wm_offload_flushes_total", 0)),
                "epistemic_delayed_payoffs_total": int(getattr(solver, "epistemic_delayed_payoffs_total", 0)),
                "stage1_educational_toys_solved": round(float(last_route1_rep.get("toys_solved_per_agent", 0.0)), 2),
                "stage1_verified_symbol_messages_total": int(last_route1_rep.get("verified_symbol_messages_total", 0)),
                "stage1_symbol_silence_decisions_total": int(last_route1_rep.get("symbol_silence_decisions_total", 0)),
                "stage1_cerebellar_corrections_total": int(last_route1_rep.get("cerebellar_corrections_total", 0)),
                "stage1_curiosity_migrations": round(float(last_route1_rep.get("curiosity_explorations_per_agent", 0.0)), 2),
                "stage1_threat_observations": round(float(last_route1_rep.get("threat_observations_per_agent", 0.0)), 2),
                "stage1_biome_diversity_coverage": round(float(last_route1_rep.get("biome_dispersion_entropy", 0.0)), 4),
                "latest_checkpoint": str(CHECKPOINTS_DIR / "checkpoint_latest.npz"),
                "best_checkpoint": str(CHECKPOINTS_DIR / "checkpoint_best.npz"),
                "latest_generation_summary": history_tail[-1] if history_tail else {},
                "recent_generations": history_tail,
            }
            tmp_hb = ARTIFACTS_DIR / "infinite_evolution_status_tmp.json"
            with open(tmp_hb, "w", encoding="utf-8") as f:
                json.dump(hb, f, indent=2)
            tmp_hb.replace(status_path)
            print(json.dumps({"heartbeat": phase_name, "generation_v2": int(target_gen), "pid": int(os.getpid()), "ts": round(time.time(), 2)}), flush=True)

        try:
            update_heartbeat("Route1_Stage1_Infinite_Procedural_Hazards")
            route1_rep = evolve_infinite_procedural_hazard_manifold(
                n_cycles=hazard_cycles,
                steps_per_cycle=120,
                seed=2026 + target_gen * 17,
            )
            last_route1_rep = dict(route1_rep)
            solver.absorb_stage1_3d_embodied_manifold()
            cleanup_neural_memory()

            update_heartbeat("Route2_Procedural_ARC_Self_Play_Breeding")
            route2_rep = run_procedural_tabletop_practice(
                solver,
                connector.multiverse_harvester,
                n_tasks=self_play_tasks,
                max_steps=6,
                generation_seed=target_gen,
                max_depth=6,
            )
            cleanup_neural_memory()

            update_heartbeat("Offline_Sleep_Dream_Replay_Consolidation")
            dream_rep = solver.offline_dream_replay(
                n_dream_cycles=dream_cycles,
                dream_lr=0.012,
            )
            cleanup_neural_memory()

            update_heartbeat("Route3_External_Infinite_ARC_Databank_Expedition")
            hunting_stream = connector.stream_hunting_grounds(generation_v2=target_gen)
            route3_train_rep = run_tabletop_databank_expedition(
                solver,
                hunting_stream,
                max_steps=6,
                freeze_plastic=False,
            )
            solver.consolidate_pfc_to_basal_ganglia()
            arc2_eval_tasks = connector.fetch_github_arc_agi_2_split(split="evaluation", limit=120)
            route3_eval_rep = run_tabletop_databank_expedition(
                solver,
                arc2_eval_tasks,
                max_steps=6,
                freeze_plastic=True,
            )
            merged_bd = dict(route3_train_rep.get("databank_breakdown", {}))
            merged_bd.update(route3_eval_rep.get("databank_breakdown", {}))
            total_streamed_all = int(route3_train_rep.get("total_external_tasks_streamed", 0)) + int(route3_eval_rep.get("total_external_tasks_streamed", 0))
            total_tr = int(route3_train_rep.get("total_train_exact_solved", 0)) + int(route3_eval_rep.get("total_train_exact_solved", 0))
            total_te = int(route3_train_rep.get("total_test_exact_solved", 0)) + int(route3_eval_rep.get("total_test_exact_solved", 0))
            route3_rep = dict(route3_train_rep)
            route3_rep["total_external_tasks_streamed"] = total_streamed_all
            route3_rep["total_train_exact_solved"] = total_tr
            route3_rep["total_test_exact_solved"] = total_te
            route3_rep["overall_external_solve_rate_pct"] = round(total_te / max(1, total_streamed_all) * 100.0, 2)
            route3_rep["databank_breakdown"] = merged_bd
            route3_rep["sample_external_solves"] = (
                list(route3_train_rep.get("sample_external_solves", []))
                + list(route3_eval_rep.get("sample_external_solves", []))
            )[:25]
            cleanup_neural_memory()

            update_heartbeat("Synaptic_Pruning_And_Manifold_Snapshot")
            pruning_rep = solver.prune_cultural_tool_library(
                max_essential_tools=24,
                max_bred_recipes=12,
                max_nested_recipes=10,
            )
            cleanup_neural_memory()

            generation_v2 = target_gen
            gens_completed += 1

            arc2_bd = route3_rep.get("databank_breakdown", {}).get("GitHub_arcprize_ARC_AGI_2_evaluation", {})
            arc2_solved = int(arc2_bd.get("test_exact_solved", 0))
            arc2_rate = float(arc2_bd.get("solve_rate_pct", 0.0))
            cur_efe = float(pruning_rep.get("mean_tool_efe_efficiency", 0.0))
            cur_train_solved = int(route3_train_rep.get("total_train_exact_solved", 0))

            is_best = (cur_train_solved > best_train_solved) or (
                cur_train_solved == best_train_solved and cur_efe > best_efe_efficiency
            )
            if cur_train_solved > best_train_solved:
                best_train_solved = cur_train_solved
            if cur_efe > best_efe_efficiency:
                best_efe_efficiency = cur_efe

            extra_meta = {
                "generation_v2": int(generation_v2),
                "train_solved": int(cur_train_solved),
                "best_train_solved": int(best_train_solved),
                "arc2_eval_solved": int(arc2_solved),
                "arc2_eval_solve_rate_pct": round(arc2_rate, 2),
                "best_arc2_solved": int(best_arc2_solved),
                "mean_tool_efe_efficiency": round(cur_efe, 4),
                "best_efe_efficiency": round(best_efe_efficiency, 4),
            }
            ckpt_saved = snapshot_generation_manifold(
                solver,
                generation_v2=generation_v2,
                is_new_best=is_best,
                extra_meta=extra_meta,
            )

            gen_elapsed = round(time.time() - gen_t0, 2)
            gen_summary = {
                "generation_v2": int(generation_v2),
                "timestamp": round(time.time(), 2),
                "elapsed_sec": gen_elapsed,
                "self_play_solve_rate_pct": route2_rep.get("self_play_solve_rate_pct", 0.0),
                "self_play_max_depth": int(route2_rep.get("self_play_max_depth", 6)),
                "glaciation_chaos_tasks_count": int(route2_rep.get("glaciation_chaos_tasks_count", 0)),
                "glaciation_chaos_solved_count": int(route2_rep.get("glaciation_chaos_solved_count", 0)),
                "arc2_eval_train_solved": int(arc2_bd.get("train_exact_solved", 0)),
                "arc2_eval_test_solved": int(arc2_solved),
                "arc2_eval_solve_rate_pct": round(arc2_rate, 2),
                "external_total_test_solved": int(route3_rep.get("total_test_exact_solved", 0)),
                "essential_tools_retained": int(len(solver.cultural_tool_library)),
                "bred_recipes_retained": int(len(solver.bred_macro_recipes)),
                "nested_recipes_retained": int(len(solver.nested_macro_recipes)),
                "tools_pruned_cumulative": int(solver.tools_pruned_total),
                "mean_tool_efe_efficiency": round(cur_efe, 4),
                "if_then_else_switches_total": int(solver.if_then_else_switches_total),
                "nested_if_then_switches_total": int(solver.nested_if_then_switches_total),
                "relational_bindings_total": int(getattr(solver, "relational_bindings_total", 0)),
                "recursive_chunks_compiled_total": int(getattr(solver, "recursive_chunks_compiled_total", 0)),
                "canvas_stitch_solves_total": int(getattr(solver, "canvas_stitch_solves_total", 0)),
                "micro_primitives_synthesized_total": int(getattr(solver, "micro_primitives_synthesized_total", 0)),
                "residual_subgoals_harvested_total": int(getattr(solver, "residual_subgoals_harvested_total", 0)),
                "residual_bridges_spliced_total": int(getattr(solver, "residual_bridges_spliced_total", 0)),
                "evolved_ast_organs_count": int(len(getattr(solver, "evolved_ast_organs", []))),
                "de_novo_codons_synthesized_count": int(len(getattr(solver, "de_novo_codon_pool", []))),
                "max_organ_codon_depth": int(max([len(o.get("codons", [])) for o in getattr(solver, "evolved_ast_organs", [])] + [0])),
                "evolving_stepping_stone_tasks_count": int(len(getattr(solver, "evolving_genome_archive", {}))),
                "ast_mutations_total": int(getattr(solver, "ast_mutations_total", 0)),
                "ast_crossovers_total": int(getattr(solver, "ast_crossovers_total", 0)),
                "adaptive_glaciation_depth": int(getattr(solver, "adaptive_glaciation_depth", 4)),
                "active_cell_ratio": round(float(getattr(solver, "last_active_cell_ratio", 0.35)), 4),
                "dormant_cell_ratio": round(float(getattr(solver, "last_dormant_cell_ratio", 0.65)), 4),
                "basal_ganglia_reflex_chunks_count": int(len(getattr(solver, "basal_ganglia_reflex_chunks", []))),
                "basal_ganglia_reflex_hits_total": int(getattr(solver, "basal_ganglia_reflex_hits_total", 0)),
                "basal_ganglia_energy_saved_total": round(float(getattr(solver, "basal_ganglia_energy_saved_total", 0.0)), 4),
                "foveal_saccades_executed_total": int(getattr(solver, "foveal_saccades_executed_total", 0)),
                "foveal_objects_perceived_total": int(getattr(solver, "foveal_objects_perceived_total", 0)),
                "one_step_subgoals_formulated_total": int(getattr(solver, "one_step_subgoals_formulated_total", 0)),
                "basal_ganglia_drawer_pulls_total": int(getattr(solver, "basal_ganglia_drawer_pulls_total", 0)),
                "dynamic_board_reconsiderations_total": int(getattr(solver, "dynamic_board_reconsiderations_total", 0)),
                "selective_object_grasps_total": int(getattr(solver, "selective_object_grasps_total", 0)),
                "intrinsic_gestalt_evaluations_total": int(getattr(solver, "intrinsic_gestalt_evaluations_total", 0)),
                "cross_example_diff_comparisons_total": int(getattr(solver, "cross_example_diff_comparisons_total", 0)),
                "in_grid_legend_reads_total": int(getattr(solver, "in_grid_legend_reads_total", 0)),
                "predicate_group_broadcasts_total": int(getattr(solver, "predicate_group_broadcasts_total", 0)),
                "surgical_guard_debugs_total": int(getattr(solver, "surgical_guard_debugs_total", 0)),
                "stage1_sensorimotor_chunks_transferred": int(getattr(solver, "stage1_sensorimotor_chunks_transferred", 0)),
                "sensorimotor_body_executions_total": int(getattr(solver, "sensorimotor_body_executions_total", 0)),
                "sensorimotor_chunks_bred_total": int(getattr(solver, "sensorimotor_chunks_bred_total", 0)),
                "cerebellar_error_corrections_total": int(getattr(solver, "cerebellar_error_corrections_total", 0)),
                "prefrontal_dynamic_chains_total": int(getattr(solver, "prefrontal_dynamic_chains_total", 0)),
                "attractor_snaps_total": int(getattr(solver, "attractor_snaps_total", 0)),
                "metacognitive_corrections_total": int(getattr(solver, "metacognitive_corrections_total", 0)),
                "verified_symbol_utterances_total": int(getattr(solver, "verified_symbol_utterances_total", 0)),
                "symbol_silence_choices_total": int(getattr(solver, "symbol_silence_choices_total", 0)),
                "shared_intentionality_groundings_total": int(getattr(solver, "shared_intentionality_groundings_total", 0)),
                "epistemic_memo_writes_total": int(getattr(solver, "epistemic_memo_writes_total", 0)),
                "wm_offload_flushes_total": int(getattr(solver, "wm_offload_flushes_total", 0)),
                "epistemic_delayed_payoffs_total": int(getattr(solver, "epistemic_delayed_payoffs_total", 0)),
                "stage1_educational_toys_solved": round(float(route1_rep.get("toys_solved_per_agent", 0.0)), 2),
                "stage1_verified_symbol_messages_total": int(route1_rep.get("verified_symbol_messages_total", 0)),
                "stage1_symbol_silence_decisions_total": int(route1_rep.get("symbol_silence_decisions_total", 0)),
                "stage1_sensorimotor_chunk_hits_total": int(route1_rep.get("sensorimotor_chunk_hits_total", 0)),
                "stage1_cerebellar_corrections_total": int(route1_rep.get("cerebellar_corrections_total", 0)),
                "stage1_curiosity_migrations": round(float(route1_rep.get("curiosity_explorations_per_agent", 0.0)), 2),
                "stage1_threat_observations": round(float(route1_rep.get("threat_observations_per_agent", 0.0)), 2),
                "stage1_biome_diversity_coverage": round(float(route1_rep.get("biome_dispersion_entropy", 0.0)), 4),
                "checkpoint_saved": ckpt_saved,
                "gc_and_cuda_cache_cleared": True,
            }
            history_tail.append(gen_summary)
            history_tail = history_tail[-25:]

            daemon_status = {
                "daemon_state": "RUNNING_IMMORTAL_EVOLUTION",
                "pid": int(os.getpid()),
                "active_generation_v2": int(generation_v2),
                "active_phase": "GENERATION_COMPLETED",
                "heartbeat_timestamp": round(time.time(), 2),
                "generation_v2": int(generation_v2),
                "best_arc2_eval_solved": int(best_arc2_solved),
                "best_arc2_eval_solve_rate_pct": round(best_arc2_solved / 120.0 * 100.0, 2),
                "best_mean_tool_efe_efficiency": round(best_efe_efficiency, 4),
                "current_cultural_tools_count": int(len(solver.cultural_tool_library)),
                "current_bred_recipes_count": int(len(solver.bred_macro_recipes)),
                "current_nested_recipes_count": int(len(solver.nested_macro_recipes)),
                "tools_pruned_cumulative": int(solver.tools_pruned_total),
                "relational_bindings_cumulative": int(getattr(solver, "relational_bindings_total", 0)),
                "recursive_chunks_compiled_total": int(getattr(solver, "recursive_chunks_compiled_total", 0)),
                "canvas_stitch_solves_total": int(getattr(solver, "canvas_stitch_solves_total", 0)),
                "micro_primitives_synthesized_total": int(getattr(solver, "micro_primitives_synthesized_total", 0)),
                "residual_subgoals_harvested_total": int(getattr(solver, "residual_subgoals_harvested_total", 0)),
                "residual_bridges_spliced_total": int(getattr(solver, "residual_bridges_spliced_total", 0)),
                "evolved_ast_organs_count": int(len(getattr(solver, "evolved_ast_organs", []))),
                "de_novo_codons_synthesized_count": int(len(getattr(solver, "de_novo_codon_pool", []))),
                "max_organ_codon_depth": int(max([len(o.get("codons", [])) for o in getattr(solver, "evolved_ast_organs", [])] + [0])),
                "evolving_stepping_stone_tasks_count": int(len(getattr(solver, "evolving_genome_archive", {}))),
                "ast_mutations_total": int(getattr(solver, "ast_mutations_total", 0)),
                "ast_crossovers_total": int(getattr(solver, "ast_crossovers_total", 0)),
                "adaptive_glaciation_depth": int(getattr(solver, "adaptive_glaciation_depth", 4)),
                "active_cell_ratio": round(float(getattr(solver, "last_active_cell_ratio", 0.35)), 4),
                "dormant_cell_ratio": round(float(getattr(solver, "last_dormant_cell_ratio", 0.65)), 4),
                "basal_ganglia_reflex_chunks_count": int(len(getattr(solver, "basal_ganglia_reflex_chunks", []))),
                "basal_ganglia_reflex_hits_total": int(getattr(solver, "basal_ganglia_reflex_hits_total", 0)),
                "basal_ganglia_energy_saved_total": round(float(getattr(solver, "basal_ganglia_energy_saved_total", 0.0)), 4),
                "foveal_saccades_executed_total": int(getattr(solver, "foveal_saccades_executed_total", 0)),
                "foveal_objects_perceived_total": int(getattr(solver, "foveal_objects_perceived_total", 0)),
                "one_step_subgoals_formulated_total": int(getattr(solver, "one_step_subgoals_formulated_total", 0)),
                "basal_ganglia_drawer_pulls_total": int(getattr(solver, "basal_ganglia_drawer_pulls_total", 0)),
                "dynamic_board_reconsiderations_total": int(getattr(solver, "dynamic_board_reconsiderations_total", 0)),
                "selective_object_grasps_total": int(getattr(solver, "selective_object_grasps_total", 0)),
                "intrinsic_gestalt_evaluations_total": int(getattr(solver, "intrinsic_gestalt_evaluations_total", 0)),
                "cross_example_diff_comparisons_total": int(getattr(solver, "cross_example_diff_comparisons_total", 0)),
                "in_grid_legend_reads_total": int(getattr(solver, "in_grid_legend_reads_total", 0)),
                "predicate_group_broadcasts_total": int(getattr(solver, "predicate_group_broadcasts_total", 0)),
                "surgical_guard_debugs_total": int(getattr(solver, "surgical_guard_debugs_total", 0)),
                "stage1_sensorimotor_chunks_transferred": int(getattr(solver, "stage1_sensorimotor_chunks_transferred", 0)),
                "sensorimotor_body_executions_total": int(getattr(solver, "sensorimotor_body_executions_total", 0)),
                "sensorimotor_chunks_bred_total": int(getattr(solver, "sensorimotor_chunks_bred_total", 0)),
                "cerebellar_error_corrections_total": int(getattr(solver, "cerebellar_error_corrections_total", 0)),
                "prefrontal_dynamic_chains_total": int(getattr(solver, "prefrontal_dynamic_chains_total", 0)),
                "attractor_snaps_total": int(getattr(solver, "attractor_snaps_total", 0)),
                "metacognitive_corrections_total": int(getattr(solver, "metacognitive_corrections_total", 0)),
                "verified_symbol_utterances_total": int(getattr(solver, "verified_symbol_utterances_total", 0)),
                "symbol_silence_choices_total": int(getattr(solver, "symbol_silence_choices_total", 0)),
                "shared_intentionality_groundings_total": int(getattr(solver, "shared_intentionality_groundings_total", 0)),
                "epistemic_memo_writes_total": int(getattr(solver, "epistemic_memo_writes_total", 0)),
                "wm_offload_flushes_total": int(getattr(solver, "wm_offload_flushes_total", 0)),
                "epistemic_delayed_payoffs_total": int(getattr(solver, "epistemic_delayed_payoffs_total", 0)),
                "stage1_educational_toys_solved": round(float(route1_rep.get("toys_solved_per_agent", 0.0)), 2),
                "stage1_verified_symbol_messages_total": int(route1_rep.get("verified_symbol_messages_total", 0)),
                "stage1_symbol_silence_decisions_total": int(route1_rep.get("symbol_silence_decisions_total", 0)),
                "stage1_sensorimotor_chunk_hits_total": int(route1_rep.get("sensorimotor_chunk_hits_total", 0)),
                "stage1_cerebellar_corrections_total": int(route1_rep.get("cerebellar_corrections_total", 0)),
                "stage1_curiosity_migrations": round(float(route1_rep.get("curiosity_explorations_per_agent", 0.0)), 2),
                "stage1_threat_observations": round(float(route1_rep.get("threat_observations_per_agent", 0.0)), 2),
                "stage1_biome_diversity_coverage": round(float(route1_rep.get("biome_dispersion_entropy", 0.0)), 4),
                "latest_checkpoint": str(CHECKPOINTS_DIR / "checkpoint_latest.npz"),
                "best_checkpoint": str(CHECKPOINTS_DIR / "checkpoint_best.npz"),
                "latest_generation_summary": gen_summary,
                "recent_generations": history_tail,
            }
            tmp_status = ARTIFACTS_DIR / "infinite_evolution_status_tmp.json"
            with open(tmp_status, "w", encoding="utf-8") as f:
                json.dump(daemon_status, f, indent=2)
            tmp_status.replace(status_path)

            existing_report = {}
            if metrics_path.exists():
                try:
                    with open(metrics_path, "r", encoding="utf-8") as f:
                        existing_report = json.load(f)
                except Exception:
                    existing_report = {}

            existing_report["Infinite_Evolution_Daemon_Status"] = daemon_status
            if "World_Expansion_Routes" not in existing_report:
                existing_report["World_Expansion_Routes"] = {}
            existing_report["World_Expansion_Routes"]["Route1_Stage1_Infinite_Procedural_Hazards"] = route1_rep
            existing_report["World_Expansion_Routes"]["Route2_Procedural_ARC_Self_Play_Breeding"] = route2_rep
            existing_report["World_Expansion_Routes"]["Route3_External_Infinite_ARC_Databank_Expedition"] = route3_rep
            existing_report["Offline_Sleep_Dream_Replay_Consolidation"] = dream_rep

            tmp_metrics = ARTIFACTS_DIR / "stage3_arc_benchmark_metrics_tmp.json"
            with open(tmp_metrics, "w", encoding="utf-8") as f:
                json.dump(existing_report, f, indent=2)
            tmp_metrics.replace(metrics_path)

            print(json.dumps(gen_summary), flush=True)
        except Exception as err:
            err_info = {
                "generation_v2": int(generation_v2),
                "error_recovered": str(err),
                "timestamp": round(time.time(), 2),
            }
            print(json.dumps(err_info), flush=True)
            time.sleep(2.0)
        finally:
            gc.collect()
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()


def diagnose_unsolved_task_biologically(task_id, task, res, test_gt):
    solved_train = bool(res.get("solved_train_exact", False))
    pred = res.get("prediction")
    alt_preds = res.get("alt_predictions_all", [])
    alt_pred = alt_preds[0] if alt_preds else None
    test_pix_acc = float(np.mean(pred == test_gt)) if (pred is not None and pred.shape == test_gt.shape) else 0.0
    alt_exact = bool(alt_pred is not None and alt_pred.shape == test_gt.shape and np.array_equal(alt_pred, test_gt))
    shape_change_req = bool(res.get("shape_change_required", False))
    test_shape_mismatch = bool(pred is None or pred.shape != test_gt.shape)
    order_gain = float(res.get("order_gain_achieved", 0.0))
    best_score = float(res.get("best_score", 0.0))
    rem_energy = float(res.get("mean_remaining_energy", 0.0))
    survival_pct = float(res.get("survival_rate_pct", 100.0))
    gaba_blocks = int(res.get("gaba_blocks_this_task", 0))
    latent_bypasses = int(res.get("latent_bypasses_this_task", 0))
    lens_warps = int(res.get("lens_warps_this_task", 0))
    foveal_objs = int(res.get("foveal_objects_count", 0))
    prog = list(res.get("program", []))

    if solved_train:
        primary_cause = "counterfactual_test_overfit_or_hidden_test_invariant"
    elif shape_change_req or test_shape_mismatch:
        primary_cause = "dynamic_canvas_shape_warping_deficit"
    elif gaba_blocks > 0 and latent_bypasses == 0:
        primary_cause = "prefrontal_gaba_trap_without_matching_latent_operator"
    elif foveal_objs >= 12 and order_gain < 0.12:
        primary_cause = "perceptual_foveal_crowding_and_multi_object_binding_limit"
    elif order_gain >= 0.08 or (best_score >= 0.72 and len(prog) > 0):
        primary_cause = "working_memory_horizon_and_atp_exhaustion_after_partial_progress"
    else:
        primary_cause = "missing_compositional_motor_affordance_stagnation"

    return {
        "task_id": str(task_id),
        "primary_cause": primary_cause,
        "solved_train_exact": solved_train,
        "alt_candidate_solved_test": alt_exact,
        "test_pixel_acc": round(test_pix_acc, 4),
        "baseline_order": round(float(res.get("baseline_order", 0.0)), 4),
        "best_train_order": round(best_score, 4),
        "order_gain_achieved": round(order_gain, 4),
        "mean_remaining_energy": round(rem_energy, 4),
        "survival_rate_pct": round(survival_pct, 2),
        "gaba_blocks": gaba_blocks,
        "latent_bypasses": latent_bypasses,
        "lens_warps": lens_warps,
        "active_ach_lens": str(res.get("active_ach_lens", "cartesian")),
        "foveal_objects_count": foveal_objs,
        "shape_change_required": shape_change_req,
        "test_shape_mismatch": test_shape_mismatch,
        "program": prog,
    }


def evaluate_stage_with_biological_diagnosis(solver, json_files, stage_name, max_steps=4):
    t_stage0 = time.time()
    train_exact_ids = []
    test_top1_ids = []
    test_top2_ids = []
    solved_details = []
    unsolved_diagnostics = []
    cause_counts = {
        "counterfactual_test_overfit_or_hidden_test_invariant": 0,
        "dynamic_canvas_shape_warping_deficit": 0,
        "prefrontal_gaba_trap_without_matching_latent_operator": 0,
        "perceptual_foveal_crowding_and_multi_object_binding_limit": 0,
        "working_memory_horizon_and_atp_exhaustion_after_partial_progress": 0,
        "missing_compositional_motor_affordance_stagnation": 0,
    }
    axis_dist = {i: 0 for i in range(8)}
    mechanism_usage = {
        "basal_ganglia_reflex_solves": 0,
        "counterfactual_gaba_bypass_solves": 0,
        "ach_topological_lens_solves": 0,
        "sensorimotor_chunk_solves": 0,
        "ast_organ_or_codon_solves": 0,
    }

    for idx_f, jf in enumerate(json_files):
        with open(jf, "r", encoding="utf-8") as f:
            task = json.load(f)
        test_pair = task["test"][0] if isinstance(task["test"], list) else task["test"]
        test_gt = np.array(test_pair["output"], dtype=np.int32)
        blind_task = make_blind_test_task(task)
        res = solver.study_and_solve_tabletop_task(blind_task, max_steps=max_steps)
        if res.get("solved_train_exact", False):
            train_exact_ids.append(jf.stem)
        pred = res.get("prediction")
        alt_preds = res.get("alt_predictions_all", [])
        alt_pred = alt_preds[0] if alt_preds else None
        top1_ok = bool(pred is not None and pred.shape == test_gt.shape and np.array_equal(pred, test_gt))
        top2_ok = top1_ok or bool(alt_pred is not None and alt_pred.shape == test_gt.shape and np.array_equal(alt_pred, test_gt))
        if top1_ok:
            test_top1_ids.append(jf.stem)
        if top2_ok:
            test_top2_ids.append(jf.stem)
            w_ax = int(res.get("winning_axis", -1))
            if w_ax in axis_dist:
                axis_dist[w_ax] += 1
            prog_strs = [str(p) for p in res.get("program", [])]
            if res.get("basal_ganglia_reflex_used", False):
                mechanism_usage["basal_ganglia_reflex_solves"] += 1
            if res.get("counterfactual_gaba_bypass_used", False):
                mechanism_usage["counterfactual_gaba_bypass_solves"] += 1
            if res.get("ach_topological_lens_used", False) or any("lens_" in s for s in prog_strs):
                mechanism_usage["ach_topological_lens_solves"] += 1
            if any("sensorimotor_chunk_" in s for s in prog_strs):
                mechanism_usage["sensorimotor_chunk_solves"] += 1
            if any(("ast_" in s or "codon" in s or "de_novo" in s) for s in prog_strs):
                mechanism_usage["ast_organ_or_codon_solves"] += 1
            solved_details.append({
                "task_id": jf.stem,
                "top1_exact": top1_ok,
                "top2_exact": top2_ok,
                "program": prog_strs,
                "winning_axis": w_ax,
                "steps_taken": int(res.get("steps_taken", 1)),
            })
        else:
            diag = diagnose_unsolved_task_biologically(jf.stem, task, res, test_gt)
            c_key = diag["primary_cause"]
            cause_counts[c_key] = cause_counts.get(c_key, 0) + 1
            unsolved_diagnostics.append(diag)

        if (idx_f + 1) % 100 == 0 or (idx_f + 1) == len(json_files):
            print(
                json.dumps({
                    "stage_progress": stage_name,
                    "processed": int(idx_f + 1),
                    "total": int(len(json_files)),
                    "train_exact_so_far": int(len(train_exact_ids)),
                    "test_top1_so_far": int(len(test_top1_ids)),
                    "test_top2_so_far": int(len(test_top2_ids)),
                    "elapsed_sec": round(time.time() - t_stage0, 2),
                }),
                flush=True,
            )

    total_n = max(1, len(json_files))
    unsolved_n = max(1, len(unsolved_diagnostics))
    cause_pcts = {
        k: round(float(v) / float(unsolved_n) * 100.0, 2)
        for k, v in cause_counts.items()
    }
    sample_by_cause = {}
    for k in cause_counts:
        sample_by_cause[k] = [d for d in unsolved_diagnostics if d["primary_cause"] == k][:8]

    return {
        "stage_name": stage_name,
        "total_tasks": int(len(json_files)),
        "elapsed_sec": round(time.time() - t_stage0, 2),
        "train_exact_solved_count": int(len(train_exact_ids)),
        "train_exact_solve_rate_pct": round(len(train_exact_ids) / total_n * 100.0, 2),
        "test_top1_solved_count": int(len(test_top1_ids)),
        "test_top1_solve_rate_pct": round(len(test_top1_ids) / total_n * 100.0, 2),
        "test_top2_solved_count": int(len(test_top2_ids)),
        "test_top2_solve_rate_pct": round(len(test_top2_ids) / total_n * 100.0, 2),
        "axis_solve_distribution": axis_dist,
        "mechanism_solve_contributions": mechanism_usage,
        "unsolved_total_count": int(len(unsolved_diagnostics)),
        "unsolved_biological_cause_counts": cause_counts,
        "unsolved_biological_cause_share_pct": cause_pcts,
        "solved_tasks_list": solved_details,
        "unsolved_samples_by_cause": sample_by_cause,
    }


def run_staged_arc1_to_arc2_diagnostic_benchmark():
    t0 = time.time()
    stage1_3d_rep = evolve_infinite_procedural_hazard_manifold(
        n_cycles=8,
        steps_per_cycle=120,
        seed=2026,
    )
    cleanup_neural_memory()

    solver = EmbodiedPuzzleSolver(use_pain=True, use_curiosity=True, use_mortality=True, plastic=True)
    latest_ckpt = CHECKPOINTS_DIR / "checkpoint_latest.npz"
    if latest_ckpt.exists():
        solver.load_evolution_checkpoint(latest_ckpt)
    solver.absorb_stage1_3d_embodied_manifold()
    cleanup_neural_memory()

    arc1_train_files = sorted(DATA_TRAIN_DIR.glob("*.json"))[:400]
    arc1_eval_files = sorted(DATA_EVAL_DIR.glob("*.json"))[:400]
    stage_a1_rep = evaluate_stage_with_biological_diagnosis(
        solver, arc1_train_files, "Stage1A_ARC_AGI_1_Training_400", max_steps=3
    )
    cleanup_neural_memory()

    prev_p_a2 = bool(solver.plastic)
    solver.plastic = False
    try:
        stage_a2_rep = evaluate_stage_with_biological_diagnosis(
            solver, arc1_eval_files, "Stage1B_ARC_AGI_1_Evaluation_400", max_steps=3
        )
    finally:
        solver.plastic = prev_p_a2
    cleanup_neural_memory()

    sleep_after_arc1 = solver.offline_dream_replay(n_dream_cycles=10, dream_lr=0.012)
    solver.consolidate_pfc_to_basal_ganglia()
    cleanup_neural_memory()

    arc2_train_dir = EXTERNAL_CACHE_DIR / "arc_agi_2_train"
    arc2_eval_dir = EXTERNAL_CACHE_DIR / "arc_agi_2_eval"
    arc2_train_files = sorted(arc2_train_dir.glob("*.json"))[:1000]
    arc2_eval_files = sorted(arc2_eval_dir.glob("*.json"))[:120]

    stage_b_rep = evaluate_stage_with_biological_diagnosis(
        solver, arc2_train_files, "Stage2_ARC_AGI_2_Training_1000", max_steps=3
    )
    cleanup_neural_memory()

    sleep_after_arc2_train = solver.offline_dream_replay(n_dream_cycles=12, dream_lr=0.012)
    solver.consolidate_pfc_to_basal_ganglia()
    cleanup_neural_memory()

    prev_p_c = bool(solver.plastic)
    solver.plastic = False
    try:
        stage_c_rep = evaluate_stage_with_biological_diagnosis(
            solver, arc2_eval_files, "Stage3_ARC_AGI_2_Evaluation_120", max_steps=4
        )
    finally:
        solver.plastic = prev_p_c
    cleanup_neural_memory()

    aggregate_causes = {
        "counterfactual_test_overfit_or_hidden_test_invariant": 0,
        "dynamic_canvas_shape_warping_deficit": 0,
        "prefrontal_gaba_trap_without_matching_latent_operator": 0,
        "perceptual_foveal_crowding_and_multi_object_binding_limit": 0,
        "working_memory_horizon_and_atp_exhaustion_after_partial_progress": 0,
        "missing_compositional_motor_affordance_stagnation": 0,
    }
    for st_rep in (stage_a1_rep, stage_a2_rep, stage_b_rep, stage_c_rep):
        for k, v in st_rep["unsolved_biological_cause_counts"].items():
            aggregate_causes[k] = aggregate_causes.get(k, 0) + int(v)
    total_unsolved_all = max(1, sum(aggregate_causes.values()))
    aggregate_cause_share_pct = {
        k: round(float(v) / float(total_unsolved_all) * 100.0, 2)
        for k, v in aggregate_causes.items()
    }

    full_report = {
        "elapsed_sec": round(time.time() - t0, 2),
        "Stage0_3D_Survival_Arena_Conditioning": stage1_3d_rep,
        "Stage1A_ARC_AGI_1_Training_400": stage_a1_rep,
        "Stage1B_ARC_AGI_1_Evaluation_400": stage_a2_rep,
        "InterStage_Sleep_Consolidation_After_ARC1": sleep_after_arc1,
        "Stage2_ARC_AGI_2_Training_1000": stage_b_rep,
        "InterStage_Sleep_Consolidation_After_ARC2_Train": sleep_after_arc2_train,
        "Stage3_ARC_AGI_2_Evaluation_120": stage_c_rep,
        "Biological_Telemetry_Totals": {
            "prefrontal_gaba_blocks_total": int(getattr(solver, "prefrontal_gaba_blocks_total", 0)),
            "latent_feature_bypasses_total": int(getattr(solver, "latent_feature_bypasses_total", 0)),
            "topological_lens_warps_total": int(getattr(solver, "topological_lens_warps_total", 0)),
            "ripple_replays_executed_total": int(getattr(solver, "ripple_replays_executed_total", 0)),
            "ib_ltp_invariant_synapses_total": int(getattr(solver, "ib_ltp_invariant_synapses_total", 0)),
            "ib_ltd_pruned_synapses_total": int(getattr(solver, "ib_ltd_pruned_synapses_total", 0)),
            "cross_example_diff_comparisons_total": int(getattr(solver, "cross_example_diff_comparisons_total", 0)),
            "basal_ganglia_reflex_hits_total": int(getattr(solver, "basal_ganglia_reflex_hits_total", 0)),
            "cerebellar_error_corrections_total": int(getattr(solver, "cerebellar_error_corrections_total", 0)),
            "prefrontal_dynamic_chains_total": int(getattr(solver, "prefrontal_dynamic_chains_total", 0)),
            "foveal_saccades_executed_total": int(getattr(solver, "foveal_saccades_executed_total", 0)),
            "v1_v2_texture_segregations_total": int(getattr(solver, "v1_v2_texture_segregations_total", 0)),
            "v1_v2_constellation_groups_total": int(getattr(solver, "v1_v2_constellation_groups_total", 0)),
            "v1_v2_hierarchical_expansions_total": int(getattr(solver, "v1_v2_hierarchical_expansions_total", 0)),
            "virtual_scratchpad_projections_total": int(getattr(solver, "virtual_scratchpad_projections_total", 0)),
            "scratchpad_inverse_backprojections_total": int(getattr(solver, "scratchpad_inverse_backprojections_total", 0)),
            "subgoal_checkpoints_locked_total": int(getattr(solver, "subgoal_checkpoints_locked_total", 0)),
            "subgoal_non_destructive_vetoes_total": int(getattr(solver, "subgoal_non_destructive_vetoes_total", 0)),
            "it_parietal_relational_bindings_total": int(getattr(solver, "it_parietal_relational_bindings_total", 0)),
            "cerebellar_valley_crossing_macros_total": int(getattr(solver, "cerebellar_valley_crossing_macros_total", 0)),
            "pfc_dyadic_bindings_total": int(getattr(solver, "pfc_dyadic_bindings_total", 0)),
            "foveal_window_zooms_total": int(getattr(solver, "foveal_window_zooms_total", 0)),
            "motor_stamp_executions_total": int(getattr(solver, "motor_stamp_executions_total", 0)),
            "ior_sequential_saccades_total": int(getattr(solver, "ior_sequential_saccades_total", 0)),
            "epistemic_atp_invested_total": round(float(getattr(solver, "epistemic_atp_invested_total", 0.0)), 4),
            "aha_dopamine_bursts_total": int(getattr(solver, "aha_dopamine_bursts_total", 0)),
            "seated_tabletop_focus_engagements_total": int(getattr(solver, "seated_tabletop_focus_engagements_total", 0)),
            "tabletop_25d_projections_total": int(getattr(solver, "tabletop_25d_projections_total", 0)),
            "head_tilt_viewpoint_shifts_total": int(getattr(solver, "head_tilt_viewpoint_shifts_total", 0)),
            "mirror_neuron_imitations_total": int(getattr(solver, "mirror_neuron_imitations_total", 0)),
            "autotelic_play_bouts_total": int(getattr(solver, "autotelic_play_bouts_total", 0)),
            "hand_regard_self_imitations_total": int(getattr(solver, "hand_regard_self_imitations_total", 0)),
            "developmental_stage": int(getattr(solver, "developmental_stage", 1)),
            "critical_period_transitions_total": int(getattr(solver, "critical_period_transitions_total", 0)),
            "ach_habit_resets_total": int(getattr(solver, "ach_habit_resets_total", 0)),
            "einstellung_fixation_prevented_total": int(getattr(solver, "einstellung_fixation_prevented_total", 0)),
            "synaptic_homeostasis_downscalings_total": int(getattr(solver, "synaptic_homeostasis_downscalings_total", 0)),
            "triple_overfit_guards_triggered_total": int(getattr(solver, "triple_overfit_guards_triggered_total", 0)),
            "external_workbench_interactions_total": int(getattr(solver, "external_workbench_interactions_total", 0)),
            "unified_brain_status": dict(getattr(solver, "loaded_brain_status", {})),
        },
        "Aggregate_Unsolved_Biological_Taxonomy": {
            "total_unsolved_across_1920_tasks": int(sum(aggregate_causes.values())),
            "cause_counts": aggregate_causes,
            "cause_share_pct": aggregate_cause_share_pct,
        },
    }

    diag_path = ARTIFACTS_DIR / "staged_arc1_arc2_biological_diagnosis.json"
    with open(diag_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)

    snapshot_generation_manifold(
        solver,
        generation_v2=1,
        is_new_best=True,
        extra_meta={
            "arc1_train_top2_solved": stage_a1_rep["test_top2_solved_count"],
            "arc1_eval_top2_solved": stage_a2_rep["test_top2_solved_count"],
            "arc2_train_top2_solved": stage_b_rep["test_top2_solved_count"],
            "arc2_eval_top2_solved": stage_c_rep["test_top2_solved_count"],
        },
    )

    print(json.dumps({
        "status": "STAGED_DIAGNOSTIC_BENCHMARK_COMPLETED",
        "elapsed_sec": full_report["elapsed_sec"],
        "arc1_train_top2_solved": stage_a1_rep["test_top2_solved_count"],
        "arc1_eval_top2_solved": stage_a2_rep["test_top2_solved_count"],
        "arc2_train_top2_solved": stage_b_rep["test_top2_solved_count"],
        "arc2_eval_top2_solved": stage_c_rep["test_top2_solved_count"],
        "aggregate_unsolved_causes": aggregate_causes,
    }, indent=2), flush=True)
    return full_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="infinite", choices=["infinite", "benchmark", "staged_diagnostic"])
    parser.add_argument("--max-generations", type=int, default=None)
    args = parser.parse_args()

    if args.mode == "benchmark":
        run_full_benchmark_once()
    elif args.mode == "staged_diagnostic":
        run_staged_arc1_to_arc2_diagnostic_benchmark()
    else:
        run_infinite_evolution_loop(max_generations=args.max_generations)


if __name__ == "__main__":
    main()
