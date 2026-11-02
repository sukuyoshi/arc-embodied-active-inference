import argparse
import gc
import json
import time
from pathlib import Path
import numpy as np

try:
    import torch
except ImportError:
    torch = None

from stage2_embodied_puzzle_solver import (
    ARTIFACTS_DIR,
    EmbodiedPuzzleSolver,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRAIN_DIR = ARTIFACTS_DIR / "external_arc_databank" / "arc_agi_2_train"
FALLBACK_TRAIN_DIR = REPO_ROOT / "arc_repo" / "data" / "training"
CHECKPOINTS_DIR = ARTIFACTS_DIR / "checkpoints"


def cleanup_memory():
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_plasticity_cycle(train_dir=None, n_tasks=10, max_steps=6, n_dream_cycles=8):
    t0 = time.time()
    if train_dir is None:
        train_dir = DEFAULT_TRAIN_DIR if DEFAULT_TRAIN_DIR.exists() else FALLBACK_TRAIN_DIR
    in_path = Path(train_dir)
    json_files = sorted(in_path.glob("*.json"))[:n_tasks]

    solver = EmbodiedPuzzleSolver(
        use_pain=True,
        use_curiosity=True,
        use_mortality=True,
        plastic=True,
        pop_size=12,
    )

    best_ckpt = CHECKPOINTS_DIR / "checkpoint_best.npz"
    latest_ckpt = CHECKPOINTS_DIR / "checkpoint_latest.npz"
    loaded_ckpt = None
    if latest_ckpt.exists():
        ok = solver.load_evolution_checkpoint(latest_ckpt)
        if ok:
            loaded_ckpt = str(latest_ckpt)
    elif best_ckpt.exists():
        ok = solver.load_evolution_checkpoint(best_ckpt)
        if ok:
            loaded_ckpt = str(best_ckpt)

    solver.absorb_stage1_3d_embodied_manifold()

    wh_norm_before = float(np.linalg.norm(solver.Wh))
    wpred_norm_before = float(np.linalg.norm(solver.Wpred))
    danger_norm = float(np.linalg.norm(getattr(solver, "stage1_danger_manifold", np.zeros(1))))

    solved_count = 0
    episodes_recorded_start = len(getattr(solver, "dream_wake_buffer", []))

    for idx, jf in enumerate(json_files):
        with open(jf, "r", encoding="utf-8") as f:
            task = json.load(f)
        res = solver.study_and_solve_tabletop_task(task, max_steps=max_steps)
        if res.get("solved_train_exact", False):
            solved_count += 1
        if (idx + 1) % 5 == 0:
            cleanup_memory()

    episodes_recorded_end = len(getattr(solver, "dream_wake_buffer", []))
    new_episodes = episodes_recorded_end - episodes_recorded_start

    dream_res = solver.offline_dream_replay(n_dream_cycles=n_dream_cycles, dream_lr=0.012)
    solver.consolidate_pfc_to_basal_ganglia()
    cleanup_memory()

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    target_latest = CHECKPOINTS_DIR / "checkpoint_latest.npz"
    solver.save_evolution_checkpoint(target_latest, generation_v2=int(solver.tasks_processed_count))
    if solved_count > 0 or not best_ckpt.exists():
        target_best = CHECKPOINTS_DIR / "checkpoint_best.npz"
        solver.save_evolution_checkpoint(target_best, generation_v2=int(solver.tasks_processed_count))

    wh_norm_after = float(np.linalg.norm(solver.Wh))
    wpred_norm_after = float(np.linalg.norm(solver.Wpred))

    elapsed = round(time.time() - t0, 2)
    summary = {
        "status": "EXPERIENCE_PLASTICITY_CYCLE_COMPLETED",
        "checkpoint_base_loaded": loaded_ckpt,
        "dataset_trained": str(in_path),
        "tasks_experienced": len(json_files),
        "tasks_solved_exact": solved_count,
        "new_wake_episodes_imprinted": new_episodes,
        "offline_dream_replay_cycles": n_dream_cycles,
        "dream_replay_stats": dream_res if isinstance(dream_res, dict) else {},
        "stage1_danger_vec_norm_bonded": danger_norm,
        "cortex_wh_norm_delta": round(wh_norm_after - wh_norm_before, 6),
        "world_model_wpred_norm_delta": round(wpred_norm_after - wpred_norm_before, 6),
        "cultural_tools_count": len(solver.cultural_tool_library),
        "bred_recipes_count": len(solver.bred_macro_recipes),
        "nested_recipes_count": len(solver.nested_macro_recipes),
        "checkpoint_latest_saved": str(target_latest),
        "elapsed_sec": elapsed,
    }

    report_path = ARTIFACTS_DIR / "experience_plasticity_cycle_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=str, default=None)
    parser.add_argument("--n-tasks", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--n-dream-cycles", type=int, default=8)
    args = parser.parse_args()

    rep = run_plasticity_cycle(
        train_dir=args.train_dir,
        n_tasks=args.n_tasks,
        max_steps=args.max_steps,
        n_dream_cycles=args.n_dream_cycles,
    )
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
