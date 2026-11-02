import argparse
import gc
import json
import os
import sys
import time
import zipfile
from pathlib import Path
import numpy as np


def auto_extract_submission_zip():
    cwd = Path.cwd()
    search_dirs = [cwd, cwd.parent, Path("/kaggle/input"), Path("/kaggle/working")]
    for sdir in search_dirs:
        if not sdir.exists():
            continue
        for zpath in list(sdir.glob("*.zip")) + list(sdir.glob("*/*.zip")):
            core_engine_path = cwd / "embodied_core_engine"
            sub_core_engine = cwd / "arc-creativity-engine" / "embodied_core_engine"
            if not core_engine_path.exists() and not sub_core_engine.exists():
                try:
                    with zipfile.ZipFile(zpath, "r") as zf:
                        zf.extractall(cwd)
                except Exception:
                    pass
    cand_paths = [
        cwd,
        cwd / "embodied_core_engine",
        cwd / "arc-creativity-engine" / "embodied_core_engine",
    ]
    for cp in cand_paths:
        if cp.exists() and str(cp) not in sys.path:
            sys.path.insert(0, str(cp))


auto_extract_submission_zip()

try:
    import torch
except ImportError:
    torch = None

from stage2_embodied_puzzle_solver import (
    ARTIFACTS_DIR,
    EmbodiedPuzzleSolver,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ARC2_EVAL_DIR = ARTIFACTS_DIR / "external_arc_databank" / "arc_agi_2_eval"
DEFAULT_ARC1_EVAL_DIR = REPO_ROOT / "arc_repo" / "data" / "evaluation"
CHECKPOINTS_DIR = ARTIFACTS_DIR / "checkpoints"


def cleanup_memory():
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


class LightweightSubmissionEngine:
    def __init__(self, checkpoint_path=None):
        self.solver = EmbodiedPuzzleSolver(
            use_pain=True,
            use_curiosity=True,
            use_mortality=True,
            plastic=False,
        )
        self.solver._test_inference_mode = True
        self.solver._mental_imagery_mode = True
        self.checkpoint_loaded = None
        if checkpoint_path is None:
            latest_ckpt = CHECKPOINTS_DIR / "checkpoint_latest.npz"
            best_ckpt = CHECKPOINTS_DIR / "checkpoint_best.npz"
            if latest_ckpt.exists():
                checkpoint_path = latest_ckpt
            elif best_ckpt.exists():
                checkpoint_path = best_ckpt
        if checkpoint_path is not None and Path(checkpoint_path).exists():
            ok = self.solver.load_evolution_checkpoint(checkpoint_path)
            if ok:
                self.checkpoint_loaded = str(checkpoint_path)
        brains_dir = Path(__file__).parent / "brains"
        if hasattr(self.solver, "unified_brains") and brains_dir.exists():
            for m_name in ["sensory_cortex", "neocortex_logic", "motor_reflex"]:
                for ext in [".onnx", ".pth", ".gguf"]:
                    b_file = brains_dir / f"{m_name}{ext}"
                    t_file = ARTIFACTS_DIR / f"{m_name}{ext}"
                    if b_file.exists() and not t_file.exists():
                        import shutil
                        shutil.copy2(b_file, t_file)
            self.solver.loaded_brain_status = self.solver.unified_brains.load_all_brains()
        self.solver.absorb_stage1_3d_embodied_manifold()
        self.solver.prune_cultural_tool_library(
            max_essential_tools=24,
            max_bred_recipes=12,
            max_nested_recipes=10,
        )

    def compute_task_adaptive_budget(self, task, base_steps=6):
        train_pairs = task.get("train", [])
        if not train_pairs:
            return base_steps
        n_pairs = len(train_pairs)
        max_dim = 0
        all_colors = set()
        diff_shape = False
        for p in train_pairs:
            inp = np.array(p.get("input", [[0]]), dtype=np.int32)
            out = np.array(p.get("output", [[0]]), dtype=np.int32)
            max_dim = max(max_dim, int(inp.shape[0]), int(inp.shape[1]), int(out.shape[0]), int(out.shape[1]))
            all_colors.update(np.unique(inp).tolist())
            all_colors.update(np.unique(out).tolist())
            if inp.shape != out.shape:
                diff_shape = True
        n_colors = len(all_colors)
        if max_dim >= 20 or (diff_shape and n_colors >= 5):
            return max(int(base_steps), 10)
        elif diff_shape or n_colors >= 4 or n_pairs <= 2:
            return max(int(base_steps), 8)
        else:
            return max(int(base_steps), 6)

    def predict_task(self, task, max_steps=None):
        if max_steps is None:
            effective_steps = self.compute_task_adaptive_budget(task)
        else:
            effective_steps = max_steps
        blind_task = make_blind_test_task(task)
        res = self.solver.study_and_solve_tabletop_task(blind_task, max_steps=effective_steps)
        preds_1 = res.get("predictions_all", [res["prediction"]])
        preds_2 = res.get("alt_predictions_all", preds_1)
        attempts = []
        for p1, p2 in zip(preds_1, preds_2):
            attempts.append({
                "attempt_1": p1.tolist() if isinstance(p1, np.ndarray) else p1,
                "attempt_2": p2.tolist() if isinstance(p2, np.ndarray) else p2,
            })
        return attempts, res

    def run_submission_pack(self, input_dir=None, output_submission_path=None, max_steps=6, limit=400):
        t0 = time.time()
        if input_dir is None:
            input_dir = DEFAULT_ARC2_EVAL_DIR if DEFAULT_ARC2_EVAL_DIR.exists() else DEFAULT_ARC1_EVAL_DIR
        in_path = Path(input_dir)
        if output_submission_path is None:
            output_submission_path = ARTIFACTS_DIR / "submission.json"
        out_path = Path(output_submission_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        json_files = sorted(in_path.glob("*.json"))[:limit]
        submission_dict = {}
        train_solved_ids = []
        test_exact_ids = []
        attempt1_exact_ids = []
        attempt2_exact_ids = []
        spike_rates = []
        thought_energies = []

        for idx, jf in enumerate(json_files):
            with open(jf, "r", encoding="utf-8") as f:
                task = json.load(f)
            step_arg = None if max_steps == 6 else max_steps
            attempts, res = self.predict_task(task, max_steps=step_arg)
            submission_dict[jf.stem] = attempts
            spike_rates.append(float(res.get("mean_spike_firing_rate", 0.0)))
            thought_energies.append(float(res.get("thought_energy_consumed", 0.0)))
            if hasattr(self.solver, "check_and_trigger_autonomous_homeostatic_sleep"):
                self.solver.check_and_trigger_autonomous_homeostatic_sleep(force_check=False)

            if res.get("solved_train_exact", False):
                train_solved_ids.append(jf.stem)
                test_list = task["test"] if isinstance(task["test"], list) else [task["test"]]
                if all("output" in tp for tp in test_list):
                    a1_ok = True
                    a2_ok = True
                    any_ok = True
                    for tp, att in zip(test_list, attempts):
                        gt = np.array(tp["output"], dtype=np.int32)
                        p1 = np.array(att["attempt_1"], dtype=np.int32)
                        p2 = np.array(att["attempt_2"], dtype=np.int32)
                        m1 = p1.shape == gt.shape and np.array_equal(p1, gt)
                        m2 = p2.shape == gt.shape and np.array_equal(p2, gt)
                        if not m1:
                            a1_ok = False
                        if not m2:
                            a2_ok = False
                        if not (m1 or m2):
                            any_ok = False
                    if a1_ok:
                        attempt1_exact_ids.append(jf.stem)
                    if a2_ok:
                        attempt2_exact_ids.append(jf.stem)
                    if any_ok:
                        test_exact_ids.append(jf.stem)

            if (idx + 1) % 25 == 0:
                cleanup_memory()

        cleanup_memory()
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(submission_dict, f)

        elapsed = round(time.time() - t0, 2)
        summary = {
            "engine": "Stage4_Lightweight_20W_Submission_Inference_Engine",
            "checkpoint_salvaged": self.checkpoint_loaded,
            "dataset_evaluated": str(in_path),
            "tasks_processed": len(json_files),
            "elapsed_sec": elapsed,
            "ms_per_task": round((elapsed * 1000.0) / max(1, len(json_files)), 2),
            "train_exact_solved_count": len(train_solved_ids),
            "test_attempt1_exact_count": len(attempt1_exact_ids),
            "test_pass_at_2_exact_count": len(test_exact_ids),
            "test_solve_rate_pct": round(len(test_exact_ids) / max(1, len(json_files)) * 100.0, 2),
            "mean_spike_firing_rate": round(float(np.mean(spike_rates)) if spike_rates else 0.0, 4),
            "mean_thought_energy": round(float(np.mean(thought_energies)) if thought_energies else 0.0, 4),
            "essential_cultural_tools_packed": len(self.solver.cultural_tool_library),
            "bred_recipes_packed": len(self.solver.bred_macro_recipes),
            "nested_recipes_packed": len(self.solver.nested_macro_recipes),
            "submission_file": str(out_path),
            "solved_task_ids": test_exact_ids,
        }
        summary_path = ARTIFACTS_DIR / "stage4_submission_verification_report.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--input-dir", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--limit", type=int, default=400)
    args = parser.parse_args()

    engine = LightweightSubmissionEngine(checkpoint_path=args.checkpoint)
    report = engine.run_submission_pack(
        input_dir=args.input_dir,
        output_submission_path=args.output,
        max_steps=args.max_steps,
        limit=args.limit,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
