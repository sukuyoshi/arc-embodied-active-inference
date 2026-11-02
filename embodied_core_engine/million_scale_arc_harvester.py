import concurrent.futures
import ctypes
import hashlib
import importlib.util
import io
import json
import os
import sys
import time
import urllib.request
import zipfile
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

ROOT_DIR = Path(__file__).resolve().parent
REPO_ROOT = ROOT_DIR.parent
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
EXTERNAL_CACHE_DIR = ARTIFACTS_DIR / "external_arc_databank"
MILLION_STREAM_CACHE_DIR = EXTERNAL_CACHE_DIR / "million_multiverse_stream"


class MillionScaleARCMultiverseHarvester:
    def __init__(self):
        self.cache_dir = EXTERNAL_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.stream_cache_dir = MILLION_STREAM_CACHE_DIR
        self.stream_cache_dir.mkdir(parents=True, exist_ok=True)
        self.arc2_eval_dir = self.cache_dir / "arc_agi_2_eval"
        self.arc2_train_dir = self.cache_dir / "arc_agi_2_train"
        self.arc2_eval_dir.mkdir(parents=True, exist_ok=True)
        self.arc2_train_dir.mkdir(parents=True, exist_ok=True)
        self.re_arc_dir = self.cache_dir / "re_arc"
        self.collection_dir = self.cache_dir / "arc_dataset_collection"
        self.re_arc_generators = {}
        self.re_arc_keys = []
        self.world_static_index = []
        self._load_re_arc_if_available()
        self._build_world_static_index()

    def ensure_world_repositories_ingested(self, full_arc2_train=True):
        stats = {}
        if not self.re_arc_dir.exists() or not any(self.re_arc_dir.rglob("generators.py")):
            self._download_and_extract_zip(
                "https://github.com/michaelhodel/re-arc/archive/refs/heads/main.zip",
                self.re_arc_dir,
            )
        self._load_re_arc_if_available()
        stats["re_arc_generators_loaded"] = len(self.re_arc_keys)

        if not self.collection_dir.exists() or len(list(self.collection_dir.rglob("*.json"))) < 100:
            self._download_and_extract_zip(
                "https://github.com/neoneye/arc-dataset-collection/archive/refs/heads/main.zip",
                self.collection_dir,
            )

        self.fetch_github_arc_agi_2_split(split="evaluation", limit=120)
        self.fetch_github_arc_agi_2_split(split="training", limit=1000 if full_arc2_train else 200)

        self._build_world_static_index()
        stats["world_static_json_tasks_indexed"] = len(self.world_static_index)
        stats["arc2_train_cached"] = len(list(self.arc2_train_dir.glob("*.json")))
        stats["arc2_eval_cached"] = len(list(self.arc2_eval_dir.glob("*.json")))
        stats["virtual_multiverse_capacity"] = 1000000
        return stats

    def _download_and_extract_zip(self, url, dest_dir):
        dest_dir.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        raw_bytes = urllib.request.urlopen(req, timeout=180).read()
        with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zf:
            zf.extractall(dest_dir)

    def _load_re_arc_if_available(self):
        if self.re_arc_generators:
            return
        gen_files = list(self.re_arc_dir.rglob("generators.py"))
        if not gen_files:
            return
        gen_path = gen_files[0]
        re_root = str(gen_path.parent)
        if re_root not in sys.path:
            sys.path.insert(0, re_root)
        try:
            spec = importlib.util.spec_from_file_location("re_arc_generators", str(gen_path))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            for name in dir(mod):
                if name.startswith("generate_"):
                    fn = getattr(mod, name)
                    if callable(fn):
                        tid = name.replace("generate_", "")
                        self.re_arc_generators[tid] = fn
            self.re_arc_keys = sorted(self.re_arc_generators.keys())
        except Exception:
            self.re_arc_generators = {}
            self.re_arc_keys = []

    def _is_valid_arc_task(self, obj):
        if not isinstance(obj, dict):
            return False
        tr = obj.get("train")
        te = obj.get("test")
        if not isinstance(tr, list) or len(tr) == 0 or te is None:
            return False
        first_tr = tr[0]
        if not isinstance(first_tr, dict) or "input" not in first_tr or "output" not in first_tr:
            return False
        test_pair = te[0] if isinstance(te, list) and len(te) > 0 else te
        if not isinstance(test_pair, dict) or "input" not in test_pair or "output" not in test_pair:
            return False
        return True

    def _compute_grid_hash(self, grid):
        arr = np.asarray(grid, dtype=np.int32)
        return hashlib.sha256(arr.tobytes()).hexdigest()

    def _collect_held_out_eval_data(self):
        held_out_ids = set()
        held_out_hashes = set()
        eval_dirs = [
            REPO_ROOT / "arc_repo" / "data" / "evaluation",
            self.arc2_eval_dir,
        ]
        for ed in eval_dirs:
            if ed.exists():
                for jf in ed.glob("*.json"):
                    held_out_ids.add(jf.stem)
                    try:
                        with open(jf, "r", encoding="utf-8") as f:
                            t_data = json.load(f)
                        for sp in ("train", "test"):
                            pairs = t_data.get(sp, [])
                            if isinstance(pairs, dict):
                                pairs = [pairs]
                            for p in pairs:
                                if isinstance(p, dict):
                                    if "input" in p:
                                        held_out_hashes.add(self._compute_grid_hash(p["input"]))
                                    if "output" in p:
                                        held_out_hashes.add(self._compute_grid_hash(p["output"]))
                    except Exception:
                        pass
        return held_out_ids, held_out_hashes

    def _is_eval_leak(self, jf, held_out_ids, held_out_hashes):
        if jf.stem in held_out_ids:
            return True
        try:
            with open(jf, "r", encoding="utf-8") as f:
                t_data = json.load(f)
            for sp in ("train", "test"):
                pairs = t_data.get(sp, [])
                if isinstance(pairs, dict):
                    pairs = [pairs]
                for p in pairs:
                    if isinstance(p, dict):
                        if "input" in p and self._compute_grid_hash(p["input"]) in held_out_hashes:
                            return True
                        if "output" in p and self._compute_grid_hash(p["output"]) in held_out_hashes:
                            return True
        except Exception:
            pass
        return False

    def _build_world_static_index(self):
        held_out_ids, held_out_hashes = self._collect_held_out_eval_data()
        index = []
        local_dirs = [
            (REPO_ROOT / "arc_repo" / "data" / "training", "ARC_AGI_1_Training"),
            (REPO_ROOT / "arc_repo" / "data" / "agi_stress_suite_v3", "External_AGI_Stress_Suite_v3"),
            (REPO_ROOT / "arc_repo" / "data" / "novel_benchmark_400", "External_Novel_Benchmark_400_v1"),
            (REPO_ROOT / "arc_repo" / "data" / "novel_benchmark_400_v2", "External_Novel_Benchmark_400_v2"),
            (self.arc2_train_dir, "GitHub_arcprize_ARC_AGI_2_training"),
        ]
        for folder, label in local_dirs:
            if folder.exists():
                for jf in sorted(folder.glob("*.json")):
                    if not self._is_eval_leak(jf, held_out_ids, held_out_hashes):
                        index.append((label, jf.stem, jf))

        if self.collection_dir.exists():
            for jf in sorted(self.collection_dir.rglob("*.json")):
                if self._is_eval_leak(jf, held_out_ids, held_out_hashes):
                    continue
                rel_lower = str(jf.relative_to(self.collection_dir)).lower()
                if "evaluation" in rel_lower:
                    continue
                parts = jf.relative_to(self.collection_dir).parts
                sub_bank = parts[2] if len(parts) > 2 else "World_ARC_Collection"
                index.append((f"WorldCollection_{sub_bank}", jf.stem, jf))

        self.world_static_index = index

    def fetch_github_arc_agi_2_split(self, split="evaluation", limit=120):
        target_dir = self.arc2_eval_dir if split == "evaluation" else self.arc2_train_dir
        cached = sorted(target_dir.glob("*.json"))
        if len(cached) < limit:
            index_url = f"https://raw.githubusercontent.com/arcprize/ARC-AGI-2/main/data/{split}.txt"
            try:
                req = urllib.request.Request(index_url, headers={"User-Agent": "Mozilla/5.0"})
                raw_txt = urllib.request.urlopen(req, timeout=15).read().decode("utf-8")
                task_ids = [line.strip() for line in raw_txt.splitlines() if line.strip()][:limit]

                def download_one(tid):
                    out_file = target_dir / f"{tid}.json"
                    if out_file.exists():
                        return
                    t_url = f"https://raw.githubusercontent.com/arcprize/ARC-AGI-2/main/data/{split}/{tid}.json"
                    try:
                        r_t = urllib.request.Request(t_url, headers={"User-Agent": "Mozilla/5.0"})
                        data_bytes = urllib.request.urlopen(r_t, timeout=12).read()
                        with open(out_file, "wb") as fw:
                            fw.write(data_bytes)
                    except Exception:
                        pass

                with concurrent.futures.ThreadPoolExecutor(max_workers=24) as pool:
                    list(pool.map(download_one, task_ids))
            except Exception:
                pass

        loaded = []
        src_label = f"GitHub_arcprize_ARC_AGI_2_{split}"
        for jf in sorted(target_dir.glob("*.json"))[:limit]:
            try:
                with open(jf, "r", encoding="utf-8") as f:
                    loaded.append({
                        "source_bank": src_label,
                        "task_id": jf.stem,
                        "task": json.load(f),
                    })
            except Exception:
                continue
        return loaded

    def sample_re_arc_task(self, global_task_idx):
        self._load_re_arc_if_available()
        if not self.re_arc_keys:
            return None
        gen_key = self.re_arc_keys[int(global_task_idx) % len(self.re_arc_keys)]
        gen_fn = self.re_arc_generators[gen_key]
        tier = (int(global_task_idx) // max(1, len(self.re_arc_keys))) % 5
        diff_bounds = [
            (0.0, 0.25),
            (0.15, 0.45),
            (0.30, 0.65),
            (0.50, 0.85),
            (0.65, 1.00),
        ][tier]
        train_pairs = []
        test_pair = None
        try:
            for k in range(4):
                ex = gen_fn(diff_bounds[0], diff_bounds[1])
                g_in = np.array(ex["input"], dtype=np.int32)
                g_out = np.array(ex["output"], dtype=np.int32)
                if g_in.ndim != 2 or g_out.ndim != 2:
                    return None
                if max(g_in.shape) > 30 or max(g_out.shape) > 30 or min(g_in.shape) < 1 or min(g_out.shape) < 1:
                    return None
                pair = {"input": g_in.tolist(), "output": g_out.tolist()}
                if k < 3:
                    train_pairs.append(pair)
                else:
                    test_pair = pair
            return {
                "source_bank": "MillionMultiverse_RE_ARC_Procedural",
                "task_id": f"REARC_{gen_key}_m{int(global_task_idx):07d}",
                "task": {
                    "id": f"REARC_{gen_key}_m{int(global_task_idx):07d}",
                    "archetype": f"re_arc_{gen_key}",
                    "train": train_pairs,
                    "test": test_pair,
                },
            }
        except Exception:
            return None

    def sample_relational_combinatorial_task(self, global_task_idx):
        rng = np.random.default_rng(1000000 + int(global_task_idx) * 97)
        task_mode = int(rng.integers(0, 8))
        train_pairs = []
        test_pair = None

        grid_h = int(rng.integers(8, 16))
        grid_w = int(rng.integers(8, 16))
        bg_col = 0

        rule_dr = int(rng.integers(1, 3))
        rule_dc = int(rng.integers(1, 3))
        rule_c0 = int(rng.integers(1, 9))
        rule_c1 = int((rule_c0 + 3) % 9 + 1)
        rule_axis = int(rng.integers(0, 2))

        for k in range(4):
            cur_h = grid_h
            cur_w = grid_w
            g_in = np.zeros((cur_h, cur_w), dtype=np.int32)
            g_out = np.zeros((cur_h, cur_w), dtype=np.int32)

            if task_mode == 0:
                bh = int(rng.integers(2, max(3, cur_h // 2 - 1)))
                bw = int(rng.integers(2, max(3, cur_w // 2 - 1)))
                r0 = int(rng.integers(0, max(1, cur_h - bh - rule_dr)))
                c0 = int(rng.integers(0, max(1, cur_w - bw - rule_dc)))
                g_in[r0 : r0 + bh, c0 : c0 + bw] = rule_c0
                r_dest = min(cur_h - bh, r0 + rule_dr)
                c_dest = min(cur_w - bw, c0 + rule_dc)
                g_out[r_dest : r_dest + bh, c_dest : c_dest + bw] = rule_c1
                arch_name = "sensorimotor_rigid_push_transport"

            elif task_mode == 1:
                mid_c = cur_w // 2
                for _ in range(int(rng.integers(2, 6))):
                    rr = int(rng.integers(0, cur_h))
                    cc = int(rng.integers(0, mid_c))
                    g_in[rr, cc] = rule_c0
                g_out = g_in.copy()
                g_out[:, cur_w - mid_c:] = np.fliplr(g_in[:, :mid_c])
                arch_name = "sensorimotor_bilateral_reflection"

            elif task_mode == 2:
                r0 = int(rng.integers(1, max(2, cur_h - 5)))
                c0 = int(rng.integers(1, max(2, cur_w - 5)))
                bh = int(rng.integers(4, min(cur_h - r0, 7)))
                bw = int(rng.integers(4, min(cur_w - c0, 7)))
                g_in[r0 : r0 + bh, c0 : c0 + bw] = rule_c0
                g_in[r0 + 1 : r0 + bh - 1, c0 + 1 : c0 + bw - 1] = bg_col
                g_out = g_in.copy()
                g_out[r0 + 1 : r0 + bh - 1, c0 + 1 : c0 + bw - 1] = rule_c1
                arch_name = "sensorimotor_cavity_filling"

            elif task_mode == 3:
                cols = list(rng.permutation(np.arange(1, 10))[:3])
                for idx_b, c in enumerate(cols):
                    b_sz = idx_b + 1
                    rr = int(rng.integers(0, cur_h - b_sz))
                    cc = int(rng.integers(0, cur_w - b_sz))
                    g_in[rr : rr + b_sz, cc : cc + b_sz] = int(c)
                target_col = int(cols[-1])
                g_out = np.where(g_in > 0, target_col, bg_col).astype(np.int32)
                arch_name = "sensorimotor_mass_rank_harmonic"

            elif task_mode == 4:
                pts = []
                for _ in range(int(rng.integers(3, 7))):
                    rr = int(rng.integers(0, cur_h))
                    cc = int(rng.integers(0, cur_w))
                    g_in[rr, cc] = rule_c0
                    pts.append((rr, cc))
                g_out = g_in.copy()
                if rule_axis == 0:
                    med_r = int(np.median([p[0] for p in pts]))
                    g_out[med_r, :] = rule_c0
                else:
                    med_c = int(np.median([p[1] for p in pts]))
                    g_out[:, med_c] = rule_c0
                arch_name = "sensorimotor_centroid_alignment"

            elif task_mode == 5:
                r1 = int(rng.integers(1, max(2, cur_h - 2)))
                c1 = int(rng.integers(1, max(2, cur_w - 2)))
                g_in[r1, c1] = rule_c0
                g_in[cur_h - 1 - r1, cur_w - 1 - c1] = rule_c1
                g_out = g_in.copy()
                g_out[r1, c1] = rule_c1
                g_out[cur_h - 1 - r1, cur_w - 1 - c1] = rule_c0
                arch_name = "sensorimotor_inversive_swap"

            elif task_mode == 6:
                bh = int(rng.integers(2, 4))
                bw = int(rng.integers(2, 4))
                r0 = int(rng.integers(0, cur_h - bh))
                c0 = int(rng.integers(0, cur_w - bw))
                g_in[r0 : r0 + bh, c0 : c0 + bw] = rule_c0
                g_out[cur_h - bh : cur_h, c0 : c0 + bw] = rule_c0
                arch_name = "sensorimotor_gravitational_settling"

            else:
                off_r = int(rng.integers(0, 2))
                off_c = int(rng.integers(0, 2))
                for rr in range(off_r, cur_h, 2):
                    for cc in range(off_c, cur_w, 2):
                        g_in[rr, cc] = rule_c0
                g_out = np.rot90(g_in, 2).astype(np.int32)
                arch_name = "sensorimotor_rotational_lattice"

            pair = {"input": g_in.tolist(), "output": g_out.tolist()}
            if k < 3:
                train_pairs.append(pair)
            else:
                test_pair = pair

        task_id = f"Multiverse_Sensorimotor_{arch_name}_{int(global_task_idx):07d}"
        return {
            "source_bank": "MillionMultiverse_Relational_Synthesis",
            "task_id": task_id,
            "task": {
                "id": task_id,
                "archetype": arch_name,
                "train": train_pairs,
                "test": test_pair,
            },
        }

    def materialize_million_stream_batch(self, batch_id=0, batch_size=250):
        batch_file = self.stream_cache_dir / f"multiverse_batch_{int(batch_id):05d}.json"
        if batch_file.exists():
            try:
                with open(batch_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass

        items = []
        base_idx = (int(batch_id) * int(batch_size)) % 1000000
        for offset in range(batch_size):
            g_idx = (base_idx + offset) % 1000000
            if offset % 2 == 0:
                sample = self.sample_re_arc_task(g_idx)
                if sample is None:
                    sample = self.sample_relational_combinatorial_task(g_idx)
            else:
                sample = self.sample_relational_combinatorial_task(g_idx)
            if sample is not None:
                items.append(sample)

        try:
            with open(batch_file, "w", encoding="utf-8") as f:
                json.dump(items, f)
        except Exception:
            pass
        return items

    def stream_generational_hunting_grounds(
        self,
        generation_v2=1,
        world_static_slice_size=220,
        multiverse_stream_size=140,
    ):
        if not self.world_static_index:
            self._build_world_static_index()

        banks = []
        n_static = len(self.world_static_index)
        if n_static > 0:
            stride = 197
            start_pos = ((int(generation_v2) - 1) * world_static_slice_size) % n_static
            selected_indices = [(start_pos + k * stride) % n_static for k in range(min(world_static_slice_size, n_static))]
            for idx in selected_indices:
                src_label, tid, jf_path = self.world_static_index[idx]
                try:
                    with open(jf_path, "r", encoding="utf-8") as f:
                        t_obj = json.load(f)
                    if self._is_valid_arc_task(t_obj):
                        banks.append({
                            "source_bank": src_label,
                            "task_id": tid,
                            "task": t_obj,
                        })
                except Exception:
                    continue

        m_items = self.materialize_million_stream_batch(
            batch_id=int(generation_v2),
            batch_size=multiverse_stream_size,
        )
        banks.extend(m_items)
        return banks


if __name__ == "__main__":
    t0 = time.time()
    harvester = MillionScaleARCMultiverseHarvester()
    ingest_stats = harvester.ensure_world_repositories_ingested(full_arc2_train=True)
    sample_batch = harvester.materialize_million_stream_batch(batch_id=0, batch_size=500)
    stream_preview = harvester.stream_generational_hunting_grounds(generation_v2=1)
    ingest_stats["initial_materialized_multiverse_tasks"] = len(sample_batch)
    ingest_stats["sample_generation_stream_count"] = len(stream_preview)
    ingest_stats["elapsed_sec"] = round(time.time() - t0, 2)
    out_meta = ARTIFACTS_DIR / "million_scale_ingestion_report.json"
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(ingest_stats, f, indent=2)
    print(json.dumps(ingest_stats, indent=2))
