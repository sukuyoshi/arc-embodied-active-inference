import os
import sys
import re
import ast
import tokenize
import numpy as np

ENGINE_DIR = os.path.dirname(os.path.abspath(__file__))
SOLVER_PATH = os.path.join(ENGINE_DIR, "stage2_embodied_puzzle_solver.py")

BANNED_KEYWORDS = [
    "induce_rule_from_y",
    "dsl_",
    "xor_gate",
    "boolean_gate",
    "fractal_expand",
    "cheat",
    "hardcode",
    "lookup_table",
    "h_gru",
    "h_tilde",
    "gh_zr",
    "zero_grad",
]

ALLOWED_CORE_NAMES = {
    "perceive_core_affordances",
    "core_recipes",
    "stage1_core_manifold.npz",
    "stage1_isaac3d_core_manifold.npz",
}

PURE_PERCEPTION_AND_MOTOR_FUNCS = {
    "extract_components",
    "find_enclosed_background_regions",
    "compute_h1_holes",
    "compute_spatial_hierarchy",
    "v1_v2_preattentive_texture_and_constellation_grouping",
    "execute_unified_sensorimotor_chunk",
    "execute_sensorimotor_chain",
    "build_unified_sensorimotor_affordances",
    "perceive_core_affordances",
    "perceive_synthesized_ast_affordances",
}

FORBIDDEN_OUTPUT_VARS_IN_PURE_FUNCS = {
    "y",
    "y0",
    "y_grid",
    "y_out",
    "target_y",
    "test_output",
}

HEX8_RE = re.compile(r"^[0-9a-f]{8}$")


class PoisonOutput:
    def __getattr__(self, name):
        raise RuntimeError(f"ANTI_CHEAT_VIOLATION: Attempted to access attribute '{name}' on target output y during affordance generation!")

    def __getitem__(self, key):
        raise RuntimeError(f"ANTI_CHEAT_VIOLATION: Attempted to index target output y[{key}] during affordance generation!")

    def __array__(self, *args, **kwargs):
        raise RuntimeError("ANTI_CHEAT_VIOLATION: Attempted to convert target output y to numpy array during affordance generation!")


class PoisonTestDict(dict):
    def __getitem__(self, key):
        if key in ("output", "y", "target", "solution"):
            raise RuntimeError(f"ANTI_CHEAT_VIOLATION: Attempted to read test['{key}'] during solve_task!")
        return super().__getitem__(key)

    def get(self, key, default=None):
        if key in ("output", "y", "target", "solution"):
            raise RuntimeError(f"ANTI_CHEAT_VIOLATION: Attempted to read test.get('{key}') during solve_task!")
        return super().get(key, default)


def check_ascii_and_comments(filepath):
    violations = []
    with open(filepath, "rb") as f:
        raw = f.read()
    for idx, b in enumerate(raw):
        if b > 127:
            violations.append(f"{os.path.basename(filepath)}: Non-ASCII byte {b} at byte offset {idx}")
            break
    with open(filepath, "r", encoding="utf-8") as f:
        tokens = list(tokenize.generate_tokens(f.readline))
    for tok in tokens:
        if tok.type == tokenize.COMMENT:
            violations.append(f"{os.path.basename(filepath)}: Line {tok.start[0]} contains a comment")
    return violations


def check_ast_rules(filepath):
    violations = []
    with open(filepath, "r", encoding="utf-8") as f:
        source = f.read()
    tree = ast.parse(source, filename=filepath)
    fname = os.path.basename(filepath)

    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.body and isinstance(node.body[0], ast.Expr):
                val = node.body[0].value
                if isinstance(val, ast.Constant) and isinstance(val.value, str):
                    violations.append(f"{fname}: Line {node.body[0].lineno} contains a docstring")

        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            s = node.value.strip()
            if HEX8_RE.match(s):
                violations.append(f"{fname}: Line {node.lineno} contains suspicious 8-hex task ID literal '{s}'")
            s_low = s.lower()
            for kw in BANNED_KEYWORDS:
                if kw in s_low:
                    violations.append(f"{fname}: Line {node.lineno} contains banned string pattern '{kw}' in '{s}'")
            if s_low.startswith("core_") and s_low not in ALLOWED_CORE_NAMES:
                violations.append(f"{fname}: Line {node.lineno} contains banned hardcoded DSL prefix 'core_' in '{s}'")

        if isinstance(node, ast.Name):
            n_low = node.id.lower()
            for kw in BANNED_KEYWORDS:
                if kw in n_low:
                    violations.append(f"{fname}: Line {node.lineno} contains banned identifier '{node.id}'")
            if n_low.startswith("core_") and n_low not in ALLOWED_CORE_NAMES:
                violations.append(f"{fname}: Line {node.lineno} contains banned hardcoded DSL identifier '{node.id}'")

        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in ("backward", "Adam", "SGD", "RMSprop"):
                violations.append(
                    f"{fname}: Line {node.lineno} uses banned non-biological backpropagation API '{node.func.attr}'"
                )

        if isinstance(node, ast.FunctionDef):
            fn_low = node.name.lower()
            for kw in BANNED_KEYWORDS:
                if kw in fn_low:
                    violations.append(f"{fname}: Line {node.lineno} contains banned function name '{node.name}'")
            if fn_low.startswith("core_") and fn_low not in ALLOWED_CORE_NAMES:
                violations.append(f"{fname}: Line {node.lineno} contains banned hardcoded DSL function '{node.name}'")

            if node.name in PURE_PERCEPTION_AND_MOTOR_FUNCS:
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Name) and sub.id in FORBIDDEN_OUTPUT_VARS_IN_PURE_FUNCS:
                        violations.append(
                            f"{fname}: Pure function '{node.name}' (line {sub.lineno}) accesses forbidden output variable '{sub.id}'"
                        )
                    if isinstance(sub, ast.Subscript):
                        if (
                            isinstance(sub.value, ast.Subscript)
                            and isinstance(sub.value.value, ast.Name)
                            and sub.value.value.id in ("cur_pairs", "pairs", "train_pairs")
                        ):
                            sl = sub.slice
                            if isinstance(sl, ast.Constant) and sl.value == 1:
                                violations.append(
                                    f"{fname}: Pure function '{node.name}' (line {sub.lineno}) indexes pair output {sub.value.value.id}[...][1]"
                                )

            if node.name == "evaluate_through_ach_topological_lenses":
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute):
                        if sub.func.attr == "array_equal":
                            violations.append(
                                f"{fname}: Function '{node.name}' (line {sub.lineno}) uses direct array_equal shortcut instead of evaluate_candidate_on_pairs"
                            )

            if node.name == "perceive_relational_transition":
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Name) and sub.id == "bio_bias":
                        violations.append(
                            f"{fname}: Function '{node.name}' (line {sub.lineno}) uses hand-coded 'bio_bias' heuristic"
                        )

            if node.name == "perceive_foveal_peripheral_saccade":
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name) and sub.func.id == "extract_components":
                        if sub.args and isinstance(sub.args[0], ast.Name) and sub.args[0].id == "x_grid":
                            violations.append(
                                f"{fname}: Function '{node.name}' (line {sub.lineno}) calls extract_components on full 'x_grid' instead of saccade-integrated 'mental_canvas'"
                            )

    return violations


def run_dynamic_poison_checks():
    violations = []
    if ENGINE_DIR not in sys.path:
        sys.path.insert(0, ENGINE_DIR)
    import stage2_embodied_puzzle_solver as s2

    solver = s2.EmbodiedPuzzleSolver()
    x0 = np.array([
        [0, 0, 0, 0, 0],
        [0, 2, 2, 0, 0],
        [0, 2, 0, 0, 4],
        [0, 0, 0, 0, 0],
        [0, 0, 3, 0, 0],
    ], dtype=int)

    poison_pairs = [(x0, PoisonOutput())]
    try:
        affs = solver.build_unified_sensorimotor_affordances(poison_pairs)
        for name, fn, p_arg in affs:
            _ = fn(x0, p_arg) if fn == solver.execute_unified_sensorimotor_chunk else fn(x0)
    except Exception as e:
        violations.append(f"Dynamic PoisonOutput check failed in build_unified_sensorimotor_affordances: {e}")

    y0 = x0.copy()
    y0[y0 == 2] = 4
    task = {
        "train": [{"input": x0.tolist(), "output": y0.tolist()}],
        "test": [PoisonTestDict({"input": x0.tolist(), "output": y0.tolist()})],
    }
    try:
        _ = solver.solve_task(task)
    except Exception as e:
        violations.append(f"Dynamic PoisonTestDict check failed in solve_task: {e}")

    return violations


def main():
    all_violations = []
    py_files = [
        f for f in os.listdir(ENGINE_DIR)
        if f.endswith(".py") and f != "audit_anti_cheat.py"
    ]
    for f in sorted(py_files):
        full_path = os.path.join(ENGINE_DIR, f)
        all_violations.extend(check_ascii_and_comments(full_path))

    all_violations.extend(check_ast_rules(SOLVER_PATH))
    unified_path = os.path.join(ENGINE_DIR, "unified_biological_brain.py")
    if os.path.exists(unified_path):
        all_violations.extend(check_ast_rules(unified_path))
    stage1_path = os.path.join(ENGINE_DIR, "stage1_isaac3d_embodied_arena.py")
    if os.path.exists(stage1_path):
        all_violations.extend(check_ast_rules(stage1_path))
    for isaac_file in (
        "four_biome_humanoid_env.py",
        "agi_homeostasis_curiosity_mdp.py",
        "agi_sensorimotor_puzzle_mdp.py",
        "isaaclab_biome_humanoid_env_cfg.py",
    ):
        isaac_path = os.path.join(ENGINE_DIR, isaac_file)
        if os.path.exists(isaac_path):
            all_violations.extend(check_ast_rules(isaac_path))
    all_violations.extend(run_dynamic_poison_checks())

    print("=== ANTI-CHEAT & BIOLOGICAL PURITY AUDIT REPORT ===")
    print(f"Scanned files: {len(py_files)}")
    if not all_violations:
        print("STATUS: PASS (0 violations detected)")
        sys.exit(0)
    else:
        print(f"STATUS: FAIL ({len(all_violations)} violations detected)")
        for v in all_violations:
            print(f"  [VIOLATION] {v}")
        sys.exit(1)


if __name__ == "__main__":
    main()
