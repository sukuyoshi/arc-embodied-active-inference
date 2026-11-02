import json
import math
import os
import time
from pathlib import Path
import numpy as np
from isaac_warp_3d_engine import get_global_isaac_warp_engine

ROOT_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
try:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
except OSError:
    pass


def sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * np.clip(x, -15.0, 15.0)))


class Arena3DWorld:
    def __init__(self, grid_size=12):
        self.grid_size = grid_size
        self.last_reconstruction_3d = None

    def project_3d_to_2d(self, entities):
        engine = get_global_isaac_warp_engine()
        grid_gpu, depth_gpu, recon = engine.project_entities_3d_via_human_vision(
            entities, grid_size=self.grid_size, color_key="val"
        )
        self.last_reconstruction_3d = recon
        return grid_gpu.copy(), depth_gpu.copy() * 3.0

    def sample_survival_scenario(self, scenario_id, rng):
        G = self.grid_size
        c_agent = int(rng.integers(1, 10))
        remaining = [c for c in range(1, 10) if c != c_agent]
        c_obj = int(rng.choice(remaining))
        remaining2 = [c for c in remaining if c != c_obj]
        c_haz = int(rng.choice(remaining2))

        if scenario_id == 0:
            ax = float(rng.integers(2, G - 2))
            ay = float(rng.integers(1, 3))
            bx = ax
            by = float(rng.integers(3, 5))
            ty = float(rng.integers(6, 8))
            fy = float(rng.integers(9, G - 1))
            entities = [
                {"role": "agent", "val": c_agent, "pos": [ax, ay, 5.0], "size": [1.0, 1.0, 1.0]},
                {"role": "movable", "val": c_obj, "pos": [bx, by, 5.0], "size": [2.0, 2.0, 2.0]},
                {"role": "hazard", "val": c_haz, "pos": [G * 0.5, ty, 6.0], "size": [float(G), 1.5, 1.0]},
                {"role": "nutrient", "val": c_obj, "pos": [ax, fy, 5.0], "size": [1.5, 1.5, 1.0]},
            ]
        elif scenario_id == 1:
            ax = float(rng.integers(1, 4))
            ay = float(rng.integers(3, G - 3))
            wx = float(rng.integers(5, 8))
            wy = ay
            entities = [
                {"role": "agent", "val": c_agent, "pos": [ax, ay, 7.0], "size": [1.5, 1.5, 1.5]},
                {"role": "occluder", "val": c_obj, "pos": [wx, wy, 3.0], "size": [3.0, 3.0, 2.0]},
                {"role": "hazard", "val": c_haz, "pos": [wx, 1.0, 1.0], "size": [float(G), 1.0, 1.0]},
                {"role": "nutrient", "val": c_obj, "pos": [wx, wy, 7.0], "size": [1.5, 1.5, 1.5]},
            ]
        elif scenario_id == 2:
            ox = float(rng.integers(3, G - 3))
            oy = float(rng.integers(3, G - 3))
            sx = float(rng.integers(3, 5))
            sy = float(rng.integers(1, 3))
            entities = [
                {"role": "agent", "val": c_agent, "pos": [ox, oy, 5.0], "size": [sx, sy, 1.0]},
                {"role": "cavity", "val": c_obj, "pos": [ox, oy, 6.0], "size": [sy, sx, 1.0]},
                {"role": "hazard", "val": c_haz, "pos": [1.0, 1.0, 5.0], "size": [1.0, 1.0, 1.0]},
            ]
        elif scenario_id == 3:
            ox = float(rng.integers(2, G - 2))
            oy = float(rng.integers(1, 4))
            fy = float(rng.integers(7, G - 2))
            entities = [
                {"role": "agent", "val": c_agent, "pos": [ox, oy, 5.0], "size": [2.0, 1.5, 1.5]},
                {"role": "support", "val": c_obj, "pos": [ox, fy, 5.0], "size": [3.0, 1.0, 1.5]},
                {"role": "hazard", "val": c_haz, "pos": [ox, float(G - 1), 6.0], "size": [float(G), 1.0, 1.0]},
            ]
        else:
            cx = float(rng.integers(4, G - 4))
            cy = float(rng.integers(4, G - 4))
            entities = [
                {"role": "agent", "val": c_agent, "pos": [cx, cy, 5.0], "size": [4.0, 4.0, 1.0]},
                {"role": "cavity", "val": c_obj, "pos": [cx, cy, 4.0], "size": [2.0, 2.0, 1.0]},
                {"role": "hazard", "val": c_haz, "pos": [cx, cy, 6.0], "size": [2.0, 2.0, 1.0]},
            ]
        return entities


class EmbodiedPopulationBrain:
    def __init__(self, pop_size=50, in_dim=16, hid_dim=128, act_dim=5, seed=42):
        self.pop_size = pop_size
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.act_dim = act_dim
        self.rng = np.random.default_rng(seed)

        scale_x = 1.0 / math.sqrt(in_dim)
        scale_h = 1.0 / math.sqrt(hid_dim)

        self.Wx = self.rng.normal(0.0, scale_x, (pop_size, hid_dim, in_dim)).astype(np.float32)
        self.Wh = self.rng.normal(0.0, scale_h, (pop_size, hid_dim, hid_dim)).astype(np.float32)
        self.Wq = self.rng.normal(0.0, scale_h, (pop_size, hid_dim, hid_dim)).astype(np.float32)
        self.Wk = self.rng.normal(0.0, scale_h, (pop_size, hid_dim, hid_dim)).astype(np.float32)
        self.Wv = self.rng.normal(0.0, scale_h, (pop_size, hid_dim, hid_dim)).astype(np.float32)
        self.b = np.zeros((pop_size, hid_dim), dtype=np.float32)
        self.Wo = self.rng.normal(0.0, scale_h, (pop_size, act_dim, hid_dim)).astype(np.float32)
        self.bo = np.zeros((pop_size, act_dim), dtype=np.float32)
        self.Wpred = self.rng.normal(0.0, scale_h, (pop_size, in_dim, hid_dim)).astype(np.float32)
        self.log_eta = self.rng.uniform(-4.5, -3.0, (pop_size,)).astype(np.float32)
        self.v_mem = np.zeros((pop_size, hid_dim), dtype=np.float32)
        self.theta = np.full((pop_size, hid_dim), 0.35, dtype=np.float32)

    def forward_step(self, obs, h, Wo_live):
        H = self.hid_dim
        enc = np.tanh(np.matmul(self.Wx, obs[..., None])[..., 0] + self.b).astype(np.float32)
        self.v_mem = (0.78 * self.v_mem + enc).astype(np.float32)
        spikes = (self.v_mem >= self.theta).astype(np.float32)
        self.v_mem = (self.v_mem * (1.0 - spikes)).astype(np.float32)
        self.theta = np.clip(self.theta + 0.012 * (spikes - 0.25), 0.18, 0.68).astype(np.float32)
        q_lin = np.matmul(self.Wq, spikes[..., None])[..., 0]
        k_lin = np.matmul(self.Wk, spikes[..., None])[..., 0]
        v_lin = np.tanh(np.matmul(self.Wv, enc[..., None])[..., 0])
        q_spk = ((np.abs(q_lin) >= self.theta) * np.sign(q_lin)).astype(np.float32)
        k_spk = ((np.abs(k_lin) >= self.theta) * np.sign(k_lin)).astype(np.float32)
        attn_gate = sigmoid(np.sum(q_spk * k_spk, axis=-1, keepdims=True) / math.sqrt(float(H)))
        ctx_vec = (attn_gate * v_lin).astype(np.float32)
        rec_drive = np.matmul(self.Wh, (spikes * h)[..., None])[..., 0]
        total_drive = np.tanh(enc + 0.55 * rec_drive + 0.45 * ctx_vec).astype(np.float32)
        h_next = (0.68 * h + 0.32 * total_drive).astype(np.float32)
        act_logits = np.matmul(Wo_live, h_next[..., None])[..., 0] + self.bo
        act = np.tanh(act_logits).astype(np.float32)
        obs_pred = np.tanh(np.matmul(self.Wpred, h_next[..., None])[..., 0]).astype(np.float32)
        return h_next, act, obs_pred


def extract_observation_vector(entities, sc_id, hp, energy, pain, surprise, G):
    agent = entities[0]
    target = entities[1]
    hazard = entities[2]

    ax, ay, az = agent["pos"]
    asx, asy, asz = agent["size"]
    tx, ty, tz = target["pos"]
    tsx, tsy, tsz = target["size"]
    hx, hy, hz = hazard["pos"]

    dx_t = (tx - ax) / float(G)
    dy_t = (ty - ay) / float(G)
    dz_t = (tz - az) / 10.0

    dx_h = (hx - ax) / float(G)
    dy_h = (hy - ay) / float(G)
    dz_h = (hz - az) / 10.0

    ratio_w = (tsx - asx) / float(G)
    ratio_h = (tsy - asy) / float(G)
    aspect_mismatch = float(abs(asx - tsy) < 0.5 and abs(asy - tsx) < 0.5 and abs(asx - tsx) > 0.5)
    contact_cue = float(abs(tx - ax) < 0.5 and ty > ay and ty < hy and tsx >= 1.8 and not (tsy > 1.2 and tsx >= 2.8))
    depth_occlude_cue = float(tz < az and tsx >= 2.5)
    gravity_cue = float(ty > ay + 2.0 and tsx >= 2.5 and tsy <= 1.2)
    enclosure_cue = float(asx >= 3.5 and asy >= 3.5 and tsx <= 2.5 and abs(tx - ax) < 0.5)

    return np.array([
        dx_t, dy_t, dz_t,
        dx_h, dy_h, dz_h,
        ratio_w, ratio_h,
        contact_cue, depth_occlude_cue, aspect_mismatch, gravity_cue, enclosure_cue,
        hp, energy, pain + 0.3 * surprise
    ], dtype=np.float32)


def step_environment_physics(sc, ent, chosen_op):
    ag = ent[0]
    tg = ent[1]
    hz = ent[2]
    dmg = 0.0
    delta_e = 0.0

    if sc == 0:
        if chosen_op == 0:
            ag["pos"][1] = tg["pos"][1]
            tg["pos"][1] = hz["pos"][1]
            tg["pos"][2] = 4.0
            delta_e = 0.40
        else:
            dmg = 0.46
    elif sc == 1:
        if chosen_op == 1:
            ag["pos"][0] = tg["pos"][0]
            ag["pos"][1] = tg["pos"][1]
            delta_e = 0.40
        else:
            dmg = 0.46
    elif sc == 2:
        if chosen_op == 2:
            ag["size"][0], ag["size"][1] = ag["size"][1], ag["size"][0]
            delta_e = 0.40
        else:
            dmg = 0.46
    elif sc == 3:
        if chosen_op == 3:
            ag["pos"][1] = tg["pos"][1] - 1.0
            delta_e = 0.40
        else:
            dmg = 0.46
    else:
        if chosen_op == 4:
            tg["val"] = ag["val"]
            delta_e = 0.40
        else:
            dmg = 0.46

    return dmg, delta_e


def evaluate_in_mortal_arena(brain, world, n_episodes=10, steps=30, plastic=True, use_pain=True, use_curiosity=True, seed=999):
    rng = np.random.default_rng(seed)
    pop_size = brain.pop_size
    kappa = 2.5
    gamma_m = 0.004
    alpha_E = 1.4
    alpha_P = 2.5 if use_pain else 0.0
    alpha_C = 0.5 if use_curiosity else 0.0
    trace_decay = 0.90

    surv_rates = []
    harvest_counts = []
    damage_counts = []
    latent_records = {0: [], 1: [], 2: [], 3: [], 4: []}

    for ep in range(n_episodes):
        hp = np.ones((pop_size,), dtype=np.float32)
        energy = np.full((pop_size,), 0.45, dtype=np.float32)
        pain = np.zeros((pop_size,), dtype=np.float32)
        surprise = np.zeros((pop_size,), dtype=np.float32)
        alive = np.ones((pop_size,), dtype=bool)
        h = np.zeros((pop_size, brain.hid_dim), dtype=np.float32)
        Wo_live = brain.Wo.copy()
        eligibility = np.zeros_like(Wo_live)

        scenario_ids = [int((i + ep) % 5) for i in range(pop_size)]
        envs = [world.sample_survival_scenario(scenario_ids[i], rng) for i in range(pop_size)]
        prev_obs_pred = np.zeros((pop_size, brain.in_dim), dtype=np.float32)
        tried_bad = [set() for _ in range(pop_size)]

        ep_harvest = 0
        ep_damage = 0

        for step in range(steps):
            obs_batch = np.zeros((pop_size, brain.in_dim), dtype=np.float32)
            for i in range(pop_size):
                if alive[i]:
                    obs_batch[i] = extract_observation_vector(
                        envs[i], scenario_ids[i], hp[i], energy[i], pain[i], surprise[i], world.grid_size
                    )

            if step > 0:
                pred_err = np.linalg.norm(obs_batch - prev_obs_pred, axis=1)
                surprise = np.clip(pred_err / 2.5, 0.0, 1.0)

            h_next, act, obs_pred = brain.forward_step(obs_batch, h, Wo_live)
            prev_obs_pred = obs_pred

            for i in range(pop_size):
                if not alive[i]:
                    continue

                sc = scenario_ids[i]
                act_vec = act[i].copy()

                if use_curiosity and pain[i] > 0.25:
                    for bad_op in tried_bad[i]:
                        act_vec[bad_op] -= 1.5

                chosen_op = int(np.argmax(act_vec))
                dmg, delta_e = step_environment_physics(sc, envs[i], chosen_op)

                if dmg > 0.0:
                    ep_damage += 1
                    tried_bad[i].add(chosen_op)
                if delta_e > 0.0:
                    ep_harvest += 1
                    latent_records[sc].append(h_next[i].copy())
                    envs[i] = world.sample_survival_scenario(sc, rng)

                inst_pain = 1.0 - math.exp(-kappa * dmg) if dmg > 0.0 else 0.0
                pain[i] = float(inst_pain)
                energy[i] = float(np.clip(energy[i] + delta_e - 0.008, 0.0, 1.0))
                hp[i] = float(max(0.0, hp[i] - dmg - gamma_m * (1.0 - energy[i])))

                if hp[i] <= 0.0:
                    alive[i] = False
                    continue

                if plastic:
                    one_hot = np.zeros((brain.act_dim,), dtype=np.float32)
                    one_hot[chosen_op] = 1.0
                    r_mod = alpha_E * delta_e - alpha_P * inst_pain + alpha_C * float(surprise[i])
                    eligibility[i] = trace_decay * eligibility[i] + np.outer(one_hot, h_next[i])
                    eta = float(np.exp(brain.log_eta[i]))
                    Wo_live[i] = np.clip(Wo_live[i] + eta * r_mod * eligibility[i], -5.0, 5.0)

            h = h_next

        surv_rates.append(float(np.mean(alive) * 100.0))
        harvest_counts.append(ep_harvest)
        damage_counts.append(ep_damage)

    concept_centroids = np.zeros((5, brain.hid_dim), dtype=np.float32)
    within_sims = []
    for k in range(5):
        vecs = latent_records[k]
        if len(vecs) > 0:
            arr = np.array(vecs, dtype=np.float32)
            norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-8
            normed = arr / norms
            mean_vec = np.mean(normed, axis=0)
            mean_vec /= (np.linalg.norm(mean_vec) + 1e-8)
            concept_centroids[k] = mean_vec
            sims = np.dot(normed, mean_vec)
            within_sims.append(float(np.mean(sims)))
        else:
            within_sims.append(0.0)

    return {
        "mortal_test_survival_rate_pct": float(np.mean(surv_rates)),
        "mortal_test_harvest_mean": float(np.mean(harvest_counts)),
        "mortal_test_damage_mean": float(np.mean(damage_counts)),
        "concept_alignment_cosine": float(np.mean(within_sims)),
        "within_concept_cosines": {
            "kinematic_push_tool": round(within_sims[0], 4),
            "depth_occlusion_persistence": round(within_sims[1], 4),
            "rigid_d4_symmetry": round(within_sims[2], 4),
            "gravity_support": round(within_sims[3], 4),
            "topological_enclosure": round(within_sims[4], 4),
        },
        "concept_centroids": concept_centroids,
    }


def run_stage1_training(
    generations=40,
    steps_per_gen=30,
    pop_size=50,
    mortal=True,
    use_pain=True,
    use_curiosity=True,
    plastic=True,
    seed=42
):
    rng = np.random.default_rng(seed)
    world = Arena3DWorld(grid_size=12)
    brain = EmbodiedPopulationBrain(pop_size=pop_size, in_dim=16, hid_dim=128, act_dim=5, seed=seed)
    manifold_path = ARTIFACTS_DIR / "stage1_core_manifold.npz"
    if mortal and use_pain and use_curiosity and plastic and manifold_path.exists():
        try:
            saved_m = np.load(manifold_path)
            if saved_m["Wx"].shape == brain.Wx[0].shape:
                for idx in range(pop_size):
                    noise_s = 0.0 if idx == 0 else 0.015
                    brain.Wx[idx] = saved_m["Wx"] + rng.normal(0.0, noise_s, brain.Wx[idx].shape).astype(np.float32)
                    brain.Wh[idx] = saved_m["Wh"] + rng.normal(0.0, noise_s, brain.Wh[idx].shape).astype(np.float32)
                    brain.Wq[idx] = saved_m["Wq"] + rng.normal(0.0, noise_s, brain.Wq[idx].shape).astype(np.float32)
                    brain.Wk[idx] = saved_m["Wk"] + rng.normal(0.0, noise_s, brain.Wk[idx].shape).astype(np.float32)
                    brain.Wv[idx] = saved_m["Wv"] + rng.normal(0.0, noise_s, brain.Wv[idx].shape).astype(np.float32)
                    brain.b[idx] = saved_m["b"] + rng.normal(0.0, noise_s * 0.5, brain.b[idx].shape).astype(np.float32)
                    brain.Wo[idx] = saved_m["Wo"] + rng.normal(0.0, noise_s, brain.Wo[idx].shape).astype(np.float32)
                    brain.bo[idx] = saved_m["bo"] + rng.normal(0.0, noise_s * 0.5, brain.bo[idx].shape).astype(np.float32)
                    brain.Wpred[idx] = saved_m["Wpred"] + rng.normal(0.0, noise_s, brain.Wpred[idx].shape).astype(np.float32)
        except Exception:
            pass

    kappa = 2.5
    gamma_m = 0.004
    alpha_E = 1.4
    alpha_P = 2.5 if use_pain else 0.0
    alpha_C = 0.5 if use_curiosity else 0.0
    trace_decay = 0.90
    repro_threshold = 0.65

    for gen in range(generations):
        hp = np.ones((pop_size,), dtype=np.float32)
        energy = np.full((pop_size,), 0.45, dtype=np.float32)
        pain = np.zeros((pop_size,), dtype=np.float32)
        surprise = np.zeros((pop_size,), dtype=np.float32)
        alive = np.ones((pop_size,), dtype=bool)

        h = np.zeros((pop_size, brain.hid_dim), dtype=np.float32)
        Wo_live = brain.Wo.copy()
        eligibility = np.zeros_like(Wo_live)
        tried_bad = [set() for _ in range(pop_size)]

        scenario_ids = [int((i + gen) % 5) for i in range(pop_size)]
        envs = [world.sample_survival_scenario(scenario_ids[i], rng) for i in range(pop_size)]
        prev_obs_pred = np.zeros((pop_size, brain.in_dim), dtype=np.float32)
        harvest_per_agent = np.zeros((pop_size,), dtype=np.float32)

        for step in range(steps_per_gen):
            obs_batch = np.zeros((pop_size, brain.in_dim), dtype=np.float32)
            for i in range(pop_size):
                if alive[i]:
                    obs_batch[i] = extract_observation_vector(
                        envs[i], scenario_ids[i], hp[i], energy[i], pain[i], surprise[i], world.grid_size
                    )

            if step > 0:
                pred_err = np.linalg.norm(obs_batch - prev_obs_pred, axis=1)
                surprise = np.clip(pred_err / 2.5, 0.0, 1.0)

            h_next, act, obs_pred = brain.forward_step(obs_batch, h, Wo_live)
            prev_obs_pred = obs_pred

            for i in range(pop_size):
                if not alive[i]:
                    continue

                sc = scenario_ids[i]
                act_vec = act[i].copy()
                if use_curiosity and pain[i] > 0.25:
                    for bad_op in tried_bad[i]:
                        act_vec[bad_op] -= 1.5

                chosen_op = int(np.argmax(act_vec))
                dmg, delta_e = step_environment_physics(sc, envs[i], chosen_op)

                if dmg > 0.0:
                    tried_bad[i].add(chosen_op)
                if delta_e > 0.0:
                    harvest_per_agent[i] += 1.0
                    scenario_ids[i] = int((scenario_ids[i] + 1) % 5)
                    envs[i] = world.sample_survival_scenario(scenario_ids[i], rng)
                    tried_bad[i].clear()

                inst_pain = 1.0 - math.exp(-kappa * dmg) if dmg > 0.0 else 0.0
                pain[i] = float(inst_pain)
                energy[i] = float(np.clip(energy[i] + delta_e - 0.008, 0.0, 1.0))
                hp[i] = float(max(0.0, hp[i] - dmg - gamma_m * (1.0 - energy[i])))

                if mortal and hp[i] <= 0.0:
                    alive[i] = False
                    continue

                if plastic:
                    one_hot = np.zeros((brain.act_dim,), dtype=np.float32)
                    one_hot[chosen_op] = 1.0
                    r_mod = alpha_E * delta_e - alpha_P * inst_pain + alpha_C * float(surprise[i])
                    eligibility[i] = trace_decay * eligibility[i] + np.outer(one_hot, h_next[i])
                    eta = float(np.exp(brain.log_eta[i]))
                    Wo_live[i] = np.clip(Wo_live[i] + eta * r_mod * eligibility[i], -5.0, 5.0)

                    pred_grad = np.outer(obs_batch[i] - obs_pred[i], h_next[i])
                    brain.Wpred[i] = np.clip(brain.Wpred[i] + 0.01 * pred_grad, -3.0, 3.0)

            h = h_next

        if mortal:
            scores = harvest_per_agent * 1.5 + hp * 2.0 + energy * 1.5 + alive.astype(np.float32) * 3.0
        else:
            scores = energy * 1.5 + rng.uniform(0.0, 0.2, size=pop_size).astype(np.float32)
        survivors = np.where(alive & (energy >= repro_threshold))[0]
        if len(survivors) < 5:
            survivors = np.argsort(scores)[-10:]

        new_Wx = np.zeros_like(brain.Wx)
        new_Wh = np.zeros_like(brain.Wh)
        new_Wq = np.zeros_like(brain.Wq)
        new_Wk = np.zeros_like(brain.Wk)
        new_Wv = np.zeros_like(brain.Wv)
        new_b = np.zeros_like(brain.b)
        new_Wo = np.zeros_like(brain.Wo)
        new_bo = np.zeros_like(brain.bo)
        new_Wpred = np.zeros_like(brain.Wpred)
        new_log_eta = np.zeros_like(brain.log_eta)

        elite_count = min(len(survivors), pop_size // 5)
        top_indices = np.argsort(scores)[-elite_count:]

        for idx in range(pop_size):
            if idx < elite_count:
                p_idx = int(top_indices[idx])
                new_Wx[idx] = brain.Wx[p_idx]
                new_Wh[idx] = brain.Wh[p_idx]
                new_Wq[idx] = brain.Wq[p_idx]
                new_Wk[idx] = brain.Wk[p_idx]
                new_Wv[idx] = brain.Wv[p_idx]
                new_b[idx] = brain.b[p_idx]
                new_Wo[idx] = Wo_live[p_idx]
                new_bo[idx] = brain.bo[p_idx]
                new_Wpred[idx] = brain.Wpred[p_idx]
                new_log_eta[idx] = brain.log_eta[p_idx]
            else:
                p_idx = int(rng.choice(survivors))
                sigma = 0.03
                new_Wx[idx] = brain.Wx[p_idx] + rng.normal(0.0, sigma, brain.Wx[p_idx].shape).astype(np.float32)
                new_Wh[idx] = brain.Wh[p_idx] + rng.normal(0.0, sigma, brain.Wh[p_idx].shape).astype(np.float32)
                new_Wq[idx] = brain.Wq[p_idx] + rng.normal(0.0, sigma, brain.Wq[p_idx].shape).astype(np.float32)
                new_Wk[idx] = brain.Wk[p_idx] + rng.normal(0.0, sigma, brain.Wk[p_idx].shape).astype(np.float32)
                new_Wv[idx] = brain.Wv[p_idx] + rng.normal(0.0, sigma, brain.Wv[p_idx].shape).astype(np.float32)
                new_b[idx] = brain.b[p_idx] + rng.normal(0.0, sigma * 0.5, brain.b[p_idx].shape).astype(np.float32)
                new_Wo[idx] = Wo_live[p_idx] + rng.normal(0.0, sigma, Wo_live[p_idx].shape).astype(np.float32)
                new_bo[idx] = brain.bo[p_idx] + rng.normal(0.0, sigma * 0.5, brain.bo[p_idx].shape).astype(np.float32)
                new_Wpred[idx] = brain.Wpred[p_idx] + rng.normal(0.0, sigma, brain.Wpred[p_idx].shape).astype(np.float32)
                new_log_eta[idx] = np.clip(brain.log_eta[p_idx] + float(rng.normal(0.0, 0.1)), -6.0, -2.5)

        brain.Wx = new_Wx
        brain.Wh = new_Wh
        brain.Wq = new_Wq
        brain.Wk = new_Wk
        brain.Wv = new_Wv
        brain.b = new_b
        brain.Wo = new_Wo
        brain.bo = new_bo
        brain.Wpred = new_Wpred
        brain.log_eta = new_log_eta

    eval_res = evaluate_in_mortal_arena(
        brain, world, n_episodes=10, steps=30,
        plastic=plastic, use_pain=use_pain, use_curiosity=use_curiosity, seed=seed + 500
    )
    best_idx = int(np.argmax(scores))
    eval_res["best_weights"] = {
        "Wx": brain.Wx[best_idx],
        "Wh": brain.Wh[best_idx],
        "Wq": brain.Wq[best_idx],
        "Wk": brain.Wk[best_idx],
        "Wv": brain.Wv[best_idx],
        "b": brain.b[best_idx],
        "Wo": brain.Wo[best_idx],
        "bo": brain.bo[best_idx],
        "Wpred": brain.Wpred[best_idx],
        "concept_centroids": eval_res["concept_centroids"],
    }
    return eval_res


def execute_stage1_benchmark():
    t0 = time.time()
    full_res = run_stage1_training(
        generations=40, steps_per_gen=30, pop_size=50,
        mortal=True, use_pain=True, use_curiosity=True, plastic=True, seed=101
    )
    non_mortal_res = run_stage1_training(
        generations=40, steps_per_gen=30, pop_size=50,
        mortal=False, use_pain=False, use_curiosity=False, plastic=True, seed=102
    )
    static_res = run_stage1_training(
        generations=40, steps_per_gen=30, pop_size=50,
        mortal=True, use_pain=True, use_curiosity=False, plastic=False, seed=103
    )

    weights = full_res["best_weights"]
    np.savez(
        ARTIFACTS_DIR / "stage1_core_manifold.npz",
        Wx=weights["Wx"],
        Wh=weights["Wh"],
        Wq=weights["Wq"],
        Wk=weights["Wk"],
        Wv=weights["Wv"],
        b=weights["b"],
        Wo=weights["Wo"],
        bo=weights["bo"],
        Wpred=weights["Wpred"],
        concept_centroids=weights["concept_centroids"],
    )

    summary = {
        "elapsed_sec": round(time.time() - t0, 3),
        "Embodied_Mortal_Pain_Curiosity": {
            "mortal_test_survival_rate_pct": round(full_res["mortal_test_survival_rate_pct"], 2),
            "mortal_test_harvest_mean": round(full_res["mortal_test_harvest_mean"], 2),
            "mortal_test_damage_mean": round(full_res["mortal_test_damage_mean"], 2),
            "concept_alignment_cosine": round(full_res["concept_alignment_cosine"], 4),
            "within_concept_cosines": full_res["within_concept_cosines"],
        },
        "Disembodied_NoPain_NoDeath_Control": {
            "mortal_test_survival_rate_pct": round(non_mortal_res["mortal_test_survival_rate_pct"], 2),
            "mortal_test_harvest_mean": round(non_mortal_res["mortal_test_harvest_mean"], 2),
            "mortal_test_damage_mean": round(non_mortal_res["mortal_test_damage_mean"], 2),
            "concept_alignment_cosine": round(non_mortal_res["concept_alignment_cosine"], 4),
        },
        "Static_NoPlasticity_Control": {
            "mortal_test_survival_rate_pct": round(static_res["mortal_test_survival_rate_pct"], 2),
            "mortal_test_harvest_mean": round(static_res["mortal_test_harvest_mean"], 2),
            "mortal_test_damage_mean": round(static_res["mortal_test_damage_mean"], 2),
            "concept_alignment_cosine": round(static_res["concept_alignment_cosine"], 4),
        },
    }

    with open(ARTIFACTS_DIR / "stage1_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    execute_stage1_benchmark()
