import ctypes
import json
import math
import os
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from four_biome_humanoid_env import IsaacLabUnified3DArena, IsaacStyle3DPhysicsArena
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


class SpikingTransformerActiveInferenceBrain:
    def __init__(self, pop_size=4, obs_dim=33, hid_dim=128, motor_dim=3, vocal_dim=2, ctx_len=6, n_candidates=5, seed=42):
        self.pop_size = pop_size
        self.obs_dim = obs_dim
        self.hid_dim = hid_dim
        self.motor_dim = motor_dim
        self.vocal_dim = vocal_dim
        self.act_dim = motor_dim + vocal_dim
        self.ctx_len = ctx_len
        self.n_candidates = n_candidates

        torch.manual_seed(seed)
        sx = 1.0 / math.sqrt(obs_dim)
        sh = 1.0 / math.sqrt(hid_dim)

        self.W_enc = torch.randn((pop_size, hid_dim, obs_dim), device=DEVICE) * sx
        self.Wq = torch.randn((pop_size, hid_dim, hid_dim), device=DEVICE) * sh
        self.Wk = torch.randn((pop_size, hid_dim, hid_dim), device=DEVICE) * sh
        self.Wv = torch.randn((pop_size, hid_dim, hid_dim), device=DEVICE) * sh

        self.Wx = torch.randn((pop_size, hid_dim, obs_dim), device=DEVICE) * sx
        self.Wx_retinal = torch.randn((pop_size, hid_dim, 16), device=DEVICE) * (1.0 / math.sqrt(16.0))
        self.W_enc_retinal = torch.randn((pop_size, hid_dim, 16), device=DEVICE) * (1.0 / math.sqrt(16.0))
        self.Wpred_retinal = torch.randn((pop_size, 16, hid_dim), device=DEVICE) * sh
        self.Wh = torch.randn((pop_size, hid_dim, hid_dim), device=DEVICE) * sh
        self.b = torch.zeros((pop_size, hid_dim), device=DEVICE)
        self.theta = torch.full((pop_size, hid_dim), 0.35, device=DEVICE)

        self.Wo = torch.randn((pop_size, self.act_dim, hid_dim), device=DEVICE) * sh
        self.bo = torch.zeros((pop_size, self.act_dim), device=DEVICE)

        shu = 1.0 / math.sqrt(hid_dim + self.act_dim)
        soa = 1.0 / math.sqrt(obs_dim + self.act_dim)
        self.Wpred_obs = torch.randn((pop_size, obs_dim, hid_dim + self.act_dim), device=DEVICE) * shu
        self.Wpred_val = torch.randn((pop_size, 2, obs_dim + self.act_dim), device=DEVICE) * soa

        self.h = torch.zeros((pop_size, hid_dim), device=DEVICE)
        self.v_mem = torch.zeros((pop_size, hid_dim), device=DEVICE)
        self.spike_ctx = torch.zeros((pop_size, ctx_len, hid_dim), device=DEVICE)
        self.val_ctx = torch.zeros((pop_size, ctx_len, hid_dim), device=DEVICE)
        self.hippo_len = 8
        self.hippo_keys = torch.zeros((pop_size, self.hippo_len, hid_dim), device=DEVICE)
        self.hippo_vals = torch.zeros((pop_size, self.hippo_len, hid_dim), device=DEVICE)
        self.hippo_acts = torch.zeros((pop_size, self.hippo_len, self.act_dim), device=DEVICE)
        self.hippo_ptr = torch.zeros((pop_size,), dtype=torch.long, device=DEVICE)
        self.mono_dim = 4
        self.macro_monologue = torch.zeros((pop_size, self.mono_dim), device=DEVICE)
        self.micro_monologue = torch.zeros((pop_size, self.mono_dim), device=DEVICE)
        self.inner_monologue = torch.zeros((pop_size, self.mono_dim), device=DEVICE)
        self.W_macro_out = torch.randn((pop_size, self.mono_dim, hid_dim), device=DEVICE) * sh
        self.W_macro_in = torch.randn((pop_size, hid_dim, self.mono_dim), device=DEVICE) * 0.20
        self.W_mono_out = torch.randn((pop_size, self.mono_dim, hid_dim), device=DEVICE) * sh
        self.W_mono_in = torch.randn((pop_size, hid_dim, self.mono_dim), device=DEVICE) * 0.22
        self.last_q_spk = torch.zeros((pop_size, hid_dim), device=DEVICE)
        self.last_k_spk = torch.zeros((pop_size, hid_dim), device=DEVICE)
        self.last_v_lin = torch.zeros((pop_size, hid_dim), device=DEVICE)
        self.last_enc = torch.zeros((pop_size, hid_dim), device=DEVICE)
        self.dream_len = 16
        self.dream_h = torch.zeros((pop_size, self.dream_len, hid_dim), device=DEVICE)
        self.dream_q = torch.zeros((pop_size, self.dream_len, hid_dim), device=DEVICE)
        self.dream_k = torch.zeros((pop_size, self.dream_len, hid_dim), device=DEVICE)
        self.dream_v = torch.zeros((pop_size, self.dream_len, hid_dim), device=DEVICE)
        self.dream_act = torch.zeros((pop_size, self.dream_len, self.act_dim), device=DEVICE)
        self.dream_reward = torch.zeros((pop_size, self.dream_len), device=DEVICE)
        self.dream_ptr = torch.zeros((pop_size,), dtype=torch.long, device=DEVICE)
        self.mutated_dream_q = torch.zeros((pop_size, 8, hid_dim), device=DEVICE)
        self.mutated_dream_k = torch.zeros((pop_size, 8, hid_dim), device=DEVICE)
        self.mutated_dream_v = torch.zeros((pop_size, 8, hid_dim), device=DEVICE)
        self.cultural_macro_proto = torch.zeros((self.mono_dim,), device=DEVICE)
        self.empathy_share_events = 0
        self.dream_mutation_events = 0
        self.verified_symbol_messages = 0
        self.symbol_silence_decisions = 0
        self.symbol_vocab_size = 16
        self.W_symbol_lexicon = torch.randn((pop_size, self.symbol_vocab_size, 11), device=DEVICE) * 0.15
        self.symbol_usage_counts = torch.zeros((self.symbol_vocab_size,), device=DEVICE)
        self.last_body_logits = torch.zeros((pop_size, 11), device=DEVICE)
        self.eligibility = torch.zeros((pop_size, self.act_dim, hid_dim), device=DEVICE)
        self.W_body_assembly = torch.randn((pop_size, 11, hid_dim), device=DEVICE) * sh
        self.body_eligibility = torch.zeros((pop_size, 11, hid_dim), device=DEVICE)
        self.prev_obs_pred = torch.zeros((pop_size, obs_dim), device=DEVICE)
        self.prev_retinal_obs = torch.zeros((pop_size, 16), device=DEVICE)
        self.prev_retinal_pred = torch.zeros((pop_size, 16), device=DEVICE)
        self.dormancy_gate_mask = torch.ones((pop_size, hid_dim), device=DEVICE)
        self.active_cell_ratio = torch.full((pop_size,), 0.35, device=DEVICE)
        self.prev_hu = torch.zeros((pop_size, hid_dim + self.act_dim), device=DEVICE)
        self.prev_oa = torch.zeros((pop_size, obs_dim + self.act_dim), device=DEVICE)
        self.last_spike_rate = torch.zeros((pop_size,), device=DEVICE)
        self.spike_fatigue = torch.zeros((pop_size,), device=DEVICE)
        self.closed_loop_3brain_steps = 0
        self.unified_brains = UnifiedBiologicalBrainSystem(
            hid_dim=hid_dim,
            artifact_dir=str(ARTIFACTS_DIR),
        )
        self.unified_brains.load_all_brains()
        self.unified_brains.sync_to_stage1_brain(self)

    def load_saved_manifold(self):
        m_path = ARTIFACTS_DIR / "stage1_isaac3d_core_manifold.npz"
        if not m_path.exists():
            return False
        try:
            m = np.load(m_path)
            attr_map = [
                ("Wx", self.Wx),
                ("W_enc", self.W_enc),
                ("Wpred_obs", self.Wpred_obs),
                ("Wpred_val", self.Wpred_val),
                ("Wq", self.Wq),
                ("Wk", self.Wk),
                ("Wv", self.Wv),
                ("Wx_retinal", self.Wx_retinal),
                ("W_enc_retinal", self.W_enc_retinal),
                ("Wpred_retinal", self.Wpred_retinal),
                ("Wh", self.Wh),
                ("Wo", self.Wo),
                ("spike_ctx", self.spike_ctx),
                ("val_ctx", self.val_ctx),
                ("hippo_keys", self.hippo_keys),
                ("hippo_vals", self.hippo_vals),
                ("W_macro_out", self.W_macro_out),
                ("W_macro_in", self.W_macro_in),
                ("W_mono_out", self.W_mono_out),
                ("W_mono_in", self.W_mono_in),
                ("W_body_assembly", self.W_body_assembly),
                ("W_symbol_lexicon", self.W_symbol_lexicon),
                ("dream_q", self.dream_q),
                ("dream_k", self.dream_k),
                ("dream_v", self.dream_v),
                ("mutated_dream_q", self.mutated_dream_q),
                ("mutated_dream_k", self.mutated_dream_k),
                ("mutated_dream_v", self.mutated_dream_v),
            ]
            with torch.no_grad():
                for k, target_t in attr_map:
                    if k in m:
                        arr_t = torch.tensor(m[k], dtype=torch.float32, device=DEVICE)
                        if arr_t.shape == target_t.shape[1:]:
                            target_t.copy_(arr_t.unsqueeze(0).expand_as(target_t).clone())
                if "cultural_macro_proto" in m:
                    c_t = torch.tensor(m["cultural_macro_proto"], dtype=torch.float32, device=DEVICE)
                    if c_t.shape == self.cultural_macro_proto.shape:
                        self.cultural_macro_proto.copy_(c_t)
            return True
        except Exception:
            return False

    def reset_states(self):
        self.h.mul_(0.35)
        self.v_mem.mul_(0.25)
        self.spike_ctx.mul_(0.35)
        self.val_ctx.mul_(0.35)
        self.macro_monologue.mul_(0.50)
        self.micro_monologue.mul_(0.25)
        self.inner_monologue.mul_(0.35)
        self.last_q_spk.zero_()
        self.last_k_spk.zero_()
        self.last_v_lin.zero_()
        self.last_enc.zero_()
        self.eligibility.mul_(0.25)
        self.body_eligibility.mul_(0.25)
        self.last_body_logits.zero_()
        self.prev_obs_pred.zero_()
        self.prev_retinal_obs.zero_()
        self.prev_retinal_pred.zero_()
        self.dormancy_gate_mask.fill_(1.0)
        self.prev_hu.zero_()
        self.prev_oa.zero_()
        self.last_spike_rate.zero_()
        self.spike_fatigue.mul_(0.30)

    def lif_spike(self, x_in, threshold):
        self.v_mem = 0.78 * self.v_mem + x_in
        spikes = (self.v_mem >= threshold).float()
        self.v_mem = self.v_mem * (1.0 - spikes)
        return spikes

    def encode_and_verify_symbol(self, agent_idx, body_vec, allow_self_imprint=False):
        if isinstance(body_vec, np.ndarray):
            m_vec = torch.tensor(body_vec, dtype=torch.float32, device=DEVICE)
        else:
            m_vec = body_vec.to(device=DEVICE, dtype=torch.float32)
        m_vec = F.normalize(m_vec, dim=-1, eps=1e-6)
        lex_m = F.normalize(self.W_symbol_lexicon[agent_idx], dim=-1, eps=1e-6)
        cos_sims = torch.mv(lex_m, m_vec)
        best_cos, best_sym_t = torch.max(cos_sims, dim=0)
        sym_id = int(best_sym_t.item())
        if float(best_cos.item()) < 0.80 and allow_self_imprint and float(torch.min(self.symbol_usage_counts).item()) < 1.0:
            sym_id = int(torch.argmin(self.symbol_usage_counts).item())
            self.W_symbol_lexicon[agent_idx, sym_id] = m_vec.clone()
            self.symbol_usage_counts[sym_id] += 1.0
        elif allow_self_imprint:
            self.W_symbol_lexicon[agent_idx, sym_id] = F.normalize(
                0.82 * self.W_symbol_lexicon[agent_idx, sym_id] + 0.18 * m_vec,
                dim=-1,
                eps=1e-6,
            )
        recon_vec = F.normalize(self.W_symbol_lexicon[agent_idx, sym_id], dim=-1, eps=1e-6)
        recon_cos = float(torch.dot(recon_vec, m_vec).item())
        if recon_cos >= 0.80:
            self.symbol_usage_counts[sym_id] += 0.05
            self.verified_symbol_messages += 1
            return sym_id, recon_cos
        self.symbol_silence_decisions += 1
        return -1, recon_cos

    def forward_active_inference(
        self,
        obs,
        arena,
        use_pain=True,
        use_curiosity=True,
        use_active_inference=True,
        metabolic_thought_mode="adaptive",
    ):
        N = self.pop_size
        H = self.hid_dim

        retinas, fov_sal, per_sal = arena.render_first_person_retinal_grids()
        ret_obs_16d = arena.extract_retinal_16d_observations(retinas, fov_sal, per_sal)
        self.prev_retinal_obs = ret_obs_16d.detach()

        if metabolic_thought_mode == "adaptive":
            v_th = torch.clamp(0.48 - 0.18 * arena.surprise - 0.16 * arena.pain + 0.26 * (1.0 - arena.energy), 0.22, 0.78).unsqueeze(-1)
        elif metabolic_thought_mode == "overthink_unbounded":
            v_th = torch.full((N, 1), 0.12, device=DEVICE)
        else:
            v_th = torch.full((N, 1), 0.45, device=DEVICE)

        macro_ctx = torch.bmm(self.W_macro_in, self.macro_monologue.unsqueeze(-1)).squeeze(-1)
        micro_ctx = torch.bmm(self.W_mono_in, self.micro_monologue.unsqueeze(-1)).squeeze(-1)
        mono_ctx = 0.55 * macro_ctx + 0.45 * micro_ctx
        ret_enc = torch.bmm(self.W_enc_retinal, ret_obs_16d.unsqueeze(-1)).squeeze(-1)
        enc_raw = torch.tanh(
            0.70 * torch.bmm(self.W_enc, obs.unsqueeze(-1)).squeeze(-1)
            + 0.30 * ret_enc
            + 0.28 * mono_ctx
        )
        if metabolic_thought_mode == "adaptive":
            active_ratio = torch.clamp(0.32 + 0.35 * arena.surprise + 0.25 * arena.pain - 0.22 * (1.0 - arena.energy), 0.20, 0.75)
            k_active = max(8, int(round(float(active_ratio.mean().item()) * H)))
            topk_vals = torch.topk(enc_raw.abs(), k=k_active, dim=-1).values[:, -1:]
            dormancy_mask = (enc_raw.abs() >= topk_vals).float()
            self.dormancy_gate_mask = dormancy_mask
            self.active_cell_ratio = dormancy_mask.mean(dim=-1)
            enc = enc_raw * dormancy_mask
            self.v_mem = self.v_mem * dormancy_mask
        else:
            self.dormancy_gate_mask.fill_(1.0)
            self.active_cell_ratio.fill_(1.0)
            enc = enc_raw
        self.last_enc = enc.detach()
        effective_theta = torch.clamp(v_th * (self.theta / 0.35), 0.15, 0.95)
        s_now = self.lif_spike(enc, effective_theta) * self.dormancy_gate_mask

        q_lin = torch.bmm(self.Wq, s_now.unsqueeze(-1)).squeeze(-1)
        k_lin = torch.bmm(self.Wk, s_now.unsqueeze(-1)).squeeze(-1)
        v_lin = torch.tanh(torch.bmm(self.Wv, enc.unsqueeze(-1)).squeeze(-1))

        q_spk = (q_lin.abs() > effective_theta).float() * torch.sign(q_lin) * self.dormancy_gate_mask
        k_spk = (k_lin.abs() > effective_theta).float() * torch.sign(k_lin) * self.dormancy_gate_mask
        self.last_q_spk = q_spk.detach()
        self.last_k_spk = k_spk.detach()
        self.last_v_lin = v_lin.detach()
        sparse_assembly_drive = (0.45 * s_now + 0.35 * q_spk + 0.20 * k_spk) * self.dormancy_gate_mask
        w_neo_body = self.unified_brains.neocortex_logic.W_body_assembly.detach().to(DEVICE)
        neo_body_drive = torch.matmul(sparse_assembly_drive, w_neo_body.T)
        body_logits = torch.tanh(
            0.78 * torch.bmm(self.W_body_assembly, sparse_assembly_drive.unsqueeze(-1)).squeeze(-1)
            + 0.22 * neo_body_drive
        )
        self.last_body_logits = body_logits.detach()
        arena.agent_body_logits = body_logits.detach()
        self.body_eligibility = 0.84 * self.body_eligibility + torch.bmm(
            body_logits.detach().unsqueeze(-1), sparse_assembly_drive.detach().unsqueeze(1)
        )

        self.spike_ctx = torch.cat([self.spike_ctx[:, 1:, :], k_spk.unsqueeze(1)], dim=1)
        self.val_ctx = torch.cat([self.val_ctx[:, 1:, :], v_lin.unsqueeze(1)], dim=1)

        attn_scores = torch.bmm(self.spike_ctx, q_spk.unsqueeze(-1)).squeeze(-1) / math.sqrt(H)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        ctx_vec = torch.bmm(attn_weights.unsqueeze(1), self.val_ctx).squeeze(1)

        hippo_scores = torch.bmm(self.hippo_keys, q_spk.unsqueeze(-1)).squeeze(-1) / math.sqrt(H)
        hippo_weights = torch.softmax(hippo_scores, dim=-1)
        hippo_ctx = torch.bmm(hippo_weights.unsqueeze(1), self.hippo_vals).squeeze(1)
        hippo_act_hint = torch.bmm(hippo_weights.unsqueeze(1), self.hippo_acts).squeeze(1)

        gx_ray = torch.bmm(self.Wx, obs.unsqueeze(-1)).squeeze(-1)
        gx_ret = torch.bmm(self.Wx_retinal, ret_obs_16d.unsqueeze(-1)).squeeze(-1)
        dendritic_in = 0.72 * gx_ray + 0.28 * gx_ret + self.b
        rec_spk = torch.bmm(self.Wh, (s_now * self.h).unsqueeze(-1)).squeeze(-1)
        total_drive = torch.tanh(dendritic_in + rec_spk + 0.38 * ctx_vec + 0.16 * hippo_ctx + 0.14 * mono_ctx)
        leak_alpha = torch.clamp(0.72 - 0.18 * arena.surprise.unsqueeze(-1) + 0.10 * (1.0 - arena.energy.unsqueeze(-1)), 0.45, 0.88)
        h_cand = leak_alpha * self.h + (1.0 - leak_alpha) * total_drive
        self.h = h_cand * self.dormancy_gate_mask + self.h * (1.0 - self.dormancy_gate_mask)
        self.theta = torch.clamp(self.theta + 0.012 * (s_now - 0.25), 0.18, 0.68)
        self.prev_retinal_pred = torch.tanh(torch.bmm(self.Wpred_retinal, self.h.unsqueeze(-1)).squeeze(-1))
        macro_step = torch.tanh(torch.bmm(self.W_macro_out, self.h.unsqueeze(-1)).squeeze(-1))
        micro_step = torch.tanh(torch.bmm(self.W_mono_out, self.h.unsqueeze(-1)).squeeze(-1))
        self.macro_monologue = 0.88 * self.macro_monologue + 0.12 * macro_step
        self.micro_monologue = 0.32 * self.micro_monologue + 0.68 * micro_step
        self.inner_monologue = 0.50 * self.macro_monologue + 0.50 * self.micro_monologue

        w_mot3d = self.unified_brains.motor_reflex.W_act_3d.weight.detach().to(DEVICE)
        mot3d_logits = torch.matmul(self.h, w_mot3d.T)
        w_j21 = self.unified_brains.motor_reflex.W_joint_21d.weight.detach().to(DEVICE)
        b_j21 = self.unified_brains.motor_reflex.W_joint_21d.bias.detach().to(DEVICE)
        arena.agent_joint_effort_21d = torch.tanh(torch.matmul(self.h, w_j21.T) + b_j21).detach()
        base_logits = 0.82 * torch.bmm(self.Wo, self.h.unsqueeze(-1)).squeeze(-1) + 0.18 * mot3d_logits + self.bo
        base_act = torch.tanh(base_logits)
        self.closed_loop_3brain_steps += 1

        if use_active_inference:
            K = self.n_candidates
            cand_acts = base_act.unsqueeze(1).repeat(1, K, 1)
            if K > 1:
                explore_sigma = (0.36 + 0.24 * arena.surprise).unsqueeze(-1).unsqueeze(-1)
                deltas = torch.randn((N, K - 1, self.act_dim), device=DEVICE) * explore_sigma
                cand_acts[:, 1:, :] = torch.clamp(cand_acts[:, 1:, :] + deltas, -1.0, 1.0)
                mono_commit = torch.sigmoid(self.inner_monologue[:, 3:4])
                cand_acts[:, -1, :] = torch.clamp(
                    (1.0 - 0.45 * mono_commit) * cand_acts[:, -1, :] + 0.45 * mono_commit * hippo_act_hint,
                    -1.0,
                    1.0,
                )

            obs_exp = obs.unsqueeze(1).expand(N, K, self.obs_dim)
            oa_cand = torch.cat([obs_exp, cand_acts], dim=-1)
            pred_val = torch.bmm(oa_cand, self.Wpred_val.transpose(1, 2))
            pred_de = pred_val[:, :, 0]
            pred_pain = torch.relu(pred_val[:, :, 1])

            hu_cand = torch.cat([self.h.unsqueeze(1).expand(N, K, H), cand_acts], dim=-1)
            pred_next_obs = torch.tanh(torch.bmm(hu_cand, self.Wpred_obs.transpose(1, 2)))
            epistemic_val_1 = torch.norm(pred_next_obs - obs_exp, dim=-1) / math.sqrt(self.obs_dim)

            gx_next = torch.bmm(pred_next_obs, self.Wx.transpose(1, 2)) + self.b.unsqueeze(1)
            enc_next = torch.tanh(torch.bmm(pred_next_obs, self.W_enc.transpose(1, 2)))
            s_next = (enc_next >= effective_theta.unsqueeze(1)).float() * self.dormancy_gate_mask.unsqueeze(1)
            rec_next = torch.bmm(s_next * self.h.unsqueeze(1), self.Wh.transpose(1, 2))
            ctx_vec_exp = ctx_vec.unsqueeze(1).expand(N, K, H)
            hippo_ctx_exp = hippo_ctx.unsqueeze(1).expand(N, K, H)
            mono_ctx_exp = mono_ctx.unsqueeze(1).expand(N, K, H)
            total_drive_next = torch.tanh(
                gx_next + rec_next + 0.38 * ctx_vec_exp + 0.16 * hippo_ctx_exp + 0.14 * mono_ctx_exp
            )
            h_next_cand = leak_alpha.unsqueeze(1) * self.h.unsqueeze(1) + (1.0 - leak_alpha.unsqueeze(1)) * total_drive_next
            act_cand_2 = torch.tanh(torch.bmm(h_next_cand, self.Wo.transpose(1, 2)) + self.bo.unsqueeze(1))

            oa_cand_2 = torch.cat([pred_next_obs, act_cand_2], dim=-1)
            pred_val_2 = torch.bmm(oa_cand_2, self.Wpred_val.transpose(1, 2))
            pred_de_2 = pred_val_2[:, :, 0]
            pred_pain_2 = torch.relu(pred_val_2[:, :, 1])
            hu_cand_2 = torch.cat([h_next_cand, act_cand_2], dim=-1)
            pred_next_obs_2 = torch.tanh(torch.bmm(hu_cand_2, self.Wpred_obs.transpose(1, 2)))
            epistemic_val_2 = torch.norm(pred_next_obs_2 - pred_next_obs, dim=-1) / math.sqrt(self.obs_dim)
            epistemic_val = epistemic_val_1 + 0.65 * epistemic_val_2

            max_pred_de = pred_de.max(dim=1).values
            intuitive_mode = (max_pred_de > 0.80).float()
            stagnation_gate = torch.sigmoid(8.0 * (0.20 - max_pred_de)) * (1.0 - intuitive_mode)
            fatigue_brake = torch.exp(-1.15 * self.spike_fatigue)
            meta_burst = stagnation_gate * fatigue_brake

            hunger_weight = (torch.clamp(1.3 - arena.energy, 0.4, 1.3) * (1.0 + 0.45 * intuitive_mode - 0.50 * meta_burst)).unsqueeze(-1)
            pain_weight = 2.85 if use_pain else 0.0
            survival_priority_gate = (
                torch.clamp((arena.hp - 0.35) / 0.65, 0.0, 1.0)
                * torch.clamp((arena.energy - 0.22) / 0.78, 0.0, 1.0)
                * torch.exp(-4.5 * arena.pain)
            )
            base_cur = 0.25 * survival_priority_gate * (1.0 - intuitive_mode)
            burst_cur = 1.45 * meta_burst * survival_priority_gate
            cur_weight = (base_cur + burst_cur).unsqueeze(-1) if use_curiosity else 0.0

            effort_pen = 0.08 * cand_acts[:, :, :2].abs().sum(dim=-1)
            efe = (
                -hunger_weight * (pred_de + 0.55 * meta_burst.unsqueeze(-1) * pred_de_2)
                + pain_weight * (pred_pain + 0.55 * meta_burst.unsqueeze(-1) * pred_pain_2)
                - cur_weight * epistemic_val
                + effort_pen
            )
            best_k = torch.argmin(efe, dim=1)
            chosen_act = cand_acts[torch.arange(N, device=DEVICE), best_k]
            rollout_spikes = float(K) * (0.10 + 0.15 * float(meta_burst.mean().item()))
        else:
            if use_curiosity:
                survival_priority_gate = (
                    torch.clamp((arena.hp - 0.35) / 0.65, 0.0, 1.0)
                    * torch.clamp((arena.energy - 0.22) / 0.78, 0.0, 1.0)
                    * torch.exp(-4.5 * arena.pain)
                )
                cur_amp = 0.18 * arena.surprise * survival_priority_gate
                chosen_act = torch.clamp(base_act + torch.randn_like(base_act) * cur_amp.unsqueeze(-1), -1.0, 1.0)
            else:
                chosen_act = base_act
            rollout_spikes = 0.10

        spike_density = (s_now.abs().mean(dim=-1) + q_spk.abs().mean(dim=-1) + k_spk.abs().mean(dim=-1)) / 3.0
        self.last_spike_rate = spike_density
        self.spike_fatigue = 0.86 * self.spike_fatigue + spike_density

        thought_tax = 0.0005 + 0.0012 * spike_density * (1.0 + 0.25 * rollout_spikes)

        motor_u = chosen_act[:, : self.motor_dim]
        vocal_s = chosen_act[:, self.motor_dim :]

        hu = torch.cat([self.h, chosen_act], dim=-1)
        self.prev_hu = hu
        self.prev_oa = torch.cat([obs, chosen_act], dim=-1)
        self.prev_obs_pred = torch.tanh(torch.bmm(self.Wpred_obs, hu.unsqueeze(-1)).squeeze(-1))
        self.prev_retinal_obs = ret_obs_16d.detach()
        self.prev_retinal_pred = torch.tanh(torch.bmm(self.Wpred_retinal, self.h.unsqueeze(-1)).squeeze(-1))
        return motor_u, vocal_s, chosen_act, thought_tax

    def hebbian_plasticity_and_mortality(
        self,
        arena,
        obs,
        act,
        next_obs,
        delta_e,
        pain,
        use_pain=True,
        use_curiosity=True,
        use_mortality=True,
        plastic=True,
        eta=0.015,
    ):
        next_retinal_obs = arena.extract_retinal_16d_observations()
        ret_pred_err = torch.norm(next_retinal_obs - self.prev_retinal_pred, dim=-1)
        pred_err = 0.65 * torch.norm(next_obs - self.prev_obs_pred, dim=-1) + 0.35 * ret_pred_err
        surprise = torch.clamp(pred_err / 2.6, 0.0, 1.0)
        arena.surprise = surprise

        if plastic:
            err_obs = (next_obs - self.prev_obs_pred).unsqueeze(-1)
            self.Wpred_obs = torch.clamp(
                self.Wpred_obs + 0.012 * torch.bmm(err_obs, self.prev_hu.unsqueeze(1)),
                -3.0,
                3.0,
            )
            err_ret = (next_retinal_obs - self.prev_retinal_pred).unsqueeze(-1)
            self.Wpred_retinal = torch.clamp(
                self.Wpred_retinal + 0.016 * torch.bmm(err_ret, self.h.detach().unsqueeze(1)),
                -3.0,
                3.0,
            )

            actual_val = torch.stack([delta_e, pain if use_pain else torch.zeros_like(pain)], dim=-1)
            pred_val_now = torch.bmm(self.Wpred_val, self.prev_oa.unsqueeze(-1)).squeeze(-1)
            val_err = (actual_val - pred_val_now).unsqueeze(-1)
            self.Wpred_val = torch.clamp(
                self.Wpred_val + 0.045 * torch.bmm(val_err, self.prev_oa.unsqueeze(1)),
                -3.5,
                3.5,
            )

            h_norm_sq = torch.sum(self.h * self.h, dim=-1, keepdim=True).unsqueeze(-1) + 1.0
            hebb_term = torch.bmm(act.unsqueeze(-1), self.h.unsqueeze(1)) / h_norm_sq
            self.eligibility = 0.85 * self.eligibility + hebb_term

            survival_priority_gate = (
                torch.clamp((arena.hp - 0.35) / 0.65, 0.0, 1.0)
                * torch.clamp((arena.energy - 0.22) / 0.78, 0.0, 1.0)
                * torch.exp(-4.5 * pain)
            )
            curiosity_drive = surprise * survival_priority_gate * (delta_e > 0.0).float() if use_curiosity else torch.zeros_like(surprise)
            r_mod = (
                2.5 * delta_e
                - (2.15 * pain if use_pain else 0.0)
                + (0.50 * curiosity_drive if use_curiosity else 0.0)
            )
            delta_Wo = eta * r_mod[:, None, None] * self.eligibility
            self.Wo = torch.clamp(0.997 * self.Wo + delta_Wo, -3.5, 3.5)
            delta_W_body = eta * r_mod[:, None, None] * self.body_eligibility
            self.W_body_assembly = torch.clamp(0.997 * self.W_body_assembly + delta_W_body, -3.0, 3.0)

            ret_hebb = torch.bmm(self.h.detach().unsqueeze(-1), self.prev_retinal_obs.unsqueeze(1))
            self.W_enc_retinal = torch.clamp(
                self.W_enc_retinal + 0.008 * r_mod[:, None, None] * ret_hebb,
                -2.8,
                2.8,
            )
            self.Wx_retinal = torch.clamp(
                self.Wx_retinal + 0.008 * r_mod[:, None, None] * ret_hebb,
                -2.8,
                2.8,
            )
            spk_rec_hebb = torch.bmm(self.h.detach().unsqueeze(-1), self.last_q_spk.unsqueeze(1))
            oja_wh = (self.h.detach().pow(2).mean(dim=-1, keepdim=True).unsqueeze(-1)) * self.Wh
            self.Wh = torch.clamp(
                self.Wh + 0.005 * r_mod[:, None, None] * (spk_rec_hebb - 0.12 * oja_wh),
                -2.6,
                2.6,
            )

            obs_pre = obs.detach()
            h_post = self.h.detach()
            enc_post = self.last_enc.detach()
            wx_hebb = torch.bmm(h_post.unsqueeze(-1), obs_pre.unsqueeze(1))
            oja_wx = h_post.pow(2).unsqueeze(-1) * self.Wx
            self.Wx = torch.clamp(
                self.Wx + 0.010 * r_mod[:, None, None] * (wx_hebb - 0.12 * oja_wx),
                -2.8,
                2.8,
            )
            enc_hebb = torch.bmm(enc_post.unsqueeze(-1), obs_pre.unsqueeze(1))
            oja_enc = enc_post.pow(2).unsqueeze(-1) * self.W_enc
            self.W_enc = torch.clamp(
                self.W_enc + 0.010 * r_mod[:, None, None] * (enc_hebb - 0.12 * oja_enc),
                -2.8,
                2.8,
            )
            self.b = torch.clamp(self.b + 0.005 * r_mod.unsqueeze(-1) * h_post - 0.001 * self.b, -1.5, 1.5)
            self.bo = torch.clamp(self.bo + 0.005 * r_mod.unsqueeze(-1) * act.detach() - 0.001 * self.bo, -1.5, 1.5)

            h_norm = torch.norm(self.h, dim=-1)
            over_agitated = (pain > 0.50) & (h_norm > 3.2)
            if use_pain and torch.any(over_agitated):
                self.Wh[over_agitated] = 0.92 * self.Wh[over_agitated]

            hippo_write_mask = (delta_e > 0.20) & (pain < 0.08) & arena.alive
            if torch.any(hippo_write_mask):
                w_idx = torch.where(hippo_write_mask)[0]
                ptrs = self.hippo_ptr[w_idx]
                self.hippo_keys[w_idx, ptrs] = self.last_k_spk[w_idx]
                self.hippo_vals[w_idx, ptrs] = self.last_v_lin[w_idx]
                self.hippo_acts[w_idx, ptrs] = act[w_idx].detach()
                self.hippo_ptr[w_idx] = (ptrs + 1) % self.hippo_len
                macro_err = (delta_e[w_idx].unsqueeze(-1) - self.macro_monologue[w_idx]).unsqueeze(-1)
                self.W_macro_out[w_idx] = torch.clamp(
                    self.W_macro_out[w_idx] + 0.012 * torch.bmm(macro_err, self.h[w_idx].detach().unsqueeze(1)),
                    -2.5,
                    2.5,
                )
                mono_err = (delta_e[w_idx].unsqueeze(-1) - self.micro_monologue[w_idx]).unsqueeze(-1)
                self.W_mono_out[w_idx] = torch.clamp(
                    self.W_mono_out[w_idx] + 0.015 * torch.bmm(mono_err, self.h[w_idx].detach().unsqueeze(1)),
                    -2.5,
                    2.5,
                )
                mentor_macro_mean = self.macro_monologue[w_idx].detach().mean(dim=0)
                self.cultural_macro_proto = 0.85 * self.cultural_macro_proto + 0.15 * mentor_macro_mean
                mentor_body = self.last_body_logits[w_idx].detach()
                mentor_body_norm = F.normalize(mentor_body, dim=-1, eps=1e-6)
                verified_mentors = []
                verified_symbols = []
                for m_pos, m_i in enumerate(w_idx):
                    m_vec = mentor_body_norm[m_pos]
                    lex_m = F.normalize(self.W_symbol_lexicon[m_i], dim=-1, eps=1e-6)
                    cos_sims = torch.mv(lex_m, m_vec)
                    best_cos, best_sym_t = torch.max(cos_sims, dim=0)
                    sym_id = int(best_sym_t.item())
                    if float(best_cos.item()) < 0.80 and float(delta_e[m_i].item()) > 0.34 and float(torch.min(self.symbol_usage_counts).item()) < 1.0:
                        sym_id = int(torch.argmin(self.symbol_usage_counts).item())
                        self.W_symbol_lexicon[m_i, sym_id] = m_vec.clone()
                        self.symbol_usage_counts[sym_id] += 1.0
                    else:
                        self.W_symbol_lexicon[m_i, sym_id] = F.normalize(
                            0.82 * self.W_symbol_lexicon[m_i, sym_id] + 0.18 * m_vec,
                            dim=-1,
                            eps=1e-6,
                        )
                    recon_vec = F.normalize(self.W_symbol_lexicon[m_i, sym_id], dim=-1, eps=1e-6)
                    recon_cos = float(torch.dot(recon_vec, m_vec).item())
                    if recon_cos >= 0.80:
                        self.symbol_usage_counts[sym_id] += 0.05
                        verified_mentors.append(int(m_i.item()))
                        verified_symbols.append(sym_id)
                    else:
                        self.symbol_silence_decisions += 1
                receiver_mask = (delta_e <= 0.02) & (arena.energy < 0.55) & arena.alive
                if torch.any(receiver_mask) and len(verified_mentors) > 0:
                    r_idx = torch.where(receiver_mask)[0]
                    for r_pos, r_i in enumerate(r_idx):
                        m_slot = r_pos % len(verified_mentors)
                        m_id = verified_mentors[m_slot]
                        sym_id = verified_symbols[m_slot]
                        self.W_symbol_lexicon[r_i, sym_id] = F.normalize(
                            0.78 * self.W_symbol_lexicon[r_i, sym_id] + 0.22 * self.W_symbol_lexicon[m_id, sym_id].detach(),
                            dim=-1,
                            eps=1e-6,
                        )
                        dec_body = self.W_symbol_lexicon[r_i, sym_id].detach()
                        dec_h = torch.tanh(torch.mv(self.W_body_assembly[r_i].detach().t(), dec_body))
                        r_ptr = int(self.hippo_ptr[r_i].item())
                        self.hippo_keys[r_i, r_ptr] = torch.sign(torch.mv(self.Wk[r_i].detach(), dec_h))
                        self.hippo_vals[r_i, r_ptr] = torch.tanh(torch.mv(self.Wv[r_i].detach(), dec_h))
                        self.hippo_acts[r_i, r_ptr] = torch.tanh(torch.mv(self.Wo[r_i].detach(), dec_h))
                        self.hippo_ptr[r_i] = (r_ptr + 1) % self.hippo_len
                        self.verified_symbol_messages += 1
                        self.empathy_share_events += 1

            dream_write_mask = ((delta_e > 0.18) | ((pain > 0.18) & (surprise > 0.20))) & arena.alive
            if torch.any(dream_write_mask):
                dw_idx = torch.where(dream_write_mask)[0]
                d_ptrs = self.dream_ptr[dw_idx]
                self.dream_h[dw_idx, d_ptrs] = self.h[dw_idx].detach()
                self.dream_q[dw_idx, d_ptrs] = self.last_q_spk[dw_idx]
                self.dream_k[dw_idx, d_ptrs] = self.last_k_spk[dw_idx]
                self.dream_v[dw_idx, d_ptrs] = self.last_v_lin[dw_idx]
                self.dream_act[dw_idx, d_ptrs] = act[dw_idx].detach()
                self.dream_reward[dw_idx, d_ptrs] = (delta_e[dw_idx] - 0.85 * pain[dw_idx]).detach()
                self.dream_ptr[dw_idx] = (d_ptrs + 1) % self.dream_len

        newly_dead = (arena.hp <= 0.0) & arena.alive
        if use_mortality and torch.any(newly_dead):
            arena.alive[newly_dead] = False
        elif not use_mortality and torch.any(newly_dead):
            arena.hp[newly_dead] = 0.25

    def offline_sleep_dream_replay(self, arena, n_sleep_cycles=4, dream_lr=0.008, mutation_rate=0.22):
        valid_mask = (self.dream_reward.abs().sum(dim=1) > 1e-4) & arena.alive
        if not torch.any(valid_mask):
            return
        v_idx = torch.where(valid_mask)[0]
        for cyc in range(n_sleep_cycles):
            for slot in range(self.dream_len):
                r_slot = self.dream_reward[v_idx, slot]
                active = r_slot.abs() > 1e-4
                if not torch.any(active):
                    continue
                idx = v_idx[active]
                r_w = torch.clamp(r_slot[active], -0.5, 1.0).unsqueeze(-1).unsqueeze(-1)
                h_d = self.dream_h[idx, slot]
                q_d = self.dream_q[idx, slot]
                k_d = self.dream_k[idx, slot]
                v_d = self.dream_v[idx, slot]
                act_d = self.dream_act[idx, slot]

                partner_slot = (slot + 5 + cyc) % self.dream_len
                partner_idx = torch.roll(idx, shifts=1, dims=0)
                thermal_q = (torch.rand_like(q_d) < mutation_rate).float() * torch.sign(torch.randn_like(q_d))
                thermal_k = (torch.rand_like(k_d) < mutation_rate).float() * torch.sign(torch.randn_like(k_d))
                q_mut = torch.sign(q_d + 0.65 * self.dream_q[partner_idx, partner_slot] + 0.45 * thermal_q)
                k_mut = torch.sign(k_d + 0.65 * self.dream_k[partner_idx, partner_slot] + 0.45 * thermal_k)
                v_mut = torch.tanh(0.60 * v_d + 0.40 * self.dream_v[partner_idx, partner_slot])
                h_mut = torch.tanh(0.60 * h_d + 0.40 * self.dream_h[partner_idx, partner_slot])
                mut_slot = slot % 8
                self.mutated_dream_q[idx, mut_slot] = q_mut.detach()
                self.mutated_dream_k[idx, mut_slot] = k_mut.detach()
                self.mutated_dream_v[idx, mut_slot] = v_mut.detach()
                self.dream_mutation_events += int(len(idx))

                qk_outer = torch.bmm(q_d.unsqueeze(-1), k_d.unsqueeze(1)) + 0.35 * torch.bmm(q_mut.unsqueeze(-1), k_mut.unsqueeze(1))
                vh_outer = torch.bmm(v_d.unsqueeze(-1), h_d.unsqueeze(1)) + 0.35 * torch.bmm(v_mut.unsqueeze(-1), h_mut.unsqueeze(1))
                ah_outer = torch.bmm(act_d.unsqueeze(-1), h_d.unsqueeze(1))
                self.Wq[idx] = torch.clamp(self.Wq[idx] + dream_lr * r_w * qk_outer, -2.5, 2.5)
                self.Wk[idx] = torch.clamp(self.Wk[idx] + dream_lr * r_w * qk_outer.transpose(1, 2), -2.5, 2.5)
                self.Wv[idx] = torch.clamp(self.Wv[idx] + dream_lr * r_w * vh_outer, -2.5, 2.5)
                pos_mask = r_slot[active] > 0.15
                if torch.any(pos_mask):
                    p_idx = idx[pos_mask]
                    rw_pos = r_slot[active][pos_mask].unsqueeze(-1).unsqueeze(-1)
                    self.Wo[p_idx] = torch.clamp(self.Wo[p_idx] + 0.5 * dream_lr * rw_pos * ah_outer[pos_mask], -3.5, 3.5)


def run_condition_benchmark(
    condition_name,
    unpredictable_mode,
    use_pain=True,
    use_curiosity=True,
    use_active_inference=True,
    metabolic_thought_mode="adaptive",
    use_mortality=True,
    plastic=True,
    comm_enabled=True,
    n_generations=8,
    steps_per_gen=150,
    seed=42,
):
    pop_size = 4
    arena = IsaacLabUnified3DArena(pop_size=pop_size, seed=seed)
    brain = SpikingTransformerActiveInferenceBrain(pop_size=pop_size, obs_dim=33, hid_dim=128, seed=seed)
    brain.load_saved_manifold()
    ubs_init = UnifiedBiologicalBrainSystem(
        hid_dim=brain.hid_dim,
        artifact_dir=str(ARTIFACTS_DIR),
        seed=seed,
    )
    loaded_init = ubs_init.load_all_brains()
    if any(loaded_init.values()):
        ubs_init.sync_to_stage1_brain(brain)

    gen_survival = []
    gen_harvest = []
    gen_damage_events = []
    gen_spike_rates = []
    gen_thought_energy = []
    gen_toys_solved = []
    gen_curiosity_explorations = []
    gen_threat_observations = []
    gen_biome_entropy = []
    danger_latents = []
    safe_latents = []
    recovery_latencies = []

    for gen in range(n_generations):
        mode = unpredictable_mode
        arena.reset_world(unpredictable_mode=mode)
        brain.reset_states()

        total_harvest = 0.0
        damage_steps = 0
        spike_accum = 0.0
        thought_tax_accum = 0.0
        shock_step = None
        recovered_step = None

        obs = arena.get_observations()
        for step in range(steps_per_gen):
            motor_u, vocal_s, act, thought_tax = brain.forward_active_inference(
                obs,
                arena,
                use_pain=use_pain,
                use_curiosity=use_curiosity,
                use_active_inference=use_active_inference,
                metabolic_thought_mode=metabolic_thought_mode,
            )
            dmg, delta_e, _ = arena.step_physics(motor_u, vocal_s, thought_tax, comm_enabled=comm_enabled)
            next_obs = arena.get_observations()

            alive_f = arena.alive.float()
            total_harvest += float((delta_e > 0.0).float().mul(alive_f).sum().item())
            damage_steps += int(((dmg > 0.05) & arena.alive).sum().item())
            spike_accum += float(brain.last_spike_rate.mean().item())
            thought_tax_accum += float(thought_tax.mean().item())

            if step >= 15 and shock_step is None and torch.any((dmg > 0.15) & arena.alive):
                shock_step = step
            if shock_step is not None and recovered_step is None and step > shock_step:
                if float((delta_e > 0.0).float().mul(alive_f).sum().item()) >= 1.0:
                    recovered_step = step - shock_step

            in_danger = ((arena.pain > 0.08) | (dmg > 0.05) | (arena.hp < 0.60)) & arena.alive
            in_safe = (arena.pain < 0.02) & (delta_e > 0.0) & arena.alive
            if torch.any(in_danger) and step % 2 == 0:
                danger_latents.append(brain.h[in_danger].detach().cpu().numpy())
            if torch.any(in_safe) and step % 4 == 0:
                safe_latents.append(brain.h[in_safe].detach().cpu().numpy())

            brain.hebbian_plasticity_and_mortality(
                arena,
                obs,
                act,
                next_obs,
                delta_e,
                arena.pain,
                use_pain=use_pain,
                use_curiosity=use_curiosity,
                use_mortality=use_mortality,
                plastic=plastic,
            )
            obs = next_obs

        surv_pct = float(arena.alive.float().mean().item() * 100.0)
        gen_survival.append(surv_pct)
        gen_harvest.append(total_harvest / float(pop_size))
        gen_damage_events.append(damage_steps / float(pop_size))
        gen_spike_rates.append(spike_accum / steps_per_gen)
        gen_thought_energy.append(thought_tax_accum)
        gen_toys_solved.append(float(arena.toys_solved_total) / float(pop_size))
        gen_curiosity_explorations.append(float(arena.curiosity_explorations_total) / float(pop_size))
        gen_threat_observations.append(float(arena.threat_observations_total) / float(pop_size))
        gen_biome_entropy.append(float(arena.biome_dispersion_entropy))
        if recovered_step is not None:
            recovery_latencies.append(recovered_step)

        if use_mortality and plastic:
            brain.offline_sleep_dream_replay(arena, n_sleep_cycles=3, dream_lr=0.008)
            fitness = arena.alive.float() * 2.0 + arena.energy + arena.hp
            top_idx = torch.argsort(fitness, descending=True)[:max(1, pop_size // 2)]
            for i in range(pop_size):
                if not arena.alive[i] or i not in top_idx:
                    p = top_idx[i % len(top_idx)]
                    brain.Wx[i] = brain.Wx[p] + torch.randn_like(brain.Wx[p]) * 0.010
                    brain.Wx_retinal[i] = brain.Wx_retinal[p] + torch.randn_like(brain.Wx_retinal[p]) * 0.010
                    brain.Wh[i] = brain.Wh[p] + torch.randn_like(brain.Wh[p]) * 0.010
                    brain.Wo[i] = brain.Wo[p] + torch.randn_like(brain.Wo[p]) * 0.012
                    brain.W_enc[i] = brain.W_enc[p] + torch.randn_like(brain.W_enc[p]) * 0.010
                    brain.W_enc_retinal[i] = brain.W_enc_retinal[p] + torch.randn_like(brain.W_enc_retinal[p]) * 0.010
                    brain.Wq[i] = brain.Wq[p] + torch.randn_like(brain.Wq[p]) * 0.008
                    brain.Wk[i] = brain.Wk[p] + torch.randn_like(brain.Wk[p]) * 0.008
                    brain.Wv[i] = brain.Wv[p] + torch.randn_like(brain.Wv[p]) * 0.008
                    brain.Wpred_obs[i] = brain.Wpred_obs[p] + torch.randn_like(brain.Wpred_obs[p]) * 0.010
                    brain.Wpred_retinal[i] = brain.Wpred_retinal[p] + torch.randn_like(brain.Wpred_retinal[p]) * 0.010
                    brain.Wpred_val[i] = brain.Wpred_val[p] + torch.randn_like(brain.Wpred_val[p]) * 0.010
                    brain.hippo_keys[i] = brain.hippo_keys[p].clone()
                    brain.hippo_vals[i] = brain.hippo_vals[p].clone()
                    brain.hippo_acts[i] = brain.hippo_acts[p].clone()
                    brain.W_macro_out[i] = brain.W_macro_out[p] + torch.randn_like(brain.W_macro_out[p]) * 0.008
                    brain.W_macro_in[i] = brain.W_macro_in[p] + torch.randn_like(brain.W_macro_in[p]) * 0.008
                    brain.W_mono_out[i] = brain.W_mono_out[p] + torch.randn_like(brain.W_mono_out[p]) * 0.008
                    brain.W_mono_in[i] = brain.W_mono_in[p] + torch.randn_like(brain.W_mono_in[p]) * 0.008
                    brain.theta[i] = brain.theta[p].clone()
                    brain.W_body_assembly[i] = brain.W_body_assembly[p] + torch.randn_like(brain.W_body_assembly[p]) * 0.010
                    brain.W_symbol_lexicon[i] = F.normalize(
                        brain.W_symbol_lexicon[p] + torch.randn_like(brain.W_symbol_lexicon[p]) * 0.008,
                        dim=-1,
                        eps=1e-6,
                    )

    mean_danger_vec = (
        np.mean(np.concatenate(danger_latents, axis=0), axis=0)
        if danger_latents
        else np.zeros((brain.hid_dim,), dtype=np.float32)
    )
    mean_safe_vec = (
        np.mean(np.concatenate(safe_latents, axis=0), axis=0)
        if safe_latents
        else np.zeros((brain.hid_dim,), dtype=np.float32)
    )

    return {
        "condition": condition_name,
        "unpredictable_mode": unpredictable_mode,
        "terminal_survival_pct": round(float(np.mean(gen_survival[-3:])), 2),
        "harvest_per_agent": round(float(np.mean(gen_harvest[-3:])), 2),
        "damage_events_per_agent": round(float(np.mean(gen_damage_events[-3:])), 2),
        "mean_spike_firing_rate": round(float(np.mean(gen_spike_rates[-3:])), 4),
        "mean_active_cell_ratio": round(float(brain.active_cell_ratio.mean().item()), 4),
        "thought_energy_consumed": round(float(np.mean(gen_thought_energy[-3:])), 4),
        "adaptation_recovery_steps": round(float(np.mean(recovery_latencies)) if recovery_latencies else 99.0, 2),
        "early_gen_survival_pct": round(float(gen_survival[min(2, len(gen_survival) - 1)]), 2),
        "final_gen_survival_pct": round(float(gen_survival[-1]), 2),
        "toys_solved_per_agent": round(float(np.mean(gen_toys_solved[-3:])), 2),
        "curiosity_explorations_per_agent": round(float(np.mean(gen_curiosity_explorations[-3:])), 2),
        "threat_observations_per_agent": round(float(np.mean(gen_threat_observations[-3:])), 2),
        "biome_dispersion_entropy": round(float(np.mean(gen_biome_entropy[-3:])), 4),
        "empathy_share_events": int(brain.empathy_share_events),
        "verified_symbol_messages": int(brain.verified_symbol_messages),
        "symbol_silence_decisions": int(brain.symbol_silence_decisions),
        "dream_mutation_events": int(brain.dream_mutation_events),
        "sensorimotor_chunk_hits_total": int(arena.sensorimotor_chunk_hits_total),
        "cerebellar_corrections_total": int(arena.cerebellar_corrections_total),
        "discovered_sensorimotor_chunks": dict(arena.discovered_sensorimotor_chunks),
        "danger_vec": mean_danger_vec,
        "safe_vec": mean_safe_vec,
        "brain": brain,
        "arena": arena,
    }


def cosine_sim(a, b):
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def evolve_infinite_procedural_hazard_manifold(n_cycles=12, steps_per_cycle=140, seed=2026):
    global DEVICE
    t0 = time.time()
    torch.manual_seed(seed)
    np.random.seed(seed)
    res_proc = run_condition_benchmark(
        "InfiniteProceduralHazard_ActiveInference_Full",
        "infinite_procedural_chaos",
        use_pain=True,
        use_curiosity=True,
        use_active_inference=True,
        metabolic_thought_mode="adaptive",
        use_mortality=True,
        plastic=True,
        comm_enabled=True,
        n_generations=n_cycles,
        steps_per_gen=steps_per_cycle,
        seed=seed,
    )
    brain = res_proc["brain"]
    arena_ref = res_proc["arena"]
    dna_genomes = [arena_ref.sample_procedural_hazard_dna(c_idx) for c_idx in range(n_cycles)]

    wx_mean = brain.Wx.mean(dim=0).detach().cpu().numpy()
    w_enc_mean = brain.W_enc.mean(dim=0).detach().cpu().numpy()
    wpred_obs_mean = brain.Wpred_obs.mean(dim=0).detach().cpu().numpy()
    wpred_val_mean = brain.Wpred_val.mean(dim=0).detach().cpu().numpy()
    wq_mean = brain.Wq.mean(dim=0).detach().cpu().numpy()
    wk_mean = brain.Wk.mean(dim=0).detach().cpu().numpy()
    wv_mean = brain.Wv.mean(dim=0).detach().cpu().numpy()
    wx_retinal_mean = brain.Wx_retinal.mean(dim=0).detach().cpu().numpy()
    w_enc_retinal_mean = brain.W_enc_retinal.mean(dim=0).detach().cpu().numpy()
    wpred_retinal_mean = brain.Wpred_retinal.mean(dim=0).detach().cpu().numpy()
    wh_mean = brain.Wh.mean(dim=0).detach().cpu().numpy()
    wo_mean = brain.Wo.mean(dim=0).detach().cpu().numpy()
    spk_ctx_mean = brain.spike_ctx.mean(dim=0).detach().cpu().numpy()
    val_ctx_mean = brain.val_ctx.mean(dim=0).detach().cpu().numpy()
    hippo_k_mean = brain.hippo_keys.mean(dim=0).detach().cpu().numpy()
    hippo_v_mean = brain.hippo_vals.mean(dim=0).detach().cpu().numpy()
    w_macro_out_mean = brain.W_macro_out.mean(dim=0).detach().cpu().numpy()
    w_macro_in_mean = brain.W_macro_in.mean(dim=0).detach().cpu().numpy()
    w_mono_out_mean = brain.W_mono_out.mean(dim=0).detach().cpu().numpy()
    w_mono_in_mean = brain.W_mono_in.mean(dim=0).detach().cpu().numpy()
    dream_q_mean = brain.dream_q.mean(dim=0).detach().cpu().numpy()
    dream_k_mean = brain.dream_k.mean(dim=0).detach().cpu().numpy()
    dream_v_mean = brain.dream_v.mean(dim=0).detach().cpu().numpy()
    mut_dream_q_mean = brain.mutated_dream_q.mean(dim=0).detach().cpu().numpy()
    mut_dream_k_mean = brain.mutated_dream_k.mean(dim=0).detach().cpu().numpy()
    mut_dream_v_mean = brain.mutated_dream_v.mean(dim=0).detach().cpu().numpy()
    cultural_macro_proto = brain.cultural_macro_proto.detach().cpu().numpy()
    w_body_assembly_mean = brain.W_body_assembly.mean(dim=0).detach().cpu().numpy()
    w_symbol_lexicon_mean = brain.W_symbol_lexicon.mean(dim=0).detach().cpu().numpy()

    np.savez(
        ARTIFACTS_DIR / "stage1_isaac3d_core_manifold.npz",
        Wx=wx_mean,
        W_enc=w_enc_mean,
        Wpred_obs=wpred_obs_mean,
        Wpred_val=wpred_val_mean,
        Wq=wq_mean,
        Wk=wk_mean,
        Wv=wv_mean,
        Wx_retinal=wx_retinal_mean,
        W_enc_retinal=w_enc_retinal_mean,
        Wpred_retinal=wpred_retinal_mean,
        Wh=wh_mean,
        Wo=wo_mean,
        spike_ctx=spk_ctx_mean,
        val_ctx=val_ctx_mean,
        hippo_keys=hippo_k_mean,
        hippo_vals=hippo_v_mean,
        W_macro_out=w_macro_out_mean,
        W_macro_in=w_macro_in_mean,
        W_mono_out=w_mono_out_mean,
        W_mono_in=w_mono_in_mean,
        W_body_assembly=w_body_assembly_mean,
        W_symbol_lexicon=w_symbol_lexicon_mean,
        dream_q=dream_q_mean,
        dream_k=dream_k_mean,
        dream_v=dream_v_mean,
        mutated_dream_q=mut_dream_q_mean,
        mutated_dream_k=mut_dream_k_mean,
        mutated_dream_v=mut_dream_v_mean,
        cultural_macro_proto=cultural_macro_proto,
        danger_vec=res_proc["danger_vec"],
    )
    np.savez(
        ARTIFACTS_DIR / "stage1_core_manifold.npz",
        Wx=wx_retinal_mean,
        Wh=wh_mean,
        Wq=wq_mean,
        Wk=wk_mean,
        Wv=wv_mean,
        spike_ctx=spk_ctx_mean,
        val_ctx=val_ctx_mean,
        danger_vec=res_proc["danger_vec"],
    )
    unified_brains = UnifiedBiologicalBrainSystem(
        hid_dim=brain.hid_dim,
        artifact_dir=str(ARTIFACTS_DIR),
        seed=seed,
    )
    unified_brains.load_all_brains()
    edu_3d_stats = unified_brains.educate_three_brains_in_3d_arena(
        res_proc["arena"],
        brain,
        n_epochs=8,
        lr=0.008,
    )
    exported_brain_paths = unified_brains.save_all_brains(
        extra_meta={
            "stage": "stage1_procedural_evolution",
            "cycles": int(n_cycles),
            **edu_3d_stats,
        }
    )

    proc_report = {
        "route": "Route1_Infinite_Procedural_3D_Hazard_Generator",
        "procedural_cycles": int(n_cycles),
        "steps_per_cycle": int(steps_per_cycle),
        "elapsed_sec": round(time.time() - t0, 2),
        "exported_unified_brains": {k: str(v) for k, v in exported_brain_paths.items()},
        "three_brain_3d_education": edu_3d_stats,
        "terminal_survival_pct": res_proc["terminal_survival_pct"],
        "early_gen_survival_pct": res_proc["early_gen_survival_pct"],
        "final_gen_survival_pct": res_proc["final_gen_survival_pct"],
        "harvest_per_agent": res_proc["harvest_per_agent"],
        "damage_events_per_agent": res_proc["damage_events_per_agent"],
        "mean_spike_firing_rate": res_proc["mean_spike_firing_rate"],
        "mean_active_cell_ratio": res_proc["mean_active_cell_ratio"],
        "thought_energy_consumed": res_proc["thought_energy_consumed"],
        "toys_solved_per_agent": res_proc["toys_solved_per_agent"],
        "curiosity_explorations_per_agent": res_proc["curiosity_explorations_per_agent"],
        "threat_observations_per_agent": res_proc["threat_observations_per_agent"],
        "biome_dispersion_entropy": res_proc["biome_dispersion_entropy"],
        "empathy_share_events": res_proc["empathy_share_events"],
        "verified_symbol_messages": int(res_proc.get("verified_symbol_messages", 0)),
        "symbol_silence_decisions": int(res_proc.get("symbol_silence_decisions", 0)),
        "verified_symbol_messages_total": int(res_proc.get("verified_symbol_messages", 0)),
        "symbol_silence_decisions_total": int(res_proc.get("symbol_silence_decisions", 0)),
        "dream_mutation_events": res_proc["dream_mutation_events"],
        "sensorimotor_chunk_hits_total": int(res_proc.get("sensorimotor_chunk_hits_total", 0)),
        "cerebellar_corrections_total": int(res_proc.get("cerebellar_corrections_total", 0)),
        "discovered_sensorimotor_chunks": res_proc.get("discovered_sensorimotor_chunks", {}),
        "procedural_dna_samples": dna_genomes[:4],
    }
    chunks_path = ARTIFACTS_DIR / "stage1_sensorimotor_chunks.json"
    merged_chunks = {}
    if chunks_path.exists():
        try:
            with open(chunks_path, "r", encoding="utf-8") as f:
                old_c = json.load(f)
            if isinstance(old_c, dict):
                merged_chunks.update(old_c)
        except Exception:
            merged_chunks = {}
    for c_k, c_v in res_proc.get("discovered_sensorimotor_chunks", {}).items():
        if c_k in merged_chunks and isinstance(merged_chunks[c_k], dict) and isinstance(c_v, dict):
            merged_chunks[c_k]["hits"] = int(merged_chunks[c_k].get("hits", 1)) + int(c_v.get("hits", 1))
        else:
            merged_chunks[c_k] = c_v
    proc_report["discovered_sensorimotor_chunks"] = merged_chunks
    with open(ARTIFACTS_DIR / "stage1_procedural_hazard_evolution.json", "w", encoding="utf-8") as f:
        json.dump(proc_report, f, indent=2)
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(merged_chunks, f, indent=2)
    return proc_report


def main():
    t0 = time.time()
    hazard_scenarios = [
        "meteor_3d",
        "abyss_collapse",
        "ambush_stalker",
        "mimic_trap",
        "vortex_storm",
        "signpost_indicator",
        "chaos_all",
    ]

    scenario_results = {}
    danger_vectors_by_hazard = {}
    last_full_brain = None
    last_full_arena = None

    for hz_mode in hazard_scenarios:
        res_full = run_condition_benchmark(
            "SpikingTransformer_ActiveInference_AdaptiveMetabolism",
            hz_mode,
            use_pain=True,
            use_curiosity=True,
            use_active_inference=True,
            metabolic_thought_mode="adaptive",
            use_mortality=True,
            plastic=True,
            comm_enabled=True,
        )
        danger_vectors_by_hazard[hz_mode] = res_full["danger_vec"]
        if hz_mode in ("signpost_indicator", "chaos_all"):
            last_full_brain = res_full["brain"]
            last_full_arena = res_full["arena"]

        res_no_ai = run_condition_benchmark(
            "NoImagination_ReactiveOnly_PaperBaseline",
            hz_mode,
            use_pain=True,
            use_curiosity=False,
            use_active_inference=False,
            metabolic_thought_mode="adaptive",
            use_mortality=True,
            plastic=True,
            comm_enabled=True,
        )
        res_overthink = run_condition_benchmark(
            "UnboundedOverthinking_MetabolicExhaustion",
            hz_mode,
            use_pain=True,
            use_curiosity=True,
            use_active_inference=True,
            metabolic_thought_mode="overthink_unbounded",
            use_mortality=True,
            plastic=True,
            comm_enabled=True,
        )
        res_static = run_condition_benchmark(
            "NoPlasticity_StaticWeights",
            hz_mode,
            use_pain=True,
            use_curiosity=True,
            use_active_inference=False,
            metabolic_thought_mode="adaptive",
            use_mortality=True,
            plastic=False,
            comm_enabled=True,
        )

        scenario_results[hz_mode] = {
            "SpikingTransformer_ActiveInference_Full": {
                k: v for k, v in res_full.items() if k not in ("danger_vec", "safe_vec", "brain", "arena")
            },
            "NoImagination_Paper_Baseline": {
                k: v for k, v in res_no_ai.items() if k not in ("danger_vec", "safe_vec", "brain", "arena")
            },
            "UnboundedOverthinking_Starvation": {
                k: v for k, v in res_overthink.items() if k not in ("danger_vec", "safe_vec", "brain", "arena")
            },
            "NoPlasticity_Static": {
                k: v for k, v in res_static.items() if k not in ("danger_vec", "safe_vec", "brain", "arena")
            },
        }

    cross_hazard_sims = []
    hz_keys = list(danger_vectors_by_hazard.keys())
    for i in range(len(hz_keys)):
        for j in range(i + 1, len(hz_keys)):
            cross_hazard_sims.append(cosine_sim(danger_vectors_by_hazard[hz_keys[i]], danger_vectors_by_hazard[hz_keys[j]]))

    mean_cross_hazard_cos = round(float(np.mean(cross_hazard_sims)), 4)

    summary = {
        "device": str(DEVICE),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "elapsed_sec": round(time.time() - t0, 2),
        "cross_hazard_unpredictable_danger_cosine_similarity": mean_cross_hazard_cos,
        "unpredictable_3d_scenarios": scenario_results,
    }

    out_json = ARTIFACTS_DIR / "stage1_isaac3d_unpredictable_hazard_report.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if last_full_brain is not None:
        wx_mean = last_full_brain.Wx.mean(dim=0).detach().cpu().numpy()
        w_enc_mean = last_full_brain.W_enc.mean(dim=0).detach().cpu().numpy()
        wpred_obs_mean = last_full_brain.Wpred_obs.mean(dim=0).detach().cpu().numpy()
        wpred_val_mean = last_full_brain.Wpred_val.mean(dim=0).detach().cpu().numpy()
        wq_mean = last_full_brain.Wq.mean(dim=0).detach().cpu().numpy()
        wk_mean = last_full_brain.Wk.mean(dim=0).detach().cpu().numpy()
        wv_mean = last_full_brain.Wv.mean(dim=0).detach().cpu().numpy()
        wx_retinal_mean = last_full_brain.Wx_retinal.mean(dim=0).detach().cpu().numpy()
        w_enc_retinal_mean = last_full_brain.W_enc_retinal.mean(dim=0).detach().cpu().numpy()
        wpred_retinal_mean = last_full_brain.Wpred_retinal.mean(dim=0).detach().cpu().numpy()
        wh_mean = last_full_brain.Wh.mean(dim=0).detach().cpu().numpy()
        wo_mean = last_full_brain.Wo.mean(dim=0).detach().cpu().numpy()
        spk_ctx_mean = last_full_brain.spike_ctx.mean(dim=0).detach().cpu().numpy()
        val_ctx_mean = last_full_brain.val_ctx.mean(dim=0).detach().cpu().numpy()
        hippo_k_mean = last_full_brain.hippo_keys.mean(dim=0).detach().cpu().numpy()
        hippo_v_mean = last_full_brain.hippo_vals.mean(dim=0).detach().cpu().numpy()
        w_macro_out_mean = last_full_brain.W_macro_out.mean(dim=0).detach().cpu().numpy()
        w_macro_in_mean = last_full_brain.W_macro_in.mean(dim=0).detach().cpu().numpy()
        w_mono_out_mean = last_full_brain.W_mono_out.mean(dim=0).detach().cpu().numpy()
        w_mono_in_mean = last_full_brain.W_mono_in.mean(dim=0).detach().cpu().numpy()
        dream_q_mean = last_full_brain.dream_q.mean(dim=0).detach().cpu().numpy()
        dream_k_mean = last_full_brain.dream_k.mean(dim=0).detach().cpu().numpy()
        dream_v_mean = last_full_brain.dream_v.mean(dim=0).detach().cpu().numpy()
        mut_dream_q_mean = last_full_brain.mutated_dream_q.mean(dim=0).detach().cpu().numpy()
        mut_dream_k_mean = last_full_brain.mutated_dream_k.mean(dim=0).detach().cpu().numpy()
        mut_dream_v_mean = last_full_brain.mutated_dream_v.mean(dim=0).detach().cpu().numpy()
        cultural_macro_proto = last_full_brain.cultural_macro_proto.detach().cpu().numpy()
        w_body_assembly_mean = last_full_brain.W_body_assembly.mean(dim=0).detach().cpu().numpy()
        w_symbol_lexicon_mean = last_full_brain.W_symbol_lexicon.mean(dim=0).detach().cpu().numpy()
        valid_d_vecs = [v for v in danger_vectors_by_hazard.values() if np.linalg.norm(v) > 1e-6]
        d_vec = np.mean(valid_d_vecs, axis=0) if valid_d_vecs else np.zeros((last_full_brain.hid_dim,), dtype=np.float32)
        np.savez(
            ARTIFACTS_DIR / "stage1_isaac3d_core_manifold.npz",
            Wx=wx_mean,
            W_enc=w_enc_mean,
            Wpred_obs=wpred_obs_mean,
            Wpred_val=wpred_val_mean,
            Wq=wq_mean,
            Wk=wk_mean,
            Wv=wv_mean,
            Wx_retinal=wx_retinal_mean,
            W_enc_retinal=w_enc_retinal_mean,
            Wpred_retinal=wpred_retinal_mean,
            Wh=wh_mean,
            Wo=wo_mean,
            spike_ctx=spk_ctx_mean,
            val_ctx=val_ctx_mean,
            hippo_keys=hippo_k_mean,
            hippo_vals=hippo_v_mean,
            W_macro_out=w_macro_out_mean,
            W_macro_in=w_macro_in_mean,
            W_mono_out=w_mono_out_mean,
            W_mono_in=w_mono_in_mean,
            W_body_assembly=w_body_assembly_mean,
            W_symbol_lexicon=w_symbol_lexicon_mean,
            dream_q=dream_q_mean,
            dream_k=dream_k_mean,
            dream_v=dream_v_mean,
            mutated_dream_q=mut_dream_q_mean,
            mutated_dream_k=mut_dream_k_mean,
            mutated_dream_v=mut_dream_v_mean,
            cultural_macro_proto=cultural_macro_proto,
            danger_vec=d_vec,
        )
        np.savez(
            ARTIFACTS_DIR / "stage1_core_manifold.npz",
            Wx=wx_retinal_mean,
            Wh=wh_mean,
            Wq=wq_mean,
            Wk=wk_mean,
            Wv=wv_mean,
            spike_ctx=spk_ctx_mean,
            val_ctx=val_ctx_mean,
            danger_vec=d_vec,
        )
        unified_brains = UnifiedBiologicalBrainSystem(
            hid_dim=last_full_brain.hid_dim,
            artifact_dir=str(ARTIFACTS_DIR),
            seed=42,
        )
        unified_brains.load_all_brains()
        edu_3d_stats = unified_brains.educate_three_brains_in_3d_arena(
            last_full_arena,
            last_full_brain,
            n_epochs=8,
            lr=0.008,
        )
        unified_brains.save_all_brains(
            extra_meta={
                "stage": "stage1_main_benchmark",
                **edu_3d_stats,
            }
        )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
