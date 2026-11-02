# Embodied Active Inference ARC Engine

Official Implementation and Research Artifacts for ARC Prize 2026 Paper Track.

Autonomous Embodied Artificial General Intelligence architecture grounded in Life-Information Theory, Karl Friston's Free Energy Principle, and Cumulative Cultural Scaffolding.

Repository: `https://github.com/sukuyoshi/arc-embodied-active-inference`  
License: MIT License

---

## Abstract

Traditional approaches to the Abstraction and Reasoning Corpus (ARC / ARC-AGI-2) rely heavily on disembodied Large Language Models (LLMs) with massive pretraining, prompt engineering, or brute-force search over handwritten Domain-Specific Languages (DSLs). These models consume thousands of watts and lack genuine grounding, suffering from severe sample inefficiency and fragility when confronted with multi-step composition or spatial novelties.

In contrast, this project presents an ultra-lightweight architecture of **459,930 parameters (<2MB)** designed for 20W-class edge neuromorphic deployment with **zero pretraining data (Tabula Rasa)**. The system is grounded in a 4-biome 3D physics survival environment where irreversible mortality (permanent termination at zero health), homeostatic drive, and visceral pain dynamically govern active inference.

Through sleep-state hippocampal replay and synaptic plasticity consolidation (LTP reinforcement and LTD pruning), the organism crystallizes high-order intuition organs in the basal ganglia. Evaluated on the official Kaggle ARC-AGI-2 benchmark (240 unseen tasks), the pipeline executed all 240 tasks in 1,406.99 seconds without runtime exceptions. Under a strict 8-step budget, the unassisted test exact-match score was 0.00 (0 / 240 unseen test tasks solved), while achieving **instantaneous one-shot exact solutions on Task 00d62c1b (enclosure fill) and Task 3c9b0459 (rigid rotation) in a single step (`steps=1, train_exact=True, test_exact=True`)**.

---

## Theoretical Pillars & Transfer Mechanism

1. **Life-Information Theory**:
   Intelligence cannot arise in an immortal or disembodied system. Irreversible mortality constitutes the ultimate boundary condition, producing intrinsic value, meaning, and true homeostatic urgency.
2. **Active Inference and Free Energy Principle (Karl Friston)**:
   Action and perception are coupled to minimize variational and expected free energy under visceral pain, energy, and surprise constraints.
3. **Visuomotor-to-Grid Grounding**:
   Foveal retinal saccades map to 2D contour/cavity extraction; physical block manipulation (push, grasp-rotate, peg insertion) maps directly to 2D matrix translation, 180° rotation, and topological enclosure fill operators.
4. **Sleep Replay and Plasticity Consolidation**:
   Sharp-wave ripple (SWR) reactivation during slow-wave sleep consolidates sensorimotor traces into basal ganglia reflex chunks and 3-tier intuition organs via dual Hebbian plasticity.

---

## Repository Structure

```text
arc-creativity-engine/
├── evaluate_engine.py                       (Standalone benchmark evaluator; 1-click verification)
├── embodied_core_engine/
│   ├── unified_biological_brain.py          (Multimodal biological brain with Spiking Transformer & Hippocampus)
│   ├── stage2_embodied_puzzle_solver.py     (Active inference puzzle solver with basal ganglia reflex & intuition)
│   ├── stage4_kaggle_submission_inference.py(Lightweight submission engine for ARC benchmark)
│   ├── run_kaggle_notebook.py               (Kaggle execution entrypoint)
│   ├── run_experience_plasticity_cycle.py   (Sleep-dream plasticity consolidation cycle)
│   ├── four_biome_humanoid_env.py           (Stage 1 3D humanoid biome environment)
│   ├── isaac_warp_3d_engine.py              (Isaac Lab & 3D tabletop spatial simulation engine)
│   └── artifacts/checkpoints/               (Trained synaptic weights and neural modules)
├── paper_track_2026/
│   ├── KAGGLE_PAPER_TRACK_WRITEUP_JA.md     (Complete Paper Track Paper in Japanese)
│   ├── KAGGLE_PAPER_TRACK_WRITEUP_EN.md     (Complete Paper Track Paper in English)
│   ├── arc_prize_2026_paper_track_reproducible_notebook.ipynb (Jupyter Reproduction Notebook)
│   └── media_gallery/                       (High-resolution architecture and telemetry figures)
├── LICENSE                                  (MIT License)
└── README.md                                (Project Documentation)
```

---

## Empirical Verification & Benchmark Telemetry

The model was evaluated across both the official Kaggle test benchmark and local evaluation sets:

- **Official Kaggle ARC-AGI-2 Hidden Test**:
  - Total Tasks Processed: 240 / 240 tasks completed in 1,406.99s (5.86s per task) without runtime failure.
  - Test Exact-Match Score: 0.00 (0 / 240 solved under strict 8-step budget).
  - Output Submission: `submission.json` (341.21 KB, strict schema compliance).
- **Verified One-Shot Intuition Solutions**:
  - `Task 00d62c1b`: `steps=1, train_exact=True, test_exact=True` (Kinematic Enclosure Fill)
  - `Task 3c9b0459`: `steps=1, train_exact=True, test_exact=True` (Rigid Body Rotation)
- **Full Local Benchmark (120 Evaluation Tasks)**:
  - 120 / 120 tasks processed in 535.95s (4.47s per task).
  - Train/Test Exact-Match Score: 0.0% (0 / 120 solved under strict 8-step budget), exactly replicating Kaggle telemetry. Complete logs are preserved in `embodied_core_engine/artifacts/arc_agi_2_eval_120_full_benchmark.json`.

---

## Quickstart & Open Reproducibility

Anyone can verify the engine and inspect one-shot intuition solutions in 5 seconds with zero external dependencies:

```bash
git clone https://github.com/sukuyoshi/arc-embodied-active-inference.git
cd arc-embodied-active-inference
python evaluate_engine.py
```

To evaluate a custom task directory:
```bash
python evaluate_engine.py --dataset_dir /path/to/tasks --limit 20 --steps 8
```

---

## Academic Writeup & Citation

For full mathematical derivations, transfer mechanics, failure mode analyses, and architectural extensions, see:
- [KAGGLE_PAPER_TRACK_WRITEUP_JA.md](paper_track_2026/KAGGLE_PAPER_TRACK_WRITEUP_JA.md) (Japanese)
- [KAGGLE_PAPER_TRACK_WRITEUP_EN.md](paper_track_2026/KAGGLE_PAPER_TRACK_WRITEUP_EN.md) (English)

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
