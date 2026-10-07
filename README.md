# Reproducing every table in the paper

This bundle contains the complete, current set of scripts needed to regenerate every table
in the paper — 16 Python scripts plus cluster orchestration helpers. It deliberately excludes
many earlier, superseded versions of these scripts that existed at various points during
development; everything here is the final, verified version that produced the numbers actually
reported.

## 0. Prerequisites

- Python 3.10+, with `numpy`, `scipy`, `scikit-learn`, `torch` installed everywhere.
- `xgboost` (optional — `compute_paper_numbers.py` skips its two rows gracefully if absent).
- `sentence-transformers` — only needed for `plan_step45_encoder_robustness.py` and for
  generating real (non-placeholder) `X_fm.npy` embeddings in step 1. Needs internet access the
  first time, to download the encoder weights.
- An Ollama server reachable at some `--host` — only needed for the four LLM scripts
  (`plan_step41`, `46`, `47`, `48`). `cluster_helpers/start_ollama.sh` is a SLURM-cluster-specific
  helper for standing one up; adapt or ignore it if you're not on that cluster.
- GPU — needed for ARC training (`plan_table1.py`, `plan_step44`, `plan_step51`) and for running
  the LLM scripts at a reasonable speed. The baseline/statistics scripts
  (`compute_paper_numbers.py`, `plan_step50`, `plan_step43`, `compute_bootstrap_cis.py`,
  `plan_step53`) are CPU-only.

## 1. Generate the data (run this first, once)

```bash
python plan_step1_data_prep_v3.py
```

Produces `data/planning_v3/` containing `episodes.json`, `X_surf.npy`, `X_fm.npy`,
`y_nsteps.npy`, `y_valid.npy`, `task_types.npy` for the seven main domains (Blocksworld,
Logistics, Mystery-Blocksworld, Depot, Rovers, Satellite, Gripper). Every other script in this
bundle reads from this directory via `--data data/planning_v3` (or a matching default).

**`X_fm.npy` note:** without `sentence-transformers` installed, this embedding falls back to a
random placeholder. Every ARC-related number, the representation-family ablation, and the
encoder-robustness table are meaningless without the real embedding — install
`sentence-transformers` before running step 1 if you want those numbers to mean anything.

## 2. Table-by-table map

| Paper table | Script(s) | Command | Needs GPU? |
|---|---|---|---|
| **Table 1 (main_results)** — Object-card, Goal count, Ridge, Ridge(z), Fisher CI | `compute_paper_numbers.py` | `python compute_paper_numbers.py data/planning_v3` | No |
| Table 1 — LogME, H-score, ERM, ARC (main + gate_blend) | `plan_table1.py` | `python plan_table1.py --n_episodes 5000 --seeds 0 1 2 42 123 --val_every 200 --patience 10 --norm pooled --y_norm domain --gate none --tag main` (repeat with `--gate blend --tag gate_blend`) | Yes |
| Table 1 — $h_\text{add}$ row | `plan_step50_relaxed_heuristic_features.py` | `python plan_step50_relaxed_heuristic_features.py --data data/planning_v3 --max_per_domain 200` | No |
| Table 1 — ARC prediction-ensembled row | `plan_step51_arc_scores_export.py` | `python plan_step51_arc_scores_export.py --seeds 0 1 2 42 123 --gate blend --tag gate_blend` | Yes |
| **Within-size table** (both main-text and appendix-detailed) | `compute_paper_numbers.py` (Ridge rows) + `plan_step44_arc_ablations.py --suite within` (ARC rows) | `python plan_step44_arc_ablations.py --suite within --seeds 0 1 2 42 123 --n_episodes 5000` | Yes (ARC rows only) |
| **Primary CI table** (Fisher intervals) | `compute_paper_numbers.py` | (same run as Table 1's deterministic rows) | No |
| **Paired bootstrap deltas** (Ridge−\|O\|, $h_\text{add}$−goal count) | `compute_bootstrap_cis.py` | `python compute_bootstrap_cis.py --data data/planning_v3` | No |
| **Leave-one-domain-out table** | `compute_paper_numbers.py` (Ridge/\|O\| rows) + `plan_step44_arc_ablations.py --suite loo` (ARC rows) | `python plan_step44_arc_ablations.py --suite loo --seeds 0 1 2 42 123 --n_episodes 5000` | Yes (ARC rows only) |
| **Learned-baselines table** (RF, XGBoost, GradBoost, SVR, LambdaMART, Ridge on $x_s$) | `compute_paper_numbers.py` | (same run as Table 1) | No |
| Learned-baselines table — RF on $x_s \Vert x_e$ | `plan_step53_rf_concat.py` | `python plan_step53_rf_concat.py --data data/planning_v3` | No |
| **Cheap-surrogates table** (goal count, init-size, init-density, obj-goal-ratio, object count) | `compute_paper_numbers.py` | (same run as Table 1) | No |
| Cheap-surrogates table — $h_\text{add}$ row and correctness verification | `plan_step50_relaxed_heuristic_features.py` | (same run as above) | No |
| **Solver-association table** (A* node/time correlation) | `plan_step43_astar_association.py` | `python plan_step43_astar_association.py --data data/planning_v3` | No |
| **Representation-family ablation** | `plan_step44_arc_ablations.py --suite repfamily` | `python plan_step44_arc_ablations.py --suite repfamily --seeds 0 1 2 42 123 --n_episodes 5000` | No (this suite is Ridge-only, no ARC training) |
| **Attention-direction ablation** | `plan_step44_arc_ablations.py --suite attn` | `python plan_step44_arc_ablations.py --suite attn --seeds 0 1 2 42 123 --n_episodes 5000` | Yes |
| **Residualization-sensitivity table** | `plan_step44_arc_ablations.py --suite resid` | `python plan_step44_arc_ablations.py --suite resid --seeds 0 1 2 42 123 --n_episodes 5000` | Yes |
| **Encoder-robustness table** | `plan_step45_encoder_robustness.py` | `python plan_step45_encoder_robustness.py --data data/planning_v3` | No (downloads 3 sentence-transformer models on first run) |
| **Custom-domains table** (Rescue/Robot Arm/Library, $\lvert\mathcal{O}\rvert$/Goal/Ridge columns) | `plan_step49_custom_domains_v3.py` | `python plan_step49_custom_domains_v3.py --n_total 200` then `python plan_step49_custom_domains_v3.py --score` | No |
| Custom-domains table — ARC column | `plan_step51_arc_scores_export.py` | (produces `results_planning_v3/arc_custom_domain_scores.json` as a side effect of the Table-1-ARC run above, IF `data/custom_domains_v3/X_fm.npy` already exists from the command above) | Yes |
| **LLM-difficulty-baseline table** (absolute/pairwise scoring) | `plan_step48_llm_difficulty_baseline.py` | `python plan_step48_llm_difficulty_baseline.py --host <ollama-url> --model qwen2.5:72b --ctx 24576` | Needs Ollama, not local GPU per se |
| **Corruption / scrambling table** | `plan_step41_scramble_real_validator.py` | `python plan_step41_scramble_real_validator.py --host <ollama-url> --model qwen2.5:72b --n 40` | Needs Ollama |
| **Routing table** | `plan_step46_routing.py` | `python plan_step46_routing.py --host <ollama-url> --model qwen2.5:72b --n 100 --arc_scores results_planning_v3/arc_scores_for_routing.json` | Needs Ollama; `--arc_scores` needs `plan_step51` to have run first |
| **LLM-capability-across-models table** | `plan_step47_llm_capability.py` | `python plan_step47_llm_capability.py --host <ollama-url> --model <qwen2.5:72b \| qwen2.5:14b \| gemma2:9b \| mistral:7b>` (run once per model) | Needs Ollama |
| **Dataset-statistics table** (per-domain N, solve %, $n^\star$ range) | printed by `plan_step1_data_prep_v3.py` itself | (no separate command — check its stdout from step 1) | No |

`plan_validator.py` is not in this table because it produces no table itself — it's the shared
state-simulating plan validator (precondition checking, goal verification by forward state
simulation) imported by `plan_step41`, `46`, `47`, and `48`. It must be in the same directory as
those four scripts.

## 3. One table this bundle cannot reproduce

The custom-domains **inter-annotator-agreement table** (Cohen's κ, percent agreement across three
human annotators) is not code-reproducible — it came from an actual human annotation session, not
a script. `plan_step49_custom_domains_v3.py` regenerates the *instances* that were annotated, but
not the annotation itself.

## 4. Suggested run order

1. `plan_step1_data_prep_v3.py` — once, first.
2. `compute_paper_numbers.py`, `plan_step50`, `plan_step43`, `compute_bootstrap_cis.py`,
   `plan_step53`, `plan_step45`, `plan_step49` — all CPU-only, can run in any order or in parallel.
3. `plan_table1.py` (both tags), then `plan_step44` (all five `--suite` values), then
   `plan_step51` (both tags) — these need a GPU, and `plan_step51`'s custom-domain scores need
   `plan_step49 --n_total 200` to have already produced `data/custom_domains_v3/`.
4. The four LLM scripts (`41`, `46`, `47`, `48`) — need a live Ollama server. `plan_step46`
   (routing) additionally needs `plan_step51`'s `arc_scores_for_routing.json` from step 3.

`cluster_helpers/` contains SLURM `.sbatch` files that encode this same dependency order for a
SLURM cluster specifically (`run_table1.sbatch`, `run_arc_ablations.sbatch`,
`run_arc_export.sbatch`, `run_llm_job.sbatch`) and `start_ollama.sh`, a helper that locates a GPU
node, loads the Ollama module, and preloads a model at the context length the difficulty-baseline
script needs. These are specific to one SLURM cluster's module and partition names and will need
adapting (or can be ignored entirely) on a different system — read them as a reference for the
command sequence, not as portable infrastructure.

## 5. A note on determinism

Every script accepts `--seeds 0 1 2 42 123` (or similar) and is intended to be run with exactly
those five seeds for any number reported as a five-seed mean. CPU-only runs of the same script
with the same seeds have been observed to reproduce reported numbers to 2–3 decimal places, with
small differences attributable to BLAS thread-count-dependent floating-point reduction order, not
to any actual change in the method. One exception: `plan_step44 --suite loo`'s Depot fold has
shown a run where four of five seeds converged to numerically-identical predictions (a real,
if unusual, property of that fold given Depot's narrow four-value target range, not a bug) —
if you see this, it is a known, previously-observed behavior, not a sign something is broken.
