# LEDGER.md — physics-consistent-feeder-generation

Authoritative log of every result, assumption, and deviation. One milestone per
section; each ended with an executed notebook (figures baked in) and the evidence
files under `results/`. Newest facts append; nothing is silently edited. Negative
results are kept deliberately — three of the eight headline findings are failures,
and each one redirected the design.

Research question: *how much physical feasibility does physics-informed training buy
a graph-conditioned generative model of distribution-feeder time series — and where
does physics actually belong in such a model?* Setting: SimBench MV feeders,
laptop-GPU scale, seeded and reproducible throughout.

---

## M0 — Scoping and environment

**Concept.** One sample = one feeder-day: 96 × 15-min steps × all buses ×
(Pd, Qd, Pg, Qg, Vm, Va). Generate the **full state** (not injections-only) so a
physics term has work to do; normalize per node to *nameplate* per-unit so every
scale is derivable from static conditioning features (transfer-safe by
construction). Physics = AC power-balance residual per generated timestep against
the conditioning topology's full Ybus. Headline experiment = ablation: identical
training ± physics term, ≥3 seeds, feasibility (MVA residual, V-limits) vs.
fidelity (validation axes of Cramer et al., IEEE Access 2022) trade-off.

**Data design.** SimBench MV, 2 train feeders + 1 held-out, full year of realistic
15-min profiles, per-step solved AC-PF (pandapower). Honest framing: SimBench's
measured-style profiles stand in for confidential operational data; benchmark data
is what makes a generator *gradeable* (full withheld truth exists). Storage folded
into net demand; topology assumed known at every timestep (topology uncertainty
named as future work, not attempted).

**Environment.** Windows 10, Python 3.11.9, torch 2.11.0+cu128 (RTX 3050 Laptop,
4 GB), torch_geometric 2.8.0, pandapower 3.5.4 + numba, simbench 1.6.2; versions
pinned in `requirements.txt`. AC-PF smoke test converged before anything else was
built.

---

## M1 — Temporal dataset ✅

**Goal:** deterministic full-year AC-PF dataset — the "real" data the generator must
imitate — on rural0 + semiurb0 (train) and comm0 (held out), with a per-step
physics gate.

**Delivered.** `gridfm/temporal_data.py` (+ `gridfm/simbench_data.py`,
`gridfm/encoding.py` reused from the GridFM thesis-recreation line), executed
`notebooks/01_temporal_dataset.ipynb`, figures + CSVs under `results/m1/`, dataset
under `data/m1/` (gitignored; regenerates exactly — profiles are the scenario
sequence, no sampling anywhere).

**Dataset:** 3 feeder-years × 35,136 15-min steps (SimBench study year, 366 days) =
**105,408 solved AC power flows, 0 failures**. Per step: full state per bus in
float64 + static nameplate (peak load P/Q, installed DER, storage flag, PQ/REF) +
edge list (g, b). Physics gate: max residual **8.0e-8 MVA** across all three years
(gate 1e-3, five orders of headroom). Runtime 11 min/feeder-year (numba).

**Findings (both found by looking, not assumed):**
1. **float32 precision floor.** The gate initially FAILED at 3.3e-3 MVA; diagnosis
   showed solver states exact to ~1e-10 and *casting the state to float32* alone
   raises the residual to 3.7–7.3e-3 MVA (per-feeder, recorded in each npz meta).
   Dataset stored float64. Consequence for M3: the physics loss runs on float32
   tensors, so generated-sample residuals must be read against a ~5e-3 MVA floor,
   not the solver's 1e-8.
2. **Reverse flow is wind-driven, not PV-driven.** MV-level DER is dominated by wind
   (rural0: 11.4 MW wind vs 0.3 MW MV-PV + 12.5 MW aggregated LV-RES). Export shows
   as day-long weather stripes and a pre-dawn band, not a clean summer-midday patch.
   Reverse-flow share of the year: **rural0 55.6%, semiurb0 33.1%, comm0 14.5%** —
   ordered exactly by DER/peak-load ratio (2.7×, 1.4×, 0.9×). The initial PV-centric
   reading of fig01 was wrong and is corrected in the notebook (honest log).
3. Voltages sit **1.00–1.06 pu** (substation setpoint holds the feeder high); the
   upper band edge pushes highest during export — the voltage-rise-under-reverse-flow
   signature of high-R/X feeders, i.e. the load–voltage coupling a physics-free
   generator can violate.
4. Statistical fingerprints for the M2 evaluation: 24/48 h autocorrelation peaks on
   a weather-decay envelope; heavy-tailed 15-min ramps (feeder-specific width);
   per-feeder voltage shapes.

**Assumptions:** base topology only; no storage on the three chosen feeders
(verified); sgen Q per SimBench defaults (no DER Q-control strategies).

---

## M2 — Generator v0: graph-conditioned normalizing flow ✅

**Goal:** end-to-end sampling of feeder-days + honest statistical grading. No
physics term yet (that is M3's controlled variable).

**Delivered.** `gridfm/dataset.py` (nameplate per-unit normalization; shared
time-PCA basis 96→K=16 per channel, 89–100% EV; BFS depth-parity coupling masks),
`gridfm/flow.py` (graph-conditioned RealNVP: GNN coupling conditioners with
series-admittance edges, static+RWPE node features, global mean context; exact
likelihood; ActNorm; tanh-clamped scales; coupling round-trip error 0.0),
`gridfm/train_flow.py`, `gridfm/metrics.py` (Cramer-2022 validation axes + network
extensions), executed `notebooks/02_generator_v0.ipynb`, `results/m2/`,
`checkpoints/v0_best.pt` + `timepca_k16.npz`.

**Training:** 1.64M params, 622 feeder-days (rural0+semiurb0 train split), 600
epochs in 9.3 min. Overfits from ~ep 155 (best val NLL/dim −1.952); early stopping
caught it. σ=0.01 dequantization noise (constant channels).

**Results (scorecard vs real validation days, 400 samples/feeder):**
- **The flow does not dominate — and the scorecard explains why that's the wrong
  bar.** Each baseline wins its engineered axis: *independent bootstrap* wins
  marginals & head-ACF because its channels ARE resampled real data — while being
  internally inconsistent (its head bears no relation to its own load/DER; fig07
  shows −10 MW export under moderate DER). *PCA-Gaussian* (the pre-deep-learning
  scenario method) wins cross-node correlation (RMSE 0.03 vs flow 0.21–0.24) by
  capturing second moments by construction, and loses tails (worst ramp W1 on both
  feeders).
- **Pre-registered plan-B trigger NOT tripped:** flow beats independent on
  cross-node correlation and Gaussian on ramps, on both feeders.
- **v0 deficiencies, on the record for M3:** head-ACF undershoot (persistence too
  weak); cross-node correlations compressed toward the diurnal floor ~0.5 (global
  mean-context too thin for feeder-wide weather); **Vm tail to ~1.10 pu vs real max
  1.062 — a physically impossible voltage that pure statistics permits**;
  overfitting → regularization queued with multi-seed judging.

**Meta-lesson (bridge to M3):** statistical fidelity metrics are gameable — a
bootstrap wins them while violating joint grid physics. The physics residual is the
metric a generator cannot game; that is the load-bearing argument for
physics-informed generation, demonstrated with numbers rather than asserted.

**Assumptions:** K=16 basis is the fidelity ceiling for all generators here (fig05b:
Pd/Qd keep 7–11% variance beyond it); Gaussian baseline = day-PCA(64) +
full-covariance; metrics on train feeders' held-out days only (transfer is M5).

---

## M3 — The physics ablation ✅

**Question:** how much physical feasibility does a physics-informed loss buy, at
what cost? **Answer: 12–17× on the residual — but partly by a pathological
mechanism, and a linear baseline exposes a better road.** Three findings, all
multi-seed.

**Design:** λ ∈ {0, 1e-3, 1e-2} × 3 seeds (42/43/44), identical protocol otherwise.
Physics penalty = mean squared full-Ybus power-balance residual (MVA²) of samples
drawn through the flow's inverse pass each step, differentiable end to end
(`gridfm/physics.py`; MATPOWER-form real arithmetic, line charging + taps
included). λ=1e-3 balances loss terms at init (measured); λ=1e-2 = dose point.
Evaluation: `gridfm/eval_m3.py` → `results/m3/m3_runs.csv` (9 runs × 2 feeders + 4
baselines).

**Findings:**
1. **The physics loss works on its own axis:** mean sample residual 10.0–18.7 MVA
   (λ=0) → 0.77–1.6 MVA (λ=1e-3), consistent across seeds and feeders; λ=1e-2
   saturates (0.46–2.2). Still ~150× above the representation floor (0.006 MVA).
2. **Mechanism partly = generator collapse (the milestone's discovery).** At λ=1e-3
   across-sample Vm variability falls to ~3% of real with a −2.2% systematic Vm
   shift; at λ=1e-2 one seed collapses in ALL channels (Pd diversity ratio 0.02 —
   essentially one repeated day with deceptively fine pooled marginals).
   Mean-squared sample residual is minimized by concentrating mass on the
   most-feasible point; NLL resists only where density is peaked. **Across-sample
   diversity (div_*_ratio) is now a first-class metric — pooled-marginal scorecards
   cannot see collapse.** Impossible-voltage share also WORSENED under the penalty
   (~1% vs ~0.02% at λ=0).
3. **The day-PCA Gaussian baseline sits AT the representation floor** (0.0055 MVA,
   healthy diversity): its samples live in the linear span of *solved* days, and at
   MV scale (Vm ±5%, small angles) the AC manifold is locally near-affine — the
   span of solutions is approximately solutions. The free flow leaves that manifold
   (17 MVA); the penalty drags it partway back at diversity cost; the linear model
   never leaves.

**Consequence — M4 architecture revision (physics-by-representation):** generate
with the model *inside a day-level subspace spanned by feasible days* (the PCA-flow
structure of Cramer et al., now understood as a physics device, not just
compression). The sample-penalty remains available as a fine-tuning term at low
weight; its collapse failure mode is documented here.

**Assumptions:** no λ sweep beyond two doses (saturation already visible); penalty
batch 8 days/step; train-feeder evaluation only. Training-history CSVs record TOTAL
loss in the train column for physics runs (~1e10 at ep 0 while 1000-MVA initial
residuals are tamed) — only val NLL is pure likelihood.

---

## M4 — Physics by representation + pseudo-measurement generation ✅

**v1 architecture** (`gridfm/dayspace.py`, `gridfm/flow_day.py`,
`gridfm/train_day.py`): feeder-day → time-PCA coeffs → whitened day-PCA (KD=64,
fitted per feeder on train days) → generative model in score space; decoding is
affine so samples stay in the span of solved days.

**Finding 1 — the flow TIES the Gaussian at this data size.** Val NLL/dim on
whitened scores: N(0,I) 1.433/1.455 (rural0/semiurb0); full-cov Gaussian identical;
trained flow 1.438–1.442 (3 seeds + a regularized variant; late-epoch val NLL
explodes = severe overfit, early stopping catches ~ep 20–60). One year of days per
feeder is statistically indistinguishable from Gaussian in this space → **v1's
generator is the Gaussian**; the flow's role (tails, multimodality) is deferred to
the many-feeder regime.

**Finding 2 — v1 delivers everything M3 promised, simultaneously, no tuning:**
resid **0.0055 MVA = the representation floor** (v0: 11–19; physics-penalty v0: ~1
at diversity cost), XCorr RMSE **0.025** (v0 0.23–0.25 — M2's weakest axis fixed
10×), W1_Vm 2e-4, ACF 3× better, **diversity 0.89–0.95, zero collapse**. Ramps
~0.11 = the subspace smoothing cost. v1_flow mildly worse than v1_gaussian
everywhere — NLL tie confirmed on metrics.

**Finding 3 — pseudo-measurement generation works and is calibrated** (rural0, 8
metered buses incl. substation, 87 unmetered, 55 held-out days, 200-sample
posteriors; conditioning is EXACT linear-Gaussian because observations are affine
in day-scores — the same algebra as weighted-least-squares state estimation, with
the day-space prior playing the role classical pseudo-measurements play in DSSE):
- Skill vs climatology (MAE, unmetered buses): Pg −85%, Va −91%, Vm −75%, Pd −49%,
  Qd −26% (Qg ≡ 0 in SimBench, excluded).
- **Calibration measured, then fixed:** raw subspace posterior overconfident
  (45–72% coverage at 90% target) because it cannot see representation error;
  adding the measured truncation std as predictive noise → 85–92%. Residual Pd/Qd
  under-coverage (85%) = heavy-tailed household noise vs Gaussian bands, noted.
- Posterior samples inherit floor physics (0.004–0.005 MVA) → usable as DSSE
  pseudo-measurements, not just plots.
- Flow Langevin refinement: ±0.5% MAE delta = zero within noise (consequence of
  Finding 1, reported).

**Honest costs of v1, named:** per-feeder day basis (zero-shot topology transfer
given up — how much target data the basis needs = M5's transfer question); Gaussian
tails; subspace smoothing of ramps; bands Gaussian-shaped.

---

## M5 — Generalization exams ✅

**Exam 1 — transfer to comm0 (held out from all training).** Shared time-PCA basis
held on the never-seen feeder (recon at floor). v1 with n target days (n = 7…311,
random subsets, kd = min(64, n−1)):
- **Physics at the floor from 7 days** (0.0083 MVA; span property — feasibility
  comes from the span, not sample count). v0 graph-flow zero-shot reference:
  transfers instantly to the unseen graph but at **84.8 MVA** — four orders of
  magnitude.
- Statistics mature at ~30–60 days (diversity 0.79→0.95 by 30 d; XCorr plateau
  0.03–0.05). Practical sentence: 1–2 months of measurements buy a faithful,
  physics-exact, calibrated scenario generator. Caveat: random-day subsets, not
  contiguous months.

**Exam 2 — summer→winter.** rural0 summer = May–Aug (123 d), exam = winter coverage
(Dec–Feb daily-mean load; winter 3.98 MW vs summer 3.26 MW):
- v1-summer reproduces summer (W1 0.06 MW) and cannot produce winter (W1 0.72;
  reference: even the full-year model's mixture sits 0.45 from winter-only).
- **The pretrain-broad fix FAILED (negative result, kept):** graph-flow B (rural0
  summer + semiurb0 full year, shared weights) is WORSE toward winter (0.97) than
  summer-only A (0.77) — it drifts toward semiurb0-flavored days. Mechanism:
  feeder-conditioned generation learns the pairing "this graph → the days seen for
  it"; nothing routes another feeder's seasons across the pairing. **Implicit
  seasonal transfer through shared weights does not happen. Design rule:
  seasonality/weather must enter as explicit exogenous covariates** (calendar,
  temperature, irradiance). Scope: A/B at fixed 200-epoch budget, no early stop;
  qualitative failure robust (fig17), exact W1s carry the caveat.

**Wrap:** the eight evidenced claims are tabulated in the README and notebook 05.

**Future work:** heavy tails and ramp sharpness (deep generative models in the
many-feeder regime, where finding 6 says they start to pay); amortized
graph-conditioned bases (bridging zero-shot transfer with manifold physics);
exogenous-covariate conditioning (weather, calendar); topology uncertainty;
policy-aware modeling of storage and controllable assets; formal privacy audits
(membership inference / DP) for synthetic-data release; scaling beyond
laptop-class hardware.
