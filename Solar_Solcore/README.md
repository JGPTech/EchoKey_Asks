# EchoKey Asks — LLM-Assisted Solar Meta-Layer (Solcore)

**Question:** *Can LLM-assisted research increase device efficiency vs. a baseline in a Solcore sandbox?*
**Short answer (for this run):** Yes — with an LLM-guided “coin → meta-layer” preset, the GaAs single-junction showed **+0.93 percentage-points** absolute efficiency vs. baseline (≈ **+3.36%** relative), driven primarily by **$V_{oc}$** and **FF** gains from passivation/lifetime improvements while keeping optics benign.

---

If you think I'm cool, donate please:  
[![Sponsor](https://img.shields.io/badge/Sponsor-Jon%20Poplett-purple?style=for-the-badge&logo=github)](https://github.com/sponsors/jgptech)  

## What this repo contains

* `ek_solar_meta_pipeline.py` — a **derivation-first, reproducible** pipeline using **Solcore**:

  * Builds a GaAs PDD junction with a thin AlGaAs window and Au back reflector.
  * Optionally inserts a **front meta-layer** defined by a coin/config (tabulated $n,k$ + thickness, plus effective transport targets).
  * Runs **TMM optics** → **PDD QE** → **illuminated IV**, and logs metrics + curves for **baseline** vs **emergent** (with meta).
* `coin_config.json` (example) — the **winning regime** used in this entry:
  `gamma_k = 0` (no added parasitic absorption), **t ≈ 95 nm** (quarter-wave-ish), improved **S_front/S_back** and **τ**.

---

## Why this answers the question

We constrained the LLM to **EchoKey v2: “asks-only”** style—no claims, just definitions, heuristics, and checks. The model proposed a **safe, physics-sane** control policy:

* **Optics neutral**: set $\gamma_k=0$ (no extra loss). Choose thickness near a **quarter-wave valley** to avoid reflection spikes.
* **Transport/passivation**: push **surface recombination velocities down** and **lifetimes up** inside PDD—lowering $J_0$, thus **$V_{oc}\uparrow$** and **FF\uparrow**.

This is exactly what the pipeline tests against an apples-to-apples **baseline**.

---

## Results (from the included run)

| Model    | $J_{sc}$ (mA/cm²) | $V_{oc}$ (V) | FF     | $\eta$ (%) |
| -------- | ----------------- | ------------ | ------ | ---------- |
| Baseline | 23.3209           | 1.0018       | 0.8776 | 27.7064    |
| Emergent | 22.9300           | 1.0448       | 0.8845 | 28.6376    |

* **Δη = +0.9311 percentage-points** (≈ **+3.36%** vs baseline).
* $J_{sc}$ dipped slightly (interference wiggle), but **$V_{oc}$** gained **~43 mV** and FF rose modestly, netting higher efficiency—**matching the analysis**.

---

## How it works (1-page)

1. **Coin → Meta mapping**
   `coin_config.json` defines spectral taps for $k(\lambda)$, thickness $t$, and effective transport/passivation targets $(S_{\rm f}, S_{\rm b}, \tau_{n,p}, \mu_{n,p})$.
   In this entry: **$\gamma_k=0$** (no taps), **$t \approx 95\text{ nm}$**.

2. **Optical registration**
   The script tabulates $n(\lambda),k(\lambda)$ for non-PDD layers (AlGaAs window, Au back) and the optional meta layer, registers them with Solcore, and builds the stack.

3. **QE + IV**

   * **TMM optics** on the given wavelength grid (AM1.5g photon flux per nm).
   * **PDD** for the GaAs junction; passivation/lifetime targets adjust $J_0$ (via $\tau_{\rm eff}^{-1}=\tau_{\rm bulk}^{-1}+2S/W$).
   * Solve illuminated IV; compute $V_{oc}, J_{sc}, \mathrm{FF}, \eta$.

4. **Compare**
   Run **baseline** (no meta) vs **emergent** (meta enabled) on the *same* grid & options. Save CSVs for metrics and JV curves.

---

## Quick start

> Python 3.10+ recommended. Solcore must be installed and able to register tabulated materials.

```bash
# 1) Install requirements
pip install numpy pandas solcore

# 2) Run with your coin (this repo’s example is shown below)
python ek_solar_meta_pipeline.py --use_solcore --coin_config coin_config.json

# 3) Outputs
# ek_runs/<run_id>/baseline_summary.csv
# ek_runs/<run_id>/emergent_summary.csv
# ek_runs/<run_id>/comparison_summary.csv
# ek_runs/<run_id>/baseline_jv.csv
# ek_runs/<run_id>/emergent_jv.csv
# ek_coin_cache/meta_optics_<hash>.json, meta_transport_<hash>.json
```

**Alternate:** supply pre-generated meta files

```bash
python ek_solar_meta_pipeline.py --use_solcore \
  --meta_optics ek_coin_cache/meta_optics_<hash>.json \
  --meta_transport ek_coin_cache/meta_transport_<hash>.json
```

---

## The “winning regime” (this entry)

```json
{
  "centers_nm": [875.0],
  "sigmas_nm":  [30.0],
  "weights":    [1.0],
  "gamma_k":    0.0,
  "thickness_m": 9.5e-8,
  "S_front_eff": 80.0,
  "S_back_eff":  220.0,
  "mu_n_eff":    30.0,
  "mu_p_eff":    10.0,
  "tau_n_eff":   5.0e-8,
  "tau_p_eff":   5.0e-8,
  "mu_n_mult":   1.25,
  "mu_p_mult":   1.25
}
```

* **Optics:** benign (no added $k$); **thickness** near quarter-wave valley.
* **Transport/passivation:** lower $S$ and higher $\tau$ → **$J_0↓$** → **$V_{oc}↑$** and **FF↑**.

---

## Outputs

* **Summaries:** baseline/emergent metrics (CSV).
* **JV curves:** voltage, current density, power (CSV).
* **Run metadata:** `run_meta.json` (repro trace).
* **Cached meta:** deterministic `meta_optics_*.json`, `meta_transport_*.json` keyed by the coin hash.

---

## Reproducibility notes

* **Deterministic** given the same coin/config and Solcore build.
* The script flips JV sign to enforce **$J(0)>0$** and computes $P_{\rm in}$ from the same AM1.5g source.
* If your Solcore version arranges `sc.iv["IV"]` differently, adapt the 2×N extraction lines accordingly.

---

## Troubleshooting

* **“Solcore … required”** → Install Solcore (`pip install solcore`) and ensure its data files are accessible.
* **Non-monotone wavelength grid** → Provide strictly increasing `wl`.
* **Weird JV signs** → The script normalizes to $J(0)>0$; check your environment if you still see negatives.

---

## License & attribution

* **Code & docs:** CC0 Public Domain.
* **Credit:** EchoKey Team — “asks-only” methodology with operator mapping ($\Cyc,\Rec,\Frac,\Reg,\Syn,\Ref,\Out$) guiding the preset and checks.

---

## TL;DR

Constrained, question-first LLM assistance selected a **safe optical preset** and **aggressive-but-plausible passivation/lifetime targets**, yielding a **repeatable efficiency lift** in a standard Solcore sandbox.

