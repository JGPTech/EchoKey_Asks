# EchoKey Asks — Correlated Quantum Engine (Exact-Stroke Demo)

A tiny, **exact** (per-stroke) microscopic engine that demonstrates how **correlations/athermality** can act as a fuel. It mirrors the bookkeeping from recent correlated-thermo frameworks while staying fully auditable and dependency-light.

* **Exact strokes**: each stroke has a constant Hamiltonian; evolution uses a **single exact exponential** (no time-stepping / Trotter).
* **Work** via **quench identities** at stroke boundaries (no ∂H/∂t integration).
* **Entropic resource**: per-cycle change
  (\Delta\sigma = I(S!:!R) + \sum_j D(\rho_{B_j}\Vert\rho^{\text{th}}*{B_j})\big|*{\text{end}} ;-; (\cdot)\big|_{\text{start}}).
  **Δσ < 0** ⇒ correlations/athermality were **spent** (athermal cycle).

---

## Files

* `ek_correlated_engine.py` — the runnable engine (NumPy only).
* `ek_correlated_engine.pdf` — math notes & bookkeeping (CC0), matching the code line-for-line.

> Both files share the same base name so they stay paired.

---

## Requirements

* Python **3.10–3.12**
* `numpy` (no other deps)

```bash
python -V
pip install --upgrade numpy
```

---

## Quick start

Run both paths (correlated “paper” method and a decorrelated “standard” control), 3 cycles, then show a scoreboard:

```bash
python ek_correlated_engine.py
```

Correlated path only:

```bash
python ek_correlated_engine.py --method paper
```

Control baseline only:

```bash
python ek_correlated_engine.py --method standard
```

A set of gentle parameters that often yields at least one **ATHERMAL** cycle (Δσ < 0) in the paper path:

```bash
python ek_correlated_engine.py --method paper --cycles 3 \
  --Th 2.2 --Tc 0.5 \
  --kappa_h 0.30 --kappa_c 0.30 --t_th 2.6 --t_rlx 0.5 \
  --g_on 0.24 --chi -0.12 --t_work 1.2
```

---

## CLI parameters (most relevant)

* `--method {paper,standard,both}`: keep correlations vs. wipe to product at cycle end (control). Default: `both`.
* `--cycles INT`: number of cycles (default: `3`).
* Bath/system physics:

  * `--Th`, `--Tc`: hot/cold temperatures (default: `2.0`, `0.5`).
  * `--omega_h`, `--omega_c`: system qubit splittings (default: `1.0`, `0.6`).
  * `--kappa_h`, `--kappa_c`: system–bath couplings during pre-thermalize; half of each in relax.
* Work stroke:

  * `--g_on`: system–system XY coupling during work.
  * `--chi`: small **joint S–bath** term during work (lets you **spend** correlations). Set `0` to disable.
* Stroke durations:

  * `--t_th`, `--t_work`, `--t_rlx`: times for pre-thermalize, work, relax.
* `--quiet_checks`: hide energy-balance residuals (by default we print them to show exactness).

---

## What the output means

Per cycle you’ll see:

* `Q_h` — **device heat from hot bath**; **positive** means hot **lost** energy to device.
* `Q_c` — **heat to cold**; positive means cold **gained** energy.
* `W_out` — **work extracted** (positive = useful work out).
* `η` — efficiency proxy (W_{\text{out}}/\max(Q_h,0)) (only meaningful if (Q_h>0)).
* `Δσ` — **entropic resource change**; **Δσ < 0** ⇒ an **ATHERMAL** cycle where correlations/athermality were consumed.
* `I(S:R)`, `D(Bh||th)`, `D(Bc||th)` — the pieces of σ at cycle end.
* **Checks**: two energy-balance residuals that should be near 0 in the `paper` path:

  * `total energy residual` (\approx 0): ((E_f - E_0) + W_{\text{out}}).
  * `split balance residual` (\approx 0): (W_{\text{out}} - (Q_h - Q_c) + (\Delta E_S + \Delta E_{\text{int}})).
    Non-zero in `standard` is expected (we apply a nonunitary reset there).

A scoreboard compares **total work**, **avg η**, and counts of **Δσ<0** cycles and declares a **winner**.

---

## Typical questions

**Why is η often 0?**
η is only defined when (Q_h>0). Many cycles in a small-bath model act more like a refrigerator/heat-pump or a correlation reservoir. That’s okay — this demo is about **Δσ** and **work** with tight bookkeeping.

**How do I make Δσ < 0 show up?**
Use a small negative `--chi` (joint S–bath handle in the work stroke) so mutual information can **decrease** while energy flows through the work channel. Without it (`chi=0`), local S-only unitaries can’t reduce (I(S!:!R)).

**Why are the residuals ~1e-15 in `paper` but ~1e-2 in `standard`?**
`paper` is closed, unitary, and exact per stroke.
`standard` forcibly **decorrelates** (nonunitary) at the cycle boundary — it’s a **control**, not physical dynamics.

---

## Reproduce the “athermal burst” (Δσ < 0 with W_out ≥ 0)

```bash
python ek_correlated_engine.py --method paper --cycles 3 \
  --Th 2.2 --Tc 0.5 \
  --kappa_h 0.30 --kappa_c 0.30 --t_th 2.6 --t_rlx 0.5 \
  --g_on 0.24 --chi -0.12 --t_work 1.2
```

Then compare with the decorrelated control:

```bash
python ek_correlated_engine.py --method standard --cycles 3 \
  --Th 2.2 --Tc 0.5 \
  --kappa_h 0.30 --kappa_c 0.30 --t_th 2.6 --t_rlx 0.5 \
  --g_on 0.24 --chi -0.12 --t_work 1.2
```

You should see at least one `ATHERMAL` cycle in the `paper` path, and none in `standard`.

---

## Extending

* Increase bath sizes (3–5 qubits/side) for more realistic relaxation.
* Add **diagonal** relative entropy (D(\rho^{\text{diag}}\Vert\rho^{\text{th}})) to isolate coherence vs populations.
* Try STA-like work Hamiltonians to suppress friction and prolong athermal operation.
* Seed non-Gibbs baths (e.g., squeezed); adapt the reference states accordingly.

---

## License

**CC0-1.0 (Public Domain).**
Do whatever you want — attribution appreciated but not required.

---

Absolutely—here’s a drop-in replacement for the **Acknowledgment** section in your `README.md`, with a precise citation (plus an optional BibTeX block).

---

### Acknowledgment

This demo is inspired by and builds on:

Milton Aguilar & Eric Lutz, **“Correlated quantum machines beyond the standard second law.”** *Science Advances* **11** (41): eadw8462, October 10, 2025. DOI: 10.1126/sciadv.adw8462. See also the preprint: arXiv:2409.07899. ([Science][1])

Any mistakes or simplifications here are our own.

<details>
<summary>BibTeX (optional)</summary>

```bibtex
@article{AguilarLutz2025SciAdv,
  author  = {Milton Aguilar and Eric Lutz},
  title   = {Correlated quantum machines beyond the standard second law},
  journal = {Science Advances},
  volume  = {11},
  number  = {41},
  pages   = {eadw8462},
  year    = {2025},
  month   = {oct},
  doi     = {10.1126/sciadv.adw8462}
}

@article{AguilarLutz2024arXiv,
  author  = {Milton Aguilar and Eric Lutz},
  title   = {Correlated quantum machines beyond the standard second law},
  journal = {arXiv},
  eprint  = {2409.07899},
  year    = {2024},
  month   = {sep},
  url     = {https://arxiv.org/abs/2409.07899}
}
```

</details>
::contentReference[oaicite:1]{index=1}

[1]: https://www.science.org/doi/10.1126/sciadv.adw8462?utm_source=chatgpt.com "Correlated quantum machines beyond the standard ..."

---
