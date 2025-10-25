# EchoKey — Correlation-Fueled Engine with Acoustic Cooling (Exact-Stroke)

**Repo contents**
- `ek_engine_acoustic.py` — exact-stroke simulator (no Lindblad), with acoustic cooling leg and optional diagnostic control
- `ek_engine_acoustic_experiment.tex` — experiment + hardware build guide
- `ek_engine_acoustic_math_notes.tex` — detailed math walkthrough and thermodynamic ledger

**License:** CC0-1.0 (public domain). Use freely.

---

## What this models

A small quantum thermodynamic cycle implemented on a superconducting platform:

- Working medium: two transmons `S_h, S_c`
- Hot reservoir: finite, `B_h = {B_h1, B_h2}`
- Cold environment: mechanical mode `M` (SAW/HBAR) + dump `D` (over-coupled cavity / phonon sink)
- Strokes (piecewise constant Hamiltonians; each solved **exactly** by a unitary):  
  1) **Pre-thermalize** (`H_th`) — light contacts build correlations/athermality  
  2) **Work** (`H_work`) — XY(XX+YY) on `S_h–S_c`; small residual contacts let **I(S:R)** drop (spend resource)  
  3) **Cooling** (`H_cool`) — red-sideband `S_c↔M` + `M↔D` bridge (export entropy)  
  4) **Relax** (`H_rlx`) — light contacts, then boundary quench back to `H_th`

**No master-equation approximations.** All energetics follow from exact quench-work identities at stroke boundaries.

---

## Install

```bash
python -V   # Python 3.10+ recommended
pip install numpy
````

No other deps required.

---

## Quick start

**Faithful correlated engine (default):**

```bash
python ek_engine_acoustic.py
```

**Tune cycles and truncations:**

```bash
python ek_engine_acoustic.py --cycles 3 --n_m 4 --n_d 4
```

**Optional diagnostic comparison (adds a decorrelating control + scoreboard):**

```bash
python ek_engine_acoustic.py --compare
```

> The *standard* control performs a **non-unitary reset** at the cycle boundary. That is not free; it is provided only as a diagnostic baseline.

**Charge a reset cost (fairer control):**

```bash
python ek_engine_acoustic.py --compare --Treset 0.5
```

This debits the control by a Landauer-style minimum erasure work
\( W_\text{reset} \ge T_\text{reset},(\sigma_\text{pre}-\sigma_\text{post}) \) (units with (k_B=1)).

---

## Interpreting the printout

Per cycle you’ll see:

* `Q_h`  — heat **from hot bath into device** (positive means into device)
* `Q_c`  — heat **to cold environment** (`Q_c = Q_M + Q_D`)
* `Q_M`  — mechanical leg (negative ⇒ **refrigeration** of `M`)
* `Q_D`  — dump leg (typically positive when exporting entropy)
* `W_out` — net mechanical work **out** (positive ⇒ engine-burst)
* `η` — efficiency reported **only** when in genuine engine mode
* `Δσ` — entropic resource change
  `Δσ < 0` ⇒ **resource consumed** (athermal cycle)
  `Δσ ≥ 0` ⇒ resource accumulated/charged
* `I(S:R)`, `D(Bh||th)`, `D(Bc||th)` — the resource ledger components
* `n_M` — mechanical occupancy (lab-friendly: sideband thermometry proxy)
* Energy checks (`residuals`) — should sit at ~1e-13 for the faithful path

**Modes (by signs):**

* *Engine-burst*: `W_out>0` **and** `Δσ<0`
* *Refrigerator*: `Q_M<0`, `W_out<0` (often `Δσ≥0`)
* *Heat-dump*: `Q_D≫0`, `W_out<0`, `Q_M≈0`

---

## Useful presets

**A) Engine-burst (consume resource in work stroke)**

```bash
python ek_engine_acoustic.py --method paper --cycles 3 \
  --g_on 0.34 --chi -0.22 \
  --g_bs 0.16 --g_md 0.22 \
  --t_th 2.2 --t_work 1.6 --t_cool 0.9 --t_rlx 0.6 \
  --n_m 4 --n_d 4
```

**B) Deep fridge (cool M hard; accept W_out<0)**

```bash
python ek_engine_acoustic.py --method paper --cycles 3 \
  --g_on 0.24 --chi -0.10 \
  --g_bs 0.28 --g_md 0.38 \
  --t_th 2.0 --t_work 1.0 --t_cool 1.6 --t_rlx 0.6 \
  --n_m 5 --n_d 5
```

---

## CLI summary

Frequencies (GHz, converted internally): `--fh --fc --fbh1 --fbh2 --fm --fd`
Temperatures (units with (k_B=1)): `--Th --Tc`
Contacts/couplings: `--kappa_h --kappa_c --g_on --chi --g_bs --g_md`
Durations: `--t_th --t_work --t_cool --t_rlx`
Truncations: `--n_m --n_d`
Control vs faithful: `--method {paper,standard}` (default `paper`), `--compare`, `--Treset`
Diagnostics: `--quiet_checks` (hide residuals)

Run `python ek_engine_acoustic.py -h` for full help.

---

## Physics guarantees (what this demo enforces)

* **Exact unitaries** on each stroke: no Markovian/Lindblad approximation.
* **Quench-work accounting**: all mechanical work is captured at boundaries.
* **Heat on baths**: measured by Hamiltonian expectation changes on `B_h` and `B_c`.
* **Resource ledger**: ( \sigma = I(S:R) + D(\rho_{B_h}|\rho^{th}*{B_h}) + D(\rho*{B_c}|\rho^{th}_{B_c}) ).
  Negative `Δσ` flags cycles that **spend correlations/athermality**.
* **Energy checks** printed per cycle for auditability.

---


## License

**CC0-1.0 (Public Domain).**
Do whatever you want — attribution appreciated but not required.

---

### Acknowledgment

This demo is inspired by and builds on:

Milton Aguilar & Eric Lutz, **“Correlated quantum machines beyond the standard second law.”** *Science Advances* **11** (41): eadw8462, October 10, 2025. DOI: 10.1126/sciadv.adw8462. See also the preprint: arXiv:2409.07899. ([Science][1])

Any mistakes or simplifications here are our own.

<details>
<summary>BibTeX</summary>

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

[1]: https://www.science.org/doi/10.1126/sciadv.adw8462?utm_source=chatgpt.com "Correlated quantum machines beyond the standard ..."

---
