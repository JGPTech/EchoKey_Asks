#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EchoKey Asks — Can I back up my statements?
Correlated Quantum Engine — EXACT stroke-based demo (no dynamics approximations)
CC0-1.0 — Public Domain

WHAT THIS IS
------------
A minimal, *exact* (per stroke) microscopic engine you can run and inspect.
It mirrors the framework in “correlated quantum machines beyond the standard second law”:
- Closed total system (working medium + finite baths) → global evolution is **unitary**.
- Piecewise-constant (square) strokes → each stroke uses a **single exact** propagator U = exp(-i H Δt).
- **Work** is accounted **exactly** via Hamiltonian **quenches** at switching instants:
    W_quench = Tr[ρ (H_after − H_before)]    (no ∂H/∂t integration, no time stepping).
- No Born/Markov/secular/weak-coupling master equations. The only “approximation”
  is finite-precision linear algebra (floating point), which we display sanity checks for.

Two paths:
  • method="paper"    : keep system–bath correlations across cycles (the physics of interest).
  • method="standard" : **control baseline only** — at the end of each cycle, forcibly
                        replace the total state by a product of the cycle-end marginals
                        (system | hot bath | cold bath). This *is not* physical time evolution;
                        it's a surgical reset to demonstrate how wiping correlations removes the
                        “entropic fuel” advantage.

WHAT’S MEASURED (per cycle)
---------------------------
- Q_h  : “device heat from the hot bath.” Sign: **Q_h > 0** means hot bath lost energy (into device).
- Q_c  : “heat to the cold bath.” Sign: **Q_c > 0** means cold bath gained energy.
- W_out: work extracted (positive = useful work out). Exactly −(sum of quench works).
- η    : efficiency proxy = W_out / max(Q_h, 0). (Meaningful only when Q_h>0.)
- Δσ   : change in “entropic content” σ between cycle **end and start**:
           σ = I(S:R) + Σ_j D(ρ_Bj || ρ_Bj^th),
         where I(S:R) is mutual information between System S and all Baths R,
         D is quantum relative entropy, and ρ_Bj^th is a (fixed) Gibbs reference at T_h/T_c.
         **Δσ < 0** means we *consumed correlations/athermal structure* — an “entropic fuel” event.
- Regime tag:
    ATHERMAL if Δσ < 0  (spent correlations/coherence to get work),
    THERMAL  otherwise.

ENERGY-BOOKKEEPING CHECKS (printed per cycle)
---------------------------------------------
We also print two residuals that should be ~ 0 (floating-point tiny):
  1) Total-energy + work consistency:
       res_total = (E_f − E_0) + W_out  (evaluated at the *same* Hamiltonian H_th)
  2) First-law-style split with interactions:
       res_split = W_out − (Q_h − Q_c) + (ΔE_S + ΔE_int),
     where ΔE_S is the system bare-energy change and ΔE_int is the *interaction energy* change
     computed at H_th (start/end Hamiltonian). Both residuals ≈ 0 confirms tight bookkeeping.

USAGE
-----
$ python ek_correlated_engine.py                      # default: run BOTH methods, 3 cycles, scoreboard
$ python ek_correlated_engine.py --method paper       # just the correlated path
$ python ek_correlated_engine.py --cycles 5           # more cycles
$ python ek_correlated_engine.py --g_on 0.30 --chi -0.20 --t_work 2.0  # tune strokes

DEFAULTS are chosen to often produce at least one ATHERMAL (Δσ < 0) cycle in the “paper” path.

DEPENDENCIES
------------
numpy only (tested on Python 3.10–3.12).

"""

import argparse
import numpy as np

# =========================
# Linear algebra utilities
# =========================

def dagger(M: np.ndarray) -> np.ndarray:
    """Conjugate transpose."""
    return M.conj().T

def hermitize(M: np.ndarray) -> np.ndarray:
    """(M + M†)/2 to clean tiny numerical asymmetries."""
    return (M + dagger(M)) / 2

def expm_hermitian(H: np.ndarray, dt: float) -> np.ndarray:
    """
    Exact unitary propagator for a constant Hamiltonian H over duration dt.
    Uses spectral decomposition; H is Hermitian.
    """
    evals, evecs = np.linalg.eigh(hermitize(H))
    U = evecs @ np.diag(np.exp(-1j * evals * dt)) @ dagger(evecs)
    return U

def kron_all(mats) -> np.ndarray:
    """Kronecker product over a list of matrices, left-to-right."""
    out = np.array([[1.0 + 0j]])
    for A in mats:
        out = np.kron(out, A)
    return out

# Pauli + identity
SIGMA_X = np.array([[0, 1],[1, 0]], dtype=complex)
SIGMA_Y = np.array([[0, -1j],[1j, 0]], dtype=complex)
SIGMA_Z = np.array([[1, 0],[0, -1]], dtype=complex)
IDENT_2 = np.eye(2, dtype=complex)

def embed(op: np.ndarray, which: int, N: int) -> np.ndarray:
    """Embed single-qubit operator `op` acting on site `which` into N-qubit space."""
    ops = [IDENT_2] * N
    ops[which] = op
    return kron_all(ops)

def partial_trace(rho: np.ndarray, keep, dims) -> np.ndarray:
    """
    Trace out all subsystems NOT in `keep` (0-based indices).
    dims: list of local dimensions (here all 2).
    Returns the reduced density matrix on the kept subsystems, ordered as in `keep`.
    """
    keep = sorted(keep)
    N = len(dims)
    D = int(np.prod(dims))
    assert rho.shape == (D, D)

    rho_t = rho.reshape(dims + dims)  # (i1..iN ; j1..jN)
    trace_out = [i for i in range(N) if i not in keep]

    # Permute so kept come first for both bra and ket indices
    perm = keep + trace_out
    rho_t = np.transpose(rho_t, axes=perm + [i + N for i in perm])

    dim_keep = int(np.prod([dims[i] for i in keep])) if keep else 1
    dim_tr   = int(np.prod([dims[i] for i in trace_out])) if trace_out else 1

    rho_t = rho_t.reshape(dim_keep, dim_tr, dim_keep, dim_tr)

    if dim_tr == 1:
        return rho_t[:, 0, :, 0]  # nothing to trace
    # Contract with identity on traced indices (δ_ij)
    return np.einsum("aibj,ij->ab", rho_t, np.eye(dim_tr))

def vn_entropy(rho: np.ndarray, tol: float = 1e-12) -> float:
    """
    Von Neumann entropy S(ρ) = −Tr[ρ log ρ] with natural log.
    Interprets a single well-prepared quantum state informationally — no ensemble needed.
    """
    evals = np.linalg.eigvalsh(hermitize(rho))
    evals = np.clip(np.real(evals), 0.0, 1.0)
    nz = evals[evals > tol]
    return float(-np.sum(nz * np.log(nz))) if nz.size else 0.0

def rel_entropy(rho: np.ndarray, sigma: np.ndarray, tol: float = 1e-12) -> float:
    """
    Quantum relative entropy D(ρ || σ) = Tr[ρ (log ρ − log σ)].
    Implements gentle support handling by minimally regularizing σ if needed.
    """
    sig = hermitize(sigma)
    ev = np.linalg.eigvalsh(sig)
    if np.min(ev) < tol:
        sig = sig + tol * np.eye(sig.shape[0], dtype=complex)

    def logm_psd(X):
        e, V = np.linalg.eigh(hermitize(X))
        e = np.clip(np.real(e), tol, None)
        return V @ np.diag(np.log(e)) @ dagger(V)

    lr = logm_psd(rho)
    ls = logm_psd(sig)
    return float(np.real(np.trace(hermitize(rho) @ (lr - ls))))

# =========================
# Exact stroke engine
# =========================

class ExactStrokeEngine:
    """
    Qubit order (N=6 total):
      0: S_h  (hot system qubit)
      1: S_c  (cold system qubit)
      2,3: hot bath qubits (Bh)
      4,5: cold bath qubits (Bc)

    Bare Hamiltonian pieces:
      H_S   = ½ ω_h Z_0  +  ½ ω_c Z_1
      H_Bh  = Σ_k ½ ω_bh[k] Z_{2,3}
      H_Bc  = Σ_k ½ ω_bc[k] Z_{4,5}
      H_B   = H_Bh + H_Bc
      H_int terms:
          H_g      = ½ (X0 X1 + Y0 Y1)        (system–system work channel)
          H_SB_th  = Σ_i X0 X_i (i in Bh)     (S_h–Bh contact)
          H_SB_tc  = Σ_i X1 X_i (i in Bc)     (S_c–Bc contact)

    Stroke Hamiltonians (constant on each stroke; exact propagators used):
      H_th   = H_S + H_B + κ_h H_SB_th + κ_c H_SB_tc     (pre-thermalization / build correlations)
      H_work = H_S + H_B + g_on H_g + χ (H_SB_th + H_SB_tc)
               (the χ term is a *small* joint S–B “power takeoff”: allows spending correlations)
      H_rlx  = H_S + H_B + ½ κ_h H_SB_th + ½ κ_c H_SB_tc (relaxation contact)

    The cycle ends with a *quench back to H_th* so start/end use the same Hamiltonian,
    which makes total-energy/work accounting exact and simple: E_f − E_0 = W_on.
    """

    def __init__(self,
                 omega_h=1.0, omega_c=0.6,
                 Th=2.0, Tc=0.5,
                 kappa_h=0.25, kappa_c=0.25,
                 g_on=0.30, chi=-0.20,
                 t_th=2.0, t_work=2.0, t_rlx=1.0,
                 method='paper',
                 print_checks=True):
        self.N = 6
        self.dims = [2] * self.N

        # indices
        self.i_Sh, self.i_Sc = 0, 1
        self.i_Bh, self.i_Bc = [2, 3], [4, 5]

        # parameters
        self.omega_h, self.omega_c = omega_h, omega_c
        self.Th, self.Tc = Th, Tc
        self.kappa_h, self.kappa_c = kappa_h, kappa_c
        self.g_on, self.chi = g_on, chi
        self.t_th, self.t_work, self.t_rlx = t_th, t_work, t_rlx
        self.method = method
        self.print_checks = print_checks

        # bath splittings (slight detunings avoid accidental resonances locking behavior)
        self.omega_bh = [omega_h * 1.00, omega_h * 1.10]
        self.omega_bc = [omega_c * 1.00, omega_c * 0.90]

        # build Hamiltonians
        self.H_S  = 0.5 * omega_h * embed(SIGMA_Z, self.i_Sh, self.N) \
                  + 0.5 * omega_c * embed(SIGMA_Z, self.i_Sc, self.N)

        self.H_Bh = sum(0.5 * w * embed(SIGMA_Z, i, self.N) for w, i in zip(self.omega_bh, self.i_Bh))
        self.H_Bc = sum(0.5 * w * embed(SIGMA_Z, i, self.N) for w, i in zip(self.omega_bc, self.i_Bc))
        self.H_B  = self.H_Bh + self.H_Bc

        self.H_g = 0.5 * (embed(SIGMA_X, self.i_Sh, self.N) @ embed(SIGMA_X, self.i_Sc, self.N)
                        +  embed(SIGMA_Y, self.i_Sh, self.N) @ embed(SIGMA_Y, self.i_Sc, self.N))
        self.H_SB_th = sum(embed(SIGMA_X, self.i_Sh, self.N) @ embed(SIGMA_X, i, self.N) for i in self.i_Bh)
        self.H_SB_tc = sum(embed(SIGMA_X, self.i_Sc, self.N) @ embed(SIGMA_X, i, self.N) for i in self.i_Bc)

        self.H_th   = self.H_S + self.H_B + self.kappa_h * self.H_SB_th + self.kappa_c * self.H_SB_tc
        self.H_work = self.H_S + self.H_B + self.g_on * self.H_g + self.chi * (self.H_SB_th + self.H_SB_tc)
        self.H_rlx  = self.H_S + self.H_B + 0.5 * self.kappa_h * self.H_SB_th + 0.5 * self.kappa_c * self.H_SB_tc

        # thermal references for baths (Gibbs at Th/Tc)
        self.rho_th_Bh = self._thermal_state(self.Bh_H(), beta=1.0 / Th)
        self.rho_th_Bc = self._thermal_state(self.Bc_H(), beta=1.0 / Tc)

        # initial *product* thermal state (no correlations)
        self.rho0 = kron_all([
            self._thermal_state(0.5 * omega_h * SIGMA_Z, beta=1.0 / Th),
            self._thermal_state(0.5 * omega_c * SIGMA_Z, beta=1.0 / Tc),
            self._thermal_state(self.Bh_H(), beta=1.0 / Th),
            self._thermal_state(self.Bc_H(), beta=1.0 / Tc),
        ])

    # --- sub-H’s and marginals
    def Bh_H(self): return partial_trace(self.H_Bh, self.i_Bh, self.dims)
    def Bc_H(self): return partial_trace(self.H_Bc, self.i_Bc, self.dims)
    def S_H (self): return partial_trace(self.H_S , [self.i_Sh, self.i_Sc], self.dims)

    def rho_S (self, rho): return partial_trace(rho, [self.i_Sh, self.i_Sc], self.dims)
    def rho_Bh(self, rho): return partial_trace(rho, self.i_Bh, self.dims)
    def rho_Bc(self, rho): return partial_trace(rho, self.i_Bc, self.dims)

    # --- thermal helper
    def _thermal_state(self, H_sub, beta):
        e, V = np.linalg.eigh(hermitize(H_sub))
        Z = np.sum(np.exp(-beta * e))
        return V @ np.diag(np.exp(-beta * e) / Z) @ dagger(V)

    # --- one exact cycle
    def run_cycle(self, rho_in):
        # propagators for constant H on each stroke
        U_th   = expm_hermitian(self.H_th,   self.t_th)
        U_work = expm_hermitian(self.H_work, self.t_work)
        U_rlx  = expm_hermitian(self.H_rlx,  self.t_rlx)

        # ----- start-of-cycle entropic content σ_i
        rho_S_i, rho_Bh_i, rho_Bc_i = self.rho_S(rho_in), self.rho_Bh(rho_in), self.rho_Bc(rho_in)
        rho_R_i = kron_all([rho_Bh_i, rho_Bc_i])
        I_i     = vn_entropy(rho_S_i) + vn_entropy(rho_R_i) - vn_entropy(rho_in)
        D_bh_i  = rel_entropy(rho_Bh_i, self.rho_th_Bh)
        D_bc_i  = rel_entropy(rho_Bc_i, self.rho_th_Bc)
        sigma_i = I_i + D_bh_i + D_bc_i

        # ----- Stroke 1: pre-thermalize/build correlations (H_th, duration t_th)
        rho1 = U_th @ rho_in @ dagger(U_th)
        # Quench: H_th -> H_work (no state change; energy jump is work-on)
        Wq1  = float(np.real(np.trace(rho1 @ (self.H_work - self.H_th))))

        # ----- Stroke 2: work stroke (H_work, duration t_work)
        rho2 = U_work @ rho1 @ dagger(U_work)
        # Quench: H_work -> H_rlx
        Wq2  = float(np.real(np.trace(rho2 @ (self.H_rlx - self.H_work))))

        # ----- Stroke 3: relaxation contact (H_rlx, duration t_rlx)
        rho3 = U_rlx @ rho2 @ dagger(U_rlx)
        # Quench: H_rlx -> H_th (close the Hamiltonian loop)
        Wq3  = float(np.real(np.trace(rho3 @ (self.H_th - self.H_rlx))))
        rho_out = rho3  # state after final evolution; post-quench H is H_th

        # Optional decorrelation baseline (control – not physical)
        if self.method == "standard":
            rho_out = kron_all([self.rho_S(rho_out), self.rho_Bh(rho_out), self.rho_Bc(rho_out)])

        # Energetics w.r.t. *bare bath pieces* (signs explained in header)
        def E(op, rho): return float(np.real(np.trace(rho @ op)))
        E_Bh_i, E_Bh_f = E(self.H_Bh, rho_in),  E(self.H_Bh, rho_out)
        E_Bc_i, E_Bc_f = E(self.H_Bc, rho_in),  E(self.H_Bc, rho_out)

        Q_h   = -(E_Bh_f - E_Bh_i)  # >0 means energy from hot bath into device
        Q_c   = +(E_Bc_f - E_Bc_i)  # >0 means energy to cold bath
        Q_in  = max(0.0, Q_h)

        # Work accounting (exact — sum of quench works)
        W_on  = Wq1 + Wq2 + Wq3
        W_out = -W_on

        # Carnot-style comparator (only meaningful if using hot/cold as thermal sources)
        eta_C = 1.0 - (self.Tc / self.Th)
        eta   = (W_out / Q_in) if Q_in > 0 else 0.0

        # ----- end-of-cycle entropic content σ_f
        rho_S_f, rho_Bh_f, rho_Bc_f = self.rho_S(rho_out), self.rho_Bh(rho_out), self.rho_Bc(rho_out)
        rho_R_f = kron_all([rho_Bh_f, rho_Bc_f])
        I_f     = vn_entropy(rho_S_f) + vn_entropy(rho_R_f) - vn_entropy(rho_out)
        D_bh_f  = rel_entropy(rho_Bh_f, self.rho_th_Bh)
        D_bc_f  = rel_entropy(rho_Bc_f, self.rho_th_Bc)
        sigma_f = I_f + D_bh_f + D_bc_f

        Delta_sigma = sigma_f - sigma_i
        regime = "ATHERMAL" if Delta_sigma < 0 else "THERMAL"

        # ----- bookkeeping checks (tight energy accounting)
        # Evaluate total energy at *same* Hamiltonian H_th at start/end
        E0 = E(self.H_th, rho_in)
        Ef = E(self.H_th, rho_out)
        res_total = (Ef - E0) + W_out  # should ~ 0

        # Split including interactions at H_th (start/end identical H makes this well-defined)
        E_S_i, E_S_f = E(self.H_S, rho_in), E(self.H_S, rho_out)
        E_int_i = E(self.H_th - self.H_S - self.H_B, rho_in)
        E_int_f = E(self.H_th - self.H_S - self.H_B, rho_out)
        dE_S   = E_S_f   - E_S_i
        dE_int = E_int_f - E_int_i
        # First-law-like split for the total closed device:
        #   0 = W_out − (Q_h − Q_c) + (ΔE_S + ΔE_int)
        res_split = W_out - (Q_h - Q_c) + (dE_S + dE_int)

        return rho_out, {
            "Q_h": Q_h, "Q_c": Q_c, "Q_in": Q_in,
            "W_out": W_out, "eta": eta, "eta_C": eta_C,
            "Delta_sigma": Delta_sigma, "regime": regime,
            "I_SR": I_f, "D_Bh": D_bh_f, "D_Bc": D_bc_f,
            "check_total_residual": res_total,
            "check_split_residual": res_split
        }

# =========================
# Runner + scoreboard
# =========================

def run_method(args, method_label):
    eng = ExactStrokeEngine(
        omega_h=args.omega_h, omega_c=args.omega_c,
        Th=args.Th, Tc=args.Tc,
        kappa_h=args.kappa_h, kappa_c=args.kappa_c,
        g_on=args.g_on, chi=args.chi,
        t_th=args.t_th, t_work=args.t_work, t_rlx=args.t_rlx,
        method=method_label, print_checks=not args.quiet_checks
    )
    rho = eng.rho0.copy()

    print("\n========================================")
    print(f"Correlated Quantum Engine — method={method_label}, cycles={args.cycles}")
    print("========================================\n")

    per_cycle = []
    for c in range(1, args.cycles + 1):
        rho, summ = eng.run_cycle(rho)
        per_cycle.append(summ)
        print(f"Cycle {c}: [{summ['regime']}]")
        print(f"  Q_h       = {summ['Q_h']:+.6f}  (device heat from hot bath; + into device)")
        print(f"  Q_c       = {summ['Q_c']:+.6f}  (heat to cold bath)")
        print(f"  W_out     = {summ['W_out']:+.6f}  (work extracted)")
        print(f"  η         = {summ['eta']:.4f}    (Carnot bound η_C={summ['eta_C']:.4f})")
        print(f"  Δσ        = {summ['Delta_sigma']:+.6f}  (<0 ⇒ entropic fuel consumed)")
        print(f"   I(S:R)   = {summ['I_SR']:+.6f}")
        print(f"   D(Bh||th)= {summ['D_Bh']:+.6f},  D(Bc||th)={summ['D_Bc']:+.6f}")
        if not args.quiet_checks:
            print(f"  [check] total energy residual  = {summ['check_total_residual']:+.3e}")
            print(f"  [check] split  balance residual= {summ['check_split_residual']:+.3e}")
        print("")

    # aggregate metrics for scoreboard
    total_work = sum(s['W_out'] for s in per_cycle)
    etas = [s['eta'] for s in per_cycle if s['eta'] > 0]
    avg_eta = (sum(etas) / len(etas)) if etas else 0.0
    neg_sigma = sum(1 for s in per_cycle if s['Delta_sigma'] < 0)

    return {
        "method": method_label,
        "per_cycle": per_cycle,
        "total_work": total_work,
        "avg_eta": avg_eta,
        "neg_sigma_count": neg_sigma,
        "cycles": args.cycles,
    }

def main():
    ap = argparse.ArgumentParser(description="Exact stroke-based correlated quantum engine demo (EchoKey Asks).")
    ap.add_argument("--method", choices=["paper", "standard", "both"], default="both",
                    help="Run the correlated engine ('paper'), the decorrelation baseline ('standard'), or both (default).")
    ap.add_argument("--cycles", type=int, default=3, help="Number of cycles to run.")
    # physics
    ap.add_argument("--omega_h", type=float, default=1.0)
    ap.add_argument("--omega_c", type=float, default=0.6)
    ap.add_argument("--Th", type=float, default=2.0)
    ap.add_argument("--Tc", type=float, default=0.5)
    ap.add_argument("--kappa_h", type=float, default=0.25)
    ap.add_argument("--kappa_c", type=float, default=0.25)
    ap.add_argument("--g_on", type=float, default=0.30, help="System–system XY work coupling during work stroke.")
    ap.add_argument("--chi", type=float, default=-0.20, help="Small joint S–bath term during work (spend correlations).")
    # timing (exact, constant-H strokes)
    ap.add_argument("--t_th", type=float, default=2.0)
    ap.add_argument("--t_work", type=float, default=2.0)
    ap.add_argument("--t_rlx", type=float, default=1.0)
    # output
    ap.add_argument("--quiet_checks", action="store_true", help="Hide energy-balance residuals.")
    args = ap.parse_args()

    methods = ["paper", "standard"] if args.method == "both" else [args.method]
    results = [run_method(args, m) for m in methods]

    if len(results) == 2:
        a, b = results
        print("\n================= SCOREBOARD =================")
        def line(res):
            return (f"{res['method']:>8} | total W_out = {res['total_work']:+.6f} | "
                    f"avg η = {res['avg_eta']:.4f} | Δσ<0 cycles = {res['neg_sigma_count']}/{res['cycles']}")
        print(line(a))
        print(line(b))
        # Winner by total work (tie-breaker: avg η)
        winner = None
        if (a["total_work"] > b["total_work"]) or (np.isclose(a["total_work"], b["total_work"]) and a["avg_eta"] > b["avg_eta"]):
            winner = a
        elif (b["total_work"] > a["total_work"]) or (np.isclose(a["total_work"], b["total_work"]) and b["avg_eta"] > a["avg_eta"]):
            winner = b
        if winner:
            print(f"Winner: {winner['method']} (by total work, tie-breaker avg η)")
        else:
            print("Winner: tie (identical total work and avg η)")
        print("==============================================\n")

    print("Done.\n")

if __name__ == "__main__":
    main()
