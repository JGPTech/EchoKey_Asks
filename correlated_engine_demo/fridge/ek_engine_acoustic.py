#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EchoKey — Correlation-Fueled Engine with Acoustic Cooling (Exact-Stroke, No Lindblad)
CC0-1.0 — Public Domain

WHAT THIS FILE MODELS (buildable with current tech)
---------------------------------------------------
Subsystems on a superconducting chip (order shown = tensor order):
  0: S_h   — transmon qubit (the “hot side” of the working medium)
  1: S_c   — transmon qubit (the “cold side” of the working medium)
  2: Bh1   — hot reservoir element #1 (e.g., lossy CPW resonator OR helper qubit)
  3: Bh2   — hot reservoir element #2 (same idea; finite bath)
  4: M     — mechanical/phononic mode (SAW/HBAR), truncated to n_m levels
  5: D     — dump mode (microwave cavity or phonon sink), truncated to n_d levels

Strokes per thermodynamic cycle (piecewise-constant Hamiltonians; each stroke is a single exact expm):
  1) Pre-thermalize  (H_th)   : Weak S↔Bh contacts + weak S_c↔M contact build correlations/athermality.
  2) Work            (H_work) : XY(=XX+YY) between S_h,S_c (tunable coupler / CR gate). Keep a small χ contact
                                so mutual information can drop during work (i.e., “spend” correlations).
  3) Cooling         (H_cool) : Red-sideband beam-splitter S_c↔M (phonon extraction) + mode-mode M↔D (export it).
  4) Relax           (H_rlx)  : Light contacts to settle. Then quench back to H_th to close the energy book.

Hardware mapping (knobs → lab)
------------------------------
  • g_on  : strength of XY(S_h↔S_c) — tunable coupler / CR, ~5–30 MHz effective.
  • χ     : small residual S–bath contact during work stroke — lets I(S:R) decrease (spend resource).
  • g_bs  : beam-splitter S_c↔M during cooling — red-sideband tone ↔ piezo coupling.
  • g_md  : mode-mode M↔D — e.g., mech ↔ over-coupled cavity / phononic waveguide.
  • t_*   : stroke durations — ns–µs, within coherence and fridge budgets.
  • n_m,n_d: oscillator truncations — keep small for tractable Hilbert space.

Exact energetics & resource ledger (no master-equation approximations)
----------------------------------------------------------------------
  Heat:
    Q_h = −Δ⟨H_Bh⟩   (+ means energy came from hot bath into device)
    Q_c = +Δ⟨H_Bc⟩   (B_c ≡ M + D; + means energy delivered to the cold environment)
      NEW: We also print Q_M and Q_D so mechanical refrigeration is visible even if D heats up.

  Work (out):
    W_out = −Σ_strokes Tr[ρ (H_after − H_before)]   # quench-work identity; exact for our stroke boundaries

  Efficiency:
    η = W_out / max(Q_h, 0) only when cycle is truly in engine mode (guarded).
    Otherwise we print “n/a” (e.g., tiny Q_h or W_out≤0).

  Entropic resource per cycle:
    σ = I(S:R) + D(ρ_Bh || ρ_Bh^th) + D(ρ_Bc || ρ_Bc^th),  with R = Bh1,Bh2,M,D and B_c ≡ M⊗D.
    Δσ = σ_f − σ_i.  Negative Δσ ⇒ athermal/correlational “fuel” was consumed (ATHERMAL cycle).

  Energy-conservation checks (printed each cycle):
    (Ef − E0) + W_out  ≈ 0             # same-H total ledger (should be ~ machine epsilon)
    W_out − (Q_h − Q_c) + (ΔE_S + ΔE_int) ≈ 0   # split ledger at H_th; also near machine epsilon

“Standard” control (optional)
-----------------------------
  The “standard” path decorrelates S from the baths at the cycle boundary to act as a **diagnostic control**.
  That non-unitary reset is NOT free physically. To make comparisons fair, you can pass --Treset
  to charge a minimum erasure work W_reset = Treset * (σ_pre − σ_post) (k_B=1 units).
  By default we DO NOT compare. Use --compare to run both paths and print a scoreboard.

Usage
-----
  # Faithful model only (default)
  python ek_engine_acoustic.py

  # Correlated path explicitly
  python ek_engine_acoustic.py --method paper --cycles 3

  # Optional diagnostic comparison (adds control + scoreboard)
  python ek_engine_acoustic.py --compare

  # Charge a reset cost in the control (fairer comparison)
  python ek_engine_acoustic.py --compare --Treset 0.5

  # Stronger cooling leg; show Q_M<0 and n_M drop
  python ek_engine_acoustic.py --method paper --cycles 3 \
    --g_on 0.28 --chi -0.18 --g_bs 0.22 --g_md 0.32 \
    --t_th 2.2 --t_work 1.3 --t_cool 1.3 --t_rlx 0.6 --n_m 4 --n_d 4
"""

import argparse
import numpy as np

# ---------- Linear algebra helpers ----------
def dagger(M): return M.conj().T
def hermitize(M): return (M + dagger(M)) / 2

def expm_hermitian(H, dt):
    """Exact unitary for a constant Hermitian H over duration dt."""
    w, V = np.linalg.eigh(hermitize(H))
    return V @ np.diag(np.exp(-1j * w * dt)) @ dagger(V)

def kron_all(mats):
    out = np.array([[1 + 0j]])
    for A in mats:
        out = np.kron(out, A)
    return out

def partial_trace(rho, keep, dims):
    """Trace out all subsystems except those in 'keep' (indices)."""
    keep = sorted(keep)
    N = len(dims)
    D = int(np.prod(dims))
    assert rho.shape == (D, D)
    rho_t = rho.reshape(dims + dims)  # (i1..iN ; j1..jN)
    trace_out = [i for i in range(N) if i not in keep]
    perm = keep + trace_out
    rho_t = np.transpose(rho_t, axes=perm + [i + N for i in perm])
    dim_keep = int(np.prod([dims[i] for i in keep])) if keep else 1
    dim_tr   = int(np.prod([dims[i] for i in trace_out])) if trace_out else 1
    rho_t = rho_t.reshape(dim_keep, dim_tr, dim_keep, dim_tr)
    if dim_tr == 1: return rho_t[:, 0, :, 0]
    return np.einsum("aibj,ij->ab", rho_t, np.eye(dim_tr))

def vn_entropy(rho, tol=1e-12):
    e = np.linalg.eigvalsh(hermitize(rho))
    e = np.clip(np.real(e), 0.0, 1.0)
    nz = e[e > tol]
    return float(-np.sum(nz * np.log(nz))) if nz.size else 0.0

def rel_entropy(rho, sigma, tol=1e-12):
    """Quantum relative entropy S(ρ||σ) with PSD guards."""
    def logm_psd(X):
        ex, V = np.linalg.eigh(hermitize(X))
        ex = np.clip(np.real(ex), tol, None)
        return V @ np.diag(np.log(ex)) @ dagger(V)
    sig = hermitize(sigma)
    ev  = np.linalg.eigvalsh(sig)
    if np.min(ev) < tol:
        sig = sig + tol * np.eye(sig.shape[0], dtype=complex)
    return float(np.real(np.trace(hermitize(rho) @ (logm_psd(rho) - logm_psd(sig)))))

# ---------- Operator blocks & embeddings ----------
def eye(d): return np.eye(d, dtype=complex)

def embed(op, idx, dims):
    ops = [eye(d) for d in dims]
    ops[idx] = op
    return kron_all(ops)

# Qubit ops
SX = np.array([[0,1],[1,0]], dtype=complex)
SY = np.array([[0,-1j],[1j,0]], dtype=complex)
SZ = np.array([[1,0],[0,-1]], dtype=complex)
SP = np.array([[0,1],[0,0]], dtype=complex)
SM = np.array([[0,0],[1,0]], dtype=complex)

# Oscillator ops
def a_op(n):
    a = np.zeros((n, n), dtype=complex)
    for k in range(1, n):
        a[k-1, k] = np.sqrt(k)
    return a
def num_op(n):  return np.diag(np.arange(0, n, 1, dtype=float)).astype(complex)

# ---------- Engine class ----------
class AcousticEngine:
    """
    Subsystem order: [S_h, S_c, Bh1, Bh2, M, D]
    Dims:            [  2 ,  2 ,  2 ,  2 , n_m, n_d]
    """
    def __init__(self,
                 # Frequencies in GHz (converted internally to rad/s; ħ=1)
                 fh=5.0, fc=6.0, fbh1=5.0, fbh2=5.5, fm=0.10, fd=0.12,
                 # Temperatures (k_B=1)
                 Th=2.0, Tc=0.5,
                 # Couplings
                 kappa_h=0.25, kappa_c=0.25,
                 g_on=0.30, chi=-0.15,
                 g_bs=0.10, g_md=0.10,
                 # Durations
                 t_th=2.0, t_work=1.5, t_cool=1.0, t_rlx=0.8,
                 # Oscillator truncations
                 n_m=3, n_d=3,
                 # Control vs faithful path
                 method='paper',
                 # Optional: charge reset cost for standard control
                 Treset=None):
        # indices and dims
        self.idx  = {"Sh":0, "Sc":1, "Bh1":2, "Bh2":3, "M":4, "D":5}
        self.dims = [2, 2, 2, 2, n_m, n_d]

        # convert GHz→rad/s
        self.omega_h   = 2*np.pi*fh
        self.omega_c   = 2*np.pi*fc
        self.omega_bh1 = 2*np.pi*fbh1
        self.omega_bh2 = 2*np.pi*fbh2
        self.omega_m   = 2*np.pi*fm
        self.omega_d   = 2*np.pi*fd

        self.Th, self.Tc = Th, Tc
        self.kappa_h, self.kappa_c = kappa_h, kappa_c
        self.g_on, self.chi = g_on, chi
        self.g_bs, self.g_md = g_bs, g_md
        self.t_th, self.t_work, self.t_cool, self.t_rlx = t_th, t_work, t_cool, t_rlx
        self.method = method
        self.Treset = Treset  # None = no reset charge (diagnostic only)

        # shorthand
        Sh, Sc, Bh1, Bh2, M, D = (self.idx[k] for k in ["Sh","Sc","Bh1","Bh2","M","D"])
        n_m, n_d = self.dims[M], self.dims[D]
        aM, aD   = a_op(n_m), a_op(n_d)
        XM       = aM + dagger(aM)

        # Bare single-site Hamiltonians (embedded)
        self.H_Sh = 0.5*self.omega_h * embed(SZ, Sh, self.dims)
        self.H_Sc = 0.5*self.omega_c * embed(SZ, Sc, self.dims)
        self.H_S  = self.H_Sh + self.H_Sc

        self.H_Bh1 = 0.5*self.omega_bh1 * embed(SZ, Bh1, self.dims)
        self.H_Bh2 = 0.5*self.omega_bh2 * embed(SZ, Bh2, self.dims)
        self.H_Bh  = self.H_Bh1 + self.H_Bh2

        self.H_M = self.omega_m * (embed(num_op(n_m), M, self.dims) + 0.5*embed(eye(n_m), M, self.dims))
        self.H_D = self.omega_d * (embed(num_op(n_d), D, self.dims) + 0.5*embed(eye(n_d), D, self.dims))
        self.H_Bc = self.H_M + self.H_D

        self.H0 = self.H_S + self.H_Bh + self.H_Bc

        # Work channel (XY) — maps to tunable coupler / cross-resonance
        self.H_g = 0.5*(embed(SX, Sh, self.dims) @ embed(SX, Sc, self.dims)
                      + embed(SY, Sh, self.dims) @ embed(SY, Sc, self.dims))

        # Contacts (microwave S_h↔Bh*, piezo S_c↔M)
        self.H_SB_hot  = (embed(SX, Sh, self.dims) @ embed(SX, Bh1, self.dims)
                        + embed(SX, Sh, self.dims) @ embed(SX, Bh2, self.dims))
        self.H_SB_cold =  embed(SX, Sc, self.dims) @ embed(XM,  M,   self.dims)

        # Cooling: beam-splitter S_c↔M (red sideband) + mode-mode M↔D (export)
        self.H_bs = (embed(SM, Sc, self.dims) @ embed(dagger(aM), M, self.dims)
                   + embed(SP, Sc, self.dims) @ embed(aM,         M, self.dims))
        self.H_md = (embed(aM,   M, self.dims) @ embed(dagger(aD), D, self.dims)
                   + embed(dagger(aM), M, self.dims) @ embed(aD,   D, self.dims))

        # Stroke Hamiltonians
        self.H_th   = self.H0 + self.kappa_h*self.H_SB_hot + self.kappa_c*self.H_SB_cold
        self.H_work = self.H0 + self.g_on*self.H_g + self.chi*(self.H_SB_hot + self.H_SB_cold)
        self.H_cool = self.H0 + self.g_bs*self.H_bs + self.g_md*self.H_md
        self.H_rlx  = self.H0 + 0.5*self.kappa_h*self.H_SB_hot + 0.5*self.kappa_c*self.H_SB_cold

        # Thermal references on subsystem spaces (NOT embedded)
        self.rho_th_Bh = self._thermal_state(self._subHam_labels(["Bh1","Bh2"]), beta=1.0/self.Th)
        self.rho_th_Bc = self._thermal_state(self._subHam_labels(["M","D"]),    beta=1.0/self.Tc)

        # Initial product thermal state (no correlations)
        rho_Sh = self._thermal_state(0.5*self.omega_h*SZ,  beta=1.0/self.Th)
        rho_Sc = self._thermal_state(0.5*self.omega_c*SZ,  beta=1.0/self.Tc)
        rho_Bh = self._thermal_state(self._subHam_labels(["Bh1","Bh2"]), beta=1.0/self.Th)
        rho_M  = self._thermal_state(self._bareHam_on_subset("M"),  beta=1.0/self.Tc)
        rho_D  = self._thermal_state(self._bareHam_on_subset("D"),  beta=1.0/self.Tc)
        self.rho0 = kron_all([rho_Sh, rho_Sc, rho_Bh, rho_M, rho_D])

    # ---------- Sub-Hamiltonians on subsystem spaces ----------
    def _bareHam_on_subset(self, which):
        if which == "M":
            n = self.dims[self.idx["M"]]
            return self.omega_m * (num_op(n) + 0.5 * eye(n))
        if which == "D":
            n = self.dims[self.idx["D"]]
            return self.omega_d * (num_op(n) + 0.5 * eye(n))
        if which in ("Sh","Sc","Bh1","Bh2"):
            omega = {"Sh":self.omega_h, "Sc":self.omega_c,
                     "Bh1":self.omega_bh1, "Bh2":self.omega_bh2}[which]
            return 0.5 * omega * SZ
        raise KeyError(which)

    def _subHam_labels(self, labels):
        """Bare many-body H on ⊗_{labels} (not embedded)."""
        sub_dims = [ (2 if lab in ("Sh","Sc","Bh1","Bh2") else self.dims[self.idx[lab]]) for lab in labels ]
        H_sub = np.zeros((int(np.prod(sub_dims)),) * 2, dtype=complex)
        for j, lab in enumerate(labels):
            h_j = self._bareHam_on_subset(lab)
            factors = []
            for k, labk in enumerate(labels):
                if k == j: factors.append(h_j)
                else:
                    d = 2 if labk in ("Sh","Sc","Bh1","Bh2") else self.dims[self.idx[labk]]
                    factors.append(eye(d))
            H_sub = H_sub + kron_all(factors)
        return H_sub

    # ---------- Reductions ----------
    def rho_S (self, rho): return partial_trace(rho, [self.idx["Sh"], self.idx["Sc"]], self.dims)
    def rho_Bh(self, rho): return partial_trace(rho, [self.idx["Bh1"], self.idx["Bh2"]], self.dims)
    def rho_Bc(self, rho): return partial_trace(rho, [self.idx["M"], self.idx["D"]], self.dims)
    def rho_R (self, rho): return partial_trace(rho, [self.idx["Bh1"], self.idx["Bh2"], self.idx["M"], self.idx["D"]], self.dims)

    # ---------- Thermal state for a given sub-H ----------
    def _thermal_state(self, H_sub, beta):
        e, V = np.linalg.eigh(hermitize(H_sub))
        Z = np.sum(np.exp(-beta * e))
        return V @ np.diag(np.exp(-beta * e) / Z) @ dagger(V)

    # ---------- One exact-stroke cycle ----------
    def run_cycle(self, rho_in):
        # Unitaries (exact per stroke)
        U_th   = expm_hermitian(self.H_th,   self.t_th)
        U_work = expm_hermitian(self.H_work, self.t_work)
        U_cool = expm_hermitian(self.H_cool, self.t_cool)
        U_rlx  = expm_hermitian(self.H_rlx,  self.t_rlx)

        # Start-of-cycle resource
        rho_S_i, rho_R_i = self.rho_S(rho_in), self.rho_R(rho_in)
        I_i    = vn_entropy(rho_S_i) + vn_entropy(rho_R_i) - vn_entropy(rho_in)
        D_bh_i = rel_entropy(self.rho_Bh(rho_in), self.rho_th_Bh)
        D_bc_i = rel_entropy(self.rho_Bc(rho_in), self.rho_th_Bc)
        sigma_i = I_i + D_bh_i + D_bc_i

        # Strokes + quench work
        rho1 = U_th   @ rho_in @ dagger(U_th);   Wq1 = float(np.real(np.trace(rho1 @ (self.H_work - self.H_th))))
        rho2 = U_work @ rho1   @ dagger(U_work); Wq2 = float(np.real(np.trace(rho2 @ (self.H_cool - self.H_work))))
        rho3 = U_cool @ rho2   @ dagger(U_cool); Wq3 = float(np.real(np.trace(rho3 @ (self.H_rlx  - self.H_cool))))
        rho4 = U_rlx  @ rho3   @ dagger(U_rlx ); Wq4 = float(np.real(np.trace(rho4 @ (self.H_th   - self.H_rlx ))))
        rho_out = rho4

        # Pre-reset resource (end of unitary strokes)
        rho_S_pre, rho_R_pre = self.rho_S(rho_out), self.rho_R(rho_out)
        I_pre    = vn_entropy(rho_S_pre) + vn_entropy(rho_R_pre) - vn_entropy(rho_out)
        D_bh_pre = rel_entropy(self.rho_Bh(rho_out), self.rho_th_Bh)
        D_bc_pre = rel_entropy(self.rho_Bc(rho_out), self.rho_th_Bc)
        sigma_pre = I_pre + D_bh_pre + D_bc_pre

        # Optional decorrelation control with reset charge
        W_reset = 0.0
        if self.method == "standard":
            rho_decorr = kron_all([ self.rho_S(rho_out),
                                    self.rho_Bh(rho_out),
                                    self.rho_Bc(rho_out) ])
            # Post-reset resource
            rho_S_post, rho_R_post = self.rho_S(rho_decorr), self.rho_R(rho_decorr)
            I_post    = vn_entropy(rho_S_post) + vn_entropy(rho_R_post) - vn_entropy(rho_decorr)
            D_bh_post = rel_entropy(self.rho_Bh(rho_decorr), self.rho_th_Bh)
            D_bc_post = rel_entropy(self.rho_Bc(rho_decorr), self.rho_th_Bc)
            sigma_post = I_post + D_bh_post + D_bc_post

            if self.Treset is not None:
                erased = max(0.0, sigma_pre - sigma_post)
                W_reset = self.Treset * erased  # Landauer-style minimum charge (k_B=1)
            rho_out = rho_decorr

        # Energetics
        def E(op, r): return float(np.real(np.trace(r @ op)))
        E_Bh_i, E_Bh_f = E(self.H_Bh, rho_in),  E(self.H_Bh, rho_out)
        E_M_i,  E_M_f  = E(self.H_M,  rho_in),  E(self.H_M,  rho_out)
        E_D_i,  E_D_f  = E(self.H_D,  rho_in),  E(self.H_D,  rho_out)
        E_Bc_i, E_Bc_f = E_M_i + E_D_i,          E_M_f + E_D_f

        Q_h = -(E_Bh_f - E_Bh_i)   # + = heat from hot bath into device
        Q_M = +(E_M_f  - E_M_i)    # + = energy to mechanical mode (cooling if negative)
        Q_D = +(E_D_f  - E_D_i)    # + = energy to dump
        Q_c = +(E_Bc_f - E_Bc_i)   # total to cold environment
        Q_in = max(0.0, Q_h)

        # Work from quench identities (exact), minus reset charge if applied
        W_on  = Wq1 + Wq2 + Wq3 + Wq4 + W_reset
        W_out = -W_on

        # Efficiency (only when truly in engine mode)
        eps = 1e-9
        eta_C = 1.0 - (self.Tc / self.Th)
        eta   = (W_out / Q_in) if (Q_in > eps and W_out > eps) else float('nan')

        # End-of-cycle resource (post reset if any)
        rho_S_f, rho_R_f = self.rho_S(rho_out), self.rho_R(rho_out)
        I_f    = vn_entropy(rho_S_f) + vn_entropy(rho_R_f) - vn_entropy(rho_out)
        D_bh_f = rel_entropy(self.rho_Bh(rho_out), self.rho_th_Bh)
        D_bc_f = rel_entropy(self.rho_Bc(rho_out), self.rho_th_Bc)
        sigma_f = I_f + D_bh_f + D_bc_f
        Delta_sigma = sigma_f - sigma_i
        regime = "ATHERMAL" if Delta_sigma < 0 else "THERMAL"

        # Energy checks at same H (H_th)
        E0 = E(self.H_th, rho_in); Ef = E(self.H_th, rho_out)
        res_total = (Ef - E0) + W_out
        H_int = self.H_th - self.H_S - self.H_Bh - self.H_Bc
        dE_S   = E(self.H_S, rho_out) - E(self.H_S, rho_in)
        dE_int = E(H_int,    rho_out) - E(H_int,    rho_in)
        res_split = W_out - (Q_h - Q_c) + (dE_S + dE_int)

        # Mechanical occupancy (lab-friendly)
        nM_i = E_M_i / self.omega_m - 0.5
        nM_f = E_M_f / self.omega_m - 0.5

        return rho_out, {
            "Q_h":Q_h, "Q_c":Q_c, "Q_M":Q_M, "Q_D":Q_D,
            "W_out":W_out, "W_reset":W_reset,
            "eta":eta, "eta_C":eta_C,
            "Delta_sigma":Delta_sigma, "regime":regime,
            "I_SR":I_f, "D_Bh":D_bh_f, "D_Bc":D_bc_f,
            "nM_i":nM_i, "nM_f":nM_f,
            "check_total_residual":res_total, "check_split_residual":res_split
        }

# ---------- Runner / CLI ----------
def run_method(args, method):
    eng = AcousticEngine(
        fh=args.fh, fc=args.fc, fbh1=args.fbh1, fbh2=args.fbh2, fm=args.fm, fd=args.fd,
        Th=args.Th, Tc=args.Tc,
        kappa_h=args.kappa_h, kappa_c=args.kappa_c,
        g_on=args.g_on, chi=args.chi, g_bs=args.g_bs, g_md=args.g_md,
        t_th=args.t_th, t_work=args.t_work, t_cool=args.t_cool, t_rlx=args.t_rlx,
        n_m=args.n_m, n_d=args.n_d,
        method=method, Treset=args.Treset
    )
    rho = eng.rho0.copy()

    print("\n========================================")
    print(f"Acoustic Correlated Engine — method={method}, cycles={args.cycles}")
    if method == "standard":
        print("Note: 'standard' is a DECORRELATING CONTROL (non-unitary reset at boundary).")
        if args.Treset is None:
            print("      No reset cost charged (diagnostic only). Use --Treset to charge minimum erasure work.")
        else:
            print(f"      Reset cost charged: W_reset >= Treset * erased_resource, with Treset={args.Treset}")
    print("========================================\n")

    per = []
    for c in range(1, args.cycles+1):
        rho, s = eng.run_cycle(rho)
        per.append(s)
        eta_str = "n/a" if (np.isnan(s["eta"])) else f"{s['eta']:.4f}"
        print(f"Cycle {c}: [{s['regime']}]")
        print(f"  Q_h       = {s['Q_h']:+.6e}  (+ into device from hot)")
        print(f"  Q_c       = {s['Q_c']:+.6e}  (+ to cold env M+D)")
        print(f"   Q_M      = {s['Q_M']:+.6e}  (mechanical mode)")
        print(f"   Q_D      = {s['Q_D']:+.6e}  (dump mode)")
        print(f"  W_out     = {s['W_out']:+.6e}")
        if method == "standard" and s["W_reset"] != 0.0:
            print(f"   W_reset  = -{s['W_reset']:.6e}  (charged decorrelation erasure)")
        print(f"  η         = {eta_str}    (η_C={s['eta_C']:.4f})")
        print(f"  Δσ        = {s['Delta_sigma']:+.6f}  (<0 ⇒ consumed resource)")
        print(f"   I(S:R)   = {s['I_SR']:+.6f}")
        print(f"   D(Bh||th)= {s['D_Bh']:+.6f},  D(Bc||th)={s['D_Bc']:+.6f}")
        print(f"  n_M       = {s['nM_i']:.4f} → {s['nM_f']:.4f}")
        if not args.quiet_checks:
            print(f"  [check] total residual  = {s['check_total_residual']:+.3e}")
            print(f"  [check] split residual  = {s['check_split_residual']:+.3e}")
        print("")

    total_W = sum(x["W_out"] for x in per)
    etas = [x["eta"] for x in per if (not np.isnan(x["eta"]))]
    avg_eta = (sum(etas)/len(etas)) if etas else 0.0
    negsig = sum(1 for x in per if x["Delta_sigma"] < 0)

    return {"method":method, "total_W":total_W, "avg_eta":avg_eta,
            "negsig":negsig, "cycles":args.cycles}

def main():
    ap = argparse.ArgumentParser(description="Exact-stroke correlation engine with acoustic cooling (EchoKey)")
    # Default: faithful correlated path only
    ap.add_argument("--method", choices=["paper","standard"], default="paper",
                    help="Run the faithful correlated engine ('paper') or the decorrelating control ('standard').")
    ap.add_argument("--compare", action="store_true",
                    help="Run BOTH (paper + standard) and show a scoreboard (diagnostic).")
    ap.add_argument("--cycles", type=int, default=3)

    # Frequencies in GHz (converted to rad/s internally)
    ap.add_argument("--fh",   type=float, default=5.0,  help="S_h freq (GHz)")
    ap.add_argument("--fc",   type=float, default=6.0,  help="S_c freq (GHz)")
    ap.add_argument("--fbh1", type=float, default=5.0,  help="Bh1 freq (GHz)")
    ap.add_argument("--fbh2", type=float, default=5.5,  help="Bh2 freq (GHz)")
    ap.add_argument("--fm",   type=float, default=0.10, help="mechanical freq (GHz)")
    ap.add_argument("--fd",   type=float, default=0.12, help="dump freq (GHz)")

    # Temperatures (k_B = 1)
    ap.add_argument("--Th", type=float, default=2.0)
    ap.add_argument("--Tc", type=float, default=0.5)
    ap.add_argument("--Treset", type=float, default=None,
                    help="Charge decorrelation erasure in 'standard' as W_reset >= Treset * erased_resource. Default: None (no charge).")

    # Couplings
    ap.add_argument("--kappa_h", type=float, default=0.25)
    ap.add_argument("--kappa_c", type=float, default=0.25)
    ap.add_argument("--g_on",    type=float, default=0.30, help="work: S_h<->S_c XY")
    ap.add_argument("--chi",     type=float, default=-0.15, help="small S-bath term during work (spend corr.)")
    ap.add_argument("--g_bs",    type=float, default=0.10, help="cool: S_c<->M beam-splitter")
    ap.add_argument("--g_md",    type=float, default=0.10, help="cool: M<->D mode-mode")

    # Durations
    ap.add_argument("--t_th",   type=float, default=2.0)
    ap.add_argument("--t_work", type=float, default=1.5)
    ap.add_argument("--t_cool", type=float, default=1.0)
    ap.add_argument("--t_rlx",  type=float, default=0.8)

    # Truncations
    ap.add_argument("--n_m", type=int, default=3)
    ap.add_argument("--n_d", type=int, default=3)

    # Printing
    ap.add_argument("--quiet_checks", action="store_true")

    args = ap.parse_args()

    methods = ["paper","standard"] if args.compare else [args.method]
    results = [run_method(args, m) for m in methods]

    if args.compare and len(results) == 2:
        a, b = results
        print("\n================= SCOREBOARD (diagnostic) =================")
        fmt = lambda r: (f"{r['method']:>8} | total W_out = {r['total_W']:+.6e} | "
                         f"avg η = {r['avg_eta']:.4f} | Δσ<0 cycles = {r['negsig']}/{r['cycles']}")
        print(fmt(a)); print(fmt(b))
        winner = None
        if (a["total_W"] > b["total_W"]) or (np.isclose(a["total_W"], b["total_W"]) and a["avg_eta"] > b["avg_eta"]):
            winner = a
        elif (b["total_W"] > a["total_W"]) or (np.isclose(a["total_W"], b["total_W"]) and b["avg_eta"] > a["avg_eta"]):
            winner = b
        if winner:
            print(f"Winner: {winner['method']}  (note: 'standard' includes a non-unitary reset; use --Treset for fair charge)")
        else:
            print("Winner: tie")
        print("===========================================================\n")

    print("Done.\n")

if __name__ == "__main__":
    main()
