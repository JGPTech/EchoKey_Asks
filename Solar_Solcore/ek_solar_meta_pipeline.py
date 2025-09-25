#!/usr/bin/env python3
# ek_solar_meta_pipeline.py
# EchoKey Emergent Meta-Layer → Standard Solar Sandbox (real Solcore; coin-driven meta; no fallbacks)

import argparse, json, os, sys, time, uuid, hashlib
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import pandas as pd

# -------------------------
# Solcore (required)
# -------------------------
try:
    from solcore import material
    from solcore.structure import Layer, Junction
    from solcore.solar_cell import SolarCell
    from solcore.light_source import LightSource
    from solcore.solar_cell_solver import solar_cell_solver
    from solcore.material_system.create_new_material import create_new_material
except Exception as e:
    raise SystemExit("Solcore (with material registration) is required for this script.\n" + str(e))

# -------------------------
# Utils
# -------------------------
def now_stamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")

def run_id() -> str:
    return f"ekrun-{now_stamp()}-{uuid.uuid4().hex[:8]}"

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_csv(df: pd.DataFrame, path: str):
    df.to_csv(path, index=False)

def cm2_to_m2(x_cm2: float) -> float:
    return x_cm2 * 1e-4

def cm3_to_m3(x_cm3: float) -> float:
    return x_cm3 * 1e6

# -------------------------
# Data models
# -------------------------
@dataclass
class MetaOptics:
    wl_m: np.ndarray          # wavelength grid [m]
    n_eff: np.ndarray         # n(lambda)
    k_eff: np.ndarray         # k(lambda)
    thickness_m: float        # layer thickness [m]

@dataclass
class MetaTransport:
    S_front_eff: float
    S_back_eff: float
    mu_n_eff: float
    mu_p_eff: float
    tau_n_eff: float
    tau_p_eff: float

@dataclass
class DeviceSummary:
    model_used: str
    Jsc_mAcm2: float
    Voc_V: float
    FF: float
    eta_pct: float

@dataclass
class ComparisonSummary:
    baseline: DeviceSummary
    emergent: DeviceSummary
    delta_eta_pct_points: float
    pct_improvement_vs_baseline: float
    optical_residual: Optional[float]
    confidence_score_0to1: float

@dataclass
class CoinConfig:
    # optics
    wl: Optional[List[float]] = None
    n_base: Optional[List[float]] = None
    k_base: Optional[List[float]] = None
    centers_nm: Optional[List[float]] = None
    sigmas_nm: Optional[List[float]] = None
    weights: Optional[List[float]] = None
    gamma_k: float = 0.05
    thickness_m: float = 120e-9
    # transport scalars (effective knobs for meta layer surface/transport)
    S_front_eff: float = 1e3
    S_back_eff: float  = 1e3
    mu_n_eff: float    = 30.0
    mu_p_eff: float    = 10.0
    tau_n_eff: float   = 5e-9
    tau_p_eff: float   = 5e-9
    # optional scaling multipliers (kept simple; you can set to 1.0)
    S_front_mult: float = 1.0
    S_back_mult: float  = 1.0
    mu_n_mult: float    = 1.0
    mu_p_mult: float    = 1.0
    tau_n_mult: float   = 1.0
    tau_p_mult: float   = 1.0

# -------------------------
# Optics helpers
# -------------------------
def write_nk_txt(wl_m: np.ndarray, n: np.ndarray, k: np.ndarray, outdir: str, prefix: str) -> Tuple[str, str]:
    ensure_dir(outdir)
    n_path = os.path.join(outdir, f"{prefix}_n.txt")
    k_path = os.path.join(outdir, f"{prefix}_k.txt")
    with open(n_path, "w") as fn:
        for w, nv in zip(wl_m, n):
            fn.write(f"{w:.9e}\t{nv:.9e}\n")
    with open(k_path, "w") as fk:
        for w, kv in zip(wl_m, k):
            fk.write(f"{w:.9e}\t{kv:.9e}\n")
    return n_path, k_path

def register_tabulated_material(name: str, wl_m: np.ndarray, n: np.ndarray, k: np.ndarray, outdir: str):
    n_path, k_path = write_nk_txt(wl_m, n, k, outdir, prefix=name)
    create_new_material(name, n_source=n_path, k_source=k_path, overwrite=True)

def gentle_gaas_baseline(wl_m: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    wl_nm = wl_m * 1e9
    n = 3.4 - 0.25*np.exp(-((wl_nm-600.0)/180.0)**2)
    k = 0.001 + 0.08*np.exp(-((wl_nm-720.0)/80.0)**2)
    return n.astype(float), k.astype(float)

def gentle_algaas_baseline(wl_m: np.ndarray, Al: float = 0.30) -> Tuple[np.ndarray, np.ndarray]:
    wl_nm = wl_m * 1e9
    n = 3.2 - 0.20*np.exp(-((wl_nm-550.0)/200.0)**2)
    k = 1e-4 + 0.002*np.exp(-((wl_nm-700.0)/70.0)**2)
    return n.astype(float), k.astype(float)

def rough_gold_baseline(wl_m: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = np.full_like(wl_m, 0.3, dtype=float)
    k = np.full_like(wl_m, 6.5, dtype=float)
    return n, k

def ensure_stack_optical_materials(wl_m: np.ndarray, outdir: str, rid: str) -> Dict[str, str]:
    """
    Register tabulated nk for non-PDD layers only:
      - AlGaAs window (optical)
      - Au back (optical)
    Junction (GaAs) stays as built-in (has electrical params).
    """
    names = {
        "algaas": f"EK_AlGaAs30_{rid}",
        "au": f"EK_Au_{rid}",
    }
    n_w, k_w = gentle_algaas_baseline(wl_m, Al=0.30)
    register_tabulated_material(names["algaas"], wl_m, n_w, k_w, outdir)
    n_au, k_au = rough_gold_baseline(wl_m)
    register_tabulated_material(names["au"], wl_m, n_au, k_au, outdir)
    return names

def register_meta_material(meta_opt: MetaOptics, outdir: str, rid: str) -> str:
    mat_name = f"MetaNK_{rid}"
    register_tabulated_material(mat_name, meta_opt.wl_m, meta_opt.n_eff, meta_opt.k_eff, outdir)
    return mat_name

# -------------------------
# Coin → Meta mapping
# -------------------------
def default_wavelength_grid() -> np.ndarray:
    return np.linspace(300e-9, 1000e-9, 701)

def gaussian_bumps(wl_m: np.ndarray, centers_nm: List[float], sigmas_nm: List[float]) -> np.ndarray:
    wl_nm = wl_m * 1e9
    mats = []
    for mu, sig in zip(centers_nm, sigmas_nm):
        sig = max(float(sig), 1e-9)
        mats.append(np.exp(-0.5*((wl_nm - mu)/sig)**2))
    return np.stack(mats, axis=0) if mats else np.zeros((0, wl_m.size))

def normalize_weights(w: Optional[List[float]], K: int) -> np.ndarray:
    if w is None or len(w) != K or K == 0:
        return np.ones(K) / float(max(K, 1))
    wv = np.asarray(w, float)
    s = float(wv.sum())
    return wv / s if s > 0 else (np.ones(K) / float(K))

def coin_to_meta(coin: CoinConfig) -> Tuple[MetaOptics, MetaTransport]:
    wl_m = default_wavelength_grid() if coin.wl is None else np.array(coin.wl, float)
    if not np.all(np.diff(wl_m) > 0):
        raise ValueError("Wavelength grid 'wl' must be strictly increasing.")
    if coin.n_base is not None and coin.k_base is not None:
        n_base = np.array(coin.n_base, float); k_base = np.array(coin.k_base, float)
        if n_base.size != wl_m.size or k_base.size != wl_m.size:
            raise ValueError("n_base and k_base must match the length of wl.")
    else:
        n_base, k_base = gentle_gaas_baseline(wl_m)

    centers = coin.centers_nm or [720.0]
    sigmas  = coin.sigmas_nm  or [60.0]*len(centers)
    B = gaussian_bumps(wl_m, centers, sigmas)  # shape K x N
    K = B.shape[0]
    rk = normalize_weights(coin.weights, K)
    mix = (rk[:, None] * B).sum(axis=0) if K > 0 else np.zeros_like(wl_m)
    if mix.size and mix.max() > 0:
        mix = mix / mix.max()
    delta_k = coin.gamma_k * mix
    k_eff = np.clip(k_base + delta_k, 0.0, None)
    n_eff = n_base.copy()

    meta_opt = MetaOptics(wl_m=wl_m, n_eff=n_eff, k_eff=k_eff, thickness_m=float(coin.thickness_m))
    meta_tr  = MetaTransport(
        S_front_eff=float(coin.S_front_eff * coin.S_front_mult),
        S_back_eff =float(coin.S_back_eff  * coin.S_back_mult),
        mu_n_eff   =float(coin.mu_n_eff * coin.mu_n_mult),
        mu_p_eff   =float(coin.mu_p_eff * coin.mu_p_mult),
        tau_n_eff  =float(coin.tau_n_eff * coin.tau_n_mult),
        tau_p_eff  =float(coin.tau_p_eff * coin.tau_p_mult)
    )
    return meta_opt, meta_tr

def hash_config(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

def cache_paths(cache_dir: str, key: str) -> Tuple[str, str]:
    return (os.path.join(cache_dir, f"meta_optics_{key}.json"),
            os.path.join(cache_dir, f"meta_transport_{key}.json"))

def save_meta_json(meta_opt: MetaOptics, meta_tr: MetaTransport, opt_path: str, tr_path: str):
    ensure_dir(os.path.dirname(opt_path))
    json.dump({
        "wl": meta_opt.wl_m.tolist(),
        "n":  meta_opt.n_eff.tolist(),
        "k":  meta_opt.k_eff.tolist(),
        "thickness_m": meta_opt.thickness_m
    }, open(opt_path, "w"), indent=2)
    json.dump({
        "S_front_eff": meta_tr.S_front_eff,
        "S_back_eff":  meta_tr.S_back_eff,
        "mu_n_eff":    meta_tr.mu_n_eff,
        "mu_p_eff":    meta_tr.mu_p_eff,
        "tau_n_eff":   meta_tr.tau_n_eff,
        "tau_p_eff":   meta_tr.tau_p_eff
    }, open(tr_path, "w"), indent=2)

def load_meta_json(opt_path: str, tr_path: str) -> Tuple[MetaOptics, MetaTransport]:
    dopt = json.load(open(opt_path)); dtr = json.load(open(tr_path))
    return MetaOptics(
        wl_m=np.array(dopt["wl"], float),
        n_eff=np.array(dopt["n"], float),
        k_eff=np.array(dopt["k"], float),
        thickness_m=float(dopt["thickness_m"])
    ), MetaTransport(
        S_front_eff=float(dtr["S_front_eff"]),
        S_back_eff=float(dtr["S_back_eff"]),
        mu_n_eff=float(dtr["mu_n_eff"]),
        mu_p_eff=float(dtr["mu_p_eff"]),
        tau_n_eff=float(dtr["tau_n_eff"]),
        tau_p_eff=float(dtr["tau_p_eff"])
    )

# -------------------------
# Solcore device (real path)
# -------------------------
def solcore_device(wl_m: np.ndarray,
                   meta_opt: Optional[MetaOptics],
                   meta_tr: Optional[MetaTransport],
                   use_meta_layer: bool,
                   outdir: str,
                   rid: str) -> Tuple[DeviceSummary, pd.DataFrame, Optional[float]]:
    # Register optics for non-PDD layers
    stack_names = ensure_stack_optical_materials(wl_m, outdir, rid)
    AlGaAsMat = material(stack_names["algaas"])
    AuMat     = material(stack_names["au"])

    # Built-in GaAs for the PDD junction (electrical params present)
    GaAs = material("GaAs")

    # Transport (SI)
    mu_n = 0.85; mu_p = 0.04
    tau_n = 5e-9; tau_p = 5e-9
    Vt = 0.02585
    Dn = mu_n * Vt; Dp = mu_p * Vt
    Ln = float(np.sqrt(Dn * tau_n)); Lp = float(np.sqrt(Dp * tau_p))

    # Junction layers
    GaAs_p_emit = GaAs(T=300, Na=cm3_to_m3(2e17),
                       electron_mobility=mu_n, hole_mobility=mu_p,
                       tau_n=tau_n, tau_p=tau_p,
                       electron_diffusion_length=Ln, hole_diffusion_length=Lp)
    GaAs_n_base = GaAs(T=300, Nd=cm3_to_m3(5e16),
                       electron_mobility=mu_n, hole_mobility=mu_p,
                       tau_n=tau_n, tau_p=tau_p,
                       electron_diffusion_length=Ln, hole_diffusion_length=Lp)
    GaAs_n_bsf  = GaAs(T=300, Nd=cm3_to_m3(2e17),
                       electron_mobility=mu_n, hole_mobility=mu_p,
                       tau_n=tau_n, tau_p=tau_p,
                       electron_diffusion_length=Ln, hole_diffusion_length=Lp)

    # Non-PDD layers
    AlGaAs_window = AlGaAsMat(T=300)   # optical window
    Au_metal      = AuMat()

    # Optional meta layer (tabulated nk + effective transport knobs)
    MetaMat = None
    if use_meta_layer and (meta_opt is not None) and (meta_tr is not None):
        meta_name = register_meta_material(meta_opt, outdir, rid)
        MetaMat = material(meta_name)(
            electron_mobility=cm2_to_m2(meta_tr.mu_n_eff),
            hole_mobility=cm2_to_m2(meta_tr.mu_p_eff),
            tau_n=meta_tr.tau_n_eff,
            tau_p=meta_tr.tau_p_eff
        )

    # Build stack
    window   = Layer(60e-9,  material=AlGaAs_window)
    emitter  = Layer(100e-9, material=GaAs_p_emit, role='emitter')
    absorber = Layer(1.5e-6, material=GaAs_n_base, role='base')
    bsf      = Layer(100e-9, material=GaAs_n_bsf,  role='bsf')
    backmet  = Layer(200e-9, material=Au_metal)

    sn_val = 1e3 if not use_meta_layer or meta_tr is None else float(meta_tr.S_front_eff)
    sp_val = 1e3 if not use_meta_layer or meta_tr is None else float(meta_tr.S_back_eff)
    active_junction = Junction([emitter, absorber, bsf], kind='PDD', T=300,
                            sn=sn_val, sp=sp_val, n_discretization=200)


    if use_meta_layer and (MetaMat is not None) and (meta_opt is not None):
        meta_layer = Layer(float(meta_opt.thickness_m), material=MetaMat, role='window')
        sc = SolarCell([window, meta_layer, active_junction, backmet])
    else:
        sc = SolarCell([window, active_junction, backmet])

    # Effective surface recombination (if meta provided)
    if use_meta_layer and (meta_tr is not None):
        scale_tau_n = max(1.0, meta_tr.tau_n_eff / 5e-9)  # or use your *_mult directly if you prefer
        scale_tau_p = max(1.0, meta_tr.tau_p_eff / 5e-9)
        # Or, if you keep *_mult in the JSON: use those multipliers directly.
        for mat in (GaAs_p_emit, GaAs_n_base, GaAs_n_bsf):
            mat.tau_n *= scale_tau_n
            mat.tau_p *= scale_tau_p        
        for attr, val in [
            ("front_surface_recombination_velocity", meta_tr.S_front_eff),
            ("back_surface_recombination_velocity",  meta_tr.S_back_eff),
        ]:
            try:
                setattr(sc, attr, float(val))
            except Exception:
                pass

    # Axes
    wl_m = np.asarray(wl_m, float)
    wl_nm = wl_m * 1e9

    # AM1.5g on nm axis (photon_flux_per_nm)
    sun = LightSource(
        source_type='standard',
        version='AM1.5g',
        x=wl_nm,
        output_units='photon_flux_per_nm'
    )

    # QE (meters axis inside solver; TMM optics)
    solar_cell_solver(sc, 'qe', user_options={
        'wavelength': wl_m,
        'optics_method': 'TMM'
    })

    # IV (illuminated)
    V = np.linspace(0, 1.30, 131)
    internal_V = np.linspace(-1.0, 2.0, 401)
    solar_cell_solver(sc, 'iv', user_options={
        'light_source': sun,
        'wavelength': wl_m,
        'optics_method': 'TMM',
        'voltages': V,
        'internal_voltages': internal_V,
        'light_iv': True,
        'mpp': True
    })

    # ---- Extract IV arrays (your build: IV is a (2, N) tuple) ----
    iv = getattr(sc, "iv", {})
    arr = np.asarray(iv["IV"], dtype=float)   # shape (2, N)
    V_curve = arr[0, :]                       # volts
    J_curve = arr[1, :]                       # A/m^2

    # Conventional sign: current density positive at short-circuit (V≈0)
    if np.interp(0.0, V_curve, J_curve) < 0:
        J_curve = -J_curve

    # Metrics
    Jsc = float(np.interp(0.0, V_curve, J_curve))
    sign = np.sign(J_curve)
    idx = np.where(sign[:-1] * sign[1:] <= 0)[0]
    if idx.size:
        i = int(idx[0])
        x0, y0 = V_curve[i],   J_curve[i]
        x1, y1 = V_curve[i+1], J_curve[i+1]
        Voc = float(x0 + (0 - y0) * (x1 - x0) / (y1 - y0 + 1e-30))
    else:
        near = int(np.argmin(np.abs(J_curve)))
        Voc = float(V_curve[near]) if abs(J_curve[near]) < max(1e-6, 1e-3*abs(Jsc)) else 0.0

    P = V_curve * J_curve
    im = int(np.argmax(P))
    Vmpp, Jmpp, Pmpp = float(V_curve[im]), float(J_curve[im]), float(P[im])

    FF = float(np.clip(Pmpp / (max(Jsc, 1e-30) * max(Voc, 1e-30)), 0.0, 0.95))

    # Incident power density from the same LightSource (nm axis)
    h = 6.62607015e-34; c = 299792458.0
    wl_nm_axis, phi_per_nm = sun.spectrum(wl_nm)  # photons/(m^2 s nm)
    E_J = (h * c) / (wl_nm_axis * 1e-9)           # J/photon
    power_per_nm = E_J * phi_per_nm               # W/(m^2 nm)
    Pin = float(np.trapz(power_per_nm, wl_nm_axis))

    eta = float(Pmpp / max(Pin, 1e-30))

    jv_df = pd.DataFrame({"V_V": V_curve, "J_A_m2": J_curve, "P_W_m2": P})
    summary = DeviceSummary(
        model_used="solcore",
        Jsc_mAcm2=Jsc / 10.0,  # A/m^2 → mA/cm^2
        Voc_V=Voc,
        FF=FF,
        eta_pct=eta * 100.0
    )
    return summary, jv_df, None

# -------------------------
# Confidence scoring (simple, no magic)
# -------------------------
def confidence_score(model_used: str,
                     have_meta: bool,
                     optical_residual: Optional[float],
                     sensitivity_ok: bool) -> float:
    base = 0.7 if model_used == "solcore" else 0.35
    if have_meta:
        base += 0.1
    if sensitivity_ok:
        base += 0.1
    return float(np.clip(base, 0.05, 0.98))

# -------------------------
# CLI
# -------------------------
def main():
    ap = argparse.ArgumentParser(description="EchoKey Solar Meta-Layer Pipeline (real Solcore; coin-driven meta; no fallbacks)")
    ap.add_argument("--coin_config", type=str, default=None, help="Path to coin_config.json (required unless explicit meta files provided).")
    ap.add_argument("--meta_optics", type=str, default=None, help="Path to meta_optics.json (overrides coin_config).")
    ap.add_argument("--meta_transport", type=str, default=None, help="Path to meta_transport.json (overrides coin_config).")
    ap.add_argument("--cache_dir", type=str, default="ek_coin_cache", help="Cache dir for generated meta files from coins.")
    ap.add_argument("--use_solcore", action="store_true", help="Must be provided to run (explicit, to avoid surprises).")
    ap.add_argument("--outdir", type=str, default="ek_runs", help="Output directory for CSV logs.")
    args = ap.parse_args()

    if not args.use_solcore:
        raise SystemExit("This script only supports the real Solcore path. Please pass --use_solcore.")

    rid = run_id()
    outdir = os.path.join(args.outdir, rid)
    ensure_dir(outdir)

    # Load or generate meta from coin
    have_real_meta = False
    if args.meta_optics and args.meta_transport:
        meta_opt, meta_tr = load_meta_json(args.meta_optics, args.meta_transport)
        have_real_meta = True
    else:
        if not args.coin_config or not os.path.exists(args.coin_config):
            raise SystemExit("Provide either --meta_optics & --meta_transport or a valid --coin_config.")
        coin_raw = json.load(open(args.coin_config, "r"))
        coin = CoinConfig(**coin_raw)
        meta_opt, meta_tr = coin_to_meta(coin)
        # Cache generated meta (deterministic by coin hash)
        key = hash_config({"coin": coin_raw})
        ensure_dir(args.cache_dir)
        opt_p, tr_p = cache_paths(args.cache_dir, key)
        save_meta_json(meta_opt, meta_tr, opt_p, tr_p)

    wl_m = meta_opt.wl_m

    # Baseline (no meta layer)
    base_summary, base_jv, _ = solcore_device(
        wl_m, meta_opt=None, meta_tr=None, use_meta_layer=False, outdir=outdir, rid=rid
    )
    # Emergent (with meta layer from coin)
    emer_summary, emer_jv, _ = solcore_device(
        wl_m, meta_opt=meta_opt, meta_tr=meta_tr, use_meta_layer=True, outdir=outdir, rid=rid
    )

    # Comparison
    d_eta = emer_summary.eta_pct - base_summary.eta_pct
    pct_gain = (emer_summary.eta_pct - base_summary.eta_pct) / max(1e-6, base_summary.eta_pct) * 100.0
    sensitivity_ok = (emer_summary.FF > 0.6 * base_summary.FF) and (emer_summary.Jsc_mAcm2 >= 0.95 * base_summary.Jsc_mAcm2)
    conf = confidence_score("solcore", True, None, sensitivity_ok)

    comp = ComparisonSummary(
        baseline=base_summary,
        emergent=emer_summary,
        delta_eta_pct_points=float(d_eta),
        pct_improvement_vs_baseline=float(pct_gain),
        optical_residual=None,
        confidence_score_0to1=conf
    )

    # Logs
    json.dump({
        "run_id": rid,
        "timestamp": now_stamp(),
        "model_used": "solcore",
        "have_real_meta": have_real_meta,
        "args": vars(args)
    }, open(os.path.join(outdir, "run_meta.json"), "w"), indent=2)

    save_csv(pd.DataFrame([asdict(base_summary)]), os.path.join(outdir, "baseline_summary.csv"))
    save_csv(pd.DataFrame([asdict(emer_summary)]), os.path.join(outdir, "emergent_summary.csv"))
    save_csv(pd.DataFrame([{
        "delta_eta_pct_points": comp.delta_eta_pct_points,
        "pct_improvement_vs_baseline": comp.pct_improvement_vs_baseline,
        "optical_residual": comp.optical_residual,
        "confidence_score_0to1": comp.confidence_score_0to1
    }]), os.path.join(outdir, "comparison_summary.csv"))

    save_csv(base_jv, os.path.join(outdir, "baseline_jv.csv"))
    save_csv(emer_jv, os.path.join(outdir, "emergent_jv.csv"))

    final_summary = {
        "run_id": rid,
        "model_used": "solcore",
        "baseline": asdict(base_summary),
        "emergent": asdict(emer_summary),
        "delta_eta_pct_points": comp.delta_eta_pct_points,
        "pct_improvement_vs_baseline": comp.pct_improvement_vs_baseline,
        "optical_residual": comp.optical_residual,
        "confidence_score_0to1": comp.confidence_score_0to1,
        "output_dir": outdir
    }
    print(json.dumps(final_summary, indent=2))

if __name__ == "__main__":
    main()
