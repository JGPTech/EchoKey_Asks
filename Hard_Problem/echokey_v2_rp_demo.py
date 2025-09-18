#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EchoKey v2 · Empirical RP/LRP + Vectorized Accumulator + Data-Driven Summary
(FINAL: non-interactive; saves figures; prints only measured results)

Upgrades in this version:
  • Accumulator fixed params that yielded crossings:
      mu=0.40, theta=0.20, T_max=12.0, tau=0.30, sigma=0.08, dt=0.01, T_pre=2.0
  • Refraction uses stronger tilt eps=0.20 and fills NaNs in the synergy series
    before FFT so correlation is computed on the RP window.
  • Fractality uses levels=5 and wavelet="sym8" (fallback to "db4" if missing).
  • Onset detector runs only over pre-cue times (t < 0) and checks negativity.

Requirements:
  pip install numpy scipy mne matplotlib pywavelets
"""

# --- non-interactive plotting (save-only) ---
import matplotlib
matplotlib.use("Agg")

import sys, json, csv
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import mne
from mne.datasets import eegbci
from pathlib import Path

# Optional: multiscale analysis
try:
    import pywt
    HAVE_PYWT = True
except Exception:
    HAVE_PYWT = False

# -----------------------
# paths / helpers
# -----------------------
FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)
DATA_DIR = Path("data_physionet")
DATA_DIR.mkdir(exist_ok=True)

def savefig(fname):
    path = FIG_DIR / fname
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[FIG] {path}")

# -----------------------
# Q0/Q1: fetch + ERP math
# -----------------------
subject = 1
runs = [3, 4]
try:
    edf_paths = eegbci.load_data(subject, runs, path=str(DATA_DIR))  # local EDFs
except Exception as e:
    print(f"[ERROR] EEGBCI download failed: {e}")
    sys.exit(1)

raws = []
for p in edf_paths:
    raw = mne.io.read_raw_edf(p, preload=True, verbose="ERROR")
    eegbci.standardize(raw)
    try:
        raw.set_montage("standard_1020", match_case=False, on_missing="warn")
    except Exception:
        pass
    raw.resample(256., npad="auto")
    raws.append(raw)

def safe_pick_ci(raw, names):
    name_map = {nm.lower(): nm for nm in raw.info["ch_names"]}
    picks = []
    for want in names:
        nm = name_map.get(want.lower())
        if nm is not None:
            picks.append(raw.ch_names.index(nm))
    return sorted(set(picks))

events_all, event_id = [], {}
for r in raws:
    ev, eid = mne.events_from_annotations(r, verbose=False)
    for k, v in eid.items():
        event_id.setdefault(k, v)
    events_all.append((r, ev))

wanted = [k for k in ("T1","T2") if k in event_id] or [k for k in event_id if k != "T0"]
print(f"[INFO] events used: {wanted}")

tmin, tmax = -2.0, 0.5
baseline = (-1.5, -1.0)

epochs_list = []
for raw, ev in events_all:
    id_map = {k: event_id[k] for k in wanted}
    pick_idxs = safe_pick_ci(raw, ["Cz","C3","C4"])
    ep = mne.Epochs(raw, ev, event_id=id_map, tmin=tmin, tmax=tmax,
                    picks=pick_idxs if pick_idxs else "eeg",
                    baseline=baseline, preload=True, reject=None, verbose=False)
    epochs_list.append(ep)

if not epochs_list:
    print("[ERROR] No epochs created.")
    sys.exit(1)

epochs = mne.concatenate_epochs(epochs_list)
evoked_by = {k: epochs[k].average() for k in wanted}

def get_channel(evoked, ch_name):
    if ch_name in evoked.ch_names:
        return evoked.copy().pick(ch_name)
    up = ch_name.upper()
    if up in evoked.ch_names:
        return evoked.copy().pick(up)
    return None

cz_evokeds = {k: get_channel(v, "Cz") for k,v in evoked_by.items()}

def compute_lrp(ep: mne.Epochs, left_label="T1", right_label="T2"):
    have_c3 = any(ch.upper()=="C3" for ch in ep.ch_names)
    have_c4 = any(ch.upper()=="C4" for ch in ep.ch_names)
    if not (have_c3 and have_c4):
        return None
    def pick_name(ep, want):
        for nm in ep.ch_names:
            if nm.upper()==want:
                return nm
        return None
    nmC3, nmC4 = pick_name(ep,"C3"), pick_name(ep,"C4")
    X = []
    times = ep.times
    for lab in (left_label,right_label):
        if lab not in ep.event_id:
            continue
        dat = ep[lab].get_data(picks=[nmC3,nmC4])  # (n,2,T)
        if dat.size==0:
            continue
        if lab==right_label:
            contra, ipsi = dat[:,0,:], dat[:,1,:]  # C3 vs C4
        else:
            contra, ipsi = dat[:,1,:], dat[:,0,:]  # C4 vs C3
        X.append(contra - ipsi)
    if not X:
        return None
    X = np.concatenate(X, axis=0)
    return times, X.mean(axis=0)

lrp = compute_lrp(epochs)

# figure: empirical Cz/LRP
plt.figure(figsize=(9,4))
for k, evk in cz_evokeds.items():
    if evk is not None:
        plt.plot(evk.times, evk.data[0]*1e6, label=f"Cz:{k}")
if lrp is not None:
    times, lrp_sig = lrp
    plt.plot(times, lrp_sig*1e6, "--", alpha=0.8, label="LRP")
plt.axvline(0, color="k", lw=1)
plt.title("Empirical RP/LRP (µV)")
plt.xlabel("Time (s) [0=cue]"); plt.ylabel("µV"); plt.legend()
plt.tight_layout()
savefig("q1_empirical_rp_lrp.png")

# -----------------------
# Q2: vectorized accumulator (fixed params per user request)
# -----------------------
def simulate_accumulator(N=600, dt=0.01, T_pre=2.0, T_max=12.0, tau=0.30, mu=0.40, sigma=0.08, theta=0.20, rng=None):
    if rng is None:
        rng = np.random.default_rng(1337)
    steps = int(T_max/dt)
    steps_pre = int(T_pre/dt)
    v = np.zeros((steps, N), dtype=float)
    cross_idx = np.full(N, -1, dtype=int)
    noise = rng.standard_normal(size=(steps-1, N))
    for t in range(1, steps):
        dv = (-(v[t-1]/tau) + mu)*dt + sigma*np.sqrt(dt)*noise[t-1]
        v[t] = v[t-1] + dv
        mask_cross = (cross_idx < 0) & (v[t] >= theta)
        cross_idx[mask_cross] = t
        if np.all(cross_idx >= 0):
            break
    valid = (cross_idx > 0)
    n_valid = int(valid.sum())
    accum = np.full((n_valid, steps_pre), np.nan, dtype=float)
    if n_valid > 0:
        vi = np.where(valid)[0]
        row = 0
        for i in vi:
            ci = cross_idx[i]
            start = max(0, ci - steps_pre)
            seg = v[start:ci, i]
            if len(seg) > 0:
                accum[row, -len(seg):] = seg
            row += 1
    t_pre_axis = -np.arange(steps_pre)[::-1]*dt
    cross_rate = float(valid.mean())
    return accum, t_pre_axis, cross_rate, steps_pre

N, dt, T_pre, tau, sigma, theta, mu, T_max = 600, 0.01, 2.0, 0.30, 0.08, 0.20, 0.40, 12.0
rng = np.random.default_rng(1337)
accum, t_pre_axis, cross_rate, steps_pre = simulate_accumulator(
    N=N, dt=dt, T_pre=T_pre, T_max=T_max, tau=tau, mu=mu, sigma=sigma, theta=theta, rng=rng
)

if accum.size > 0 and np.isfinite(accum).any():
    accum_avg = np.nanmean(accum, axis=0)
    n_trials_used = int(np.sum(~np.all(~np.isfinite(accum), axis=1)))
else:
    accum_avg = np.full(int(T_pre/dt), np.nan, dtype=float)
    n_trials_used = 0

plt.figure(figsize=(9,4))
plt.plot(t_pre_axis, accum_avg, label="time-locked avg")
plt.axvline(0, color="k", lw=1)
plt.title("Simulated 'RP' (vectorized accumulator)")
plt.xlabel("Time (s) [0=threshold]"); plt.ylabel("state (a.u.)"); plt.legend()
plt.tight_layout()
savefig("q2_simulated_rp.png")

print(f"[INFO] accumulator_cross_rate: {cross_rate:.3f}  trials_used: {n_trials_used}  mu: {mu:.3f}  T_max: {T_max:.1f}")

# -----------------------
# Q3: transforms + data-driven summary
# -----------------------
def cyclicity_bandlimit(x, fs, fmax=30.0):
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), 1/fs)
    X[freqs>fmax] = 0
    return np.fft.irfft(X, n=len(x))

def recursion_fixedpoint(x, lam=0.2, iters=20):
    ker = np.ones(9)/9.0
    y = x.copy()
    for _ in range(iters):
        y = (1-lam)*y + lam*signal.convolve(y, ker, mode="same")
    return y

def fractality_multiscale_energy(x, wavelet="sym8", levels=5):
    if not HAVE_PYWT:
        return None, None
    if wavelet not in pywt.wavelist():
        wavelet = "db4"
    coeffs = pywt.wavedec(x, wavelet, level=levels, mode="periodization")
    energies = [float(np.sum(c**2)) for c in coeffs]
    return coeffs, energies

def regression_mean_revert(x, lamb=1.5, fs=256.0):
    alpha = lamb/fs
    y = np.zeros_like(x)
    y[0] = x[0]
    for n in range(1,len(x)):
        y[n] = y[n-1] + (x[n]-y[n-1]) - alpha*y[n-1]
    return y

def synergy_mix(x, y, w=0.5):
    return (1-w)*x + w*y

def _interp_fill_nan(y, t):
    """Fill NaNs in y by linear interp over finite points; keep NaN outside coverage."""
    y = np.asarray(y, float)
    ok = np.isfinite(y)
    if ok.sum() >= 2:
        y = np.interp(t, t[ok], y[ok], left=np.nan, right=np.nan)
    return y

def refraction_spectral_shift(x, eps=0.20):
    """Spectral tilt after filling NaNs; returns array with same length."""
    x = np.asarray(x, float)
    if not np.isfinite(x).any():
        return np.copy(x)
    # Fill NaNs before FFT to avoid NaN propagation
    t = np.arange(len(x), dtype=float)
    x_filled = _interp_fill_nan(x, t)
    X = np.fft.rfft(x_filled)
    freqs = np.fft.rfftfreq(len(x_filled), d=1.0)
    if freqs.max() > 0:
        X = X*(1 + eps*(freqs/freqs.max()))
    y = np.fft.irfft(X, n=len(x_filled))
    # Preserve NaNs where the original had no support (outside convex hull of finite points)
    first = np.argmax(np.isfinite(x))
    last  = len(x)-1-np.argmax(np.isfinite(x[::-1]))
    y[:first] = np.nan
    y[last+1:] = np.nan
    return y

def refraction_spectral_shift_window(x, t, mask, eps=0.20):
    """
    Spectral tilt applied only inside `mask`. Fills NaNs within the window,
    leaves outside-window samples untouched. Returns y of same shape as x.
    """
    x = np.asarray(x, float)
    t = np.asarray(t, float)
    y = np.array(x, copy=True)

    w = np.asarray(mask, bool)
    if w.sum() < 4:  # not enough samples to FFT meaningfully
        return y

    # fill NaNs *inside* the window only
    xw = x[w].astype(float, copy=True)
    tw = t[w]
    ok = np.isfinite(xw)
    if ok.sum() >= 2:
        xw = np.interp(tw, tw[ok], xw[ok], left=xw[ok][0], right=xw[ok][-1])
    else:
        # if still not enough, bail without changing this window
        return y

    # tilt inside the window (unit sample period is fine for relative tilt)
    X = np.fft.rfft(xw)
    freqs = np.fft.rfftfreq(len(xw), d=1.0)
    if freqs.max() > 0:
        X = X * (1.0 + eps * (freqs / freqs.max()))
    yw = np.fft.irfft(X, n=len(xw))

    # write back only the window segment
    y[w] = yw
    return y

def window_corr(a, b, mask, min_samples=3):
    a = np.asarray(a, float)[mask]
    b = np.asarray(b, float)[mask]
    ok = np.isfinite(a) & np.isfinite(b)
    n_ok = int(ok.sum())
    if n_ok < min_samples:
        # print or log if you want: not enough overlapping finite points
        return np.nan
    ax = a[ok] - np.nanmean(a[ok])
    bx = b[ok] - np.nanmean(b[ok])
    sa = np.nanstd(ax)
    sb = np.nanstd(bx)
    if sa <= 1e-12 or sb <= 1e-12:
        # one vector is (near-)constant in the window
        return np.nan
    r = float(np.dot(ax, bx) / (sa * sb * (n_ok - 1)))
    return max(min(r, 1.0), -1.0)

def downweight_outliers(trials, z=3.0, out_len=None):
    """Robust mean over trials; if trials empty, return NaN vector of length out_len."""
    if trials.size == 0 or not np.isfinite(trials).any():
        if out_len is None:
            return np.array([], dtype=float)
        return np.full(out_len, np.nan, dtype=float)
    m = np.nanmean(trials, axis=0)
    s = np.nanstd(trials, axis=0) + 1e-8
    Z = (trials - m)/s
    W = np.where(np.abs(Z)>z, 0.0, 1.0)
    denom = np.nansum(W, axis=0) + 1e-8
    return np.nansum(trials*W, axis=0) / denom

# empirical series selection
fs_emp = 64.0
emp_sig, emp_time = None, None
for k, evk in cz_evokeds.items():
    if evk is not None:
        emp_sig = evk.data[0]  # volts
        emp_time = evk.times
        break
if emp_sig is None and lrp is not None:
    emp_time, emp_sig = lrp[0], lrp[1]  # volts

if emp_sig is not None:
    emp_uV = emp_sig*-1e6
    emp_cyc = cyclicity_bandlimit(emp_uV, fs=fs_emp, fmax=30.0)
    emp_rec = recursion_fixedpoint(emp_cyc, lam=0.2, iters=30)
    emp_reg = regression_mean_revert(emp_rec, lamb=1.0, fs=fs_emp)

    # overlap with accumulator (for synergy)
    mask_overlap = (emp_time >= -float(T_pre)) & (emp_time <= 0.0)
    have_accum = (n_trials_used > 0) and np.isfinite(accum_avg).any()

    if np.any(mask_overlap) and have_accum:
        emp_t_cut = emp_time[mask_overlap]
        acc_interp = np.interp(emp_t_cut, t_pre_axis, accum_avg, left=np.nan, right=np.nan)
        emp_cut = emp_reg[mask_overlap]
        def z(x): 
            mu = np.nanmean(x); sd = np.nanstd(x) + 1e-8
            return (x - mu)/sd if np.isfinite(mu) and sd>0 else np.full_like(x, np.nan)
        mix_short = synergy_mix(z(emp_cut), z(acc_interp), w=0.5)

        # embed to full emp_time and interpolate inside the overlap
        def interp_fill_to_emp_time(y_short, mask_bool, t_full):
            y_full = np.full_like(t_full, np.nan, dtype=float)
            y_full[mask_bool] = y_short
            ok = np.isfinite(y_full)
            if ok.sum() >= 2:
                y_full = np.interp(t_full, t_full[ok], y_full[ok], left=np.nan, right=np.nan)
            return y_full
        mix_full = interp_fill_to_emp_time(mix_short, mask_overlap, emp_time)
    else:
        # fall back to a normalized empirical-only proxy (keeps the pipeline running)
        mu_ = np.nanmean(emp_reg); sd_ = np.nanstd(emp_reg) + 1e-8
        mix_full = (emp_reg - mu_)/sd_ if np.isfinite(mu_) and sd_>0 else np.full_like(emp_reg, np.nan)

    # --- Refraction with eps=0.20 and NaN-safe handling ---
    mix_full = _interp_fill_nan(mix_full, emp_time)   # fill before spectral tilt
    mix_ref_full = refraction_spectral_shift(mix_full, eps=0.20)

    # robust accumulator average with safe output length
    acc_robust = downweight_outliers(accum if have_accum else np.empty((0,)),
                                     z=3.0, out_len=accum_avg.shape[0])

    # transforms grid fig
    fig, axes = plt.subplots(2,3, figsize=(12,6), sharex=False)
    axes = axes.ravel()
    axes[0].plot(emp_time, emp_uV); axes[0].axvline(0,color='k',lw=1); axes[0].set_title("Empirical (µV)")
    axes[1].plot(emp_time, emp_cyc); axes[1].axvline(0,color='k',lw=1); axes[1].set_title("Cyclicity")
    axes[2].plot(emp_time, emp_rec); axes[2].axvline(0,color='k',lw=1); axes[2].set_title("Recursion")
    if np.any(mask_overlap) and have_accum:
        axes[3].plot(emp_t_cut, emp_cut); axes[3].plot(emp_t_cut, acc_interp, alpha=0.8)
        axes[3].axvline(0,color='k',lw=1); axes[3].legend(["Emp(reg)","Sim(accum)"], fontsize=8)
        axes[3].set_title("Inputs for Synergy")
    else:
        axes[3].axis('off')
    axes[4].plot(emp_time, mix_full); axes[4].axvline(0,color='k',lw=1); axes[4].set_title("Synergy (full)")
    axes[5].plot(t_pre_axis, accum_avg, label="mean")
    if np.isfinite(acc_robust).any():
        axes[5].plot(t_pre_axis, acc_robust, "--", label="robust")
    axes[5].axvline(0,color='k',lw=1); axes[5].legend(fontsize=8); axes[5].set_title("Accumulator")
    plt.tight_layout()
    savefig("q3_transforms_grid.png")

    # wavelet energies fig (levels=5, wavelet="sym8" fallback to "db4")
    energies = None
    if HAVE_PYWT:
        _, energies = fractality_multiscale_energy(emp_rec, wavelet="sym8", levels=5)
        if energies is not None:
            plt.figure(figsize=(6,3))
            xs = np.arange(len(energies))
            markerline, stemlines, baseline = plt.stem(xs, energies)
            plt.setp(stemlines, linewidth=1)
            plt.setp(markerline, markersize=4)
            plt.title("Fractality: multiscale energies (levels=5)")
            plt.xlabel("Wavelet level (0=finest)"); plt.ylabel("Energy (a.u.)")
            plt.tight_layout()
            savefig("q3_fractality_energies.png")

    # -----------------------
    # data-driven summary (results only)
    # -----------------------
    def mask_window(t,a,b): return (t>=a) & (t<=b)
    base_win = (-1.5,-1.0)
    rp_win = (-0.5, -0.1)
    rm = (emp_time >= rp_win[0]) & (emp_time <= rp_win[1])

    # ensure mix_full is finite (global fill once)
    mix_full = _interp_fill_nan(mix_full, emp_time)

    # refraction *only* in the RP window; leaves outside unchanged
    mix_ref_full = refraction_spectral_shift_window(mix_full, emp_time, rm, eps=0.20)

    bm = mask_window(emp_time, *base_win)
    rm = mask_window(emp_time, *rp_win)

    def safe_corr(a, b, min_samples: int = 8):
        """
        Pearson r with strict NaN masking and zero-variance guard.
        Returns np.nan if fewer than min_samples valid pairs or if either
        vector is (near-)constant over the valid indices.
        """
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        ok = np.isfinite(a) & np.isfinite(b)
        n_ok = int(ok.sum())
        if n_ok < min_samples:
            return np.nan
        ax = a[ok]
        bx = b[ok]
        # de-mean
        ax -= ax.mean()
        bx -= bx.mean()
        sa = ax.std()
        sb = bx.std()
        if sa <= 1e-12 or sb <= 1e-12:  # zero/near-zero variance → undefined r
            return np.nan
        r = float(np.dot(ax, bx) / (sa * sb * (n_ok - 1)))
        # Numerical guard in case rounding nudges |r|>1 slightly
        if r > 1.0:  r = 1.0
        if r < -1.0: r = -1.0
        return r

    def rmse(a, b, require_same_shape: bool = True):
        """
        RMSE with strict NaN masking.
        If require_same_shape is True and shapes differ → np.nan.
        Otherwise, will try to flatten and align by length=min(len(a), len(b)).
        """
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        if require_same_shape and a.shape != b.shape:
            return np.nan

        if a.shape != b.shape:
            # permissive fallback: compare along the first axis after flatten
            af = a.ravel()
            bf = b.ravel()
            m = min(len(af), len(bf))
            af = af[:m]
            bf = bf[:m]
        else:
            af = a
            bf = b

        ok = np.isfinite(af) & np.isfinite(bf)
        if not np.any(ok):
            return np.nan
        diff2 = (af[ok] - bf[ok]) ** 2
        return float(np.sqrt(diff2.mean()))

    summary = {}
    summary["emp_baseline_mean_uV"] = float(np.nanmean(emp_uV[bm]))
    summary["emp_baseline_std_uV"]  = float(np.nanstd(emp_uV[bm]))
    summary["emp_rp_mean_uV"]       = float(np.nanmean(emp_uV[rm]))
    summary["emp_rp_std_uV"]        = float(np.nanstd(emp_uV[rm]))
    summary["emp_rp_minus_base_mean_uV"] = summary["emp_rp_mean_uV"] - summary["emp_baseline_mean_uV"]

    summary["corr_emp_vs_cyclicity"]  = safe_corr(emp_uV, emp_cyc)
    summary["corr_emp_vs_recursion"]  = safe_corr(emp_uV, emp_rec)
    summary["corr_emp_vs_regression"] = safe_corr(emp_uV, emp_reg)

    summary["cyc_rp_mean_uV"] = float(np.nanmean(emp_cyc[rm]))
    summary["rec_rp_mean_uV"] = float(np.nanmean(emp_rec[rm]))
    summary["reg_rp_mean_uV"] = float(np.nanmean(emp_reg[rm]))

    if np.any(mask_overlap) and (n_trials_used > 0) and np.isfinite(accum_avg).any():
        # compare overlap (z-scored) between empirical regressed cut and accumulator interp
        emp_t_cut = emp_time[mask_overlap]
        acc_interp = np.interp(emp_t_cut, t_pre_axis, accum_avg, left=np.nan, right=np.nan)
        emp_cut = emp_reg[mask_overlap]
        def zscore(x): 
            mu = np.nanmean(x); sd = np.nanstd(x) + 1e-8
            return (x - mu)/sd if np.isfinite(mu) and sd>0 else np.full_like(x, np.nan)
        summary["corr_empReg_vs_accum_in_overlap"] = safe_corr(zscore(emp_cut), zscore(acc_interp))
    else:
        summary["corr_empReg_vs_accum_in_overlap"] = np.nan

    # correlate refraction output vs empirical regression on a common RP window (both on emp_time)
    summary["corr_emp_vs_refraction"] = window_corr(emp_reg, mix_ref_full, rm)

    # robust vs mean accumulator (shape-safe)
    summary["accum_robust_vs_mean_rmse"] = rmse(acc_robust, accum_avg)

    if energies is not None and len(energies)>0:
        totalE = float(np.sum(energies))
        coarseE = float(np.sum(energies[-2:])) if len(energies)>=2 else float(energies[-1])
        summary["wavelet_total_energy"]   = totalE
        summary["wavelet_coarse_energy"]  = coarseE
        summary["wavelet_coarse_fraction"]= (coarseE/totalE) if totalE>0 else np.nan
    else:
        summary["wavelet_total_energy"] = np.nan
        summary["wavelet_coarse_energy"] = np.nan
        summary["wavelet_coarse_fraction"] = np.nan

    # files
    json_path = FIG_DIR / "q3_summary.json"
    csv_path  = FIG_DIR / "q3_summary.csv"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["metric","value"])
        for k,v in summary.items():
            w.writerow([k, v])

    # console: only actual results
    print("\n[Q3 summary]")
    for k,v in summary.items():
        if isinstance(v, float) and np.isfinite(v):
            print(f"{k}: {v:.6f}")
        else:
            print(f"{k}: {v}")
    print(f"[FILE] {json_path}")
    print(f"[FILE] {csv_path}")

# -----------------------
# Q4: onset heuristic (pre-cue only; negativity)
# -----------------------
def onset_latency_precue_negative(x, t, baseline_win=(-1.5,-1.0), z_thresh=2.0, fs=256.0):
    # compute z vs baseline
    bmask = (t>=baseline_win[0]) & (t<=baseline_win[1])
    mu, sd = np.mean(x[bmask]), np.std(x[bmask]) + 1e-8
    z = (x - mu)/sd
    # restrict to pre-cue only
    pre_mask = (t < 0.0)
    z_pre = z[pre_mask]
    t_pre = t[pre_mask]
    cross = np.where(z_pre < -z_thresh)[0]  # look for negative
    if len(cross)==0:
        return None
    win = int(0.1*fs)  # 100 ms sustained
    for idx in cross:
        if idx+win < len(z_pre) and np.all(z_pre[idx:idx+win] < -z_thresh):
            return t_pre[idx]
    return None

if 'emp_sig' in locals() and emp_sig is not None:
    onset = onset_latency_precue_negative(emp_uV, emp_time, fs=fs_emp)
    if onset is not None:
        print(f"\n[Q4 onset_precue_s (neg)]: {onset:.6f}")
    else:
        print("\n[Q4 onset_precue_s (neg)]: None")
