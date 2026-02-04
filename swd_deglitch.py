# -*- coding: utf-8 -*-
"""
Created on Tue Feb  3 21:30:54 2026

@author: Andong Lu
"""

# swd_deglitch.py
# Windowed SWD glitch+spike modelling and subtraction.
#
# Outputs written by the driver (run_deglitch.py):
#   - deglitched_data.mseed : cleaned traces y(t) = x(t) - g(t)
#   - output/<CH>/glitches_time.txt : merged glitch intervals (UTC)

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from obspy import Stream, Trace, UTCDateTime, read

from SWTools import SWdecomp


# ----------------------------
# User-facing parameters
# ----------------------------
@dataclass
class DeglitchConfig:
    # Windowing
    window_len: float = 600.0
    overlap: float = 100.0
    edge_guard: float = 50.0  # ignore detections close to window edges

    # Selection criteria
    dur_criterion: float = 50.0   # seconds
    amp_criterion: float = 5.0    # MAD multiplier

    # SWD settings
    max_iter_glitch: int = 10
    max_iter_spike: int = 1
    err_tol: float = 0.1

    # Energy-based duration bounds (fraction of cumulative energy)
    energy_lo: float = 0.0005
    energy_hi: float = 0.9995

    # Spike search padding around each detected glitch (seconds)
    spike_pad_before: float = 1.0
    spike_pad_after: float = 1.0
    spike_mask_halfwidth: float = 1.0  # seconds masked around current max spike


# ----------------------------
# Helpers
# ----------------------------
def _mad_threshold(x: np.ndarray, k: float) -> float:
    """Robust amplitude threshold: median + k*MAD."""
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return med + k * mad


def _combine_with_overlap(base: np.ndarray, new: np.ndarray, overlap_samples: int) -> np.ndarray:
    """
    Stitch two adjacent window-series assuming (overlap_samples + 1) samples overlap.
    This preserves the behaviour of your original combine_with_overlap.
    """
    result_length = len(base) + len(new) - (overlap_samples + 1)
    out = np.zeros(result_length, dtype=float)
    out[: len(base)] += base
    out[-len(new) :] += new
    return out


def _duration_energy(
    comps: np.ndarray,
    t: np.ndarray,
    window_start: UTCDateTime,
    energy_lo: float,
    energy_hi: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate per-component duration using cumulative energy bounds.
    Returns:
      durations (s), start_times (UTCDateTime), end_times (UTCDateTime)
    """
    ncomp = comps.shape[1]
    durations = np.zeros(ncomp, dtype=float)
    start_times = np.empty(ncomp, dtype=object)
    end_times = np.empty(ncomp, dtype=object)

    for i in range(ncomp):
        sig = comps[:, i].astype(float)
        peak = np.max(np.abs(sig))
        if peak == 0:
            start_times[i] = window_start
            end_times[i] = window_start
            durations[i] = 0.0
            continue

        sig = sig / peak
        energy = sig**2
        cum = np.cumsum(energy)
        total = cum[-1] if cum[-1] != 0 else 1.0
        norm = cum / total

        s_idx = int(np.searchsorted(norm, energy_lo))
        e_idx = int(np.searchsorted(norm, energy_hi))
        s_idx = np.clip(s_idx, 0, len(t) - 1)
        e_idx = np.clip(e_idx, 0, len(t) - 1)

        ts = float(t[s_idx])
        te = float(t[e_idx])

        start_times[i] = window_start + ts
        end_times[i] = window_start + te
        durations[i] = te - ts

    return durations, start_times, end_times


def _merge_intervals(starts: np.ndarray, ends: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Merge overlapping [start, end] intervals (UTCDateTime arrays)."""
    if len(starts) == 0:
        return starts, ends

    order = np.argsort(starts)
    s = starts[order]
    e = ends[order]

    ms, me = [s[0]], [e[0]]
    for i in range(1, len(s)):
        if s[i] <= me[-1]:
            me[-1] = max(me[-1], e[i])
        else:
            ms.append(s[i])
            me.append(e[i])

    return np.asarray(ms, dtype=object), np.asarray(me, dtype=object)


# ----------------------------
# Core: process one Trace
# ----------------------------
def deglitch_trace_swd(tr: Trace, cfg: DeglitchConfig) -> Tuple[Trace, np.ndarray]:
    """
    Process one trace with windowed SWD.

    Returns:
      tr_clean  : deglitched trace y(t) = x(t) - g(t)
      intervals : (N, 2) array of ISO timestamps [start, end]
    """
    tr_raw = tr.copy()
    tr_clean = tr.copy()

    sr = float(tr.stats.sampling_rate)
    starttime = tr.stats.starttime
    endtime = tr.stats.endtime

    # Full-length removal series g(t) (stitched from windows)
    glitch_series = np.array([], dtype=float)

    glitch_starts: List[str] = []
    glitch_ends: List[str] = []

    nwin = 1
    win_s = starttime + (nwin - 1) * (cfg.window_len - cfg.overlap)

    while win_s <= endtime:
        win_e = win_s + cfg.window_len
        if win_e > endtime:
            win_e = endtime

        # Extract window
        tr_w = tr_raw.copy().trim(starttime=win_s, endtime=win_e)
        t = tr_w.times()
        x = tr_w.data.astype(float)

        # SWD decomposition (candidate glitch components)
        swd = SWdecomp(x, t, MaxIter=cfg.max_iter_glitch, ErrTol=cfg.err_tol, target_type="glitch")

        # Duration and absolute time bounds for all SW components
        durations, st_abs, et_abs = _duration_energy(swd.comps, t, win_s, cfg.energy_lo, cfg.energy_hi)

        # Duration criterion
        idx_dur = set(np.where(durations < cfg.dur_criterion)[0].tolist())

        # Core-of-window criterion: avoid edge artefacts
        idx_core = set(
            np.where(
                (st_abs + 0.5 * durations < win_e - cfg.edge_guard)
                & (et_abs - 0.5 * durations > win_s + cfg.edge_guard)
            )[0].tolist()
        )

        # Background estimate (long components) and amplitude criterion
        idx_bg = np.where(durations > cfg.window_len * 0.5)[0]
        background = np.sum(swd.comps[:, idx_bg], axis=1) if idx_bg.size > 0 else 0.0

        residual = x - np.mean(x) - background
        amp_thr = _mad_threshold(residual, cfg.amp_criterion)

        idx_amp = set(np.where(np.max(np.abs(swd.comps), axis=0) > amp_thr)[0].tolist())

        # Final glitch component indices
        indices = sorted(list(idx_dur & idx_core & idx_amp))

        g_comps = swd.comps[:, indices] if len(indices) > 0 else np.zeros((len(t), 0), dtype=float)
        g_st = np.asarray(st_abs[indices], dtype=object)
        g_et = np.asarray(et_abs[indices], dtype=object)

        # Spike modelling around each detected glitch region
        if len(indices) > 0:
            x_minus_glitch = x - (np.sum(g_comps, axis=1) if g_comps.shape[1] > 0 else 0.0)
            amp_bg = float(np.mean(x_minus_glitch))

            for i in range(len(g_st)):
                # seconds from window start
                sp_s = (g_st[i] - win_s) - cfg.spike_pad_before
                sp_e = (g_et[i] - win_s) + cfg.spike_pad_after
                sp_s = max(0.0, float(sp_s))
                sp_e = min(float(t[-1]), float(sp_e))

                x_sp = x_minus_glitch.copy()
                x_sp[: int(sp_s * sr)] = amp_bg
                x_sp[int(sp_e * sr) :] = amp_bg

                amp_x_sp = np.abs(x_sp - amp_bg)

                while np.max(amp_x_sp) > amp_thr:
                    kmax = int(np.argmax(amp_x_sp))
                    hw = int(cfg.spike_mask_halfwidth * sr)
                    i0 = max(0, kmax - hw)
                    i1 = min(len(x_sp), kmax + hw)

                    x_focus = x_sp.copy()
                    x_focus[:i0] = amp_bg
                    x_focus[i1:] = amp_bg

                    swd_s = SWdecomp(x_focus, t, MaxIter=cfg.max_iter_spike, ErrTol=cfg.err_tol, target_type="spike")
                    spike_comp = swd_s.comps.reshape(-1)

                    g_comps = np.hstack((g_comps, swd_s.comps))

                    x_sp = x_sp - spike_comp
                    amp_x_sp = np.abs(x_sp - amp_bg)

        # Record merged intervals for reporting (glitch + spikes)
        if g_comps.shape[1] > 0:
            _, st_all, et_all = _duration_energy(g_comps, t, win_s, cfg.energy_lo, cfg.energy_hi)
            st_m, et_m = _merge_intervals(np.asarray(st_all, dtype=object), np.asarray(et_all, dtype=object))
            glitch_starts.extend([tt.isoformat() for tt in st_m])
            glitch_ends.extend([tt.isoformat() for tt in et_m])

        # Window removal contribution g_win(t)
        g_win = np.sum(g_comps, axis=1) if g_comps.shape[1] > 0 else np.zeros(len(t), dtype=float)

        # Stitch into full-length glitch_series
        if glitch_series.size == 0:
            glitch_series = g_win.copy()
        else:
            glitch_series = _combine_with_overlap(glitch_series, g_win, int(cfg.overlap * sr))

        nwin += 1
        win_s = starttime + (nwin - 1) * (cfg.window_len - cfg.overlap)

    # Apply subtraction: y = x - g
    tr_clean.data = tr_raw.data.astype(float)

    nmin = min(len(tr_clean.data), len(glitch_series))
    if nmin > 0:
        tr_clean.data[:nmin] = tr_clean.data[:nmin] - glitch_series[:nmin]

    intervals = (
        np.array([glitch_starts, glitch_ends]).T
        if len(glitch_starts) > 0
        else np.zeros((0, 2), dtype=str)
    )
    return tr_clean, intervals


# ----------------------------
# Driver: process a file
# ----------------------------
def deglitch_mseed_file(
    mseed_path: str,
    out_dir: str,
    channels: List[str],
    cfg: DeglitchConfig,
    out_clean_name: str = "deglitched_data.mseed",
) -> Dict[str, Dict[str, object]]:
    """
    Read one mseed, process selected channels, and write:
      - output/deglitched_data.mseed (clean traces)
      - output/<CH>/glitches_time.txt (merged intervals per channel)

    Returns a dict of per-channel results.
    """
    st_in: Stream = read(mseed_path)
    os.makedirs(out_dir, exist_ok=True)

    st_clean = Stream()
    results: Dict[str, Dict[str, object]] = {}

    for ch in channels:
        tr_list = st_in.select(channel=ch)
        if len(tr_list) == 0:
            continue

        tr = tr_list[0]
        tr_c, intervals = deglitch_trace_swd(tr, cfg)

        ch_dir = os.path.join(out_dir, ch)
        os.makedirs(ch_dir, exist_ok=True)

        np.savetxt(
            os.path.join(ch_dir, "glitches_time.txt"),
            intervals,
            fmt="%s",
            delimiter="\t",
        )

        st_clean.append(tr_c)

        results[ch] = {
            "trace_clean": tr_c,
            "intervals": intervals,
        }

    st_clean.write(os.path.join(out_dir, out_clean_name), format="MSEED")
    return results
