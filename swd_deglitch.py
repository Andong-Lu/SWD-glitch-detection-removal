# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 22:29:27 2026

@author: Andong Lu

Windowed SWD glitch and spike modeling and subtraction.
"""
#%%
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from itertools import repeat
from multiprocessing import current_process
from typing import Dict, List, Optional, Tuple

import numpy as np
from obspy import Stream, Trace, UTCDateTime, read
from tqdm import tqdm

from SWTools import SWdecomp

#%%
# Processing profiles
GLITCH_PURSUIT_PROFILE = "A: FFT-derived lower-frequency bound"
MIN_GLITCH_FREQUENCY_HZ = 0.0
SPIKE_SEARCH_PROFILE = (
    "conservative reciprocal-peak groups; search around main-lobe start"
)
_MIN_GLITCH_ZETA = 0.01

#%%
# ----------------------------
# Configuration defaults
# ----------------------------
@dataclass
class DeglitchConfig:
    # Windowing
    window_len: float = 600.0
    overlap: float = 100.0
    edge_guard: float = 50.0  # ignore detections close to window edges
    parallel_windows: bool = True
    max_window_workers: int = 2
    show_inner_swd_progress: bool = False

    # Selection criteria
    dur_criterion: float = 50.0   # seconds
    amp_criterion: float = 5.0    # MAD multiplier

    # SWD settings
    max_iter_glitch: int = 10
    max_iter_spike: int = 1
    err_tol: float = 0.1

    # Local glitch fitting
    tau_top_num_glitch: int = 10
    tau_peak_distance: float = 5.0
    tau_peak_prominence_mad: float = 2.0

    local_fit_padding: float = 3.0
    local_fit_outside_penalty: float = 0.2

    # Energy bounds
    energy_lo: float = 0.0005
    energy_hi: float = 0.9995

    # Spike fitting
    spike_anchor_search_before: float = 2.0
    spike_anchor_search_after: float = 2.0
    spike_pad_before: float = 1.0
    spike_pad_after: float = 1.0
    spike_mask_halfwidth: float = 1.0  # seconds masked around current max spike
    max_spike_loops: int = 5


@dataclass
class _WindowTask:
    index: int
    data: np.ndarray
    time: np.ndarray
    start_timestamp: float
    end_timestamp: float
    sampling_rate: float


@dataclass
class _GlitchGroup:
    anchor_timestamp: float
    model: np.ndarray
    interval_start: float
    interval_end: float


@dataclass
class _WindowResult:
    index: int
    start_timestamp: float
    end_timestamp: float
    sample_count: int
    groups: List[_GlitchGroup]

#%%
# ----------------------------
# Helpers
# ----------------------------
def _mad_threshold(x: np.ndarray, k: float) -> float:
    """Robust amplitude threshold: median + k*MAD."""
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return med + k * mad


def _locate_spike_window(
    residual: np.ndarray,
    background: float,
    t: np.ndarray,
    glitch_start_rel: float,
    amplitude_threshold: float,
    cfg: DeglitchConfig,
) -> Tuple[float, float, Optional[float], Optional[float], Optional[float]]:
    """Locate a spike near the glitch start and return its narrow fitting window."""
    search_start = max(0.0, glitch_start_rel - cfg.spike_anchor_search_before)
    search_end = min(float(t[-1]), glitch_start_rel + cfg.spike_anchor_search_after)
    search_mask = (t >= search_start) & (t <= search_end)

    if not np.any(search_mask):
        return search_start, search_end, None, None, None

    search_indices = np.flatnonzero(search_mask)
    local_amplitude = np.abs(residual[search_indices] - background)
    peak_local_index = int(np.argmax(local_amplitude))
    if local_amplitude[peak_local_index] <= amplitude_threshold:
        return search_start, search_end, None, None, None

    spike_anchor = float(t[search_indices[peak_local_index]])
    fit_start = max(0.0, spike_anchor - cfg.spike_pad_before)
    fit_end = min(float(t[-1]), spike_anchor + cfg.spike_pad_after)
    return search_start, search_end, spike_anchor, fit_start, fit_end


def _crossing_time(y0: float, y1: float, t0: float, t1: float) -> float:
    """Linearly interpolate a zero crossing between two samples."""
    denominator = abs(y0) + abs(y1)
    if denominator <= np.finfo(float).eps:
        return 0.5 * (t0 + t1)
    return t0 + abs(y0) / denominator * (t1 - t0)


def _dominant_lobe_groups(
    comps: np.ndarray,
    t: np.ndarray,
    fallback_starts: np.ndarray,
    fallback_ends: np.ndarray,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    List[Tuple[str, str]],
    np.ndarray,
    List[List[int]],
    List[int],
]:
    """Build conservative dominant-lobe groups."""
    ncomp = comps.shape[1]
    lobe_starts = np.asarray(fallback_starts, dtype=float).copy()
    lobe_ends = np.asarray(fallback_ends, dtype=float).copy()
    peak_times = np.zeros(ncomp, dtype=float)
    boundary_sources: List[Tuple[str, str]] = []
    reliable = np.zeros(ncomp, dtype=bool)

    for component_index in range(ncomp):
        component = comps[:, component_index]
        peak_index = int(np.argmax(np.abs(component)))
        peak_times[component_index] = float(t[peak_index])
        left_source = "energy fallback"
        right_source = "energy fallback"

        left_values = component[:-1]
        right_values = component[1:]
        crossings = np.flatnonzero(
            (left_values * right_values < 0)
            | ((left_values == 0) & (right_values != 0))
            | ((left_values != 0) & (right_values == 0))
        )
        before = crossings[crossings < peak_index]
        after = crossings[crossings >= peak_index]

        if before.size > 0:
            crossing_index = int(before[-1])
            lobe_starts[component_index] = _crossing_time(
                float(component[crossing_index]),
                float(component[crossing_index + 1]),
                float(t[crossing_index]),
                float(t[crossing_index + 1]),
            )
            left_source = "zero crossing"
        else:
            abs_component = np.abs(component)
            local_minima = np.flatnonzero(
                (abs_component[1:-1] <= abs_component[:-2])
                & (abs_component[1:-1] <= abs_component[2:])
                & (
                    (abs_component[1:-1] < abs_component[:-2])
                    | (abs_component[1:-1] < abs_component[2:])
                )
            ) + 1
            before_minima = local_minima[local_minima < peak_index]
            if before_minima.size > 0:
                minimum_index = int(before_minima[-1])
                lobe_starts[component_index] = float(t[minimum_index])
                left_source = "local |SW| minimum"

        if after.size > 0:
            crossing_index = int(after[0])
            lobe_ends[component_index] = _crossing_time(
                float(component[crossing_index]),
                float(component[crossing_index + 1]),
                float(t[crossing_index]),
                float(t[crossing_index + 1]),
            )
            right_source = "zero crossing"
        else:
            abs_component = np.abs(component)
            local_minima = np.flatnonzero(
                (abs_component[1:-1] <= abs_component[:-2])
                & (abs_component[1:-1] <= abs_component[2:])
                & (
                    (abs_component[1:-1] < abs_component[:-2])
                    | (abs_component[1:-1] < abs_component[2:])
                )
            ) + 1
            after_minima = local_minima[local_minima > peak_index]
            if after_minima.size > 0:
                minimum_index = int(after_minima[0])
                lobe_ends[component_index] = float(t[minimum_index])
                right_source = "local |SW| minimum"

        boundary_sources.append((left_source, right_source))
        reliable[component_index] = (
            left_source != "energy fallback"
            and right_source != "energy fallback"
        )

    energies = np.sum(comps**2, axis=0)
    ungrouped = set(range(ncomp))
    groups: List[List[int]] = []

    while ungrouped:
        reference = max(ungrouped, key=lambda index: float(energies[index]))
        group = [reference]
        ungrouped.remove(reference)

        if reliable[reference]:
            for candidate in list(ungrouped):
                same_group = (
                    reliable[candidate]
                    and lobe_starts[reference] <= peak_times[candidate] <= lobe_ends[reference]
                    and lobe_starts[candidate] <= peak_times[reference] <= lobe_ends[candidate]
                )
                if same_group:
                    group.append(candidate)
                    ungrouped.remove(candidate)

        groups.append(group)

    groups.sort(key=lambda group: min(lobe_starts[index] for index in group))
    main_components = [
        max(group, key=lambda index: float(energies[index])) for group in groups
    ]
    return (
        lobe_starts,
        lobe_ends,
        peak_times,
        boundary_sources,
        reliable,
        groups,
        main_components,
    )


def _run_glitch_swd(x_swd: np.ndarray, t: np.ndarray, cfg: DeglitchConfig) -> SWdecomp:
    """Run localized SWD using the configured glitch-search parameters."""
    return SWdecomp(
        x_swd,
        t,
        MaxIter=cfg.max_iter_glitch,
        ErrTol=cfg.err_tol,
        target_type="glitch",
        MinZeta=_MIN_GLITCH_ZETA,
        MinGlitchFreq=MIN_GLITCH_FREQUENCY_HZ,
        TauTopNum=cfg.tau_top_num_glitch,
        TauPeakDistance=cfg.tau_peak_distance,
        TauPeakProminenceMAD=cfg.tau_peak_prominence_mad,
        LocalFitPadding=cfg.local_fit_padding,
        LocalFitOutsidePenalty=cfg.local_fit_outside_penalty,
        ShowProgress=cfg.show_inner_swd_progress,
    )


def _combine_with_overlap(base: np.ndarray, new: np.ndarray, overlap_samples: int) -> np.ndarray:
    """Stitch adjacent overlapping windows."""
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
    """Estimate component intervals from cumulative energy."""
    ncomp = comps.shape[1]
    durations = np.zeros(ncomp, dtype=float)
    start_times = np.empty(ncomp, dtype=object)
    end_times = np.empty(ncomp, dtype=object)

    for i in range(ncomp):
        sig = comps[:, i].astype(float)
        energy = sig**2
        total = np.sum(energy)
        if total <= 0:
            start_times[i] = window_start
            end_times[i] = window_start
            durations[i] = 0.0
            continue

        norm = np.cumsum(energy) / total

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
    """Merge overlapping intervals."""
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


def _component_anchor_times(
    comps: np.ndarray,
    raw_window: np.ndarray,
    t: np.ndarray,
    window_start: UTCDateTime,
    sampling_rate: float,
) -> np.ndarray:
    """Locate a stable raw-signal anchor near each fitted component peak."""
    anchors = np.empty(comps.shape[1], dtype=float)
    search_halfwidth = max(1, int(round(sampling_rate)))

    for i in range(comps.shape[1]):
        model_peak = int(np.argmax(np.abs(comps[:, i])))
        i0 = max(0, model_peak - search_halfwidth)
        i1 = min(len(t), model_peak + search_halfwidth + 1)
        local_raw = raw_window[i0:i1] - np.median(raw_window[i0:i1])
        anchor_index = i0 + int(np.argmax(np.abs(local_raw)))
        anchors[i] = float(window_start + float(t[anchor_index]))

    return anchors


def _deduplicate_handover_groups(
    window_results: List[_WindowResult],
    cfg: DeglitchConfig,
    sampling_rate: float,
) -> None:
    """Remove duplicate groups at adjacent-window handovers."""
    tolerance = 1.0 / sampling_rate + np.finfo(float).eps

    for previous, current in zip(window_results[:-1], window_results[1:]):
        if current.index != previous.index + 1:
            continue

        handover = current.start_timestamp + cfg.edge_guard
        previous_remove = set()
        current_remove = set()

        for prev_index, prev_group in enumerate(previous.groups):
            if abs(prev_group.anchor_timestamp - handover) > tolerance:
                continue

            for curr_index, curr_group in enumerate(current.groups):
                if abs(curr_group.anchor_timestamp - handover) > tolerance:
                    continue
                if abs(prev_group.anchor_timestamp - curr_group.anchor_timestamp) > tolerance:
                    continue

                anchor = 0.5 * (
                    prev_group.anchor_timestamp + curr_group.anchor_timestamp
                )
                if anchor >= handover:
                    previous_remove.add(prev_index)
                else:
                    current_remove.add(curr_index)

        previous.groups = [
            group
            for index, group in enumerate(previous.groups)
            if index not in previous_remove
        ]
        current.groups = [
            group
            for index, group in enumerate(current.groups)
            if index not in current_remove
        ]


def _window_glitch_model(result: _WindowResult) -> np.ndarray:
    """Sum retained glitch groups for one window."""
    if not result.groups:
        return np.zeros(result.sample_count, dtype=float)
    return np.sum([group.model for group in result.groups], axis=0)


def _build_window_tasks(tr_raw: Trace, cfg: DeglitchConfig) -> List[_WindowTask]:
    """Extract deterministic overlapping windows from the unchanged raw trace."""
    tasks: List[_WindowTask] = []
    sr = float(tr_raw.stats.sampling_rate)
    starttime = tr_raw.stats.starttime
    endtime = tr_raw.stats.endtime
    step = cfg.window_len - cfg.overlap
    overlap_samples = int(cfg.overlap * sr)

    if step <= 0:
        raise ValueError("window_len must be greater than overlap.")

    window_index = 0
    win_s = starttime
    while win_s <= endtime:
        win_e = min(win_s + cfg.window_len, endtime)
        tr_w = tr_raw.copy().trim(starttime=win_s, endtime=win_e)
        t = tr_w.times()

        # Skip windows that add no samples.
        is_useful = window_index == 0 or len(t) > overlap_samples + 1
        if len(t) >= 2 and is_useful:
            tasks.append(
                _WindowTask(
                    index=window_index,
                    data=tr_w.data.astype(float),
                    time=t,
                    start_timestamp=float(win_s),
                    end_timestamp=float(win_e),
                    sampling_rate=sr,
                )
            )

        window_index += 1
        win_s = starttime + window_index * step

    return tasks


def _process_window_task(task: _WindowTask, cfg: DeglitchConfig) -> _WindowResult:
    """Detect glitches and spikes in one independent window."""
    x = task.data
    t = task.time
    sr = task.sampling_rate
    win_s = UTCDateTime(task.start_timestamp)
    win_e = UTCDateTime(task.end_timestamp)

    x_swd = x - np.mean(x)
    swd = _run_glitch_swd(x_swd, t, cfg)

    durations, st_abs, et_abs = _duration_energy(
        swd.comps,
        t,
        win_s,
        cfg.energy_lo,
        cfg.energy_hi,
    )

    pass_duration = durations < cfg.dur_criterion
    pass_core = (
        (st_abs + 0.5 * durations < win_e - cfg.edge_guard)
        & (et_abs - 0.5 * durations > win_s + cfg.edge_guard)
    )

    idx_bg = np.where(durations > cfg.window_len * 0.5)[0]
    background = np.sum(swd.comps[:, idx_bg], axis=1) if idx_bg.size > 0 else 0.0

    residual = x_swd - background
    amp_thr = _mad_threshold(residual, cfg.amp_criterion)

    pass_amplitude = np.max(np.abs(swd.comps), axis=0) > amp_thr
    indices = np.flatnonzero(pass_duration & pass_core & pass_amplitude)

    g_comps = (
        swd.comps[:, indices]
        if indices.size > 0
        else np.zeros((len(t), 0), dtype=float)
    )
    glitch_comps = g_comps.copy()
    g_st = np.asarray(st_abs[indices], dtype=object)
    g_et = np.asarray(et_abs[indices], dtype=object)
    anchor_times = _component_anchor_times(glitch_comps, x, t, win_s, sr)
    group_spikes: List[List[np.ndarray]] = [[] for _ in range(len(g_st))]
    fallback_starts = np.asarray([float(start - win_s) for start in g_st])
    fallback_ends = np.asarray([float(end - win_s) for end in g_et])
    lobe_starts, _, _, _, _, _, main_components = _dominant_lobe_groups(
        glitch_comps,
        t,
        fallback_starts,
        fallback_ends,
    )

    if indices.size > 0:
        for i in main_components:
            x_minus_glitch = x_swd - (
                np.sum(g_comps, axis=1) if g_comps.shape[1] > 0 else 0.0
            )
            amp_bg = float(np.mean(x_minus_glitch))

            spike_search_center = float(lobe_starts[i])
            _, _, _, sp_s, sp_e = _locate_spike_window(
                x_minus_glitch,
                amp_bg,
                t,
                spike_search_center,
                amp_thr,
                cfg,
            )
            if sp_s is None or sp_e is None:
                continue

            x_sp = x_minus_glitch.copy()
            x_sp[t < sp_s] = amp_bg
            x_sp[t > sp_e] = amp_bg
            amp_x_sp = np.abs(x_sp - amp_bg)

            iter_sp = 0
            while np.max(amp_x_sp) > amp_thr and iter_sp < cfg.max_spike_loops:
                kmax = int(np.argmax(amp_x_sp))
                hw = int(cfg.spike_mask_halfwidth * sr)
                i0 = max(0, kmax - hw)
                i1 = min(len(x_sp), kmax + hw)

                x_focus = x_sp.copy()
                x_focus[:i0] = amp_bg
                x_focus[i1:] = amp_bg

                swd_s = SWdecomp(
                    x_focus,
                    t,
                    MaxIter=cfg.max_iter_spike,
                    ErrTol=cfg.err_tol,
                    target_type="spike",
                    ShowProgress=cfg.show_inner_swd_progress,
                )

                spike_comp = swd_s.comps.reshape(-1)
                if not np.any(np.abs(spike_comp) > 0):
                    break

                g_comps = np.hstack((g_comps, swd_s.comps))
                group_spikes[i].append(spike_comp)
                x_sp = x_sp - spike_comp
                amp_x_sp = np.abs(x_sp - amp_bg)
                iter_sp += 1

    groups: List[_GlitchGroup] = []
    for i in range(glitch_comps.shape[1]):
        group_model = glitch_comps[:, i].copy()
        if group_spikes[i]:
            group_model += np.sum(group_spikes[i], axis=0)

        groups.append(
            _GlitchGroup(
                anchor_timestamp=float(anchor_times[i]),
                model=group_model,
                interval_start=float(g_st[i]),
                interval_end=float(g_et[i]),
            )
        )

    return _WindowResult(
        index=task.index,
        start_timestamp=task.start_timestamp,
        end_timestamp=task.end_timestamp,
        sample_count=len(t),
        groups=groups,
    )


def _run_window_tasks(
    tasks: List[_WindowTask],
    cfg: DeglitchConfig,
) -> List[_WindowResult]:
    """Process windows in order."""
    if not tasks:
        return []

    workers = min(max(1, int(cfg.max_window_workers)), len(tasks))
    use_parallel = (
        cfg.parallel_windows
        and workers > 1
        and current_process().name == "MainProcess"
    )

    if use_parallel:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            ordered_results = executor.map(
                _process_window_task,
                tasks,
                repeat(cfg),
                chunksize=1,
            )
            return list(
                tqdm(
                    ordered_results,
                    total=len(tasks),
                    desc="Completed windows",
                    unit="window",
                )
            )

    return [
        _process_window_task(task, cfg)
        for task in tqdm(tasks, desc="Completed windows", unit="window")
    ]

#%%
# ----------------------------
# Process one trace
# ----------------------------
def deglitch_trace_swd(tr: Trace, cfg: DeglitchConfig) -> Tuple[Trace, np.ndarray]:
    """Return a cleaned trace and detected-glitch intervals."""
    tr_raw = tr.copy()
    tr_clean = tr.copy()

    tasks = _build_window_tasks(tr_raw, cfg)
    window_results = sorted(
        _run_window_tasks(tasks, cfg),
        key=lambda result: result.index,
    )
    _deduplicate_handover_groups(
        window_results,
        cfg,
        float(tr.stats.sampling_rate),
    )

    glitch_series = np.array([], dtype=float)
    glitch_starts: List[UTCDateTime] = []
    glitch_ends: List[UTCDateTime] = []

    for result in window_results:
        glitch_starts.extend(
            UTCDateTime(group.interval_start) for group in result.groups
        )
        glitch_ends.extend(
            UTCDateTime(group.interval_end) for group in result.groups
        )
        g_win = _window_glitch_model(result)
        if glitch_series.size == 0:
            glitch_series = g_win.copy()
        else:
            glitch_series = _combine_with_overlap(
                glitch_series,
                g_win,
                int(cfg.overlap * tr.stats.sampling_rate),
            )

    tr_clean.data = tr_raw.data.astype(float)
    nmin = min(len(tr_clean.data), len(glitch_series))
    if nmin > 0:
        tr_clean.data[:nmin] = tr_clean.data[:nmin] - glitch_series[:nmin]

    if glitch_starts:
        merged_starts, merged_ends = _merge_intervals(
            np.asarray(glitch_starts, dtype=object),
            np.asarray(glitch_ends, dtype=object),
        )
        intervals = np.array(
            [
                [start.isoformat(), end.isoformat()]
                for start, end in zip(merged_starts, merged_ends)
            ],
            dtype=str,
        )
    else:
        intervals = np.zeros((0, 2), dtype=str)
    return tr_clean, intervals

#%%
# ----------------------------
# Process one file
# ----------------------------
def deglitch_mseed_file(
    mseed_path: str,
    out_dir: str,
    channels: Optional[List[str]],
    cfg: DeglitchConfig,
    out_clean_name: str = "deglitched_data.mseed",
) -> Dict[str, Dict[str, object]]:
    """Process selected channels and write outputs."""
    st_in: Stream = read(mseed_path)
    os.makedirs(out_dir, exist_ok=True)

    if channels is None:
        channels = sorted({tr.stats.channel for tr in st_in})

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

    if len(st_clean) > 0:
        st_clean.write(os.path.join(out_dir, out_clean_name), format="MSEED")

    return results
