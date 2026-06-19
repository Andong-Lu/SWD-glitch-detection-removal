# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 22:30:17 2026

@author: Andong Lu
"""
#%%
from tqdm import tqdm
import numpy as np
from scipy.optimize import least_squares
from scipy.signal import find_peaks

#%%
class SWdecomp:
    def __init__(self, raw_wave, time, **kwargs):
        self.raw_wave = raw_wave
        self.t = time
        self.dt = time[1] - time[0]
        self.params, self.comps, self.comps_er, self.comps_er_cumsum = self._swd(**kwargs)

    def _swd(self,
             target_type = 'spike',
             OmegaNum=50,
             TauTopNum=5,
             IniTim=0,
             TwoWay=0,
             ErrTol=0.1,
             MinSW=6,
             ZetaList=np.array([0.01, 0.1, 1, 10]),
             PhisList=np.array([0, np.pi/2, np.pi, 3*np.pi/2]),
             MaxIter=50,
             MaxGridRefine=5,
             MinZeta=0.0001,
             MinGlitchFreq=0.0,
             TauPeakDistance=0.0,
             TauPeakProminenceMAD=0.0,
             LocalFitPadding=0.0,
             LocalFitOutsidePenalty=0.0,
             ShowProgress=True):
        """Perform shock waveform decomposition."""

        # Remove the mean and shift time to zero.
        t = self.t - self.t[0]
        y0 = self.raw_wave - np.mean(self.raw_wave)
        r = y0.copy()
        energy0 = np.sum(r**2)

        selected_params = []
        selected_waveforms = []

        for _ in tqdm(
            range(MaxIter),
            desc="SWdecomp iterations",
            unit="iter",
            disable=not ShowProgress,
        ):
            # Select candidate residual peaks.
            abs_r = np.abs(r)
            use_distinct_peaks = TauPeakDistance > 0 or TauPeakProminenceMAD > 0

            if use_distinct_peaks:
                peak_kwargs = {}

                if TauPeakDistance > 0:
                    peak_kwargs["distance"] = max(
                        1,
                        int(round(TauPeakDistance / self.dt)),
                    )

                residual_mad = np.median(np.abs(r - np.median(r)))
                if TauPeakProminenceMAD > 0 and residual_mad > 0:
                    peak_kwargs["prominence"] = TauPeakProminenceMAD * residual_mad

                peak_indices, _ = find_peaks(abs_r, **peak_kwargs)

                if peak_indices.size > 0:
                    peak_order = peak_indices[np.argsort(-abs_r[peak_indices])]
                    tau_candidates = t[peak_order[:TauTopNum]]
                else:
                    tau_candidates = t[np.argsort(-abs_r)[:TauTopNum]]
            else:
                tau_candidates = t[np.argsort(-abs_r)[:TauTopNum]]

            local_bounds = (
                self.local_fit_bounds(
                    tau_candidates,
                    t[0],
                    t[-1],
                    LocalFitPadding,
                )
                if target_type == "glitch"
                else {}
            )

            # Set frequency bounds.
            freqs = np.fft.fftfreq(len(t), d=self.dt)
            spectrum = np.fft.fft(r)
            pos = freqs > 0
            P = np.abs(spectrum[pos])
            f_pos = freqs[pos]
            if P.size == 0 or P.max() == 0:
                break
            peaks = np.where(P > P.max()/10)[0]
            if peaks.size == 0:
                break

            if target_type == 'spike':
                high_hz = f_pos[min(peaks.max()-1, len(f_pos)-1)]
            elif target_type == 'glitch':
                high_hz = min(1, f_pos[min(peaks.max()-1, len(f_pos)-1)])
            else:
                raise ValueError("target_type must be 'glitch' or 'spike'.")

            low_hz = min(f_pos[max(peaks.min()+1, 0)], high_hz/100)

            if target_type == 'glitch' and MinGlitchFreq > 0:
                low_hz = max(low_hz, min(MinGlitchFreq, high_hz * 0.9))

            local_Omega = OmegaNum
            success = False
            amp_thresh = 1e-3 * np.linalg.norm(r)
            zeta_candidates = np.asarray(ZetaList, dtype=float)
            zeta_candidates = zeta_candidates[zeta_candidates >= MinZeta]
            if zeta_candidates.size == 0:
                zeta_candidates = np.array([MinZeta], dtype=float)

            for _ in range(MaxGridRefine):
                # Build the candidate grid.
                freq_grid = np.logspace(
                    np.log10(low_hz),
                    np.log10(high_hz),
                    local_Omega
                )
                omegas = 2 * np.pi * freq_grid

                W, TAU, ZETA, PHI = np.meshgrid(
                    omegas, tau_candidates, zeta_candidates, PhisList,
                    indexing='ij'
                )
                cand_params = np.column_stack((
                    np.ones(W.size),
                    W.ravel(),
                    np.full(W.size, IniTim),
                    TAU.ravel(),
                    ZETA.ravel(),
                    PHI.ravel()
                ))

                atoms = self.recon(cand_params, t)
                norms = np.linalg.norm(atoms, axis=0)
                valid = norms > np.finfo(float).eps
                if not np.any(valid):
                    local_Omega *= 2
                    continue
                atoms[:, ~valid] = 0
                atoms[:, valid] /= norms[valid]

                valid_indices = np.flatnonzero(valid)
                if target_type == "glitch":
                    ranked = []
                    for tau_value in tau_candidates:
                        same_tau = valid_indices[
                            np.isclose(cand_params[valid_indices, 3], tau_value)
                        ]
                        if same_tau.size == 0:
                            continue

                        left, right = local_bounds[float(tau_value)]
                        mask = (t >= left) & (t <= right)
                        local_atoms = atoms[mask][:, same_tau]
                        local_norms = np.linalg.norm(local_atoms, axis=0)
                        usable = local_norms > np.finfo(float).eps
                        if not np.any(usable):
                            continue

                        scores = np.full(same_tau.size, -np.inf)
                        scores[usable] = np.abs(
                            r[mask] @ local_atoms[:, usable]
                        ) / local_norms[usable]
                        best = int(np.argmax(scores))
                        ranked.append((scores[best], same_tau[best], mask))

                    if not ranked:
                        local_Omega *= 2
                        continue

                    _, idx, fit_mask = max(ranked, key=lambda item: item[0])
                    fit_atom = atoms[fit_mask, idx]
                    amp = np.dot(r[fit_mask], fit_atom / np.linalg.norm(fit_atom))
                else:
                    dots = np.abs(r @ atoms[:, valid_indices])
                    idx = valid_indices[int(np.argmax(dots))]
                    fit_mask = np.ones(len(t), dtype=bool)
                    amp = np.dot(r, atoms[:, idx])

                omega, t0, tau, zeta, phi = cand_params[idx, 1:]
                lb = [
                    -np.inf,
                    low_hz * 2 * np.pi - 1e-6,
                    -tau * TwoWay,
                    -tau * TwoWay,
                    MinZeta,
                    -np.inf,
                ]
                ub = [
                    np.inf,
                    high_hz * 2 * np.pi + 1e-6,
                    t.max(),
                    t.max(),
                    10000,
                    np.inf,
                ]

                def resid(x):
                    wf_fit = self.recon(x, t).flatten()
                    if target_type != "glitch":
                        return wf_fit - r

                    fit_error = wf_fit[fit_mask] - r[fit_mask]
                    outside = ~fit_mask
                    if LocalFitOutsidePenalty <= 0 or not np.any(outside):
                        return fit_error
                    return np.concatenate((
                        fit_error,
                        np.sqrt(LocalFitOutsidePenalty) * wf_fit[outside],
                    ))

                x_opt = least_squares(
                    resid,
                    [amp, omega, IniTim, tau, zeta, phi],
                    bounds=(lb, ub),
                ).x

                if abs(x_opt[0]) > amp_thresh:
                    wf = self.recon(x_opt, t).flatten()
                    success = True
                    break

                local_Omega *= 2

            if not success:
                break

            r -= wf
            selected_params.append(x_opt)
            selected_waveforms.append(wf)

            if np.sum(r**2) < ErrTol * energy0 and len(selected_params) >= MinSW:
                break

        # Sort components by energy.
        energies = np.array([np.sum(w**2) for w in selected_waveforms])
        order = np.argsort(-energies)
        params = (
            np.array(selected_params)[order]
            if selected_params
            else np.empty((0, 6), dtype=float)
        )
        comps = (
            np.array(selected_waveforms).T[:, order]
            if selected_waveforms
            else np.empty((len(t), 0), dtype=float)
        )
        er = energies[order] / energy0
        er_cum = np.cumsum(er)

        return params, comps, er, er_cum
#%%
    @staticmethod
    def local_fit_bounds(tau_candidates, t_min, t_max, padding=0.0):
        """Return midpoint-bounded local fitting regions for candidate peaks."""
        candidates = np.unique(np.asarray(tau_candidates, dtype=float))
        candidates.sort()
        bounds = {}

        if candidates.size == 1:
            bounds[float(candidates[0])] = (float(t_min), float(t_max))
            return bounds

        for i, center in enumerate(candidates):
            left = (
                t_min
                if i == 0
                else 0.5 * (candidates[i - 1] + center)
            )
            right = (
                t_max
                if i == candidates.size - 1
                else 0.5 * (center + candidates[i + 1])
            )
            bounds[float(center)] = (
                max(float(t_min), float(left - padding)),
                min(float(t_max), float(right + padding)),
            )

        return bounds

    @staticmethod
    def recon(params, t, batch_size=2000):

        if params.size == 0:
            return np.empty((len(t), 0))

        else:
            if params.ndim == 1:
                params = params.reshape(1, -1)

            n_total = len(params)
            results = []

            for start in range(0, n_total, batch_size):
                end = min(start + batch_size, n_total)
                batch = params[start:end]

                amp, omega, t0, tau, zeta, phi = batch[:, 0], batch[:, 1], batch[:, 2], batch[:, 3], batch[:, 4], batch[:, 5]
                t_shifted = t[np.newaxis, :] - t0[:, np.newaxis]
                z = np.zeros_like(t_shifted)
                eps = 1e-8

                ip_mask = tau > 0
                in_mask = tau < 0
                iz_mask = tau == 0

                if ip_mask.any():
                    ip_indices = np.where(ip_mask)[0]
                    shifted = t_shifted[ip_indices]
                    exponent = (zeta[ip_indices, None] * omega[ip_indices, None] *
                                (tau[ip_indices, None] - shifted) +
                                zeta[ip_indices, None] * tau[ip_indices, None] * omega[ip_indices, None] *
                                (np.log(np.clip(shifted, eps, None)) -
                                 np.log(np.clip(tau[ip_indices, None], eps, None))))
                    z[ip_indices] = (amp[ip_indices, None] *
                                     np.exp(exponent) *
                                     np.cos(omega[ip_indices, None] * shifted + phi[ip_indices, None]) *
                                     (shifted >= 0))

                if in_mask.any():
                    in_indices = np.where(in_mask)[0]
                    shifted = t_shifted[in_indices]
                    exponent = -zeta[in_indices, None] * omega[in_indices, None] * shifted
                    z[in_indices] = (amp[in_indices, None] *
                                     np.exp(exponent) *
                                     np.cos(omega[in_indices, None] * shifted + phi[in_indices, None]) *
                                     (shifted < 0).astype(float))

                if iz_mask.any():
                    iz_indices = np.where(iz_mask)[0]
                    shifted = t_shifted[iz_indices]
                    exponent = -zeta[iz_indices, None] * omega[iz_indices, None] * shifted
                    z[iz_indices] = (amp[iz_indices, None] *
                                     np.exp(exponent) *
                                     np.cos(omega[iz_indices, None] * shifted + phi[iz_indices, None]) *
                                     (shifted >= 0))

                results.append(z)

        return np.concatenate(results, axis=0).T
