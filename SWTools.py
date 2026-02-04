# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 16:52:09 2025

@author: Andong Lu
"""
import time
from tqdm import tqdm
import numpy as np
from scipy.optimize import least_squares

class SWs:
    def __init__(self, raw_wave, time, params):
        self.raw_wave = raw_wave
        self.t = time
        self.dt = time[1]-time[0]
        self.params=params
        self.comps=SWdecomp.recon(params, time)

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
             MaxGridRefine=5):
        """
        Perform Shock Waveform Decomposition (_swd) with these termination conditions:

        1) No positive-frequency content or no spectral peaks above threshold.
        2) Failed to find a valid atom after MaxGridRefine grid refinements.
        3) Residual energy criterion met.
        4) Maximum iterations exceeded.
        """
        # Notify start and record time
        print("\nStarting decomposition...")
        start_time = time.time()

        # Preprocessing: shift time to start at zero and remove mean
        t = self.t - self.t[0]
        y0 = self.raw_wave - np.mean(self.raw_wave)
        r = y0.copy()
        energy0 = np.sum(r**2)

        selected_params = []
        selected_waveforms = []

        # Main decomposition loop with progress bar
        for iteration in tqdm(range(MaxIter), desc="SWdecomp iterations", unit="iter"):
            # 1. pick time-shift candidates based on residual peaks
            tau_candidates = t[np.argsort(-np.abs(r))[:TauTopNum]]

            # 2. determine frequency bounds via FFT
            freqs = np.fft.fftfreq(len(t), d=self.dt)
            spectrum = np.fft.fft(r)
            pos = freqs > 0
            P = np.abs(spectrum[pos])
            f_pos = freqs[pos]
            if P.size == 0 or P.max() == 0:
                print("\nTerminated: no positive-frequency content in residual.")
                break
            peaks = np.where(P > P.max()/10)[0]
            if peaks.size == 0:
                print("\nTerminated: no significant spectral peaks above threshold.")
                break

            if target_type == 'spike':
                high_hz = f_pos[min(peaks.max()-1, len(f_pos)-1)]
            if target_type == 'glitch':
                high_hz = min(1, f_pos[min(peaks.max()-1, len(f_pos)-1)])

            low_hz = min(f_pos[max(peaks.min()+1, 0)], high_hz/100)

            # prepare for grid-refinement attempts
            local_Omega = OmegaNum
            success = False
            amp_thresh = 1e-3 * np.linalg.norm(r)

            for refine in range(MaxGridRefine):
                # 3. generate equal log-spaced frequency grid
                freq_grid = np.logspace(
                    np.log10(low_hz),
                    np.log10(high_hz),
                    local_Omega
                )
                omegas = 2 * np.pi * freq_grid

                # build candidates: Cartesian product of (omega, tau, zeta, phi)
                W, TAU, ZETA, PHI = np.meshgrid(
                    omegas, tau_candidates, ZetaList, PhisList,
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

                # compute atoms and norms; valid atoms have norm > eps
                atoms = self.recon(cand_params, t)
                norms = np.linalg.norm(atoms, axis=0)
                valid = norms > np.finfo(float).eps
                if not np.any(valid):
                    # refine grid resolution and retry
                    local_Omega *= 2
                    continue
                atoms[:, ~valid] = 0
                atoms /= norms

                # pick best candidate by inner product
                dots = np.abs(r @ atoms)
                idx = np.argmax(dots)
                omega, t0, tau, zeta, phi = cand_params[idx, 1:]
                amp = np.dot(r, atoms[:, idx])

                # local least-squares refinement
                lb = [-np.inf, low_hz * 2 * np.pi - 1e-6, -tau * TwoWay, -tau * TwoWay, 0.0001, -np.inf]
                ub = [np.inf, high_hz * 2 * np.pi + 1e-6, t.max(), t.max(), 10000, np.inf]

                def resid(x): return self.recon(x, t).flatten() - r

                x_opt = least_squares(
                    resid,
                    [amp, omega, IniTim, tau, zeta, phi],
                    bounds=(lb, ub)
                ).x

                # check amplitude significance
                if abs(x_opt[0]) > amp_thresh:
                    success = True
                    break
                # else refine grid and retry
                local_Omega *= 2

            if not success:
                # concise termination message for negligible atoms
                print("\nTerminated: no meaningful atom found after grid refinements.")
                break

            # 4. update residual and record component
            wf = self.recon(x_opt, t).flatten()
            r -= wf
            selected_params.append(x_opt)
            selected_waveforms.append(wf)

            # 5. stopping criteria: residual energy low or enough components
            if np.sum(r**2) < ErrTol * energy0 and len(selected_params) >= MinSW:
                print("Completed: residual energy below threshold.")
                break
        else:
            # only executed if for-loop completes without a break
            print("Completed: reached maximum iterations.")

        # final sorting by energy contribution
        energies = np.array([np.sum(w**2) for w in selected_waveforms])
        order = np.argsort(-energies)
        params = np.array(selected_params)[order]
        comps = np.array(selected_waveforms).T[:, order] if selected_waveforms else np.array([])
        er = energies[order] / energy0
        er_cum = np.cumsum(er)

        # report elapsed time
        elapsed = time.time() - start_time
        print(f"Finished decomposition in {elapsed:.2f} seconds.")

        return params, comps, er, er_cum

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
