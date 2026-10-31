# -*- coding: utf-8 -*-
"""
fitting.py
----------
Statistical fitting routines for Single Particle Tracking (SPT) jump distance 
distributions (JDD) and diffusion coefficients using Maximum Likelihood Estimation (MLE).

Includes formulations for:
  - Pure 2D Brownian Motion
  - Anomalous 2D Diffusion (via C-compiled QUADPACK + Bessel K0)
  - Rayleigh Distribution with Localization Error
  - Multi-component Mixture Models with Multi-Start Restarts & Degeneracy Guard
  - Gamma Mixture Models for Diffusion Coefficients
"""

import functools
import numpy as np
from scipy.stats import gamma, rayleigh, norm
from scipy.optimize import minimize, curve_fit
from scipy.integrate import quad
from scipy.special import kv  # C-compiled Modified Bessel Function of the 2nd kind (K_nu)


# =========================================================================
# JUMP DISTANCE PDF FORMULATIONS & FAST SCIPY INTEGRATION
# =========================================================================

def rayleigh_pdf(r, D, dt, loc_err=0.0):
    """
    Rayleigh Jump Distance PDF incorporating localization error:
    P(r) = (r / s^2) * exp(-r^2 / (2 * s^2))
    where s^2 = 2*D*dt + 2*(loc_err^2)
    """
    r = np.asarray(r, dtype=np.float64)
    var_term = 2.0 * D * dt + 2.0 * (loc_err ** 2)
    if var_term <= 1e-12:
        return np.zeros_like(r)
    return (r / var_term) * np.exp(-(r ** 2) / (2.0 * var_term))


def pure_2d_pdf(r, D, dt):
    """
    Pure 2D Brownian Motion PDF:
    P(r) = (r / (2 * D * dt)) * exp(-r^2 / (4 * D * dt))
    """
    r = np.asarray(r, dtype=np.float64)
    denom = 2.0 * D * dt
    if denom <= 1e-12:
        return np.zeros_like(r)
    return (r / denom) * np.exp(-(r ** 2) / (2.0 * denom))


@functools.lru_cache(maxsize=65536)
def _cached_anomalous_single(r_val, D_val, alpha_val, dt_val):
    """
    Fast anomalous 2D diffusion contour integration using C-compiled QUADPACK
    (scipy.integrate.quad) and C-compiled Bessel K0 (scipy.special.kv).
    """
    if D_val <= 0 or alpha_val <= 0 or r_val <= 0:
        return 0.0

    def integrand(u):
        p = (u - 1.0) - 1e-6j
        term1 = np.exp(1j * p * dt_val)
        term2 = p ** (alpha_val - 1.0)
        arg_bessel = (r_val / np.sqrt(D_val)) * (p ** (alpha_val / 2.0))
        
        bessel_val = kv(0, arg_bessel)
        val_complex = term1 * term2 * bessel_val
        return np.real(val_complex)

    try:
        res, _ = quad(integrand, 0.0, 500.0, limit=50, epsabs=1e-5, epsrel=1e-4)
        val = (r_val / (2.0 * np.pi * D_val)) * res
        return float(val) if np.isfinite(val) and val >= 0 else 0.0
    except Exception:
        return 0.0


def anomalous_2d_pdf_single(r_val, D, alpha, dt):
    """Evaluates anomalous diffusion density with coarsened rounding for LRU cache hits."""
    r_round = round(float(r_val), 3)
    D_round = round(float(D), 3)
    alpha_round = round(float(alpha), 2)
    dt_round = round(float(dt), 4)
    return _cached_anomalous_single(r_round, D_round, alpha_round, dt_round)


def anomalous_2d_pdf(r, D, alpha, dt):
    """Vectorized wrapper for anomalous_2d_pdf_single over an array of jump distances r."""
    r_arr = np.asarray(r, dtype=np.float64)
    return np.array([anomalous_2d_pdf_single(x, D, alpha, dt) for x in r_arr])


# =========================================================================
# JUMP DISTANCE MLE FITTING
# =========================================================================

def fit_jump_distances_mle(jumps, dt, num_components=1, loc_error=0.0, 
                           model_type="Rayleigh Distribution (with Loc Precision)",
                           cancel_check=None):
    """
    Fits jump distance distributions using MLE and multi-start optimization.
    Flags models containing degenerate (<1% weight) components.
    """
    jumps = np.asarray(jumps, dtype=np.float64)
    jumps = jumps[jumps > 0]
    N = len(jumps)
    if N < 10:
        return None

    r_min = float(np.min(jumps))
    r_max = float(np.max(jumps))
    var_loc = 2.0 * (loc_error ** 2)

    D_min = max(1e-7, (r_min ** 2 - var_loc) / (4.0 * dt))
    D_max = max(D_min * 100.0, (r_max ** 2 - var_loc) / (2.0 * dt))

    unique_jumps, counts = np.unique(np.round(jumps, 6), return_counts=True)

    def unpack_params(params):
        if model_type in ["Rayleigh Distribution (with Loc Precision)", "Pure 2D Brownian Motion"]:
            D_list = params[:num_components]
            alpha_list = [1.0] * num_components
            raw_w = params[num_components:]

        elif model_type == "Anomalous 2D Diffusion":
            D_list = [params[2 * i] for i in range(num_components)]
            alpha_list = [params[2 * i + 1] for i in range(num_components)]
            raw_w = params[2 * num_components:]

        elif model_type == "Mixed (Pure Brownian + Anomalous)":
            if num_components == 1:
                # 1-component Mixed is 1 Pure Brownian state (no mix possible with 1 state)
                D_list = [params[0]]
                alpha_list = [1.0]
                raw_w = []
            else:
                # K components: (K-1) Pure Brownian states + 1 Anomalous state
                D_list = params[:num_components]
                alpha_list = [1.0] * (num_components - 1) + [params[num_components]]
                raw_w = params[num_components + 1:]
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        if num_components == 1:
            weights = np.array([1.0])
        else:
            exp_w = np.exp(np.clip(np.append(raw_w, 0.0), -20.0, 20.0))
            weights = exp_w / np.sum(exp_w)

        return D_list, alpha_list, weights

    def neg_log_likelihood(params):
        if cancel_check is not None and cancel_check():
            raise InterruptedError("Fitting canceled by user.")

        D_list, alpha_list, weights = unpack_params(params)
        pdf_total = np.zeros_like(unique_jumps, dtype=np.float64)

        for i in range(num_components):
            if model_type == "Rayleigh Distribution (with Loc Precision)":
                pdf_comp = rayleigh_pdf(unique_jumps, D_list[i], dt, loc_err=loc_error)
            elif model_type == "Pure 2D Brownian Motion":
                pdf_comp = pure_2d_pdf(unique_jumps, D_list[i], dt)
            else:
                if abs(alpha_list[i] - 1.0) < 1e-4:
                    pdf_comp = pure_2d_pdf(unique_jumps, D_list[i], dt)
                else:
                    pdf_comp = anomalous_2d_pdf(unique_jumps, D_list[i], alpha_list[i], dt)

            pdf_total += weights[i] * pdf_comp

        pdf_total = np.maximum(pdf_total, 1e-12)
        log_pdf = np.log(pdf_total)
        if not np.all(np.isfinite(log_pdf)):
            return 1e12
        return -np.sum(counts * log_pdf)

    def iter_callback(xk):
        if cancel_check is not None and cancel_check():
            raise InterruptedError("Fitting canceled by user.")

    # --- MULTI-START INITIAL GUESS GENERATION ---
    x0_candidates = []
    alpha_seeds = [0.8, 0.6, 1.2] if ("Anomalous" in model_type or "Mixed" in model_type) else [1.0]

    for a_init in alpha_seeds:
        if num_components > 1:
            seed_D_1 = np.geomspace(max(D_min, 1e-4), min(D_max, 50.0), num_components)
            seed_D_2 = np.linspace(max(D_min, 1e-4), min(D_max, 20.0), num_components)
            d_seeds = [seed_D_1, seed_D_2]
        else:
            d_seeds = [[np.sqrt(D_min * D_max)]]

        for init_D in d_seeds:
            if model_type in ["Rayleigh Distribution (with Loc Precision)", "Pure 2D Brownian Motion"]:
                x0_candidates.append(list(init_D) + [0.0] * (num_components - 1))
                bounds = [(D_min, D_max)] * num_components + [(-10.0, 10.0)] * (num_components - 1)
                num_free_params = 2 * num_components - 1

            elif model_type == "Anomalous 2D Diffusion":
                cand = []
                bounds = []
                for i in range(num_components):
                    cand.extend([init_D[i], a_init])
                    bounds.extend([(D_min, D_max), (0.15, 1.85)])
                cand.extend([0.0] * (num_components - 1))
                bounds.extend([(-10.0, 10.0)] * (num_components - 1))
                x0_candidates.append(cand)
                num_free_params = 3 * num_components - 1

            elif model_type == "Mixed (Pure Brownian + Anomalous)":
                if num_components == 1:
                    x0_candidates.append([init_D[0]])
                    bounds = [(D_min, D_max)]
                    num_free_params = 1
                else:
                    cand = list(init_D) + [a_init] + [0.0] * (num_components - 1)
                    bounds = [(D_min, D_max)] * num_components + [(0.15, 1.85)] + [(-10.0, 10.0)] * (num_components - 1)
                    x0_candidates.append(cand)
                    num_free_params = 2 * num_components

    best_res = None
    best_nll = float('inf')

    for x0 in x0_candidates[:3]:
        try:
            res = minimize(
                neg_log_likelihood, 
                x0, 
                bounds=bounds, 
                method='L-BFGS-B', 
                callback=iter_callback,
                options={'maxiter': 300, 'ftol': 1e-5, 'gtol': 1e-4}
            )
            if res.fun < best_nll:
                best_nll = res.fun
                best_res = res
        except InterruptedError:
            return None
        except Exception:
            continue

    if best_res is None:
        try:
            best_res = minimize(
                neg_log_likelihood, 
                x0_candidates[0], 
                bounds=bounds, 
                method='Nelder-Mead', 
                callback=iter_callback,
                options={'maxiter': 600, 'fatol': 1e-5}
            )
        except InterruptedError:
            return None

    D_opt, alpha_opt, w_opt = unpack_params(best_res.x)
    nll = best_res.fun
    bic = num_free_params * np.log(N) + 2.0 * nll

    has_degenerate = bool(np.any(w_opt < 0.01))

    def total_pdf_func(r_eval):
        r_arr = np.asarray(r_eval, dtype=np.float64)
        pdf_eval = np.zeros_like(r_arr)
        for i in range(num_components):
            if model_type == "Rayleigh Distribution (with Loc Precision)":
                pdf_eval += w_opt[i] * rayleigh_pdf(r_arr, D_opt[i], dt, loc_err=loc_error)
            elif model_type == "Pure 2D Brownian Motion":
                pdf_eval += w_opt[i] * pure_2d_pdf(r_arr, D_opt[i], dt)
            else:
                if abs(alpha_opt[i] - 1.0) < 1e-4:
                    pdf_eval += w_opt[i] * pure_2d_pdf(r_arr, D_opt[i], dt)
                else:
                    pdf_eval += w_opt[i] * anomalous_2d_pdf(r_arr, D_opt[i], alpha_opt[i], dt)
        return pdf_eval

    comp_pdfs = []
    for i in range(num_components):
        def make_comp(d_val, a_val, w_val):
            if model_type == "Rayleigh Distribution (with Loc Precision)":
                return lambda r_eval: w_val * rayleigh_pdf(np.asarray(r_eval, dtype=np.float64), d_val, dt, loc_err=loc_error)
            elif model_type == "Pure 2D Brownian Motion":
                return lambda r_eval: w_val * pure_2d_pdf(np.asarray(r_eval, dtype=np.float64), d_val, dt)
            else:
                if abs(a_val - 1.0) < 1e-4:
                    return lambda r_eval: w_val * pure_2d_pdf(np.asarray(r_eval, dtype=np.float64), d_val, dt)
                return lambda r_eval: w_val * anomalous_2d_pdf(np.asarray(r_eval, dtype=np.float64), d_val, a_val, dt)
        comp_pdfs.append(make_comp(D_opt[i], alpha_opt[i], w_opt[i]))

    param_dicts = [
        {'D': D_opt[i], 'alpha': alpha_opt[i], 'weight': w_opt[i]}
        for i in range(num_components)
    ]

    return {
        'components': num_components,
        'model_type': model_type,
        'bic': bic,
        'log_lh': -nll,
        'log_likelihood': -nll,
        'has_degenerate': has_degenerate,
        'params': param_dicts,
        'pdf_func': total_pdf_func,
        'component_pdfs': comp_pdfs
    }

# =========================================================================
# GAUSSIAN & LINEAR REGRESSION UTILITIES
# =========================================================================

def gaussian_1d(x, amplitude, mean, stddev, offset=0):
    return offset + amplitude * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2))


def gaussian_2d(xy, amplitude, x0, y0, sigma_x, sigma_y, offset=0):
    x, y = xy
    inner = ((x - x0) ** 2) / (2 * sigma_x ** 2) + ((y - y0) ** 2) / (2 * sigma_y ** 2)
    return offset + amplitude * np.exp(-inner)


def fit_straight_line(x, y):
    """Linear fit for MSD vs tau curves."""
    if len(x) < 2:
        return 0.0, 0.0
    p = np.polyfit(x, y, 1)
    return float(p[0]), float(p[1])