# -*- coding: utf-8 -*-
"""
fitting.py
----------
Statistical fitting routines for Single Particle Tracking (SPT) jump distance 
distributions (JDD) and diffusion coefficients using Maximum Likelihood Estimation (MLE).

Includes formulations for:
  - Pure 2D Brownian Motion
  - Anomalous 2D Diffusion (via Fourier-Laplace Bessel integral)
  - Rayleigh Distribution with Localization Error
  - Multi-component Mixture Models
  - Gamma Mixture Models for Diffusion Coefficients
"""

import numpy as np
import mpmath
import functools
from scipy.stats import gamma, rayleigh, norm
from scipy.optimize import minimize, curve_fit

# Precision for mpmath contour integration
mpmath.dps = 5


# =========================================================================
# JUMP DISTANCE PDF FORMULATIONS & CACHING
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


@functools.lru_cache(maxsize=16384)
def _cached_anomalous_single(r_val, D_val, alpha_val, dt_val):
    """Internal memoized calculator for anomalous 2D diffusion contour integration."""
    if D_val <= 0 or alpha_val <= 0 or r_val <= 0:
        return 0.0
    try:
        f = lambda p: mpmath.exp(1j * p * dt_val) * ((1j * p) ** (alpha_val - 1)) * mpmath.besselk(0, ((r_val / np.sqrt(D_val)) * (1j * p) ** (alpha_val / 2)))
        I = mpmath.quad(f, [-1 - (1j * 1e-6), 500 - (1j * 1e-6)])
        val = np.float64(np.abs(((r_val) / (2.0 * np.pi * D_val)) * mpmath.re(I)))
        return val if np.isfinite(val) else 0.0
    except Exception:
        return 0.0


def anomalous_2d_pdf_single(r_val, D, alpha, dt):
    """
    Evaluates the 2D Anomalous Diffusion probability density at a single jump distance r_val
    using contour integration with caching.
    """
    r_round = round(float(r_val), 5)
    D_round = round(float(D), 6)
    alpha_round = round(float(alpha), 4)
    dt_round = round(float(dt), 6)
    return _cached_anomalous_single(r_round, D_round, alpha_round, dt_round)


def anomalous_2d_pdf(r, D, alpha, dt):
    """
    Vectorized wrapper for anomalous_2d_pdf_single over an array of jump distances r.
    """
    r_arr = np.asarray(r, dtype=np.float64)
    return np.array([anomalous_2d_pdf_single(x, D, alpha, dt) for x in r_arr])


# =========================================================================
# JUMP DISTANCE MLE FITTING
# =========================================================================

def fit_jump_distances_mle(jumps, dt, num_components=1, loc_error=0.0, 
                           model_type="Rayleigh Distribution (with Loc Precision)",
                           cancel_check=None):
    """
    Fits single- or multi-component jump distance distributions using Maximum Likelihood Estimation (MLE)
    and evaluates model quality via BIC. Bounds are calculated dynamically from min and max jumps.
    """
    jumps = np.asarray(jumps, dtype=np.float64)
    jumps = jumps[jumps > 0]
    N = len(jumps)
    if N < 10:
        return None

    # Calculate min/max jump distance to establish physical bounds on D
    r_min = float(np.min(jumps))
    r_max = float(np.max(jumps))
    var_loc = 2.0 * (loc_error ** 2)

    D_min = max(1e-7, (r_min ** 2 - var_loc) / (4.0 * dt))
    D_max = max(D_min * 100.0, (r_max ** 2 - var_loc) / (2.0 * dt))

    # Vectorize Likelihood computation using rounded unique jump distances and frequencies
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
            D_list = params[:num_components]
            alpha_list = [1.0] * (num_components - 1) + [params[num_components]]
            raw_w = params[num_components + 1:]
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        if num_components == 1:
            weights = np.array([1.0])
        else:
            exp_w = np.exp(np.append(raw_w, 0.0))
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

        pdf_total = np.maximum(pdf_total, 1e-15)
        return -np.sum(counts * np.log(pdf_total))

    def iter_callback(xk):
        if cancel_check is not None and cancel_check():
            raise InterruptedError("Fitting canceled by user.")

    if num_components > 1:
        init_D = np.geomspace(max(D_min, 1e-5), min(D_max, 100.0), num_components)
    else:
        init_D = [np.sqrt(D_min * D_max)]

    if model_type in ["Rayleigh Distribution (with Loc Precision)", "Pure 2D Brownian Motion"]:
        x0 = list(init_D) + [0.0] * (num_components - 1)
        bounds = [(D_min, D_max)] * num_components + [(-10.0, 10.0)] * (num_components - 1)
        num_free_params = 2 * num_components - 1

    elif model_type == "Anomalous 2D Diffusion":
        x0 = []
        bounds = []
        for i in range(num_components):
            x0.extend([init_D[i], 0.8])
            bounds.extend([(D_min, D_max), (0.01, 2.0)])
        x0.extend([0.0] * (num_components - 1))
        bounds.extend([(-10.0, 10.0)] * (num_components - 1))
        num_free_params = 3 * num_components - 1

    elif model_type == "Mixed (Pure Brownian + Anomalous)":
        x0 = list(init_D) + [0.8] + [0.0] * (num_components - 1)
        bounds = [(D_min, D_max)] * num_components + [(0.01, 2.0)] + [(-10.0, 10.0)] * (num_components - 1)
        num_free_params = 2 * num_components

    try:
        res = minimize(neg_log_likelihood, x0, bounds=bounds, method='L-BFGS-B', callback=iter_callback)
        if not res.success:
            res = minimize(neg_log_likelihood, x0, method='Nelder-Mead', callback=iter_callback)
    except InterruptedError:
        return None

    D_opt, alpha_opt, w_opt = unpack_params(res.x)
    nll = res.fun
    bic = num_free_params * np.log(N) + 2.0 * nll

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
        'params': param_dicts,
        'pdf_func': total_pdf_func,
        'component_pdfs': comp_pdfs
    }


# =========================================================================
# GAMMA DIFFUSION COEFFICIENT FIT
# =========================================================================

def fit_gamma_diffusion_mle(d_data, num_components=1, cancel_check=None):
    """
    Fits a 1-to-N component Gamma mixture model to diffusion coefficients (D)
    using Maximum Likelihood Estimation (MLE).
    """
    d_data = np.asarray(d_data)
    d_data = d_data[d_data > 0]
    n_samples = len(d_data)

    if n_samples == 0:
        return None

    if num_components == 1:
        shape, loc, scale = gamma.fit(d_data, floc=0)
        mean_d = shape * scale
        log_lh = float(np.sum(gamma.logpdf(d_data, shape, loc=0, scale=scale)))
        k_params = 2
        bic = k_params * np.log(n_samples) - 2 * log_lh

        pdf_func = lambda x: gamma.pdf(x, shape, loc=0, scale=scale)

        return {
            'components': 1,
            'bic': bic,
            'log_lh': log_lh,
            'params': [{'weight': 1.0, 'mean_D': mean_d, 'shape': shape, 'scale': scale}],
            'pdf_func': pdf_func,
            'component_pdfs': [pdf_func]
        }

    quantiles = np.quantile(d_data, np.linspace(0.1, 0.9, num_components))
    x0 = [0.0] * (num_components - 1) + [4.0] * num_components + [max(q / 4.0, 1e-4) for q in quantiles]

    def unpack_params(params):
        w_raw = np.append(params[:num_components - 1], 0.0)
        e_w = np.exp(w_raw - np.max(w_raw))
        weights = e_w / np.sum(e_w)

        shapes = np.maximum(params[num_components - 1 : 2 * num_components - 1], 0.1)
        scales = np.maximum(params[2 * num_components - 1 : 3 * num_components - 1], 1e-5)
        return weights, shapes, scales

    def nll(params):
        if cancel_check is not None and cancel_check():
            raise InterruptedError("Fitting canceled by user.")
        weights, shapes, scales = unpack_params(params)
        pdf = np.zeros_like(d_data, dtype=float)
        for w, sh, sc in zip(weights, shapes, scales):
            pdf += w * gamma.pdf(d_data, sh, loc=0, scale=sc)
        pdf = np.maximum(pdf, 1e-300)
        return -np.sum(np.log(pdf))

    def iter_callback(xk):
        if cancel_check is not None and cancel_check():
            raise InterruptedError("Fitting canceled by user.")

    try:
        res = minimize(nll, x0, method='L-BFGS-B', callback=iter_callback)
    except InterruptedError:
        return None

    weights, shapes, scales = unpack_params(res.x)

    log_lh = float(-res.fun)
    num_params = (num_components - 1) + 2 * num_components
    bic = num_params * np.log(n_samples) - 2 * log_lh

    param_list = []
    comp_pdfs = []
    for w, sh, sc in zip(weights, shapes, scales):
        mean_d = sh * sc
        param_list.append({'weight': float(w), 'mean_D': float(mean_d), 'shape': float(sh), 'scale': float(sc)})
        comp_pdfs.append(lambda x, sh=sh, sc=sc: gamma.pdf(x, sh, loc=0, scale=sc))

    def mix_pdf(x):
        val = np.zeros_like(x, dtype=float)
        for w, sh, sc in zip(weights, shapes, scales):
            val += w * gamma.pdf(x, sh, loc=0, scale=sc)
        return val

    return {
        'components': num_components,
        'bic': bic,
        'log_lh': log_lh,
        'params': param_list,
        'pdf_func': mix_pdf,
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