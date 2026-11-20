# -*- coding: utf-8 -*-
"""
fitting.py
----------
Statistical fitting routines for Single Particle Tracking (SPT) jump distance
distributions (JDD) and diffusion coefficients using Maximum Likelihood Estimation (MLE).

Includes formulations for:
  - Gamma Mixture Models for Diffusion Coefficients (with optional fixed D_static & shape)
  - Pure 2D Brownian Motion
  - Anomalous 2D Diffusion (via C-compiled QUADPACK + Bessel K0)
  - Rayleigh Distribution with Localization Error
  - Multi-component Mixture Models with Multi-Start Restarts, Degeneracy Guard,
    Asymptotic Fisher-Information Fitting Errors, and Non-Parametric Bootstrapping
"""

import functools
import numpy as np
from scipy.stats import gamma, rayleigh, norm
from scipy.optimize import minimize, curve_fit
from scipy.integrate import quad
from scipy.special import kv  # C-compiled Modified Bessel Function of the 2nd kind (K_nu)


# =========================================================================
# ASYMPTOTIC (HESSIAN) & BOOTSTRAP UNCERTAINTY ESTIMATION
# =========================================================================

def _compute_asymptotic_fit_errors(nll_fn, x_opt, bounds, unpack_flat_fn):
    """
    Computes asymptotic MLE standard errors via the inverse observed Fisher
    Information Matrix (numerical Hessian of NLL at x_opt) propagated to the
    unpacked physical parameters (D, shape/alpha, weights) via the Delta method:
        Cov(phi) = J @ inv(H) @ J.T
    """
    x_opt = np.asarray(x_opt, dtype=np.float64)
    p = len(x_opt)
    phi_opt = np.asarray(unpack_flat_fn(x_opt), dtype=np.float64)
    m = len(phi_opt)

    if p == 0:
        return np.zeros(m, dtype=np.float64)

    # Adaptive central-difference step sizes strictly inside bounds
    steps = np.zeros(p, dtype=np.float64)
    pinned = np.zeros(p, dtype=bool)
    for i in range(p):
        lb, ub = bounds[i]
        h = max(1e-5, 1e-4 * max(abs(x_opt[i]), 1e-2))
        max_left = (x_opt[i] - lb) * 0.45 if np.isfinite(lb) else h
        max_right = (ub - x_opt[i]) * 0.45 if np.isfinite(ub) else h
        h_safe = min(h, max_left, max_right)
        if h_safe < 1e-8:
            pinned[i] = True
            steps[i] = 1e-6
        else:
            steps[i] = h_safe

    # Numerical Hessian of NLL wrt optimizer variables x
    H = np.zeros((p, p), dtype=np.float64)
    f0 = float(nll_fn(x_opt))

    for i in range(p):
        if pinned[i]:
            H[i, i] = 1e12
            continue
        ei = np.zeros(p, dtype=np.float64)
        ei[i] = steps[i]
        f_plus = float(nll_fn(x_opt + ei))
        f_minus = float(nll_fn(x_opt - ei))
        H[i, i] = (f_plus - 2.0 * f0 + f_minus) / (steps[i] ** 2)

        for j in range(i + 1, p):
            if pinned[j]:
                continue
            ej = np.zeros(p, dtype=np.float64)
            ej[j] = steps[j]
            f_pp = float(nll_fn(x_opt + ei + ej))
            f_pm = float(nll_fn(x_opt + ei - ej))
            f_mp = float(nll_fn(x_opt - ei + ej))
            f_mm = float(nll_fn(x_opt - ei - ej))
            val = (f_pp - f_pm - f_mp + f_mm) / (4.0 * steps[i] * steps[j])
            H[i, j] = val
            H[j, i] = val

    # Invert Hessian safely with eigenvalue floor for numerical stability
    try:
        eigvals, eigvecs = np.linalg.eigh(H)
        eigvals_clipped = np.maximum(eigvals, 1e-8)
        cov_x = (eigvecs / eigvals_clipped) @ eigvecs.T
    except np.linalg.LinAlgError:
        cov_x = np.linalg.pinv(H)

    # Numerical Jacobian J = d(phi)/d(x)
    J = np.zeros((m, p), dtype=np.float64)
    for i in range(p):
        if pinned[i]:
            continue
        ei = np.zeros(p, dtype=np.float64)
        ei[i] = steps[i]
        phi_plus = np.asarray(unpack_flat_fn(x_opt + ei), dtype=np.float64)
        phi_minus = np.asarray(unpack_flat_fn(x_opt - ei), dtype=np.float64)
        J[:, i] = (phi_plus - phi_minus) / (2.0 * steps[i])

    cov_phi = J @ cov_x @ J.T
    fit_err = np.sqrt(np.maximum(np.diag(cov_phi), 0.0))
    return fit_err


def _compute_bootstrap_errors(data, x_opt, bounds, make_nll_fn, unpack_flat_fn,
                              n_bootstrap=100, seed=42, cancel_check=None,
                              progress_callback=None):
    """
    Non-parametric bootstrap uncertainty estimation. Resamples raw observations
    with replacement, warm-starts L-BFGS-B from x_opt, and computes the sample
    standard deviation across replicates.
    """
    phi_opt = np.asarray(unpack_flat_fn(x_opt), dtype=np.float64)
    m = len(phi_opt)

    if len(x_opt) == 0:
        return np.zeros(m, dtype=np.float64)
    if n_bootstrap is None or n_bootstrap <= 0:
        return np.full(m, np.nan, dtype=np.float64)

    rng = np.random.default_rng(seed)
    N = len(data)
    boot_records = []

    for b in range(n_bootstrap):
        if cancel_check is not None and cancel_check():
            raise InterruptedError("Fitting canceled by user.")
        if progress_callback is not None and (b % 10 == 0 or b == n_bootstrap - 1):
            progress_callback(b + 1, n_bootstrap)

        boot_sample = rng.choice(data, size=N, replace=True)
        boot_nll = make_nll_fn(boot_sample)

        try:
            res_b = minimize(
                boot_nll,
                x_opt,
                bounds=bounds,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-5, 'gtol': 1e-4}
            )
            if np.isfinite(res_b.fun):
                boot_records.append(unpack_flat_fn(res_b.x))
        except InterruptedError:
            raise
        except Exception:
            continue

    if len(boot_records) < 2:
        return np.full(m, np.nan, dtype=np.float64)

    boot_arr = np.asarray(boot_records, dtype=np.float64)
    return np.nanstd(boot_arr, axis=0, ddof=1)


# =========================================================================
# DIFFUSION COEFFICIENT GAMMA MIXTURE MLE FITTING
# =========================================================================

def fit_gamma_diffusion_mle(d_values, num_components=1, fix_d_static=False,
                            d_static=0.045, fix_shape=False, shape_val=4.0,
                            n_bootstrap=100, cancel_check=None,
                            progress_callback=None):
    """
    Fits a K-component Gamma mixture model to diffusion coefficients (D > 0)
    using Maximum Likelihood Estimation (MLE).

    Parameters
    ----------
    d_values : array-like
        Positive apparent diffusion coefficients (um^2/s).
    num_components : int
        Number of Gamma mixture states K.
    fix_d_static : bool
        If True, pins State 1's diffusion coefficient to `d_static`.
    d_static : float
        Fixed static diffusion coefficient (um^2/s) when `fix_d_static=True`.
    fix_shape : bool
        If True, pins all Gamma shape parameters to `shape_val` (scale = D / shape).
    shape_val : float
        Fixed Gamma shape parameter when `fix_shape=True`.
    n_bootstrap : int
        Number of non-parametric bootstrap resamples for parameter error bars (0 = off).
    """
    d_values = np.asarray(d_values, dtype=np.float64)
    d_values = d_values[np.isfinite(d_values) & (d_values > 0)]
    N = len(d_values)
    if N < 10:
        return None

    d_min = max(1e-6, float(np.min(d_values)) * 0.5)
    d_max = max(d_min * 100.0, float(np.max(d_values)) * 1.5)
    n_free_D = (num_components - 1) if fix_d_static else num_components
    n_free_shape = 0 if fix_shape else num_components

    def unpack_params(params):
        idx = 0
        if fix_d_static:
            D_free = list(params[idx:idx + n_free_D])
            D_list = np.asarray([float(d_static)] + D_free, dtype=np.float64)
            idx += n_free_D
        else:
            D_list = np.asarray(params[idx:idx + n_free_D], dtype=np.float64)
            idx += n_free_D

        if fix_shape:
            shape_list = np.full(num_components, float(shape_val), dtype=np.float64)
        else:
            shape_list = np.asarray(params[idx:idx + n_free_shape], dtype=np.float64)
            idx += n_free_shape

        raw_w = params[idx:]
        if num_components == 1:
            weights = np.array([1.0], dtype=np.float64)
        else:
            exp_w = np.exp(np.clip(np.append(raw_w, 0.0), -20.0, 20.0))
            weights = exp_w / np.sum(exp_w)

        return D_list, shape_list, weights

    def unpack_sorted_flat(params):
        D_list, shape_list, weights = unpack_params(params)
        if num_components > 1:
            if fix_d_static:
                order = np.concatenate(([0], 1 + np.argsort(D_list[1:])))
            else:
                order = np.argsort(D_list)
            D_list = D_list[order]
            shape_list = shape_list[order]
            weights = weights[order]
        return np.concatenate([D_list, shape_list, weights])

    def make_nll_fn(sample_arr):
        def nll_fn(params):
            if cancel_check is not None and cancel_check():
                raise InterruptedError("Fitting canceled by user.")
            D_list, shape_list, weights = unpack_params(params)
            pdf_total = np.zeros_like(sample_arr, dtype=np.float64)
            for i in range(num_components):
                scale_i = D_list[i] / shape_list[i]
                if scale_i <= 1e-12:
                    return 1e12
                pdf_total += weights[i] * gamma.pdf(sample_arr, a=shape_list[i], scale=scale_i)
            pdf_total = np.maximum(pdf_total, 1e-15)
            log_pdf = np.log(pdf_total)
            if not np.all(np.isfinite(log_pdf)):
                return 1e12
            return -float(np.sum(log_pdf))
        return nll_fn

    neg_log_likelihood = make_nll_fn(d_values)

    def iter_callback(xk):
        if cancel_check is not None and cancel_check():
            raise InterruptedError("Fitting canceled by user.")

    # Multi-start initial seeds for free D parameters
    if n_free_D == 0:
        d_seeds = [[]]
    elif n_free_D == 1:
        mobile_vals = d_values[d_values > d_static * 1.1] if fix_d_static else d_values
        if len(mobile_vals) < 3:
            mobile_vals = d_values
        d_seeds = [[float(np.median(mobile_vals))], [float(np.mean(mobile_vals))]]
    else:
        low_d = max(d_min * 1.1, d_static * 1.15) if fix_d_static else max(d_min * 1.1, np.percentile(d_values, 5))
        high_d = max(low_d * 2.0, min(d_max * 0.9, np.percentile(d_values, 95)))
        quantiles = np.linspace(0.15, 0.85, n_free_D)
        q_seed = np.clip(np.quantile(d_values, quantiles), d_min * 1.1, d_max * 0.9)
        g_seed = np.geomspace(low_d, high_d, n_free_D)
        l_seed = np.linspace(low_d, high_d, n_free_D)
        d_seeds = [q_seed, g_seed, l_seed]

    bounds = (
        [(d_min, d_max)] * n_free_D
        + [(1.0, 25.0)] * n_free_shape
        + [(-10.0, 10.0)] * (num_components - 1)
    )
    num_free_params = len(bounds)

    if num_free_params == 0:
        best_x = np.array([], dtype=np.float64)
        best_nll = neg_log_likelihood(best_x)
    else:
        best_res = None
        best_nll = float('inf')
        shape_seeds = [float(shape_val)] if fix_shape else [4.0, 2.5]

        for init_D in d_seeds:
            for init_s in shape_seeds:
                x0 = (
                    list(init_D)
                    + ([init_s] * n_free_shape)
                    + ([0.0] * (num_components - 1))
                )
                try:
                    res = minimize(
                        neg_log_likelihood,
                        x0,
                        bounds=bounds,
                        method='L-BFGS-B',
                        callback=iter_callback,
                        options={'maxiter': 400, 'ftol': 1e-7, 'gtol': 1e-5}
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
                x0_fb = (
                    list(d_seeds[0])
                    + ([4.0] * n_free_shape)
                    + ([0.0] * (num_components - 1))
                )
                best_res = minimize(
                    neg_log_likelihood,
                    x0_fb,
                    bounds=bounds,
                    method='Nelder-Mead',
                    callback=iter_callback,
                    options={'maxiter': 800, 'fatol': 1e-6}
                )
            except InterruptedError:
                return None

        best_x = np.asarray(best_res.x, dtype=np.float64)
        best_nll = float(best_res.fun)

    # Compute asymptotic fitting errors & bootstrap errors
    fit_err_flat = _compute_asymptotic_fit_errors(
        neg_log_likelihood, best_x, bounds, unpack_sorted_flat
    )
    try:
        boot_err_flat = _compute_bootstrap_errors(
            d_values, best_x, bounds, make_nll_fn, unpack_sorted_flat,
            n_bootstrap=n_bootstrap, cancel_check=cancel_check,
            progress_callback=progress_callback
        )
    except InterruptedError:
        return None

    sorted_flat = unpack_sorted_flat(best_x)
    D_opt = sorted_flat[:num_components]
    shape_opt = sorted_flat[num_components:2 * num_components]
    w_opt = sorted_flat[2 * num_components:]

    D_fit_err = fit_err_flat[:num_components]
    shape_fit_err = fit_err_flat[num_components:2 * num_components]
    w_fit_err = fit_err_flat[2 * num_components:]

    D_boot_err = boot_err_flat[:num_components]
    shape_boot_err = boot_err_flat[num_components:2 * num_components]
    w_boot_err = boot_err_flat[2 * num_components:]

    if fix_d_static:
        D_fit_err[0] = 0.0
        D_boot_err[0] = 0.0
    if fix_shape:
        shape_fit_err[:] = 0.0
        shape_boot_err[:] = 0.0
    if num_components == 1:
        w_fit_err[0] = 0.0
        w_boot_err[0] = 0.0

    bic = num_free_params * np.log(N) + 2.0 * best_nll
    aic = 2.0 * num_free_params + 2.0 * best_nll
    has_degenerate = bool(np.any(w_opt < 0.01))

    def total_pdf_func(d_eval):
        d_arr = np.asarray(d_eval, dtype=np.float64)
        pdf_eval = np.zeros_like(d_arr)
        for i in range(num_components):
            pdf_eval += w_opt[i] * gamma.pdf(d_arr, a=shape_opt[i], scale=D_opt[i] / shape_opt[i])
        return pdf_eval

    comp_pdfs = []
    for i in range(num_components):
        def make_comp(d_val, s_val, w_val):
            return lambda d_eval: w_val * gamma.pdf(
                np.asarray(d_eval, dtype=np.float64), a=s_val, scale=d_val / s_val
            )
        comp_pdfs.append(make_comp(D_opt[i], shape_opt[i], w_opt[i]))

    param_dicts = [
        {
            'D': float(D_opt[i]),
            'mean_D': float(D_opt[i]),
            'D_fit_err': float(D_fit_err[i]),
            'D_boot_err': float(D_boot_err[i]),
            'is_static': bool(fix_d_static and i == 0),
            'shape': float(shape_opt[i]),
            'n': float(shape_opt[i]),
            'shape_fit_err': float(shape_fit_err[i]),
            'shape_boot_err': float(shape_boot_err[i]),
            'is_fixed_shape': bool(fix_shape),
            'scale': float(D_opt[i] / shape_opt[i]),
            'weight': float(w_opt[i]),
            'weight_fit_err': float(w_fit_err[i]),
            'weight_boot_err': float(w_boot_err[i]),
        }
        for i in range(num_components)
    ]

    return {
        'components': num_components,
        'bic': float(bic),
        'aic': float(aic),
        'log_lh': -best_nll,
        'log_likelihood': -best_nll,
        'has_degenerate': has_degenerate,
        'params': param_dicts,
        'pdf_func': total_pdf_func,
        'component_pdfs': comp_pdfs,
    }


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


def anomalous_2d_pdf(r, D, alpha, dt, M=32):
    """
    Exact 2D Subdiffusive CTRW / Fractional Diffusion Jump Distance PDF:
        P(r | D, alpha, dt) = L^{-1}{ (r/D) * s^(alpha-1) * K_0((r/sqrt(D)) * s^(alpha/2)) }(dt)
    Evaluated via vectorized Fixed Talbot contour inversion (Abate & Valko, 2004).
    Valid for 0 < alpha <= 1; smoothly reduces to Pure 2D Brownian Motion at alpha = 1.
    """
    r = np.asarray(r, dtype=np.float64)
    scalar_input = (r.ndim == 0)
    r_1d = np.atleast_1d(r)

    if D <= 1e-12 or dt <= 1e-12 or alpha <= 1e-6:
        out = np.zeros_like(r_1d)
        return float(out[0]) if scalar_input else out

    alpha_eff = min(float(alpha), 1.0)

    # Exact closed-form limit at alpha = 1 (Pure 2D Brownian Motion)
    if abs(alpha_eff - 1.0) < 1e-6:
        denom = 2.0 * D * dt
        out = (r_1d / denom) * np.exp(-(r_1d ** 2) / (2.0 * denom))
        out = np.where((r_1d > 0) & np.isfinite(out), out, 0.0)
        return float(out[0]) if scalar_input else out

    k = np.arange(1, M, dtype=np.float64)
    theta = k * np.pi / M
    cot_theta = 1.0 / np.tan(theta)

    r_talbot = (2.0 * M) / (5.0 * dt)
    s = r_talbot * theta * (cot_theta + 1j)
    sigma = theta + (theta * cot_theta - 1.0) * cot_theta
    ds_factor = 1.0 + 1j * sigma

    r_col = np.maximum(r_1d[:, None], 1e-12)
    s_row = s[None, :]
    sqrt_D = np.sqrt(D)

    arg = (r_col / sqrt_D) * (s_row ** (0.5 * alpha_eff))
    with np.errstate(over='ignore', invalid='ignore'):
        bessel_val = kv(0, arg)
        integrand = (
            np.exp(dt * s_row)
            * (r_col / D)
            * (s_row ** (alpha_eff - 1.0))
            * bessel_val
            * ds_factor[None, :]
        )

    integrand_real = np.where(np.isfinite(integrand), np.real(integrand), 0.0)
    sum_terms = np.sum(integrand_real, axis=1)

    s0 = r_talbot
    r_safe = np.maximum(r_1d, 1e-12)
    with np.errstate(over='ignore', invalid='ignore'):
        bessel0 = kv(0, (r_safe / sqrt_D) * (s0 ** (0.5 * alpha_eff)))
        term0 = 0.5 * np.exp(dt * s0) * (r_safe / D) * (s0 ** (alpha_eff - 1.0)) * bessel0
    term0 = np.where(np.isfinite(term0), term0, 0.0)

    pdf = (r_talbot / M) * (term0 + sum_terms)
    pdf = np.where((r_1d > 0) & np.isfinite(pdf) & (pdf > 0), pdf, 0.0)
    return float(pdf[0]) if scalar_input else pdf

# =========================================================================
# CVE ESTIMATOR
# =========================================================================

def compute_track_cve(x_um, y_um, frames, dt, R=1.0 / 6.0, fixed_sigma2=None):
    """
    Computes the 2D Covariance-Based Estimator (CVE) of diffusion coefficient D
    and localization error variance sigma^2 (Vestergaard et al., PRE 89, 022726, 2014).

    Parameters
    ----------
    x_um, y_um : 1D array-like
        Particle coordinates in micrometers, sorted by frame.
    frames : 1D array-like
        Integer frame indices corresponding to (x_um, y_um).
    dt : float
        Effective time interval between consecutive frames in seconds.
    R : float
        Motion blur coefficient in [0, 0.25]. Default 1/6 (full exposure).
    fixed_sigma2 : float or None
        If None, estimates D and sigma^2 jointly via Eqs. (14)-(15).
        If float (in um^2), estimates D with known sigma^2 via Eq. (16).

    Returns
    -------
    dict with keys: 'D', 'sigma2', 'n_steps', 'n_pairs', 'se_D'
    or None if the track has too few contiguous displacements (< 2 steps).
    """
    if dt <= 0:
        return None

    x = np.asarray(x_um, dtype=np.float64)
    y = np.asarray(y_um, dtype=np.float64)
    f = np.asarray(frames, dtype=np.int64)

    if len(x) < 3:
        return None

    df = np.diff(f)
    dx = np.diff(x)
    dy = np.diff(y)

    # Enforce strictly contiguous single-frame displacements (no gap bridging)
    valid_step = (df == 1)
    valid_pair = (df[:-1] == 1) & (df[1:] == 1)

    n_steps = int(np.sum(valid_step))
    n_pairs = int(np.sum(valid_pair))

    if n_steps < 2 or n_pairs < 1:
        return None

    dx_v = dx[valid_step]
    dy_v = dy[valid_step]

    msd_x = float(np.mean(dx_v ** 2))
    msd_y = float(np.mean(dy_v ** 2))

    cov_x = float(np.mean(dx[:-1][valid_pair] * dx[1:][valid_pair]))
    cov_y = float(np.mean(dy[:-1][valid_pair] * dy[1:][valid_pair]))

    if fixed_sigma2 is None:
        # Unknown localization error: Eqs. (14) & (15) averaged over x, y
        D_x = msd_x / (2.0 * dt) + cov_x / dt
        D_y = msd_y / (2.0 * dt) + cov_y / dt
        D = 0.5 * (D_x + D_y)

        sig2_x = R * msd_x + (2.0 * R - 1.0) * cov_x
        sig2_y = R * msd_y + (2.0 * R - 1.0) * cov_y
        sigma2 = 0.5 * (sig2_x + sig2_y)
    else:
        # Known / pooled localization error: Eq. (16) averaged over x, y
        denom = 2.0 * (1.0 - 2.0 * R) * dt
        if denom <= 0:
            return None
        D_x = (msd_x - 2.0 * fixed_sigma2) / denom
        D_y = (msd_y - 2.0 * fixed_sigma2) / denom
        D = 0.5 * (D_x + D_y)
        sigma2 = float(fixed_sigma2)

    # Theoretical standard error of D_hat for 2D (Eqs. 17 & 18, scaled by 1/sqrt(2))
    se_D = np.nan
    if D > 0 and n_steps > 0:
        sig2_pos = max(sigma2, 0.0)
        eps = sig2_pos / (D * dt) - 2.0 * R
        if fixed_sigma2 is None:
            var_D = (D ** 2 / 2.0) * (
                (6.0 + 4.0 * eps + 2.0 * eps ** 2) / n_steps
                + (4.0 * (1.0 + eps) ** 2) / (n_steps ** 2)
            )
        else:
            var_D = (D ** 2 * (2.0 + 4.0 * eps + 3.0 * eps ** 2)) / (
                2.0 * n_steps * ((1.0 - 2.0 * R) ** 2)
            )
        if var_D >= 0:
            se_D = float(np.sqrt(var_D))

    return {
        "D": float(D),
        "sigma2": float(sigma2),
        "n_steps": n_steps,
        "n_pairs": n_pairs,
        "se_D": se_D,
    }


def compute_pooled_sigma2(sigma2_values, n_steps_values):
    """
    Computes the step-weighted ensemble localization error variance (in um^2)
    for 2-Pass Pooled CVE (Vestergaard et al., Sec. III C 3).

    Averages unclipped per-track sigma^2 estimates first to avoid upward
    truncation bias, then clamps the pooled variance at >= 0.0.
    """
    sig2 = np.asarray(sigma2_values, dtype=np.float64)
    weights = np.asarray(n_steps_values, dtype=np.float64)

    valid = np.isfinite(sig2) & np.isfinite(weights) & (weights > 0)
    if not np.any(valid):
        return None

    pooled = float(np.sum(weights[valid] * sig2[valid]) / np.sum(weights[valid]))
    return max(0.0, pooled)

# =========================================================================
# JUMP DISTANCE MLE FITTING
# =========================================================================

def fit_jump_distances_mle(jumps, dt, num_components=1, loc_error=0.0,
                           model_type="Rayleigh Distribution (with Loc Precision)",
                           fix_d_static=False, d_static=0.045,
                           n_bootstrap=100, cancel_check=None,
                           progress_callback=None):
    """
    Fits jump distance distributions using MLE, multi-start optimization,
    optional fixed State 1 D_static, asymptotic Fisher-Information fitting errors,
    and non-parametric bootstrap errors.
    """
    jumps = np.asarray(jumps, dtype=np.float64)
    jumps = jumps[np.isfinite(jumps) & (jumps > 0)]
    N = len(jumps)
    if N < 10:
        return None

    r_min = float(np.min(jumps))
    r_max = float(np.max(jumps))
    var_loc = 2.0 * (loc_error ** 2)

    dt_eff_min = (dt ** 0.05) if ("Anomalous" in model_type and dt < 1.0) else dt
    dt_eff_max = (dt ** 0.05) if ("Anomalous" in model_type and dt > 1.0) else dt
    D_min = max(1e-7, (r_min ** 2 - var_loc) / (4.0 * dt_eff_min))
    D_max = max(D_min * 100.0, (r_max ** 2 - var_loc) / (2.0 * dt_eff_max))
    n_free_D = (num_components - 1) if fix_d_static else num_components

    def unpack_params(params):
        if model_type in ["Rayleigh Distribution (with Loc Precision)", "Pure 2D Brownian Motion"]:
            D_free = list(params[:n_free_D])
            D_list = ([float(d_static)] + D_free) if fix_d_static else D_free
            alpha_list = [1.0] * num_components
            raw_w = params[n_free_D:]

        elif model_type == "Anomalous 2D Diffusion":
            idx = 0
            D_list = []
            alpha_list = []
            for i in range(num_components):
                if fix_d_static and i == 0:
                    D_list.append(float(d_static))
                else:
                    D_list.append(float(params[idx]))
                    idx += 1
                alpha_list.append(float(params[idx]))
                idx += 1
            raw_w = params[idx:]

        elif model_type == "Mixed (Pure Brownian + Anomalous)":
            if num_components == 1:
                D_list = [float(d_static)] if fix_d_static else [float(params[0])]
                alpha_list = [1.0]
                raw_w = []
            else:
                D_free = list(params[:n_free_D])
                D_list = ([float(d_static)] + D_free) if fix_d_static else D_free
                alpha_list = [1.0] * (num_components - 1) + [float(params[n_free_D])]
                raw_w = params[n_free_D + 1:]
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        if num_components == 1:
            weights = np.array([1.0], dtype=np.float64)
        else:
            exp_w = np.exp(np.clip(np.append(raw_w, 0.0), -20.0, 20.0))
            weights = exp_w / np.sum(exp_w)

        return np.asarray(D_list, dtype=np.float64), np.asarray(alpha_list, dtype=np.float64), weights

    def unpack_sorted_flat(params):
        D_list, alpha_list, weights = unpack_params(params)
        if num_components > 1:
            if model_type == "Mixed (Pure Brownian + Anomalous)":
                # Keep Anomalous state in the last slot (num_components - 1)
                n_pure = num_components - 1
                if n_pure > 1:
                    if fix_d_static:
                        pure_order = np.concatenate(([0], 1 + np.argsort(D_list[1:n_pure])))
                    else:
                        pure_order = np.argsort(D_list[:n_pure])
                    order = np.concatenate((pure_order, [n_pure]))
                else:
                    order = np.arange(num_components)
            else:
                if fix_d_static:
                    order = np.concatenate(([0], 1 + np.argsort(D_list[1:])))
                else:
                    order = np.argsort(D_list)

            D_list = D_list[order]
            alpha_list = alpha_list[order]
            weights = weights[order]

        return np.concatenate([D_list, alpha_list, weights])

    def make_nll_fn(jump_arr):
        u_jumps, counts = np.unique(np.round(jump_arr, 6), return_counts=True)

        def nll_fn(params):
            if cancel_check is not None and cancel_check():
                raise InterruptedError("Fitting canceled by user.")

            D_list, alpha_list, weights = unpack_params(params)
            pdf_total = np.zeros_like(u_jumps, dtype=np.float64)

            for i in range(num_components):
                if model_type == "Rayleigh Distribution (with Loc Precision)":
                    pdf_comp = rayleigh_pdf(u_jumps, D_list[i], dt, loc_err=loc_error)
                elif model_type == "Pure 2D Brownian Motion":
                    pdf_comp = pure_2d_pdf(u_jumps, D_list[i], dt)
                else:
                    if abs(alpha_list[i] - 1.0) < 1e-4:
                        pdf_comp = pure_2d_pdf(u_jumps, D_list[i], dt)
                    else:
                        pdf_comp = anomalous_2d_pdf(u_jumps, D_list[i], alpha_list[i], dt)

                pdf_total += weights[i] * pdf_comp

            pdf_total = np.maximum(pdf_total, 1e-12)
            log_pdf = np.log(pdf_total)
            if not np.all(np.isfinite(log_pdf)):
                return 1e12
            return -float(np.sum(counts * log_pdf))

        return nll_fn

    neg_log_likelihood = make_nll_fn(jumps)

    def iter_callback(xk):
        if cancel_check is not None and cancel_check():
            raise InterruptedError("Fitting canceled by user.")

    # --- MULTI-START INITIAL GUESS GENERATION ---
    x0_candidates = []
    alpha_seeds = [0.85, 0.60, 0.35] if ("Anomalous" in model_type or "Mixed" in model_type) else [1.0]
    
    for a_init in alpha_seeds:
        if n_free_D == 0:
            d_seeds = [[]]
        elif n_free_D == 1:
            base_d = max(np.sqrt(D_min * D_max), d_static * 1.5 if fix_d_static else 0.0)
            d_seeds = [[min(base_d, D_max * 0.9)]]
        else:
            low_d = max(D_min, d_static * 1.2, 1e-4) if fix_d_static else max(D_min, 1e-4)
            seed_D_1 = np.geomspace(low_d, max(low_d * 5.0, min(D_max, 50.0)), n_free_D)
            seed_D_2 = np.linspace(low_d, max(low_d * 5.0, min(D_max, 20.0)), n_free_D)
            d_seeds = [seed_D_1, seed_D_2]

        for init_D in d_seeds:
            if model_type in ["Rayleigh Distribution (with Loc Precision)", "Pure 2D Brownian Motion"]:
                x0_candidates.append(list(init_D) + [0.0] * (num_components - 1))
                bounds = [(D_min, D_max)] * n_free_D + [(-10.0, 10.0)] * (num_components - 1)

            elif model_type == "Anomalous 2D Diffusion":
                cand = []
                bounds = []
                d_idx = 0
                for i in range(num_components):
                    if not (fix_d_static and i == 0):
                        d_seed_a = np.clip(init_D[d_idx] * (dt ** (1.0 - a_init)), D_min * 1.05, D_max * 0.95)
                        cand.append(float(d_seed_a))
                        bounds.append((D_min, D_max))
                        d_idx += 1
                    cand.append(a_init)
                    bounds.append((0.05, 1.0))
                cand.extend([0.0] * (num_components - 1))
                bounds.extend([(-10.0, 10.0)] * (num_components - 1))
                x0_candidates.append(cand)

            elif model_type == "Mixed (Pure Brownian + Anomalous)":
                if num_components == 1:
                    if fix_d_static:
                        x0_candidates.append([])
                        bounds = []
                    else:
                        x0_candidates.append([init_D[0]])
                        bounds = [(D_min, D_max)]
                else:
                    scaled_init_D = list(init_D)
                    scaled_init_D[-1] = float(
                        np.clip(scaled_init_D[-1] * (dt ** (1.0 - a_init)), D_min * 1.05, D_max * 0.95)
                    )
                    cand = scaled_init_D + [a_init] + [0.0] * (num_components - 1)
                    bounds = (
                        [(D_min, D_max)] * n_free_D
                        + [(0.05, 1.0)]
                        + [(-10.0, 10.0)] * (num_components - 1)
                    )
                    x0_candidates.append(cand)

    num_free_params = len(bounds)

    if num_free_params == 0:
        best_x = np.array([], dtype=np.float64)
        nll = neg_log_likelihood(best_x)
    else:
        best_res = None
        best_nll = float('inf')

        for x0 in x0_candidates:
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

        best_x = np.asarray(best_res.x, dtype=np.float64)
        nll = float(best_res.fun)

    # Compute asymptotic fitting errors & bootstrap errors
    fit_err_flat = _compute_asymptotic_fit_errors(
        neg_log_likelihood, best_x, bounds, unpack_sorted_flat
    )
    try:
        boot_err_flat = _compute_bootstrap_errors(
            jumps, best_x, bounds, make_nll_fn, unpack_sorted_flat,
            n_bootstrap=n_bootstrap, cancel_check=cancel_check,
            progress_callback=progress_callback
        )
    except InterruptedError:
        return None

    sorted_flat = unpack_sorted_flat(best_x)
    D_opt = sorted_flat[:num_components]
    alpha_opt = sorted_flat[num_components:2 * num_components]
    w_opt = sorted_flat[2 * num_components:]

    D_fit_err = fit_err_flat[:num_components]
    alpha_fit_err = fit_err_flat[num_components:2 * num_components]
    w_fit_err = fit_err_flat[2 * num_components:]

    D_boot_err = boot_err_flat[:num_components]
    alpha_boot_err = boot_err_flat[num_components:2 * num_components]
    w_boot_err = boot_err_flat[2 * num_components:]

    if fix_d_static:
        D_fit_err[0] = 0.0
        D_boot_err[0] = 0.0
    if model_type in ["Rayleigh Distribution (with Loc Precision)", "Pure 2D Brownian Motion"]:
        alpha_fit_err[:] = 0.0
        alpha_boot_err[:] = 0.0
    elif model_type == "Mixed (Pure Brownian + Anomalous)":
        if num_components == 1:
            alpha_fit_err[0] = 0.0
            alpha_boot_err[0] = 0.0
        else:
            alpha_fit_err[:-1] = 0.0
            alpha_boot_err[:-1] = 0.0
    if num_components == 1:
        w_fit_err[0] = 0.0
        w_boot_err[0] = 0.0

    bic = num_free_params * np.log(N) + 2.0 * nll
    aic = 2.0 * num_free_params + 2.0 * nll
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
        {
            'D': float(D_opt[i]),
            'D_fit_err': float(D_fit_err[i]),
            'D_boot_err': float(D_boot_err[i]),
            'is_static': bool(fix_d_static and i == 0),
            'alpha': float(alpha_opt[i]),
            'alpha_fit_err': float(alpha_fit_err[i]),
            'alpha_boot_err': float(alpha_boot_err[i]),
            'weight': float(w_opt[i]),
            'weight_fit_err': float(w_fit_err[i]),
            'weight_boot_err': float(w_boot_err[i]),
        }
        for i in range(num_components)
    ]

    return {
        'components': num_components,
        'model_type': model_type,
        'bic': float(bic),
        'aic': float(aic),
        'log_lh': -nll,
        'log_likelihood': -nll,
        'has_degenerate': has_degenerate,
        'params': param_dicts,
        'pdf_func': total_pdf_func,
        'component_pdfs': comp_pdfs
    }

# =========================================================================
# MSD VS TIME LAG CURVE FITTING
# =========================================================================
from scipy.special import jnp_zeros

# Precompute the first 20 positive zeros of J_1'(alpha_{1m}) = 0 (Bickel, 2007)
_ALPHA_1M_SQ = jnp_zeros(1, 20) ** 2
_BESSEL_WEIGHTS = 8.0 / (_ALPHA_1M_SQ * (_ALPHA_1M_SQ - 1.0))


def _msd_linear_2d(tau, D, offset):
    """2D Brownian MSD with static localization offset: MSD(tau) = 4*D*tau + 4*sigma^2."""
    return 4.0 * D * tau + offset


def _msd_confined_2d(tau, D, r, offset, spherical_projection=False):
    """
    Exact 2D confined diffusion MSD in a circular domain of radius r
    (Bickel, Physica A 2007; Jin et al., Sci. Adv. 2021):
        MSD(tau) = r^2 * [1 - 8 * sum_{m=1..inf} exp(-alpha_{1m}^2 * D * tau / r^2)
                                              / (alpha_{1m}^2 * (alpha_{1m}^2 - 1))] + 4*sigma^2

    If spherical_projection=True, applies the 4/5 geometric factor for the 2D
    projection of a 3D spherical body lying within the microscope depth of field.
    """
    tau = np.asarray(tau, dtype=np.float64)
    r_sq = max(r * r, 1e-12)
    decay = np.exp(-np.outer(tau, _ALPHA_1M_SQ) * (D / r_sq))
    msd_norm = 1.0 - decay @ _BESSEL_WEIGHTS
    geom_factor = 0.8 if spherical_projection else 1.0
    return geom_factor * r_sq * msd_norm + offset


def fit_msd_curve(tau, msd, model_type="Linear Fit (4D·τ + 4σ²)",
                  num_points=4, loc_err=0.0, spherical_projection=None):
    """
    Fits an ensemble MSD vs. time-lag curve over the first `num_points` valid lags
    and evaluates the fitted model across the full `tau` array.

    Parameters
    ----------
    tau : array-like
        Time lags in seconds.
    msd : array-like
        Mean squared displacements in um^2.
    model_type : str
        "Linear Fit (4D·τ + 4σ²)" or "Confined 2D Diffusion (Circular Domain r)"
        (also detects "3D" / "Sphere" in `model_type` if `spherical_projection` is None).
    num_points : int
        Number of initial lag points to include in the fit (default: 4).
    loc_err : float
        1D localization precision sigma in um (static offset = 4 * sigma^2 = L^2).
    spherical_projection : bool or None
        If True, uses the 4/5 * r^2 plateau limit for a 3D sphere projected onto 2D.
        If None, inferred from `model_type` (defaults to False for a 2D circular domain).

    Returns
    -------
    dict or None
    """
    tau = np.asarray(tau, dtype=np.float64)
    msd = np.asarray(msd, dtype=np.float64)

    valid = np.isfinite(tau) & np.isfinite(msd) & (tau > 0)
    tau_v = tau[valid]
    msd_v = msd[valid]

    if len(tau_v) < 2:
        return None

    n_fit = min(max(2, int(num_points)), len(tau_v))
    tau_fit = tau_v[:n_fit]
    msd_fit = msd_v[:n_fit]

    sigma_um = float(loc_err) if (loc_err is not None and np.isfinite(loc_err)) else 0.0
    offset = 4.0 * (sigma_um ** 2)

    model_str = str(model_type)
    is_confined = "Confined" in model_str
    if spherical_projection is None:
        is_spherical = ("Sphere" in model_str) or ("3D" in model_str)
    else:
        is_spherical = bool(spherical_projection)

    if is_confined:
        geom_factor = 0.8 if is_spherical else 1.0

        # Initial guesses from early slope and plateau
        slope_est = (msd_fit[1] - msd_fit[0]) / max(tau_fit[1] - tau_fit[0], 1e-9)
        d0 = max(1e-5, slope_est / 4.0)
        plateau_est = max(msd_fit[-1] - offset, 1e-4) / geom_factor
        r0 = max(0.05, np.sqrt(plateau_est) * 1.1)

        try:
            popt, pcov = curve_fit(
                lambda t, D, r: _msd_confined_2d(
                    t, D, r, offset, spherical_projection=is_spherical
                ),
                tau_fit,
                msd_fit,
                p0=[d0, r0],
                bounds=([0.0, 1e-4], [np.inf, 100.0]),
                maxfev=5000,
            )
        except Exception:
            return None

        D_opt = float(popt[0])
        r_opt = float(popt[1])
        r_nm = r_opt * 1000.0

        d_err = float(np.sqrt(pcov[0, 0])) if np.isfinite(pcov[0, 0]) else np.nan
        r_err_nm = float(np.sqrt(pcov[1, 1]) * 1000.0) if np.isfinite(pcov[1, 1]) else np.nan

        y_pred_fit = _msd_confined_2d(
            tau_fit, D_opt, r_opt, offset, spherical_projection=is_spherical
        )
        ss_res = float(np.sum((msd_fit - y_pred_fit) ** 2))
        ss_tot = float(np.sum((msd_fit - np.mean(msd_fit)) ** 2))
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else np.nan

        fit_curve = _msd_confined_2d(
            tau_v, D_opt, r_opt, offset, spherical_projection=is_spherical
        )
        d_str = f"{D_opt:.4f} ± {d_err:.4f}" if np.isfinite(d_err) else f"{D_opt:.4f}"
        r_str = f"{r_nm:.1f} ± {r_err_nm:.1f}" if np.isfinite(r_err_nm) else f"{r_nm:.1f}"
        r2_str = f", R² = {r2:.3f}" if np.isfinite(r2) else ""
        geom_tag = " [3D sphere proj.]" if is_spherical else ""

        return {
            "model_type": "Confined 2D",
            "D": D_opt,
            "D_err": d_err,
            "r_um": r_opt,
            "r_nm": r_nm,
            "r_err_nm": r_err_nm,
            "diameter_nm": 2.0 * r_nm,
            "spherical_projection": is_spherical,
            "num_points": n_fit,
            "r_squared": r2,
            "fit_curve": fit_curve,
            "summary": f"D = {d_str} μm²/s, r = {r_str} nm{geom_tag}{r2_str}",
        }

    # Linear 2D MSD Fit: 4*D*tau + 4*sigma^2
    d0 = max(1e-5, (msd_fit[-1] - offset) / (4.0 * tau_fit[-1]))
    try:
        popt, pcov = curve_fit(
            lambda t, D: _msd_linear_2d(t, D, offset),
            tau_fit,
            msd_fit,
            p0=[d0],
            bounds=(0.0, np.inf),
            maxfev=2000,
        )
    except Exception:
        return None

    D_opt = float(popt[0])
    d_err = float(np.sqrt(pcov[0, 0])) if np.isfinite(pcov[0, 0]) else np.nan

    y_pred_fit = _msd_linear_2d(tau_fit, D_opt, offset)
    ss_res = float(np.sum((msd_fit - y_pred_fit) ** 2))
    ss_tot = float(np.sum((msd_fit - np.mean(msd_fit)) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else np.nan

    fit_curve = _msd_linear_2d(tau_v, D_opt, offset)
    d_str = f"{D_opt:.4f} ± {d_err:.4f}" if np.isfinite(d_err) else f"{D_opt:.4f}"
    r2_str = f", R² = {r2:.3f}" if np.isfinite(r2) else ""

    return {
        "model_type": "Linear",
        "D": D_opt,
        "D_err": d_err,
        "num_points": n_fit,
        "r_squared": r2,
        "fit_curve": fit_curve,
        "summary": f"D = {d_str} μm²/s{r2_str}",
    }