import numpy as np
from scipy.optimize import minimize
from scipy.stats import gamma

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def fit_gamma_diffusion_mle(d_values: np.ndarray, num_components: int = 1):
    """
    Fits a 1 to N component Gamma mixture model to diffusion coefficients (D) using MLE.
    Returns parameter estimates, log-likelihood, BIC, and callable PDF.
    """
    d_clean = d_values[~np.isnan(d_values) & (d_values > 0)]
    n_obs = len(d_clean)
    if n_obs < 5:
        return None

    if num_components == 1:
        shape, loc, scale = gamma.fit(d_clean, floc=0)
        log_likelihood = float(np.sum(gamma.logpdf(d_clean, shape, loc=0, scale=scale)))
        k_params = 2  # shape, scale
        bic = k_params * np.log(n_obs) - 2 * log_likelihood

        return {
            "components": 1,
            "params": [{"weight": 1.0, "shape": shape, "scale": scale, "mean_D": shape * scale}],
            "log_likelihood": log_likelihood,
            "bic": bic,
            "n_obs": n_obs,
            "pdf_func": lambda x: gamma.pdf(x, shape, loc=0, scale=scale)
        }

    else:
        k = num_components
        k_params = (k - 1) + 2 * k  # (k-1) weights + k shapes + k scales

        mean_d = np.mean(d_clean)
        var_d = np.var(d_clean) if np.var(d_clean) > 0 else 0.1

        # Initial parameters setup
        init_v = np.zeros(k)
        init_shapes = np.full(k, max(1.0, (mean_d ** 2) / var_d))
        scales_grid = np.linspace(mean_d / (2 * k), mean_d * 1.5, k)
        
        x0 = np.concatenate([init_v, init_shapes, scales_grid])

        def nll(params):
            v = params[:k]
            shapes = params[k:2*k]
            scales = params[2*k:]

            if np.any(shapes <= 1e-4) or np.any(scales <= 1e-4):
                return 1e12

            weights = softmax(v)
            pdf_matrix = np.column_stack([
                gamma.pdf(d_clean, shapes[i], loc=0, scale=scales[i]) for i in range(k)
            ])
            mixture_pdf = np.dot(pdf_matrix, weights)
            return -np.sum(np.log(np.maximum(mixture_pdf, 1e-300)))

        res = minimize(nll, x0, method='Nelder-Mead', options={'maxiter': 2000 * k})

        opt_v = res.x[:k]
        weights = softmax(opt_v)
        opt_shapes = np.abs(res.x[k:2*k])
        opt_scales = np.abs(res.x[2*k:])
        
        log_likelihood = -float(res.fun)
        bic = k_params * np.log(n_obs) - 2 * log_likelihood

        comp_details = [
            {"weight": weights[i], "shape": opt_shapes[i], "scale": opt_scales[i], "mean_D": opt_shapes[i] * opt_scales[i]}
            for i in range(k)
        ]
        comp_details.sort(key=lambda c: c['mean_D'])

        def mixture_pdf(x):
            pdf_stack = np.column_stack([
                gamma.pdf(x, comp['shape'], loc=0, scale=comp['scale']) for comp in comp_details
            ])
            w_arr = np.array([comp['weight'] for comp in comp_details])
            return np.dot(pdf_stack, w_arr)

        return {
            "components": k,
            "params": comp_details,
            "log_likelihood": log_likelihood,
            "bic": bic,
            "n_obs": n_obs,
            "pdf_func": mixture_pdf
        }


def fit_jump_distances_mle(r_jumps: np.ndarray, dt: float, num_components: int = 1, loc_error: float = 0.0):
    """
    Fits 1 to N component 2D Rayleigh Brownian jump distance distributions P(r, dt).
    P(r, dt) = sum(w_i * [2r / msd_i] * exp[-r^2 / msd_i])
    where msd_i = 4 * D_i * dt + 4 * sigma^2
    """
    r_clean = r_jumps[~np.isnan(r_jumps) & (r_jumps > 0)]
    n_obs = len(r_clean)
    if n_obs < 5:
        return None

    def Rayleigh_msd_pdf(r, D):
        msd = 4.0 * D * dt + 4.0 * (loc_error ** 2)
        if msd <= 0:
            return np.zeros_like(r)
        return (2.0 * r / msd) * np.exp(-(r ** 2) / msd)

    if num_components == 1:
        def nll(params):
            D = params[0]
            if D <= 0:
                return 1e12
            p = Rayleigh_msd_pdf(r_clean, D)
            return -np.sum(np.log(np.maximum(p, 1e-300)))

        res = minimize(nll, x0=[0.1], method='L-BFGS-B', bounds=[(1e-6, None)])
        opt_D = float(res.x[0])
        log_likelihood = -float(res.fun)
        k_params = 1
        bic = k_params * np.log(n_obs) - 2 * log_likelihood

        return {
            "components": 1,
            "params": [{"weight": 1.0, "D": opt_D}],
            "log_likelihood": log_likelihood,
            "bic": bic,
            "n_obs": n_obs,
            "pdf_func": lambda x: Rayleigh_msd_pdf(x, opt_D)
        }

    else:
        k = num_components
        k_params = (k - 1) + k  # (k-1) weights + k diffusion coefficients

        init_v = np.zeros(k)
        init_D = np.linspace(0.01, 1.0, k)
        x0 = np.concatenate([init_v, init_D])

        def nll(params):
            v = params[:k]
            ds = params[k:]
            if np.any(ds <= 1e-6):
                return 1e12

            weights = softmax(v)
            pdf_matrix = np.column_stack([Rayleigh_msd_pdf(r_clean, ds[i]) for i in range(k)])
            mixture_pdf = np.dot(pdf_matrix, weights)
            return -np.sum(np.log(np.maximum(mixture_pdf, 1e-300)))

        res = minimize(nll, x0, method='Nelder-Mead', options={'maxiter': 2000 * k})

        weights = softmax(res.x[:k])
        opt_ds = np.abs(res.x[k:])
        
        log_likelihood = -float(res.fun)
        bic = k_params * np.log(n_obs) - 2 * log_likelihood

        comp_details = [
            {"weight": weights[i], "D": opt_ds[i]} for i in range(k)
        ]
        comp_details.sort(key=lambda c: c['D'])

        def mixture_pdf(x):
            pdf_stack = np.column_stack([Rayleigh_msd_pdf(x, c['D']) for c in comp_details])
            w_arr = np.array([c['weight'] for c in comp_details])
            return np.dot(pdf_stack, w_arr)

        return {
            "components": k,
            "params": comp_details,
            "log_likelihood": log_likelihood,
            "bic": bic,
            "n_obs": n_obs,
            "pdf_func": mixture_pdf
        }