from __future__ import annotations

from typing import Dict, Tuple, TypedDict

import jax
import jax.numpy as jnp
from jax import Array, lax, random

from .graphon import (
    make_spline_knots,
    spline_design_matrix,
    spline_penalty_matrix,
    ridge_reg_spline,
)

# ============================================================
# Helpers (JAX-safe)
# ============================================================

def _safe_quantile(x: Array, q: float, default: float = 0.0) -> Array:
    if x.size > 0:
        return jnp.quantile(x, q)
    else:
        return default

def _winsorize_log1p(y_log1p: Array, lo: Array, hi: Array) -> Array:
    return jnp.clip(y_log1p, lo, hi)

def _bin_ids_from_t(t: Array, n_bins: int) -> Array:
    t = jnp.clip(t, 0.0, 1.0 - 1e-12)
    b = jnp.floor(t * n_bins).astype(jnp.int32)
    return jnp.clip(b, 0, n_bins - 1)

def _prepare_bin_pools(values: Array, bin_id: Array, n_bins: int) -> Tuple[Array, Array, Array]:
    #remove values for args!
    perm = jnp.argsort(bin_id)
    bin_sorted = bin_id[perm]
    counts = jnp.bincount(bin_sorted, length=n_bins)
    offsets = jnp.concatenate([jnp.array([0], dtype=jnp.int32), jnp.cumsum(counts[:-1])])
    return perm, offsets, counts

def _sample_from_bin_pool(
        key: Array,
        values: Array,
        perm: Array,
        offsets: Array,
        counts: Array,
        bin_id_new: Array,
        fallback_values: Array,
) -> Tuple[Array, Array]:
    N_new = bin_id_new.shape[0]
    key, sub1, sub2 = random.split(key, 3)

    cnt = counts[bin_id_new]
    cnt_safe = jnp.maximum(cnt, 1)

    u = random.randint(sub1, shape=(N_new,), minval=0, maxval=cnt_safe)
    pick_pos = offsets[bin_id_new] + u
    picked_idx = perm[pick_pos]
    drawn = values[picked_idx]

    #needs a local fallback!
    fbN = fallback_values.shape[0]
    fb_idx = random.randint(sub2, shape=(N_new,), minval=0, maxval=jnp.maximum(fbN, 1))
    fb_draw = fallback_values[fb_idx]

    drawn = jnp.where(cnt > 0, drawn, fb_draw)
    return key, drawn

def _log_affine_calibrate(
        y_new_pos: Array,
        y_obs_pos: Array,
        eps: float,
        b_cap: float = 5.0,
        q1: float = 0.05,
        q2: float = 0.95,
) -> Tuple[Array, Array]:
    def _do(_):
        m_new = jnp.quantile(y_new_pos, q1)
        m_obs = jnp.quantile(y_obs_pos, q1)
        q_new = jnp.quantile(y_new_pos, q2)
        q_obs = jnp.quantile(y_obs_pos, q2)

        denom = jnp.maximum(q_new - m_new, eps)
        b = (q_obs - m_obs) / denom
        b = jnp.clip(b, 1.0 / b_cap, b_cap)
        a = m_obs - b * m_new
        return a, b

    a, b = lax.cond(
        (y_new_pos.size > 0) & (y_obs_pos.size > 0),
        _do,
        lambda _: (jnp.array(0.0), jnp.array(1.0)),
        operand=None,
        )
    return a, b
#---------------------
# MLE
#----------------------
class GPDResult(TypedDict):
    u: float
    q: float
    Nu: int
    xi: float
    beta: float
    loglik: float
    xi_ci: tuple[float, float]
    beta_ci: tuple[float, float]
    success: bool

def _gpd_loglikelihood(
    params: Array,
    excess: Array
    ) -> Array:
    xi = params[0]
    beta = jnp.exp(params[1])
    nu = excess.shape[0]
    z = 1 + xi * excess / beta
    ll_general = (
        -nu * jnp.log(beta)
        - (1.0 + 1.0 / xi)
        * jnp.sum(jnp.log(z))
    )
    ll_exp = (
        -nu * jnp.log(beta)
        - jnp.sum(excess) / beta
    )
    ll = lax.cond(
        jnp.abs(xi) < 1e-6,
        lambda _: ll_exp,
        lambda _: ll_general,
        operand=None
    )
    valid = jnp.all(z > 0)
    ll = lax.cond(
        valid,
        lambda _: ll,
        lambda _: -jnp.inf,
        operand=None
    )
    return -ll
def gpd_MLE(
    pbar: Array,
    u: Array | None = None,
    q: float = 0.95,
    ) -> GPDResult:

    pos_pbar = pbar[pbar > 0.0]


    uq = jnp.quantile(
        pos_pbar,
        q,
    )
    u = u if u is not None else uq
    mask = pos_pbar > u
    excess = (
        pos_pbar[mask]
        - u
    )
    Nu = excess.shape[0]
    xi0 = 0.1
    beta0 = jnp.mean(excess)
    log_beta0 = jnp.log(beta0)
    x0 = jnp.asarray([
        xi0,
        log_beta0,
    ])
    res = minimize(
        _gpd_loglikelihood,
        x0,
        args=(excess,),
        method="BFGS",
    )
    xi_hat = res.x[0]
    beta_hat = jnp.exp(
        res.x[1]
    )
    success = bool(
        res.success
    )
    loglik = -res.fun
    z = 1.96
    cov_theta = res.hess_inv
    se_xi = jnp.sqrt(
        cov_theta[0, 0]
    )
    se_log_beta = jnp.sqrt(
        cov_theta[1, 1]
    )
    xi_ci = (
        float(xi_hat - z * se_xi),
        float(xi_hat + z * se_xi),
    )
    log_beta_hat = jnp.log(
        beta_hat
    )
    beta_ci = (
        float(
            jnp.exp(
                log_beta_hat
                - z * se_log_beta
            )
        ),

        float(
            jnp.exp(
                log_beta_hat
                + z * se_log_beta
            )
        ),
    )
    return {
        "u": float(u),
        "q": float(q),
        "Nu": int(Nu),
        "xi": float(xi_hat),
        "beta": float(beta_hat),
        "loglik": float(loglik),
        "xi_ci": xi_ci,
        "beta_ci": beta_ci,
        "success": success,
    }
def gpd_hill_estimator(
    pbar: Array,
    k: int,
) -> Array:

    pos_pbar = pbar[pbar > 0.0]

    x = jnp.sort(
        pos_pbar,
        descending=True,
    )

    threshold = x[k - 1]

    j = jnp.arange(x.shape[0])

    mask = j < (k - 1)

    logs = jnp.where(
        mask,
        jnp.log(x / threshold),
        0.0,
    )

    xi_hill = jnp.sum(logs) / (k - 1)

    return xi_hill
#---------------------
# End of MLE
#----------------------
#Replace this with MLE!
def _fit_gpd_moments(excess: Array, eps: float = 1e-12) -> Tuple[Array, Array]:
    m = jnp.mean(excess)
    v = jnp.var(excess)

    r = v / jnp.maximum(m * m, eps)
    xi = 0.5 * (1.0 - 1.0 / jnp.maximum(r, 1.0 + 1e-6))
    xi = jnp.clip(xi, -0.2, 0.45)
    beta = m * (1.0 - xi)
    beta = jnp.maximum(beta, eps)
    return xi, beta

def _sample_gpd_excess(key: Array, xi: Array, beta: Array, shape) -> Tuple[Array, Array]:
    #typing: xi, beta will need to be float!
    key, sub = random.split(key)
    U = random.uniform(sub, shape=shape, minval=1e-12, maxval=1.0 - 1e-12)

    e = lax.cond(
        jnp.abs(xi) > 1e-6,
        lambda _: (beta / xi) * ((1.0 - U) ** (-xi) - 1.0),
        lambda _: -beta * jnp.log(1.0 - U),
        operand=None,
        )
    return key, e

# ============================================================
# 1) Fit size function with hurdle + residual pools (+ tail + p_tail_bin)
# ============================================================

def fit_size_function_hurdle(
        t_l: Array,
        pbar: Array,
        K: int = 15,
        degree: int = 3,
        ridge_alpha: float = 1e-3,
        penalty_order: int = 2,
        eps: float = 1e-12,
        resid_winsor_q: float = 0.999, #too high?
        n_bins: int = 30,
        k_tail: int = 40,
        min_tail_n: int = 20,
        fit_tail_prob: bool = True,
        tail_prob_a: float = 0.5,
        tail_prob_b: float = 200.0,
) -> Dict:
    #asarray not needed in the next two lines
    pbar = jnp.asarray(pbar)
    t_l = jnp.asarray(t_l)

    y = jnp.log1p(jnp.maximum(pbar, eps))

    knots_l, _ = make_spline_knots(t_l, t_l, K=K, degree=degree) 
    B = spline_design_matrix(t_l, knots_l, degree=degree)
    Omega = spline_penalty_matrix(B.shape[1], order=penalty_order)

    theta = ridge_reg_spline(B, y[:, None], alpha=ridge_alpha, Omega=Omega)[0]
    mu = B @ theta
    resid = y - mu

    r_lo = _safe_quantile(resid, 1.0 - resid_winsor_q, default=0.0)
    r_hi = _safe_quantile(resid, resid_winsor_q, default=0.0)
    resid_w = jnp.clip(resid, r_lo, r_hi)

    y_pos = y[pbar > 0.0]
    #Use if statement instead of jnp.where!
    y_hi = jnp.where(y_pos.size > 0, jnp.max(y_pos), jnp.max(y))
    y_lo = jnp.where(y_pos.size > 0, jnp.min(y_pos), jnp.min(y))

    bin_id = _bin_ids_from_t(t_l, n_bins=n_bins)

    is_zero = (pbar == 0.0).astype(jnp.float32)
    counts = jnp.bincount(bin_id, length=n_bins).astype(jnp.float32)
    zeros = jnp.bincount(bin_id, weights=is_zero, length=n_bins)
    pi0_bin = zeros / jnp.maximum(counts, 1.0) #!!

    mask_pos = (pbar > 0.0)
    resid_pos = resid_w[mask_pos]
    bin_pos = bin_id[mask_pos]
    perm_r, offsets_r, counts_r = _prepare_bin_pools(resid_pos, bin_pos, n_bins=n_bins)

    N_pos = y_pos.shape[0]
    qu = 1.0 - (k_tail / jnp.maximum(N_pos, 1))
    qu = jnp.clip(qu, 0.9, 0.999) #!!! upper end too high!?!
    u_thr = jnp.quantile(y_pos, qu) if N_pos > 0 else jnp.array(0.0)
    excess = (y_pos - u_thr)
    excess = excess[excess > 0.0]

    xi, beta = lax.cond(
        excess.size >= min_tail_n,
        lambda _: _fit_gpd_moments(excess, eps=eps),
        lambda _: (jnp.array(0.1), jnp.array(0.1)), #would be questionable with MLE
        operand=None,
        )
    tail = {
        'u': u_thr,
        'xi': xi,
        'beta': beta,
        'qu': qu,
        'k_tail': k_tail,
        'N_pos': N_pos,
        'n_exc': k_tail,
    }

    if fit_tail_prob:
        is_tail = (y > u_thr).astype(jnp.float32)
        tail_counts = jnp.bincount(bin_id, weights=is_tail, length=n_bins).astype(jnp.float32)
        p_tail_bin = (tail_counts + tail_prob_a) / (counts + tail_prob_a + tail_prob_b) #shrinkage is too agressive
    else:
        p_tail_bin = jnp.zeros((n_bins,), dtype=jnp.float32)

    return {
        'knots': knots_l,
        'degree': degree,
        'K': K,
        'ridge_alpha': ridge_alpha,
        'penalty_order': penalty_order,
        'theta': theta,
        'eps': eps,

        'resid_global': resid_pos,
        'resid_sigma': jnp.std(resid_pos),
        'n_bins': n_bins,
        'pi0_bin': pi0_bin,
        'resid_pos': resid_pos,
        'bin_pos': bin_pos,
        'perm_r': perm_r,
        'offsets_r': offsets_r,
        'counts_r': counts_r,

        'y_lo': y_lo,
        'y_hi': y_hi,

        'tail': tail,
        'p_tail_bin': p_tail_bin,
    }

# ============================================================
# 2) Predict mu(t)
# ============================================================

def predict_log1p_pbar(size_model: Dict, t_new: Array) -> Array:
    B_new = spline_design_matrix(t_new, size_model['knots'], degree=size_model['degree'])
    return B_new @ size_model['theta']

# ============================================================
# 3) Sample pbar with hurdle + residual bootstrap + calibration + EVT (guardrail/mixture/off)
# ============================================================

def sample_pbar_hurdle(
        key: Array,
        size_model: Dict,
        t_new: Array,
        pbar_obs: Array,
        noise_scale: float = 1.0,
        calibrate_log_affine: bool = True,
        cal_q1: float = 0.05,
        cal_q2: float = 0.95,
        b_cap: float = 5.0,
        use_evt_guardrail: bool = True,
        p_tail_refresh: float = 0.05,
        guardrail_mult: float = 3.0,
        evt_mode: str = 'guardrail',
        evt_mix_mult: float = 1.0,
        evt_mix_shrink: float = 0.7,
        evt_p_tail_cap: float = 0.01,
        evt_mix_use_global: bool = True,
        enforce_pos_sum: bool = False,
        scale_cap: float = 5.0,
        eps: float = 1e-12,
) -> Tuple[Array, Array, Array]:
    t_new = jnp.asarray(t_new) #remove casting
    pbar_obs = jnp.asarray(pbar_obs) #remove casting

    mu_y = predict_log1p_pbar(size_model, t_new)
    N = mu_y.shape[0]

    bin_id_new = _bin_ids_from_t(t_new, n_bins=size_model['n_bins'])
    pi0_new = size_model['pi0_bin'][bin_id_new]

    key, sub = random.split(key)
    u_unif = random.uniform(sub, shape=(N,))
    active = (u_unif > pi0_new)

    key, resid_draw = _sample_from_bin_pool(
        key=key,
        values=size_model['resid_pos'],
        perm=size_model['perm_r'],
        offsets=size_model['offsets_r'],
        counts=size_model['counts_r'],
        bin_id_new=bin_id_new,
        fallback_values=size_model['resid_global'],
    )

    y_body = mu_y + noise_scale * resid_draw  #noise_scale not the only control - it is one of many
    y_body = jnp.where(active, y_body, 0.0)

    if calibrate_log_affine:
        y_obs_pos = jnp.log1p(jnp.maximum(pbar_obs[pbar_obs > 0.0], eps))
        y_new_pos = y_body[active]
        a, b = _log_affine_calibrate(
            y_new_pos=y_new_pos,
            y_obs_pos=y_obs_pos,
            eps=eps,
            b_cap=b_cap,
            q1=cal_q1,
            q2=cal_q2,
        )
        y_body = jnp.where(active, a + b * y_body, 0.0)

    y_body = jnp.where(
        active,
        _winsorize_log1p(y_body, size_model['y_lo'], size_model['y_hi']),
        0.0,
    )

    pbar_pos_obs = pbar_obs[pbar_obs > 0.0]
    tail = size_model.get('tail', None)

    y_new = y_body

    if (tail is not None) and (pbar_pos_obs.size > 0) and (evt_mode != 'off'):
        u_thr = tail['u']
        xi = tail['xi']
        beta = tail['beta']

        if evt_mode == 'guardrail':
            if use_evt_guardrail: #we do not need this arg
                y_max_obs = jnp.max(jnp.log1p(jnp.maximum(pbar_pos_obs, eps)))

                in_tail = active & (y_body > u_thr)

                key, sub = random.split(key)
                do_refresh = in_tail & (random.uniform(sub, shape=y_body.shape) < p_tail_refresh)

                key, exc = _sample_gpd_excess(key, xi, beta, shape=y_body.shape)
                y_evt = u_thr + exc

                y_new = jnp.where(do_refresh, jnp.maximum(y_body, y_evt), y_body)

                y_guard = y_max_obs + jnp.log(guardrail_mult)
                y_new = jnp.where(active, jnp.minimum(y_new, y_guard), 0.0)

        elif evt_mode == 'mixture':
            p_tail_bin = size_model.get('p_tail_bin', None)
            if p_tail_bin is None:
                p_tail = jnp.zeros((N,), dtype=jnp.float32)
            else:
                p_tail = p_tail_bin[bin_id_new]
                p_tail = jnp.clip(evt_mix_mult * p_tail, 0.0, 1.0)

                if evt_mix_use_global:
                    N_pos = jnp.maximum(size_model['tail']['N_pos'], 1)
                    k_tail = jnp.maximum(size_model['tail']['k_tail'], 1)
                    p_global = jnp.clip(k_tail / N_pos, 0.0, 1.0)
                    p_tail = evt_mix_shrink * p_tail + (1.0 - evt_mix_shrink) * p_global

                p_tail = jnp.minimum(p_tail, evt_p_tail_cap)

            key, sub = random.split(key)
            u = random.uniform(sub, shape=(N,))
            do_tail = active & (u < p_tail)

            key, exc = _sample_gpd_excess(key, xi, beta, shape=(N,))
            y_tail = u_thr + exc

            y_max_obs = jnp.max(jnp.log1p(jnp.maximum(pbar_pos_obs, eps)))
            y_guard = y_max_obs + jnp.log(guardrail_mult)
            y_tail = jnp.minimum(y_tail, y_guard)

            y_new = jnp.where(do_tail, y_tail, y_body)

        else:
            raise ValueError("evt_mode must be one of: 'guardrail', 'mixture', 'off'")

    pbar_new = jnp.where(active, jnp.expm1(jnp.maximum(y_new, 0.0)), 0.0)
    pbar_new = jnp.maximum(pbar_new, 0.0)

    if enforce_pos_sum and (pbar_pos_obs.size > 0): #this hinders EVT
        target = jnp.sum(pbar_pos_obs)
        current = jnp.sum(pbar_new[pbar_new > 0.0])
        scale = target / jnp.maximum(current, eps)
        scale = jnp.clip(scale, 1.0 / scale_cap, scale_cap)
        pbar_new = jnp.where(pbar_new > 0.0, pbar_new * scale, 0.0)

    return pbar_new, mu_y, pi0_new
