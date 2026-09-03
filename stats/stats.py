from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array, lax
from jax.scipy.stats import rankdata
from typing import Dict, Tuple

from .utils import (
    gini,
    hill_estimator,
    entropy,
    tail_exposure,
    core_ratio,
    skewness,
    kurtosis,
    sum_topk,
    median_of_postitive,
)


# -------------------------
# Primitive
# -------------------------

def strengths(L: Array) -> Tuple[Array, Array]:
    r = jnp.sum(L, axis=1)
    c = jnp.sum(L, axis=0)
    return r, c


# -------------------------
# Matrix-level statistics
# -------------------------

def sparsity(L: Array) -> Array:
    N = L.shape[0]
    nnz = jnp.sum(L != 0)
    density = nnz / (N * (N - 1))
    return 1 - density


def isolates(L: Array) -> Array:
    r, c = strengths(L)
    return jnp.sum((r == 0) & (c == 0))


def leading_eigenvalue_operator(
        A_mul,
        n: int,
        n_iter: int = 50,
        eps: float = 1e-12,
) -> Array:
    v = jnp.ones((n,))
    v = v / jnp.maximum(jnp.linalg.norm(v), eps)

    def body(_, v):
        w = A_mul(v)
        nrm = jnp.maximum(jnp.linalg.norm(w), eps)
        return w / nrm

    v = lax.fori_loop(0, n_iter, body, v)
    w = A_mul(v)
    lam = jnp.vdot(v, w) / jnp.maximum(jnp.vdot(v, v), eps)

    return jnp.real(lam)


def leading_singular_value(
        A: Array,
        n_iter: int = 50,
        eps: float = 1e-12,
) -> Array:
  
    v = jnp.ones((A.shape[1],), dtype=A.dtype)
    v = v / jnp.maximum(jnp.linalg.norm(v), eps)

    def body(_, v):
        w = A.T @ (A @ v)
        nrm = jnp.maximum(jnp.linalg.norm(w), eps)
        return w / nrm

    v = lax.fori_loop(0, n_iter, body, v)
    Av = A @ v
    sigma = jnp.linalg.norm(Av)

    return sigma


def spectral_stats(L: Array, r: int = 12, top_eigs: int = 1) -> Dict[str, Array]:
    
    # _, S, _ = jnp.linalg.svd(L, full_matrices=False)

    largest_sv = leading_singular_value(L, n_iter=60)

    # leading_energy_r = jnp.sum(S[:r] ** 2) / jnp.sum(S ** 2)

    fro_norm = jnp.sqrt(jnp.sum(L * L))

    # nuc_norm = jnp.sum(S)

    def _A_mul(x):
        return 0.5 * (L @ x + L.T @ x)

    lambda_max_sym = leading_eigenvalue_operator(
        _A_mul,
        L.shape[0],
        n_iter=60,
    )

    # A = 0.5 * (L + L.T)
    # eigvals = jnp.linalg.eigvalsh(A)
    # lambda_top_sym = eigvals[-top_eigs:][::-1]

    out: Dict[str, Array] = {
        'largest_sv': largest_sv,
        # 'leading_energy_r': leading_energy_r,
        'fro_norm': fro_norm,
        # 'nuc_norm': nuc_norm,
        'lambda_max_sym': lambda_max_sym,
    }

    # if top_eigs > 1:
    #     out['lambda_top_sym'] = lambda_top_sym

    return out


# -------------------------
# Strength-based statistics
# -------------------------

def node_level_stats(r: Array, c: Array, ep: float = 0.1) -> Dict[str, Array]:
   
    N = r.shape[0]

    frac_zero_out = jnp.sum(r == 0) / N
    frac_zero_in = jnp.sum(c == 0) / N

    eps_r = ep * median_of_postitive(r)
    eps_c = ep * median_of_postitive(c)

    weakly_connected_out = jnp.sum((r > 0) & (r < eps_r)) / N
    weakly_connected_in = jnp.sum((c > 0) & (c < eps_c)) / N

    return {
        'frac_zero_out': frac_zero_out,
        'frac_zero_in': frac_zero_in,
        'weakly_connected_out': weakly_connected_out,
        'weakly_connected_in': weakly_connected_in,
    }


def dist_stats(
        r_pos: Array,
        c_pos: Array,
        k_hill_r: int,
        k_hill_c: int,
        qLow: float = 0.05,
        qHigh: float = 0.95,
) -> Dict[str, Array]:
    
    median_r = jnp.median(r_pos)
    median_c = jnp.median(c_pos)

    q_rLow = jnp.nanquantile(r_pos, qLow)
    q_rHigh = jnp.nanquantile(r_pos, qHigh)
    q_cLow = jnp.nanquantile(c_pos, qLow)
    q_cHigh = jnp.nanquantile(c_pos, qHigh)

    std_r = jnp.std(r_pos)
    std_c = jnp.std(c_pos)

    min_r = jnp.min(r_pos)
    max_r = jnp.max(r_pos)
    min_c = jnp.min(c_pos)
    max_c = jnp.max(c_pos)

    CV_r = std_r / jnp.mean(r_pos)
    CV_c = std_c / jnp.mean(c_pos)

    skew_r = skewness(r_pos)
    skew_c = skewness(c_pos)
    kurt_r = kurtosis(r_pos)
    kurt_c = kurtosis(c_pos)

    gini_r = gini(r_pos)
    gini_c = gini(c_pos)

    hill_r = hill_estimator(r_pos, k=k_hill_r)
    hill_c = hill_estimator(c_pos, k=k_hill_c)

    ent_r = entropy(r_pos)
    ent_c = entropy(c_pos)

    return {
        'median_r': median_r,
        'median_c': median_c,
        'q_rLow': q_rLow,
        'q_rHigh': q_rHigh,
        'q_cLow': q_cLow,
        'q_cHigh': q_cHigh,
        'std_r': std_r,
        'std_c': std_c,
        'min_r': min_r,
        'max_r': max_r,
        'min_c': min_c,
        'max_c': max_c,
        'CV_r': CV_r,
        'CV_c': CV_c,
        'skew_r': skew_r,
        'skew_c': skew_c,
        'kurt_r': kurt_r,
        'kurt_c': kurt_c,
        'gini_r': gini_r,
        'gini_c': gini_c,
        'hill_r': hill_r,
        'hill_c': hill_c,
        'entropy_r': ent_r,
        'entropy_c': ent_c,
    }


def concentration_stats(
        r: Array,
        c: Array,
        top_r: int = 5,
        top_c: int = 5,
) -> Dict[str, Array]:
   
    N = r.shape[0]
    R = jnp.sum(r)
    C = jnp.sum(c)

    r_sorted = jnp.sort(r)[::-1]
    c_sorted = jnp.sort(c)[::-1]

    hhi_r = jnp.where(R > 0, jnp.sum((r / R) ** 2), jnp.nan)
    hhi_c = jnp.where(C > 0, jnp.sum((c / C) ** 2), jnp.nan)

    CR_r = jnp.where(R > 0, sum_topk(r_sorted, top_r) / R, jnp.nan)
    CR_c = jnp.where(C > 0, sum_topk(c_sorted, top_c) / C, jnp.nan)

    top1 = jnp.maximum(1, jnp.floor(0.01 * N).astype(int))
    top5 = jnp.maximum(1, jnp.floor(0.05 * N).astype(int))

    top1_r = jnp.where(R > 0, sum_topk(r_sorted, top1) / R, jnp.nan)
    top1_c = jnp.where(C > 0, sum_topk(c_sorted, top1) / C, jnp.nan)

    top5_r = jnp.where(R > 0, sum_topk(r_sorted, top5) / R, jnp.nan)
    top5_c = jnp.where(C > 0, sum_topk(c_sorted, top5) / C, jnp.nan)

    return {
        'hhi_r': hhi_r,
        'hhi_c': hhi_c,
        'CR_r': CR_r,
        'CR_c': CR_c,
        'top1_r': top1_r,
        'top1_c': top1_c,
        'top5_r': top5_r,
        'top5_c': top5_c,
    }


def corr_pearson(x: Array, y: Array) -> Array:
    x_mean = jnp.mean(x)
    y_mean = jnp.mean(y)
    num = jnp.sum((x - x_mean) * (y - y_mean))
    den = jnp.sqrt(jnp.sum((x - x_mean) ** 2)) * jnp.sqrt(jnp.sum((y - y_mean) ** 2))
    return jnp.where(den > 0, num / den, jnp.nan)


def corr_spearman(x: Array, y: Array) -> Array:
    X = rankdata(x, method='average')
    Y = rankdata(y, method='average')
    return corr_pearson(X, Y)


def correlation_stats(r: Array, c: Array) -> Dict[str, Array]:
   
    return {
        'pearson': corr_pearson(r, c),
        'spearman': corr_spearman(r, c),
    }


def weighted_risk_stats(L: Array, highq: float = 0.95) -> Dict[str, Array]:
   
    r, c = strengths(L)
    tail_exp_r = tail_exposure(r, highq)
    tail_exp_c = tail_exposure(c, highq)
    scr = core_ratio(L, highq)
    return {
        'tail_share_out': tail_exp_r,
        'tail_share_in': tail_exp_c,
        'scr_out': scr,
    }
