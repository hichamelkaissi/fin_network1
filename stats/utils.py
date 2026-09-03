import jax.numpy as jnp
from jax import Array
from typing import Dict, Tuple
import jax

def gini(x_pos: Array) -> Array:
    x_pos = jnp.sort(x_pos)
    n = x_pos.shape[0]
    idx = jnp.arange(1, n + 1)
    return (2 * jnp.sum(idx * x_pos) / (n * jnp.sum(x_pos))) - (n + 1) / n

def hill_estimator(x_pos, k: int):
    x_sorted = jnp.sort(x_pos)[::-1]
    N = x_sorted.shape[0]

    def _ok(_):
        logx = jnp.log(x_sorted)
        logxk = logx[k]
        idx = jnp.arange(N)
        s = jnp.sum(jnp.where(idx < k, logx - logxk, 0.0))
        return jnp.where(s > 0, k / s, jnp.nan)

    def _bad(_):
        return jnp.nan

    return jax.lax.cond(N > k, _ok, _bad, operand=None)


def entropy(x_pos: Array) -> Array:
    p = x_pos / jnp.sum(x_pos)
    return -jnp.sum(p * jnp.log(p))

def sum_topk(sorted_vals, k):
  N = sorted_vals.shape[0]
  idx = jnp.arange(N)
  mask = idx < k
  return jnp.sum(jnp.where(mask, sorted_vals, 0.0))

def median_of_postitive(x):
    x_pos = jnp.where(x > 0, x, jnp.inf)
    xs = jnp.sort(x_pos)
    npos = jnp.sum(x > 0)

    n = jnp.maximum(npos, 1)
    i_hi = (n - 1) // 2
    i_lo = (n - 1) - i_hi

    a = jnp.take(xs, i_lo, mode="clip")
    b = jnp.take(xs, i_hi, mode="clip")
    med = 0.5 * (a + b)

    return jnp.where(npos > 0, med, jnp.nan)


def tail_exposure(x, q: float):
    x_nan = jnp.where(x > 0, x, jnp.nan)
    xq = jnp.nanquantile(x_nan, q)
    X = jnp.sum(x)
    has_pos = jnp.sum(x > 0) > 0
    finite_q = jnp.isfinite(xq)

    mask = has_pos & finite_q & (x > 0) & (x >= xq)
    tail_sum = jnp.sum(jnp.where(mask, x, 0.0))

    return jnp.where((X > 0) & has_pos & finite_q, tail_sum / X, jnp.nan)


def core_ratio(L: Array, q: float):
    x = jnp.sum(L, axis=1)
    total = jnp.sum(L)
    x_pos = jnp.where(x > 0, x, jnp.nan)
    xq = jnp.nanquantile(x_pos, q)
    core = (x >= xq) & (x > 0)
    mask = core[:, None] & core[None, :]
    ecore = jnp.sum(jnp.where(mask, L, 0.0))
    return jnp.where(
        (total > 0),
        ecore / total,
        jnp.nan
    )


def skewness(x_pos: Array) -> Array:
    mu = jnp.mean(x_pos)
    sigma = jnp.std(x_pos)
    return jnp.where(
        sigma > 0,
        jnp.mean(((x_pos - mu) / sigma) ** 3),
        jnp.nan
    )

def kurtosis(x_pos: Array) -> Array:
    mu = jnp.mean(x_pos)
    sigma = jnp.std(x_pos)
    return jnp.where(
        sigma > 0,
        jnp.mean(((x_pos - mu) / sigma) ** 4) - 3,
        jnp.nan
    )
