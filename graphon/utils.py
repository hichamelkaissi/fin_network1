from __future__ import annotations

from functools import partial
import jax
import jax.numpy as jnp
from jax import Array, lax, random
from typing import Tuple, List
import logging, os, sys
from logging.handlers import RotatingFileHandler
import subprocess
import psutil



from .sample import (
    drichlet_rows_gamma
)

__all__ = [
    "power_iter_sigma_max",
    "winsorise_by_row",
    "regularise",
    "get_svd",
    "normalise_orientation",
    "make_grid_points",
    "make_block_CV",
    "align_svd_signs",
    "loss",
    "row_stochasticize",
    "clean_Pi",
    "compose_L_from_Pi_pbar",
]

@jax.jit
def power_iter_sigma_max(A: Array, seed: int = 123, max_iter: int = 50) -> Array:
    M = A.T @ A                    # PSD matrix whose eigenvalues = singular^2
    key = random.PRNGKey(seed)
    b_k = random.normal(key, shape=(M.shape[0],))
    b_k = b_k / jnp.linalg.norm(b_k)
    def iter_body(_: int, b: Array) -> Array:
        Mb = M @ b
        Mb_norm = jnp.linalg.norm(Mb)
        return Mb / Mb_norm

    b_k = lax.fori_loop(0, max_iter, iter_body, b_k)
    Mb_final = M @ b_k
    lambda_max = jnp.dot(b_k, Mb_final)        # Rayleigh quotient
    sigma_max = jnp.sqrt(lambda_max)           # singular value = sqrt(eigenvalue)
    return sigma_max


def winsorise_by_row(x: Array, q_high: float = 0.995) -> Array:
    cap = jnp.quantile(x, q_high)
    return jnp.clip(x, a_min=None, a_max=cap)

winsorise_by_row = jax.vmap(winsorise_by_row, in_axes=0, out_axes=0)

def regularise(
        A: Array,
        kappa_r: float = 1.0,
        kappa_c: float = 1.0,
        eps: float = 1e-12,
) -> Tuple[Array, Tuple[Array, Array]]:
    N = A.shape[0]
    A = A.at[jnp.diag_indices(N)].set(0.0)

    r = A.sum(axis=1)
    c = A.sum(axis=0)

    # JIT-safe: includes zeros
    r_med = jnp.median(r)
    c_med = jnp.median(c)

    tau_r = kappa_r * r_med
    tau_c = kappa_c * c_med

    sr = jnp.power(r + tau_r + eps, -0.5)
    sc = jnp.power(c + tau_c + eps, -0.5)

    A_tilde = (sr[:, None] * A) * sc[None, :]
    return A_tilde, (sr, sc)

@jax.jit
def get_svd(A_tilde: Array) -> Tuple[Array, Array, Array]:
    U, s, Vh = jnp.linalg.svd(A_tilde, full_matrices=False)
    V = Vh.T
    return U, s, V


def normalise_orientation(top_v: Array) -> Array:
    sum_top_v = jnp.sum(top_v, axis=0)

    signs = jnp.where(sum_top_v < 0, -1., 1)

    top_v = top_v * signs

    return top_v

def make_grid_points(G: int = 200) -> Tuple[Array, Array, float]:

    a = jnp.arange(G)
    x = (a + .5) / G
    y = x
    quad_weight = 1/G

    return x, y, quad_weight

def make_block_CV(N: int, K: int = 4, seed: int = 123) -> Tuple[List[Array], Array]:
    key = random.PRNGKey(seed)

    base= N//K

    r = N % K

    perm = random.permutation(key, N)

    start = 0
    blocks = []

    for k in range(K):
        size = base + (1 if k < r else 0)
        blocks.append(perm[start:start+size])

        start += size

    return blocks, perm

def align_svd_signs(U: Array, V: Array) -> Tuple[Array, Array]:
    idx = jnp.argmax(jnp.abs(U), axis=0)

    signs = jnp.sign(U[idx, jnp.arange(U.shape[1])])
    signs = jnp.where(signs == 0, 1.0, signs)

    U_aligned = U * signs
    V_aligned = V * signs

    return U_aligned, V_aligned

def loss(A_true: Array, A_pred: Array) -> Tuple[Array, Array, Array, Array]:
    N = A_true.shape[0]
    mask_offdiag = 1.0 - jnp.eye(N)

    #pos_mask = (A_true > 0) * mask_offdiag
    #zero_mask = (A_true == 0) * mask_offdiag

    pos_mask = (A_true > 0)
    zero_mask = (A_true == 0)

    omega_p = jnp.sum(pos_mask)
    omega_z = jnp.sum(zero_mask)

    sse_p = jnp.sum(((A_pred - A_true) ** 2) * pos_mask)
    sse_z = jnp.sum((A_pred ** 2) * zero_mask)

    return sse_p, sse_z, omega_p, omega_z

def row_stochasticize(A: Array, method: str = "softplus", tau: float = 1.0, eps: float = 1e-12) -> Array:
    if method == "softplus":
        A_pos = jax.nn.softplus(A)
        return A_pos / jnp.maximum(jnp.sum(A_pos, axis=1, keepdims=True), eps)

    if method == "softmax":
        return jax.nn.softmax(A / tau, axis=1)

    if method == "clip":
        A_pos = jnp.maximum(A, 0.0)
        return A_pos / jnp.maximum(jnp.sum(A_pos, axis=1, keepdims=True), eps)

    raise ValueError("Unknown method.")

@jax.jit
def clean_Pi(Pi: Array, tau: float = 1e-12, eps: float = 1e-12) -> Array:

    N = Pi.shape[0]

    Pi = jnp.maximum(Pi, 0.0)

    Pi = Pi * (1.0 - jnp.eye(N))

    Pi = jnp.where(Pi > tau, Pi, 0)

    rs = jnp.sum(Pi, axis=1, keepdims=True)
    Pi = jnp.where(rs > eps, Pi / rs, 0)

    return Pi

def compose_L_from_Pi_pbar(Pi: Array, pbar: Array) -> Array:
    return Pi * pbar[:, None]

@jax.jit
def density_offdiag_positive(A: Array) -> Array:
    n = A.shape[0]
    pos_all = jnp.sum(A > 0)
    pos_diag = jnp.sum(jnp.diag(A) > 0)
    denom = n * (n - 1)
    return (pos_all - pos_diag) / denom

@partial(jax.jit, static_argnames=("M",))
def eval_density_L_for_kappa_streaming(key, W, kappa, pbar_new, M=1):
    def body(i, acc):
        k = random.fold_in(key, i)
        Pi = drichlet_rows_gamma(k, W, kappa)
        Pi = clean_Pi(Pi)
        L  = compose_L_from_Pi_pbar(Pi, pbar_new)
        d  = density_offdiag_positive(L)
        return acc + d

    acc = lax.fori_loop(0, M, body, 0.0)
    return acc / M

@partial(jax.jit, static_argnames=("max_iter","M"))
def bisection_kappa_for_L_density(
        key, W, pbar_new, target_d,
        k_lo, k_hi,
        tol=1e-3,
        max_iter=30,
        M=3,
):
    lo = jnp.log(k_lo)
    hi = jnp.log(k_hi)

    def cond(state):
        key, lo, hi, i = state
        mid = 0.5 * (lo + hi)
        k_mid = jnp.exp(mid)
        key1, _ = random.split(key)
        d_mid = eval_density_L_for_kappa_streaming(key1, W, k_mid, pbar_new, M=M)
        return (jnp.abs(d_mid - target_d) > tol) & (i < max_iter)

    def body(state):
        key, lo, hi, i = state
        mid = 0.5 * (lo + hi)
        k_mid = jnp.exp(mid)

        key1, key2, key3 = random.split(key, 3)
        d_lo  = eval_density_L_for_kappa_streaming(key1, W, jnp.exp(lo), pbar_new, M=M)
        d_mid = eval_density_L_for_kappa_streaming(key2, W, k_mid,       pbar_new, M=M)

        s_lo  = d_lo  - target_d
        s_mid = d_mid - target_d

        keep_left = (s_lo * s_mid) <= 0.0
        hi2 = jnp.where(keep_left, mid, hi)
        lo2 = jnp.where(keep_left, lo,  mid)
        return (key3, lo2, hi2, i + 1)

    key, lo, hi, _ = lax.while_loop(cond, body, (key, lo, hi, 0))
    return key, jnp.exp(0.5 * (lo + hi))



def maybe_block(x):
    if hasattr(x, 'block_until_ready'):
        return x.block_until_ready()
    return x
