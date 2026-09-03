from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional

import jax
import jax.numpy as jnp
from jax import Array
from .utils import align_svd_signs

def left_coordinate(U: Array, r: int = 20) -> Tuple[Array, Array]:
    N = U.shape[0]

    
    Z = U[:, :r]                           # (N, r)

    
    Zc = Z - jnp.mean(Z, axis=0, keepdims=True)

    
    _, _, Wt = jnp.linalg.svd(Zc, full_matrices=False)
    w = Wt[0, :]                           # (r,)
    #w2 = Wt[1, :]                          # (r, )

    
    t = Zc @ w                             # (N,)
    #t = t @ w
    
    order = jnp.argsort(t)
    ranks = jnp.empty_like(order).at[order].set(jnp.arange(N))
    u_hat = ranks / (N - 1)

    return u_hat, t

def right_coordinate(V: Array, r: int = 20) -> Tuple[Array, Array]:
    N = V.shape[0]

    Z = V[:, :r]  # (N, r)
    Zc = Z - jnp.mean(Z, axis=0, keepdims=True)

    _, _, Wt = jnp.linalg.svd(Zc, full_matrices=False)
    w = Wt[0, :]  # (r,)

    t = Z @ w     # (N,)

    order = jnp.argsort(t)
    ranks = jnp.empty_like(order).at[order].set(jnp.arange(N))
    v_hat = ranks / (N - 1)

    return v_hat, t

def make_spline_knots(tl: Array, tr: Array, K: int = 15, degree: int = 3) -> Tuple[Array, Array]:
    tt = jnp.concatenate([tl, tr])

    qs = jnp.linspace(0.0, 1.0, K)
    base_knots = jnp.quantile(tt, qs)
    base_knots = jnp.maximum.accumulate(base_knots)

    t0_l, tM_l = jnp.min(tl), jnp.max(tl)
    t0_r, tM_r = jnp.min(tr), jnp.max(tr)

    left_idx_l  = jnp.full((degree,), t0_l)
    right_idx_l = jnp.full((degree,), tM_l)
    left_idx_r  = jnp.full((degree,), t0_r)
    right_idx_r = jnp.full((degree,), tM_r)

    knots_l = jnp.concatenate([left_idx_l, base_knots, right_idx_l])
    knots_r = jnp.concatenate([left_idx_r, base_knots, right_idx_r])
    return knots_l, knots_r

def _bspline_0d(x: Array, knots: Array, degree: int = 3) -> Array:
    M = knots.shape[0] - 1
    idx = jnp.arange(M)

    in_interval = (x >= knots[idx]) & (x < knots[idx + 1])

    in_last = (idx == (M - 1)) & (x == knots[-1])

    return jnp.where(in_interval | in_last, 1.0, 0.0)

def spline_1d(x: Array, knots: Array, degree: int = 3) -> Array:
    B_prev = _bspline_0d(x, knots, degree=degree)

    for p in range(1, degree + 1):
        L = B_prev.shape[0]
        M = L - 1
        idx = jnp.arange(M)

        ti    = knots[idx]
        ti_p  = knots[idx + p]
        ti_1  = knots[idx + 1]
        ti_p1 = knots[idx + p + 1]

        denom1 = ti_p - ti
        denom2 = ti_p1 - ti_1

        a = jnp.where(denom1 > 0.0, (x - ti) / denom1, 0.0)
        b = jnp.where(denom2 > 0.0, (ti_p1 - x) / denom2, 0.0)

        B_prev = a * B_prev[idx] + b * B_prev[idx + 1]

    return B_prev

def spline_design_matrix(t: Array, knots: Array, degree: int = 3) -> Array:
    return jax.vmap(lambda x: spline_1d(x, knots, degree))(t)

def spline_penalty_matrix(M: int, order: int = 2) -> Array:
    D = jnp.eye(M)

    def first_diff(A: Array) -> Array:
        # (k-1, k): A[i+1]-A[i]
        return A[1:, :] - A[:-1, :]

    for _ in range(order):
        D = first_diff(D)

    Omega = D.T @ D
    return Omega


def ridge_reg_spline(B: Array, U: Array, alpha: float = 1e-3, Omega: Optional[Array] = None) -> Array:
    alpha = alpha + 1e-8
    M = B.shape[1]

    if Omega is None:
        Omega = jnp.eye(M)

    G = B.T @ B + alpha * Omega
    b = B.T @ U

    L = jnp.linalg.cholesky(G)
    y = jnp.linalg.solve(L, b)
    theta = jnp.linalg.solve(L.T, y)

    return theta.T

def graphon_kernel(
        Xs: Array,
        Ys: Array,
        knots_l: Array,
        knots_r: Array,
        theta_l: Array,
        theta_r: Array,
        top_s: Array,
) -> Array:
    BL = spline_design_matrix(Xs, knots_l, degree=3)  # (Gx, Ml)
    BR = spline_design_matrix(Ys, knots_r, degree=3)  # (Gy, Mr)

    psi_l = BL @ theta_l.T   # (Gx, R)
    psi_r = BR @ theta_r.T   # (Gy, R)

    psi_l_scaled = psi_l * top_s  # (Gx, R)

    W = psi_l_scaled @ psi_r.T    # (Gx, Gy)

    return W

def fit_CV_graphon(
        t_l: Array,
        t_r: Array,
        BL: Array,
        BR: Array,
        knots_l: Array,
        knots_r: Array,
        U: Array,
        V: Array,
        S: Array,
        Omega_L: Array,
        Omega_R: Array,
        r: int = 5,
        alpha: float = 1e-3,
) -> Array:

    x = t_l
    y = t_r

    top_u = U[:,:r]
    #top_u = normalise_orientation(top_u)
    top_v = V[:,:r]
    #top_v = normalise_orientation(top_v)
    top_s = S[:r]

    top_u, top_v = align_svd_signs(top_u, top_v)

    theta_l = ridge_reg_spline(BL, top_u, alpha=alpha, Omega=Omega_L)
    theta_r = ridge_reg_spline(BR, top_v, alpha=alpha, Omega=Omega_R)

    A_pred = graphon_kernel(x, y, knots_l, knots_r, theta_l, theta_r, top_s)

    return A_pred

@dataclass
class GraphonSplineModel:
    rank: int
    alpha: float
    degree: int
    knots_l: Array
    knots_r: Array
    theta_l: Array
    theta_r: Array
    top_s: Array
    sr: Optional[Array] = None
    sc: Optional[Array] = None

def fit_final_model(
        A: Array,
        U: Array,
        V: Array,
        S: Array,
        r_coord: int = 1,
        rank: int = 17,
        alpha: float = 1e-3,
        K_knots: int = 15,
        degree: int = 3,
        Omega_order: int = 2,
        sr: Optional[Array] = None,
        sc: Optional[Array] = None,
) -> GraphonSplineModel:
    t_l, _ = left_coordinate(U, r=r_coord)
    t_r, _ = right_coordinate(V, r=r_coord)

    knots_l, knots_r = make_spline_knots(t_l, t_r, K= K_knots, degree=degree)
    BL = spline_design_matrix(t_l, knots_l, degree=degree)
    BR = spline_design_matrix(t_r, knots_r, degree=degree)

    Omega_L = spline_penalty_matrix(BL.shape[1], order=Omega_order)
    Omega_R = spline_penalty_matrix(BR.shape[1], order=Omega_order)

    top_u = U[:, :rank]    # (N, rank)

    top_v = V[:, :rank]
    top_s = S[:rank]       # (rank,)
    top_u, top_v = align_svd_signs(top_u, top_v)

    theta_l = ridge_reg_spline(BL, top_u, alpha=alpha, Omega=Omega_L)  # expect (rank, M_L)
    theta_r = ridge_reg_spline(BR, top_v, alpha=alpha, Omega=Omega_R)  # expect (rank, M_R)

    return GraphonSplineModel(
        rank=rank,
        alpha=alpha,
        degree=degree,
        knots_l=knots_l,
        knots_r=knots_r,
        theta_l=theta_l,
        theta_r=theta_r,
        top_s=top_s,
        sr=sr,
        sc=sc
    )

def fitted_matrix_on_training_nodes(model: GraphonSplineModel, t_l: Array, t_r: Array) -> Array:
    return graphon_kernel(
        t_l, t_r,
        model.knots_l, model.knots_r,
        model.theta_l, model.theta_r,
        model.top_s
    )

