
from functools import partial
import pandas as pd
import numpy as np
from jax import Array, lax, random
import jax
import jax.numpy as jnp
from typing import Tuple, Any

from .utils import (
    col_loss,
    row_loss,
    scale_row,
    scale_col,
    logit,
    sigma,
    
)

@partial(jax.jit, static_argnames=('max_iter',))
def RAS_max_e(P, IA_hat, IL, epsilon: float = 1e-8, max_iter: int = 1000):
    l_c = col_loss(P, IL)
    l_r = row_loss(P, IA_hat)

    # +1 because we write at it+1
    rLosses = jnp.full((max_iter + 1,), jnp.nan).at[0].set(l_r)
    cLosses = jnp.full((max_iter + 1,), jnp.nan).at[0].set(l_c)

    def body_fun(carry):
        l_c, l_r, rLosses, cLosses, P, it = carry
        P = scale_row(P, IA_hat)
        P = scale_col(P, IL)
        l_c = col_loss(P, IL)
        l_r = row_loss(P, IA_hat)
        it1 = it + 1
        rLosses = rLosses.at[it1].set(l_r)
        cLosses = cLosses.at[it1].set(l_c)
        return l_c, l_r, rLosses, cLosses, P, it1

    def cond_fun(carry):
        l_c, l_r, rLosses, cLosses, _, it = carry

        under_cap = it < max_iter

        not_converged = jnp.logical_or(l_r > epsilon, l_c > epsilon)

        def still_moving(_):
            r_rel = jnp.abs(rLosses[it] - rLosses[it - 1]) / jnp.maximum(jnp.abs(rLosses[it - 1]), 1.0)
            c_rel = jnp.abs(cLosses[it] - cLosses[it - 1]) / jnp.maximum(jnp.abs(cLosses[it - 1]), 1.0)
            # continue if either is still improving
            return jnp.logical_or(r_rel > epsilon, c_rel > epsilon)

        warmup = 1
        moving = lax.cond(it >= warmup, still_moving, lambda _: True, operand=None)

        return jnp.logical_and(under_cap, jnp.logical_and(not_converged, moving))

    l_c, l_r, rLosses, cLosses, P_final, it_final = lax.while_loop(
        cond_fun, body_fun, (l_c, l_r, rLosses, cLosses, P, 0)
    )
    return l_c, l_r, rLosses, cLosses, P_final, it_final

# --------------------
# Fitness-based
#---------------------

def extract_arrays(
    df: pd.DataFrame,
    dtype: Any = jnp.float64,
) -> Tuple[Array, Array, Array, Array, Array, Array, Array]:

    required = [
        "IA", "IL", "TotalAssets", "TotalLiabilities",
        "CashAndBalances", "Repos", "FedFundsPurchased"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}")

    def col(name: str) -> Array:
        x = pd.to_numeric(df[name], errors="coerce").to_numpy()
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        return jnp.asarray(x, dtype=dtype)

    IA   = col("IA")
    IL   = col("IL")
    TA   = col("TotalAssets")
    TL   = col("TotalLiabilities")
    cash = col("CashAndBalances")
    repo = col("Repos")
    FFP  = col("FedFundsPurchased")

    n = IA.shape[0]
    for nm, arr in [("IL", IL), ("TA", TA), ("TL", TL),
                    ("cash", cash), ("repo", repo), ("FFP", FFP)]:
        if arr.shape != (n,):
            raise ValueError(f"{nm} has shape {arr.shape}, expected {(n,)}")

    return IA, IL, TA, TL, cash, repo, FFP

@jax.jit
def rescaled_features(
    IA: Array,
    IL: Array,
    TA: Array,
    TL: Array,
    cash: Array,
    repo: Array,
    FFP: Array,
    eps_denom: float = 1e-12,
    rel_mismatch_tol: float = 1e-6,
) -> Tuple[Array, Array, Array, Array, Array, Array, Array, Array, Array]:

    TA = jnp.maximum(TA, eps_denom)
    TL = jnp.maximum(TL, eps_denom)

    IA   = jnp.maximum(IA, 0.0)
    IL   = jnp.maximum(IL, 0.0)
    cash = jnp.maximum(cash, 0.0)
    repo = jnp.maximum(repo, 0.0)
    FFP  = jnp.maximum(FFP, 0.0)

    sumIA = jnp.sum(IA)
    sumIL = jnp.sum(IL)

    denom = jnp.maximum(sumIA, eps_denom)
    rel_mismatch = jnp.abs(sumIL - sumIA) / denom

    IA_tilde = IA * jnp.sqrt(sumIL / sumIA)
    IL_tilde = IL * jnp.sqrt(sumIA / sumIL)

    return IA, IL, TA, TL, cash, repo, FFP, IA_tilde, IL_tilde

@jax.jit
def make_type_vector(
    cash: Array,
    TA: Array,
    FFP: Array,
    repo: Array,
    TL: Array,
    IA_tilde: Array,
    IL: Array,
    eps: float = 1e-12,
) -> Tuple[Array, Array, Array, Array]:
    TA_safe = jnp.maximum(TA, eps)
    TL_safe = jnp.maximum(TL, eps)

    l_raw = cash / TA_safe
    w_raw = (FFP + repo) / TL_safe
    n_raw = IA_tilde - IL

    def z(a: Array) -> Array:
        a = jnp.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
        mu = jnp.mean(a)
        sd = jnp.maximum(jnp.std(a), eps)
        return (a - mu) / sd

    l = z(l_raw)
    w = z(w_raw)
    n = z(n_raw)
    log_TA = z(jnp.log1p(TA_safe))

    x = jnp.stack([log_TA, l, w, n], axis=1)  # (N, 4)
    return x, l, w, n

@jax.jit
def make_fitness(
    IA_tilde: Array,
    IL: Array,
    l: Array,
    w: Array,
    n: Array,
    a1: float = 1.0, a2: float = 0.5, a3: float = 0.5, a4: float = 0.25,
    b1: float = 1.0, b2: float = 0.5, b3: float = 0.5, b4: float = 0.25,
    floor: float = 1e-6,
    cap: float = 1e6,
    eps_norm: float = 1e-12,
) -> Tuple[Array, Array]:

    L = a1 * jnp.log1p(IA_tilde) + a2 * l - a3 * w + a4 * n
    B = b1 * jnp.log1p(IL)       + b2 * w - b3 * l - b4 * n

    L = L - jnp.max(L)
    B = B - jnp.max(B)

    phi = jnp.clip(jnp.exp(L), floor, cap)
    psi = jnp.clip(jnp.exp(B), floor, cap)

    phi = phi / jnp.maximum(jnp.mean(phi), eps_norm)
    psi = psi / jnp.maximum(jnp.mean(psi), eps_norm)

    return phi, psi

@jax.jit
def prob_matrix(
    phi: Array,
    psi: Array,
    x: Array,          
    TA: Array,         
    z: float,          
    gamma: float = 0.5,
    alpha: float = 0.3,
    eta1: float = 0.2,
    eta2: float = 0.2,
    eta3: float = 0.5,
    eps: float = 1e-12,
) -> Tuple[Array, Array]:
    N = x.shape[0]

   
    diff = x[:, None, :] - x[None, :, :]         
    d2 = jnp.sum(diff * diff, axis=-1)           

  
    offdiag_mask = ~jnp.eye(N, dtype=bool)
    d2_off = jnp.where(offdiag_mask, d2, jnp.nan)

   
    scale = jnp.nanmedian(d2_off)
    scale = jnp.where(jnp.isfinite(scale) & (scale > eps), scale, 1.0)

    d2n = d2 / scale
    S = jnp.exp(-gamma * d2n)
    S = S.at[jnp.arange(N), jnp.arange(N)].set(0.0)

   
    # S_mix in [(1-alpha), (1-alpha)+alpha] = [1-alpha, 1] if S in [0,1]
    S_mix = (1.0 - alpha) + alpha * S

    cap_lambda = z * jnp.outer(phi, psi) * S_mix
    p = cap_lambda / (1.0 + cap_lambda)
    p = p.at[jnp.arange(N), jnp.arange(N)].set(0.0)

    q_core = jnp.quantile(TA, 0.9)
    core = (TA >= q_core).astype(p.dtype)         # (N,)

    logit_p = logit(p, eps=eps)
    logit_p = (
        logit_p
        + eta1 * core[:, None]
        + eta2 * core[None, :]
        + eta3 * (core[:, None] * core[None, :])
    )
    p = sigma(logit_p)
    p = p.at[jnp.arange(N), jnp.arange(N)].set(0.0)

    return p, S

@partial(jax.jit, static_argnames=("max_tries",))
def build_masked_prior_until_ok(
    IA_tilde,
    IL,
    P_base,          
    p,
    key,
    tau=1e-6,
    eps_support=0.0,
    max_tries=5000,
):
    N = IL.shape[0]
    diag = jnp.arange(N)

    def once(key):
        key, sub = random.split(key)
        A = random.bernoulli(sub, p=p, shape=(N, N))
        A = A.at[diag, diag].set(False)

        P = P_base * A.astype(P_base.dtype)
        P = P.at[diag, diag].set(0.0)

        row_support = jnp.sum(P > eps_support, axis=1)
        col_support = jnp.sum(P > eps_support, axis=0)
        row_has = row_support > 0
        col_has = col_support > 0

        row_mass = jnp.sum(P, axis=1)
        col_mass = jnp.sum(P, axis=0)

        row_ok = jnp.all((IA_tilde <= 0) | (row_has & (row_mass >= tau * IA_tilde)))
        col_ok = jnp.all((IL      <= 0) | (col_has & (col_mass >= tau * IL)))
        ok = row_ok & col_ok
        return P, A, ok, key

    # warm start (and carry updated key)
    P0, A0, ok0, key0 = once(key)

    def cond(carry):
        _, _, ok, _, tries = carry
        return (~ok) & (tries < max_tries)

    def body(carry):
        _, _, _, key, tries = carry
        P, A, ok, key = once(key)
        return (P, A, ok, key, tries + 1)

    carry0 = (P0, A0, ok0, key0, jnp.array(1, jnp.int32))
    P, A, ok, key, tries = lax.while_loop(cond, body, carry0)

    return P, A, ok, tries, key

def solve_z_prob_matrix(
    phi: Array,
    psi: Array,
    x: Array,
    TA: Array,
    d_target: float,
    z_lo: float = 1e-16,
    z_hi: float = 1e6,
    max_iter: int = 60,
    tol: float = 1e-6,
) -> Array:
   
    lo = jnp.asarray(z_lo, dtype=jnp.float64)
    hi = jnp.asarray(z_hi, dtype=jnp.float64)
    d_target = jnp.asarray(d_target, dtype=jnp.float64)

    d_lo = density_from_prob_matrix(lo, phi, psi, x, TA)
    d_hi = density_from_prob_matrix(hi, phi, psi, x, TA)
    f_lo = float(d_lo - d_target)
    f_hi = float(d_hi - d_target)

   
    if f_lo >= 0.0 and f_hi >= 0.0:
        return lo
    if f_lo <= 0.0 and f_hi <= 0.0:
        return hi

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        d_mid = density_from_prob_matrix(mid, phi, psi, x, TA)
        f_mid = float(d_mid - d_target)

        if float(jnp.abs(hi - lo)) < tol:
            break

        if f_mid < 0.0:
            lo = mid
        else:
            hi = mid

    return 0.5 * (lo + hi)

@jax.jit
def offdiag_density_from_p(p: Array) -> Array:
   
    N = p.shape[0]
    off_sum = jnp.sum(p) - jnp.trace(p)
    denom = jnp.maximum(N * (N - 1), 1)
    return off_sum / denom


@jax.jit
def density_from_prob_matrix(z: Array, phi: Array, psi: Array, x: Array, TA: Array) -> Array:
   
    p, _ = prob_matrix(phi, psi, x, TA, z=z)
    return offdiag_density_from_p(p)

def pick_density_by_ras_loss(
    phi: Array, psi: Array, x: Array, TA: Array,
    IA_tilde: Array, IL: Array,
    key: Array,
    d_grid,
    loss_tol: float = 1e-6,      
    ras_iters_probe: int = 800,  
    build_tries_cap: int = 500,  
    tau: float = 1e-6,
):
   
    d_grid = sorted(d_grid)

    sum_row = float(jnp.sum(IA_tilde))
    sum_col = float(jnp.sum(IL))
    print("sum(IA_tilde)=", sum_row, "sum(IL)=", sum_col, "ratio=", sum_row / max(sum_col, 1e-30))

    for d in d_grid:
        z = solve_z_prob_matrix(phi, psi, x, TA, d_target=d)
        p, S = prob_matrix(phi, psi, x, TA, z=z)

        N = p.shape[0]
        diag = jnp.arange(N)
        p = p.at[diag, diag].set(0.0)

        sumIL = jnp.maximum(jnp.sum(IL), 1e-12)
        coef = IL / sumIL
        P_build = jnp.outer(IA_tilde, coef).at[diag, diag].set(0.0)

        key, sub = random.split(key)
        P, A, ok, tries, key = build_masked_prior_until_ok(
            IA_tilde, IL, P_build, p, key=sub, 
            tau = tau, eps_support=0.0, max_tries=build_tries_cap
        )

        if not ok:
            print(f"d={d:.4f} z={float(z):.4g} build_ok=False tries={tries}")
            continue

        
        l_c, l_r, rLosses, cLosses, E, it_final = RAS_max_e(P, IA_tilde, IL, epsilon=1e-6, max_iter=ras_iters_probe)
        E.block_until_ready()

        lr = float(l_r)
        lc = float(l_c)
        itf = int(it_final)

        
        lr_rel = lr / max(float(jnp.sum(IA_tilde)), 1e-30)
        lc_rel = lc / max(float(jnp.sum(IL)), 1e-30)

        achieved = float(offdiag_density_from_p(p))
        print(
            f"d={d:.4f} achieved={achieved:.4f} z={float(z):.4g} "
            f"tries={tries} it={itf} row_loss={lr:.3e} col_loss={lc:.3e} "
            f"row_rel={lr_rel:.3e} col_rel={lc_rel:.3e}"
        )

        if (lr_rel <= loss_tol) and (lc_rel <= loss_tol):
            print(f"SELECTED min-d by RAS: d*={d:.4f}")
            return d, float(z), P_build, P, A, key


    raise ValueError("No density met the RAS loss tolerance on this grid; expand d_grid upward or relax loss_tol.")
