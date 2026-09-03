import jax
import jax.numpy as jnp
from jax import lax, Array

def strategic_EN_game(
        L: Array,
        e: Array,
        theta: Array,
        alpha: float = 1.0,
        beta: float = 1.0,
        tol: float = 1e-10,
        eps: float = 1e-12,
):
    n = L.shape[0]
    max_rounds = n

    pbar = jnp.sum(L, axis=1)

    Pi = jnp.where(pbar[:, None] > 0.0, L / pbar[:, None], 0.0)
    M = Pi.T

    D_hist = jnp.zeros((max_rounds, n), dtype=bool)
    new_hist = jnp.zeros((max_rounds, n), dtype=bool)

    s0 = jnp.ones(n)

    a0 = alpha * e + beta * (M @ pbar)
    a0_pos = jnp.maximum(a0, 0.0)
    d0 = jnp.logical_and(pbar > 0.0, a0_pos < (pbar - eps))

    D_hist = D_hist.at[0].set(d0)
    new_hist = new_hist.at[0].set(d0)

    init_state = (s0, D_hist, new_hist, jnp.int32(1), jnp.bool_(True))

    def cond_fun(state):
        _, _, _, k, changed = state
        return jnp.logical_and(changed, k < max_rounds)

    def body_fun(state):
        s, D_curr, new_curr, k, _ = state

        p = s * pbar
        a = alpha * e + beta * (M @ p)
        a_pos = jnp.maximum(a, 0.0)

        theta_pos = theta > 0.0
        theta_safe = jnp.where(theta_pos, theta, 1.0)

        s_raw = 1.0 - pbar / (2.0 * theta_safe)
        s_raw = jnp.where(theta_pos, s_raw, 1.0)
        s_raw = jnp.clip(s_raw, 0.0, 1.0)

        safe_ratio = jnp.where(pbar > 0.0, a_pos / pbar, 1.0)
        upper = jnp.maximum(0.0, jnp.minimum(1.0, safe_ratio))

        s_new = jnp.minimum(s_raw, upper)
        default_raw = jnp.logical_and(pbar > 0.0, a_pos < (pbar - eps))

        default_prev = D_curr[k - 1]
        default_new = jnp.logical_or(default_prev, default_raw)
        new_defaults = jnp.logical_and(default_new, jnp.logical_not(default_prev))

        D_next = D_curr.at[k].set(default_new)
        new_next = new_curr.at[k].set(new_defaults)

        diff = jnp.max(jnp.abs(s_new - s))
        changed = diff > tol

        return (s_new, D_next, new_next, k + 1, changed)

    s_final, D_hist_full, new_hist_full, k_final, _ = lax.while_loop(
        cond_fun, body_fun, init_state
    )

    rounds_taken = k_final
    p_final = jnp.nan_to_num(s_final * pbar, nan=0.0)
    p_final = jnp.clip(p_final, 0.0, pbar)

    return p_final, s_final, pbar, M, D_hist_full, new_hist_full, rounds_taken

def strategic_RV_fda(
        L: jax.Array,
        e: jax.Array,
        theta: jax.Array,
        alpha: float = 1.0,
        beta: float = 1.0,
):
    n = L.shape[0]
    max_rounds = n

    pbar = jnp.sum(L, axis=1)

    Pi = jnp.where(pbar[:, None] > 0.0, L / pbar[:, None], 0.0)
    M = Pi.T

    theta_safe = jnp.where(theta > 0.0, theta, 1.0)
    c_raw = pbar - (pbar * pbar) / (2.0 * theta_safe)
    c = jnp.clip(c_raw, 0.0, pbar)

    p0 = c

    usable0 = alpha * e + beta * (M @ p0)
    d0 = usable0 < c

    D_hist = jnp.zeros((max_rounds, n), dtype=bool)
    new_hist = jnp.zeros((max_rounds, n), dtype=bool)
    D_hist = D_hist.at[0].set(d0)
    new_hist = new_hist.at[0].set(d0)

    init_val = (d0, p0, D_hist, new_hist, jnp.int32(1), jnp.bool_(True))

    def cond_fun(carry):
        _, _, _, _, k, changed = carry
        return jnp.logical_and(changed, k < max_rounds)

    def body_fun(carry):
        d, p, D_curr, new_curr, k, _ = carry

        d_f  = d.astype(p.dtype)     # 1 defaulting
        nd_f = 1.0 - d_f             # 1 solvent

        A = jnp.eye(n, dtype=p.dtype) - beta * (d_f[:, None] * M * d_f[None, :])

        b = nd_f * c + d_f * (alpha * e + beta * (M @ (nd_f * c)))

        p_new = jnp.linalg.solve(A, b)

        usable = alpha * e + beta * (M @ p_new)
        d_raw = usable < c

        d_new = jnp.logical_or(d, d_raw)
        new_defaults = jnp.logical_and(d_new, jnp.logical_not(d))
        changed = jnp.any(new_defaults)

        D_next = D_curr.at[k].set(d_new)
        new_next = new_curr.at[k].set(new_defaults)

        return (d_new, p_new, D_next, new_next, k + 1, changed)

    d_final, p_raw, D_hist_full, new_hist_full, k_final, _ = lax.while_loop(
        cond_fun, body_fun, init_val
    )

    rounds_taken = k_final

    usable_final = alpha * e + beta * (M @ p_raw)
    p_final = jnp.minimum(c, usable_final)
    p_final = jnp.clip(p_final, 0.0, c)

    return p_final, D_hist_full, new_hist_full, rounds_taken

def br_residual(
        L: Array, e: Array,
        theta: float, s: Array,
        alpha: float = 1.0,
        beta: float = 1.0
) -> Array:
    pbar = jnp.sum(L, axis=1)
    Pi = jnp.where(pbar[:, None] > 0, L / pbar[:, None], 0.0)
    M = Pi.T

    p = s * pbar
    a = alpha * e + beta * (M @ p)

    theta_pos = theta > 0.0
    theta_safe = jnp.where(theta_pos, theta, 1.0)

    s_raw = 1.0 - pbar / (2.0 * theta_safe)
    s_raw = jnp.where(theta_pos, s_raw, 1.0)
    s_raw - jnp.clip(s_raw, 0.0, 1.0)
    safe_ratio = jnp.where(pbar > 0.0, a / pbar, 1.0)
    upper = jnp.maximum(0, jnp.minimum(1.0, safe_ratio))
    br = jnp.clip(s_raw, 0.0, upper)
    return jnp.max(jnp.abs(s - br))

def strategic_RV_game_liquid_cost(
        L: Array,
        e: Array,
        theta: Array,          
        phi: Array,            
        alpha: float = 1.0,
        beta: float = 1.0,
        tol: float = 1e-10,
        eps: float = 1e-12,
):
   
    n = L.shape[0]
    max_rounds = n

    pbar = jnp.sum(L, axis=1)
    Pi = jnp.where(pbar[:, None] > 0.0, L / pbar[:, None], 0.0)
    M = Pi.T

    D_hist = jnp.zeros((max_rounds, n), dtype=bool)
    new_hist = jnp.zeros((max_rounds, n), dtype=bool)

    s0 = jnp.ones(n)

    a0 = alpha * e + beta * (M @ (pbar))
    a0_pos = jnp.maximum(a0, 0.0)
    d0 = jnp.logical_and(pbar > 0.0, a0_pos < (pbar - eps))

    D_hist = D_hist.at[0].set(d0)
    new_hist = new_hist.at[0].set(d0)

    t0 = jnp.int32(1)
    changed0 = jnp.bool_(True)
    init_state = (s0, D_hist, new_hist, t0, changed0)

    def cond_fun(state):
        _, _, _, t, changed = state
        return jnp.logical_and(changed, t < max_rounds)

    def body_fun(state):
        s, D_curr, new_curr, t, _ = state

        p = s * pbar
        a = alpha * e + beta * (M @ p)
        a_pos = jnp.maximum(a, 0.0)

        thresh = jnp.where(pbar > 0.0, a_pos / pbar, 1.0)
        thresh = jnp.clip(thresh, 0.0, 1.0)

        theta_safe = jnp.maximum(theta, eps)
        sA = 1.0 - pbar / (2.0 * theta_safe)
        sB = 1.0 - (1.0 + jnp.maximum(phi, 0.0)) * pbar / (2.0 * theta_safe)

        sA = jnp.clip(sA, 0.0, 1.0)
        sB = jnp.clip(sB, 0.0, 1.0)

        sA_proj = jnp.minimum(sA, thresh)          # enforce s <= a/pbar
        sB_proj = jnp.maximum(sB, thresh)          # enforce s >= a/pbar

        def Ui(s_candidate):
            p_c = s_candidate * pbar
            short = jnp.maximum(p_c - a_pos, 0.0)
            return a_pos - p_c - phi * short - theta * (1.0 - s_candidate) ** 2

        UA = Ui(sA_proj)
        UB = Ui(sB_proj)

        s_br = jnp.where(UB > UA, sB_proj, sA_proj)

        s_new = s_br

        d_raw = jnp.logical_and(pbar > 0.0, a_pos < (pbar - eps))

        d_prev = D_curr[t - 1]
        d_new = jnp.logical_or(d_prev, d_raw)
        new_d = jnp.logical_and(d_new, jnp.logical_not(d_prev))

        D_next = D_curr.at[t].set(d_new)
        new_next = new_curr.at[t].set(new_d)

        diff = jnp.max(jnp.abs(s_new - s))
        changed = diff > tol

        return (s_new, D_next, new_next, t + 1, changed)

    s_final, D_hist_full, new_hist_full, t_final, _ = lax.while_loop(
        cond_fun, body_fun, init_state
    )


    rounds_taken = t_final
    p_final = jnp.clip(jnp.nan_to_num(s_final * pbar, nan=0.0), 0.0, pbar)

    return p_final, s_final, pbar, M, D_hist_full, new_hist_full, rounds_taken
