import jax
import jax.numpy as jnp
from jax import Array, lax
from .utils import spectral_radius


def EN_rogers_veraart_fda(
        L: jax.Array,
        e: jax.Array,
        alpha: float = 1.0,
        beta: float = 1.0,
):

    n = L.shape[0]
    max_rounds = n  # hard cap (worst case), but we will exit early

    p_bar = jnp.sum(L, axis=1)

    Pi = jnp.where(p_bar[:, None] > 0.0, L / p_bar[:, None], 0.0)
    M = Pi.T

    p0 = p_bar

    usable0 = alpha * e + beta * (M @ p0)
    d0 = usable0 < p_bar

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

        d_f = d.astype(p.dtype)          # 1 if defaulting, 0 if solvent
        nd_f = 1.0 - d_f                 # 1 if solvent, 0 if defaulting

        A = jnp.eye(n, dtype=p.dtype) - beta * (d_f[:, None] * M * d_f[None, :])

        b = nd_f * p_bar + d_f * (alpha * e + beta * (M @ (nd_f * p_bar)))

        p_new = jnp.linalg.solve(A, b)

        usable = alpha * e + beta * (M @ p_new)
        d_raw = usable < p_bar

        d_new = jnp.logical_or(d, d_raw)
        new_defaults = jnp.logical_and(d_new, jnp.logical_not(d))
        changed = jnp.any(new_defaults)

        D_next = D_curr.at[k].set(d_new)
        new_next = new_curr.at[k].set(new_defaults)

        return (d_new, p_new, D_next, new_next, k + 1, changed)

    d_final, p_final_raw, D_hist_full, new_hist_full, k_final, _ = lax.while_loop(
        cond_fun, body_fun, init_val
    )

    rounds_taken = k_final  # includes round 0

    usable_final = alpha * e + beta * (M @ p_final_raw)
    p_final = jnp.minimum(p_bar, usable_final)
    p_final = jnp.clip(p_final, 0.0, p_bar)


    return p_final, D_hist_full, new_hist_full, rounds_taken



def EN_stats(
        L: Array,
        e: Array,
        p_final: Array,
        D_hist: Array,
        new_hist: Array,
        rounds_taken: Array,
        alpha: float = 1.0,
        beta: float = 1.0,
        spectral_n_iter: int = 50,
):
    n = L.shape[0]
    max_rounds = D_hist.shape[0]

    T = jnp.clip(rounds_taken, 1, max_rounds)

    p_bar = jnp.sum(L, axis=1)
    inv_pbar = jnp.where(p_bar > 0.0, 1.0 / p_bar, 0.0)

    def M_mul(x: Array) -> Array:
        return L.T @ (x * inv_pbar)

    D_final = lax.dynamic_index_in_dim(D_hist, T - 1, axis=0, keepdims=False)
    n_default_final = jnp.sum(D_final)
    default_rate = n_default_final / n

    shortfall = jnp.clip(p_bar - p_final, 0.0)
    total_shortfall = jnp.sum(shortfall)
    total_liabilities = jnp.sum(p_bar)
    shortfall_frac = jnp.where(total_liabilities > 0.0, total_shortfall / total_liabilities, 0.0)

    inflow_promised = M_mul(p_bar)
    inflow_received = M_mul(p_final)
    creditor_loss = jnp.clip(inflow_promised - inflow_received, 0.0)
    total_creditor_loss = jnp.sum(creditor_loss)

    usable_final = alpha * e + beta * M_mul(p_final)

    round_mask = (jnp.arange(max_rounds) < T)

    any_new_round = jnp.any(new_hist, axis=1)
    cascade_rounds = jnp.sum((any_new_round & round_mask).astype(jnp.int32))

    loss_sorted = jnp.sort(creditor_loss)
    k = jnp.minimum(10, n).astype(jnp.int32)

    idx_last = (n - 10) + jnp.arange(10)
    idx_last = jnp.clip(idx_last, 0, n - 1)
    last10 = loss_sorted[idx_last]
    take_mask = (jnp.arange(10) >= (10 - k))
    topk_sum = jnp.sum(jnp.where(take_mask, last10, 0.0))
    top10_loss_share = jnp.where(total_creditor_loss > 0.0, topk_sum / total_creditor_loss, 0.0)

    sx = jnp.sum(loss_sorted)
    idx = jnp.arange(1, n + 1, dtype=loss_sorted.dtype)
    gini_creditor_loss = jnp.where(
        sx > 0.0,
        (2.0 * jnp.sum(idx * loss_sorted) / (n * sx)) - (n + 1.0) / n,
        0.0,
        )

    loss_to_survivors = jnp.sum(jnp.where(~D_final, creditor_loss, 0.0))
    loss_to_defaulters = jnp.sum(jnp.where(D_final, creditor_loss, 0.0))

    avg_default_severity = jnp.where(
        n_default_final > 0,
        jnp.sum(jnp.where(D_final, shortfall, 0.0)) / n_default_final,
        0.0,
        )

    usable_at_full_payment = alpha * e + beta * M_mul(p_bar)
    initial_shortfall = jnp.sum(jnp.clip(p_bar - usable_at_full_payment, 0.0))
    amplification = jnp.where(initial_shortfall > 0.0, total_shortfall / initial_shortfall, 0.0)

    initial_defaults = lax.dynamic_index_in_dim(new_hist, 0, axis=0, keepdims=False)
    endogenous_defaults = jnp.logical_and(D_final, jnp.logical_not(initial_defaults))
    n_endogenous_defaults = jnp.sum(endogenous_defaults)
    endogenous_default_share = jnp.where(
        n_default_final > 0,
        n_endogenous_defaults / n_default_final,
        0.0,
        )

    buffers = usable_final - p_bar
    n_survivors = jnp.sum(~D_final)

    avg_buffer_survivors = jnp.where(
        n_survivors > 0,
        jnp.sum(jnp.where(~D_final, buffers, 0.0)) / n_survivors,
        0.0,
        )
    avg_buffer_defaulters = jnp.where(
        n_default_final > 0,
        jnp.sum(jnp.where(D_final, buffers, 0.0)) / n_default_final,
        0.0,
        )

    new_counts_all = jnp.sum(new_hist, axis=1).astype(jnp.int32)
    cum_counts_all = jnp.sum(D_hist, axis=1).astype(jnp.int32)

    new_counts = jnp.where(round_mask, new_counts_all, 0)
    last_cum = lax.dynamic_index_in_dim(cum_counts_all, T - 1, axis=0, keepdims=False)
    cum_counts = jnp.where(round_mask, cum_counts_all, last_cum)

    peak_new_defaults = jnp.max(new_counts)
    peak_round = jnp.argmax(new_counts)

    first_wave_new_defaults = jnp.where(T > 1, new_counts_all[1], 0)

    any_new_i32 = (any_new_round & round_mask).astype(jnp.int32)
    last_wave_round = jnp.max(jnp.where(any_new_i32 == 1, jnp.arange(max_rounds), 0))

    last_new = lax.dynamic_index_in_dim(new_counts_all, T - 1, axis=0, keepdims=False)
    prev_new = lax.dynamic_index_in_dim(new_counts_all, jnp.maximum(T - 2, 0), axis=0, keepdims=False)
    late2_new_defaults = jnp.where(T >= 2, prev_new + last_new, last_new)

    new_hist_valid = jnp.where(round_mask[:, None], new_hist, False)
    any_default = jnp.any(new_hist_valid, axis=0)
    tau_raw = jnp.argmax(new_hist_valid, axis=0).astype(jnp.int32)
    tau = jnp.where(any_default, tau_raw, T)

    n_default_tau = jnp.sum(any_default)
    avg_tau_default = jnp.where(
        n_default_tau > 0,
        (jnp.sum(jnp.where(any_default, tau, 0)).astype(jnp.float32)
         / n_default_tau.astype(jnp.float32)),
        0.0,
        )

    tau_consistent = jnp.all(any_default == D_final)

    # Spectral radius of M (structure-only). If you want beta-scaled, pass beta=beta.
    spectralM = spectral_radius(L, beta=1.0, n_iter=spectral_n_iter)

    return {
        'D_final': D_final,
        'n_default': n_default_final,
        'default_rate': default_rate,
        'p_bar': p_bar,
        'shortfall': shortfall,
        'total_shortfall': total_shortfall,
        'shortfall_frac': shortfall_frac,
        'cascade_rounds': cascade_rounds,
        'inflow_promised': inflow_promised,
        'inflow_received': inflow_received,
        'creditor_loss': creditor_loss,
        'total_creditor_loss': total_creditor_loss,

        'top10_loss_share': top10_loss_share,
        'gini_creditor_loss': gini_creditor_loss,
        'loss_to_survivors': loss_to_survivors,
        'loss_to_defaulters': loss_to_defaulters,
        'avg_default_severity': avg_default_severity,

        'initial_shortfall': initial_shortfall,
        'amplification': amplification,
        'initial_defaults': initial_defaults,
        'endogenous_defaults': endogenous_defaults,
        'n_endogenous_defaults': n_endogenous_defaults,
        'endogenous_default_share': endogenous_default_share,
        'buffers': buffers,
        'avg_buffer_survivors': avg_buffer_survivors,
        'avg_buffer_defaulters': avg_buffer_defaulters,

        'new_defaults_by_round': new_counts,
        'cum_defaults_by_round': cum_counts,
        'peak_new_defaults': peak_new_defaults,
        'peak_round': peak_round,
        'first_wave_new_defaults': first_wave_new_defaults,
        'last_wave_round': last_wave_round,
        'late2_new_defaults': late2_new_defaults,
        'tau': tau,
        'avg_tau_default': avg_tau_default,
        'n_default_tau': n_default_tau,
        'tau_consistent': tau_consistent,

        'spectralM': spectralM,
    }
