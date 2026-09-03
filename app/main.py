from __future__ import annotations

import hashlib
import time
import os

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import typer
from jax import Array, random

from app.theapp import get_EN_stats, get_game_stats, get_net_stats
from clearing_game.game_nash import br_residual
from common.utils import (
    index_by_quarter,
    load_config,
    load_reconstruction_data,
    require,
)
from graphon.graphon import (
    left_coordinate,
    fit_final_model,
    graphon_kernel,
)
from graphon.sample import drichlet_rows_gamma
from graphon.utils import (
    bisection_kappa_for_L_density,
    clean_Pi,
    compose_L_from_Pi_pbar,
    density_offdiag_positive,
    get_svd,
    regularise,
    winsorise_by_row,
)
from stats.stats import strengths

jax.config.update('jax_enable_x64', True)


# -------------------------
# Helpers
# -------------------------

def to_numpy_tree(tree):
    return jax.tree.map(lambda x: np.asarray(jax.device_get(x)), tree)


def sparsemax(z: Array, axis: int = -1) -> Array:
    z_sorted = jnp.flip(jnp.sort(z, axis=axis), axis=axis)
    z_cumsum = jnp.cumsum(z_sorted, axis=axis)

    d = z.shape[axis]
    k = jnp.arange(1, d + 1, dtype=z.dtype)

    shape = [1] * z.ndim
    shape[axis] = d
    k = jnp.reshape(k, shape)

    support = 1.0 + k * z_sorted > z_cumsum
    k_z = jnp.sum(support, axis=axis, keepdims=True)

    tau_sum = jnp.take_along_axis(
        z_cumsum,
        k_z.astype(jnp.int32) - 1,
        axis=axis,
        )
    tau = (tau_sum - 1.0) / jnp.maximum(k_z, 1.0)

    return jnp.maximum(z - tau, 0.0)


def stochasticise_rows_sparsemax(sample: Array, eps: float = 1e-12) -> Array:
    sample_pos = jnp.maximum(sample, 0.0)
    p = sparsemax(sample_pos, axis=1)
    den = jnp.sum(p, axis=1, keepdims=True)
    return jnp.where(den > 0, p / jnp.maximum(den, eps), 0.0)


def deterministic_rank_match(x_new: Array, values_obs: Array) -> Array:
    order_new = jnp.argsort(x_new)
    sorted_vals = jnp.sort(values_obs)

    out = jnp.empty_like(sorted_vals)
    out = out.at[order_new].set(sorted_vals)
    return out


# -------------------------
# Main
# -------------------------

def main(
        MODEL: str = typer.Option(...),
        quarter: str = typer.Option(...),
        config: str = typer.Option('config.toml'),
):
    cfg = load_config(config)

    network_root = require(cfg, 'paths', 'network_root')
    ffiec_root = require(cfg, 'paths', 'ffiec_root')
    net_stats_root = require(cfg, 'paths', 'net_stats_root')
    en_stats_root = require(cfg, 'paths', 'en_stats_root')
    game_stats_root = require(cfg, 'paths', 'game_stats_root')

    comparison_keys = require(cfg, 'stats', 'comparison_keys')
    kappa_grid = require(cfg, 'kappas', 'kappa_grid')

    os.makedirs(net_stats_root, exist_ok=True)
    os.makedirs(en_stats_root, exist_ok=True)
    os.makedirs(game_stats_root, exist_ok=True)

    print('--- Loading data ---')

    Lpaths = index_by_quarter(f'{network_root}/{MODEL}', 'npy')
    epaths = index_by_quarter(ffiec_root, 'csv')

    L = jnp.asarray(np.load(Lpaths[quarter]))
    df = load_reconstruction_data(epaths[quarter])
    e_obs = jnp.asarray(df['opCF_in'].to_numpy())

    N = L.shape[0]
    eps = 1e-12

    print(f'N = {N}')

    # -------------------------
    # Preprocessing
    # -------------------------
    print('--- Preprocessing ---')

    pbar_obs = jnp.sum(L, axis=1)

    Pi = jnp.where(
        pbar_obs[:, None] > 0,
        L / jnp.maximum(pbar_obs[:, None], eps),
        0.0,
        )

    Pi_w = winsorise_by_row(Pi)
    Pi_tilde, _ = regularise(Pi_w)

    # -------------------------
    # SVD
    # -------------------------
    print('--- SVD ---')
    t0 = time.perf_counter()
    U, S, V = get_svd(Pi_tilde)
    jax.tree.map(lambda x: x.block_until_ready(), (U, S, V))
    print(f'SVD time: {time.perf_counter() - t0:.3f}s')

    t_l, _ = left_coordinate(U, r=1)

    # -------------------------
    # Graphon fit
    # -------------------------
    print('--- Graphon fit ---')
    t0 = time.perf_counter()

    gmodel = fit_final_model(
        Pi_tilde,
        U,
        V,
        S,
        r_coord=1,
        rank=12,
        alpha=1e-3,
    )

    jax.tree.map(lambda x: x.block_until_ready(), gmodel)

    print(f'Graphon fit time: {time.perf_counter() - t0:.3f}s')

    # -------------------------
    # Seed
    # -------------------------
    seed = int(hashlib.md5(f'{MODEL}-{quarter}'.encode()).hexdigest()[:8], 16)
    key = random.PRNGKey(seed)

    print(f'seed = {seed}')

    # -------------------------
    # Calibration
    # -------------------------
    print('--- Calibration ---')

    key, skx, sky = random.split(key, 3)

    Xs_cal = jnp.sort(random.uniform(skx, (N,)))
    Ys_cal = jnp.sort(random.uniform(sky, (N,)))

    sample_cal = graphon_kernel(
        Xs_cal,
        Ys_cal,
        gmodel.knots_l,
        gmodel.knots_r,
        gmodel.theta_l,
        gmodel.theta_r,
        gmodel.top_s,
    )

    weights_cal = stochasticise_rows_sparsemax(sample_cal)

    pbar_cal = deterministic_rank_match(Xs_cal, pbar_obs)

    target_d = density_offdiag_positive(L)

    key, skk = random.split(key)
    key, kappa_star = bisection_kappa_for_L_density(
        skk,
        weights_cal,
        pbar_cal,
        target_d,
        k_lo=0.5,
        k_hi=1e5,
        tol=1e-3,
        max_iter=25,
        M=3,
    )

    print(f'target_d = {float(target_d):.6f}')
    print(f'kappa_star = {float(kappa_star):.6f}')

    # -------------------------
    # ONE DRAW
    # -------------------------
    print('--- One draw ---')

    key, skx, sky = random.split(key, 3)

    Xs = jnp.sort(random.uniform(skx, (N,)))
    Ys = jnp.sort(random.uniform(sky, (N,)))

    print('Kernel...')
    sample = graphon_kernel(
        Xs,
        Ys,
        gmodel.knots_l,
        gmodel.knots_r,
        gmodel.theta_l,
        gmodel.theta_r,
        gmodel.top_s,
    )

    print('Sparsemax...')
    weights = stochasticise_rows_sparsemax(sample)

    print('Sampling Pi...')
    key, skd = random.split(key)
    Pi_new = drichlet_rows_gamma(skd, weights, kappa_star)
    Pi_new = clean_Pi(Pi_new)

    print('Deterministic matching...')
    pbar_det = deterministic_rank_match(Xs, pbar_obs)
    e_det = deterministic_rank_match(Xs, e_obs)

    print('Compose L...')
    L_new = compose_L_from_Pi_pbar(Pi_new, pbar_det)
    L_new.block_until_ready()

    # -------------------------
    # Network stats
    # -------------------------
    print('--- Network stats ---')

    r, c = strengths(L_new)
    r_pos = r[r > 0]
    c_pos = c[c > 0]

    k_hill_r = int(np.floor(np.sqrt(int(r_pos.shape[0]))))
    k_hill_c = int(np.floor(np.sqrt(int(c_pos.shape[0]))))

    net_stats = get_net_stats(
        L_new,
        r_pos,
        c_pos,
        k_hill_r=k_hill_r,
        k_hill_c=k_hill_c,
    )

    net_stats_np = to_numpy_tree(net_stats)
    net_df = pd.DataFrame([{k: float(net_stats_np[k]) for k in net_stats_np}])
    net_df.to_csv(f'{net_stats_root}/stats_{quarter}.csv', index=False)

    print('Saved network stats')

    # -------------------------
    # EN stats
    # -------------------------
    print('--- EN stats ---')

    en_stats = get_EN_stats(L_new, e_det)
    en_stats_np = to_numpy_tree(en_stats)

    en_df = pd.DataFrame([{k: float(en_stats_np[k]) for k in comparison_keys}])
    en_df.to_csv(f'{en_stats_root}/en_{quarter}.csv', index=False)

    print('Saved EN stats')

    # -------------------------
    # Game stats
    # -------------------------
    print('--- Game stats ---')

    game_rows = []

    for kappa in kappa_grid:
        print(f'kappa = {kappa}')

        theta = kappa * pbar_det

        game_stats, s_final = get_game_stats(L_new, e_det, theta)
        res = br_residual(L_new, e_det, theta, s_final)

        game_stats_np = to_numpy_tree(game_stats)

        row = {k: float(game_stats_np[k]) for k in comparison_keys}
        row['kappa'] = kappa
        row['NE_resid'] = float(res)

        game_rows.append(row)

    game_df = pd.DataFrame(game_rows)
    game_df.to_csv(f'{game_stats_root}/game_{quarter}.csv', index=False)

    print('Saved game stats')

    print('--- DONE ---')


if __name__ == '__main__':
    typer.run(main)