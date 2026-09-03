from __future__ import annotations

import hashlib
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import typer
from jax import random, Array

from app.theapp import get_EN_stats, get_game_stats, get_net_stats
from clearing_game.game_nash import br_residual
from common.utils import (
    index_by_quarter,
    load_config,
    load_reconstruction_data,
    require,
)
from graphon.graphon import (
    fit_final_model,
    graphon_kernel,
    left_coordinate,
)
from graphon.pbar_sample import (
    fit_size_function_hurdle,
    sample_pbar_hurdle,
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


def sparsemax(
        z: Array,
        axis: int = -1,
) -> Array:
    '''
    Closed-form sparsemax projection onto the probability simplex.

    Returns a row-stochastic matrix with exact zeros possible.
    '''
    z_sorted = jnp.flip(jnp.sort(z, axis=axis), axis=axis)
    z_cumsum = jnp.cumsum(z_sorted, axis=axis)

    d = z.shape[axis]
    k = jnp.arange(1, d + 1, dtype=z.dtype)

    shape = [1] * z.ndim
    shape[axis] = d
    k = jnp.reshape(k, shape)

    support = 1.0 + k * z_sorted > z_cumsum
    k_z = jnp.sum(support, axis=axis, keepdims=True)

    tau_sum = jnp.take_along_axis(z_cumsum, k_z.astype(jnp.int32) - 1, axis=axis)
    tau = (tau_sum - 1.0) / jnp.maximum(k_z, 1.0)

    p = jnp.maximum(z - tau, 0.0)
    return p


def to_numpy_tree(tree):
    return jax.tree.map(lambda x: np.asarray(jax.device_get(x)), tree)


def ras_balance(
        L_prior: Array,
        IL_target: Array,
        IA_target: Array,
        n_iter: int = 500,
        tol: float = 1e-10,
        eps: float = 1e-12,
) -> Array:
    '''
    Standard multiplicative RAS balancing on L.

    Targets
    -------
    IL_target : row sums target (total liabilities)
    IA_target : column sums target (total assets)

    Notes
    -----
    - Preserves zero pattern up to eps floor.
    - If target totals differ slightly, IA_target is rescaled to match IL_target.
    '''
    total_IL = jnp.sum(IL_target)
    total_IA = jnp.sum(IA_target)
    IA_target = jnp.where(
        total_IA > eps,
        IA_target * (total_IL / jnp.maximum(total_IA, eps)),
        IA_target,
        )

    L_curr = jnp.maximum(L_prior, eps)

    for _ in range(n_iter):
        # Row scaling
        row_sums = jnp.sum(L_curr, axis=1)
        r = IL_target / jnp.maximum(row_sums, eps)
        L_curr = r[:, None] * L_curr

        # Column scaling
        col_sums = jnp.sum(L_curr, axis=0)
        s = IA_target / jnp.maximum(col_sums, eps)
        L_curr = L_curr * s[None, :]

        # Convergence check
        row_err = jnp.max(jnp.abs(jnp.sum(L_curr, axis=1) - IL_target))
        col_err = jnp.max(jnp.abs(jnp.sum(L_curr, axis=0) - IA_target))
        err = jnp.maximum(row_err, col_err)

        if float(err) < tol:
            break

    return L_curr


def L_to_Pi(
        L: Array,
        eps: float = 1e-12,
) -> Array:
    '''
    Recover row-stochastic Pi from liabilities matrix L.
    '''
    pbar = jnp.sum(L, axis=1)
    Pi = jnp.where(
        pbar[:, None] > 0,
        L / jnp.maximum(pbar[:, None], eps),
        0.0,
        )
    return clean_Pi(Pi)


def main(
        MODEL: str = typer.Option(...),
        quarter: str = typer.Option(...),
        config: str = 'config.toml',
        ras_iter: int = typer.Option(500, help='Maximum RAS iterations'),
        ras_tol: float = typer.Option(1e-10, help='RAS convergence tolerance'),
):
    cfg = load_config(config)

    network_root = require(cfg, 'paths', 'network_root')
    ffiec_root = require(cfg, 'paths', 'ffiec_root')
    net_stats_root = require(cfg, 'paths', 'net_stats_root')
    en_stats_root = require(cfg, 'paths', 'en_stats_root')
    game_stats_root = require(cfg, 'paths', 'game_stats_root')

    comparison_keys = require(cfg, 'stats', 'comparison_keys')
    kappa_grid = require(cfg, 'kappas', 'kappa_grid')

    print('--- Loading data ---')

    Lpaths = index_by_quarter(f'{network_root}/{MODEL}', 'npy')
    epaths = index_by_quarter(ffiec_root, 'csv')

    L = jnp.asarray(np.load(Lpaths[quarter]))
    df = load_reconstruction_data(epaths[quarter])
    e = jnp.asarray(df['opCF_in'].to_numpy())

    N = L.shape[0]
    print(f'N = {N}')

    # -------------------------
    # Preprocessing
    # -------------------------
    print('--- Preprocessing Pi ---')
    eps = 1e-12
    pbar = jnp.sum(L, axis=1)

    Pi = jnp.where(
        pbar[:, None] > 0,
        L / jnp.maximum(pbar[:, None], eps),
        0.0,
        )

    Pi_w = winsorise_by_row(Pi)
    Pi_tilde, _ = regularise(Pi_w)

    # -------------------------
    # SVD
    # -------------------------
    print('--- SVD ---')
    t0 = time.time()
    U, S, V = get_svd(Pi_tilde)
    U.block_until_ready()
    print(f'SVD time: {time.time() - t0:.2f}s')

    # -------------------------
    # Graphon fit
    # -------------------------
    print('--- Graphon fit ---')
    gmodel = fit_final_model(
        Pi_tilde,
        U,
        V,
        S,
        r_coord=1,
        rank=2,
        alpha=1e-3,
    )

    t_l, _ = left_coordinate(U, r=1)

    # -------------------------
    # Size models
    # -------------------------
    print('--- Fit size models ---')
    size_model = fit_size_function_hurdle(t_l, pbar)

    # -------------------------
    # Seed
    # -------------------------
    seed = int(hashlib.md5(f'{MODEL}-{quarter}'.encode()).hexdigest()[:8], 16)
    key = random.PRNGKey(seed)

    # -------------------------
    # Calibration
    # -------------------------
    print('--- Calibration ---')
    target_d = density_offdiag_positive(L)

    key, skx, sky = random.split(key, 3)
    Xs = jnp.sort(random.uniform(skx, (N,)))
    Ys = jnp.sort(random.uniform(sky, (N,)))

    sample = graphon_kernel(
        Xs,
        Ys,
        gmodel.knots_l,
        gmodel.knots_r,
        gmodel.theta_l,
        gmodel.theta_r,
        gmodel.top_s,
    )

    print('--- Row normalization ---')
    weights = sparsemax(sample)

    key, skp = random.split(key)
    pbar_new, _, _ = sample_pbar_hurdle(
        key=skp,
        size_model=size_model,
        t_new=Xs,
        pbar_obs=pbar,
    )

    key, skk = random.split(key)
    key, kappa_star = bisection_kappa_for_L_density(
        skk,
        weights,
        pbar_new,
        target_d,
        k_lo=0.5,
        k_hi=1e5,
    )

    print(f'kappa_star = {float(kappa_star):.4f}')

    # -------------------------
    # One draw
    # -------------------------
    print('--- Sampling ---')
    key, skd = random.split(key)
    Pi_new = drichlet_rows_gamma(skd, weights, kappa_star)
    Pi_new = clean_Pi(Pi_new)

    key, skp = random.split(key)
    pbar_new, _, _ = sample_pbar_hurdle(
        key=skp,
        size_model=size_model,
        t_new=Xs,
        pbar_obs=pbar,
    )

    # Prior liabilities from graphon sample
    L_prior = compose_L_from_Pi_pbar(Pi_new, pbar_new)

    # -------------------------
    # RAS layer on L
    # -------------------------
    print('--- RAS balancing on L_prior ---')
    # Targets based on ORIGINAL L
    IL_target = jnp.sum(L, axis=1)   # total liabilities
    IA_target = jnp.sum(L, axis=0)   # total assets

    L_new = ras_balance(
        L_prior=L_prior,
        IL_target=IL_target,
        IA_target=IA_target,
        n_iter=ras_iter,
        tol=ras_tol,
        eps=eps,
    )

    Pi_new = L_to_Pi(L_new, eps=eps)
    pbar_new = jnp.sum(L_new, axis=1)

    # -------------------------
    # Network stats
    # -------------------------
    print('--- Network stats ---')
    r, c = strengths(L_new)
    r_pos = r[r > 0]
    c_pos = c[c > 0]

    k_hill_r = int(np.sqrt(r_pos.shape[0]))
    k_hill_c = int(np.sqrt(c_pos.shape[0]))

    stats = get_net_stats(L_new, r_pos, c_pos, k_hill_r, k_hill_c)
    stats = {k: float(v) for k, v in stats.items()}
    stats['quarter'] = quarter
    stats['model'] = MODEL
    stats['ras_iter'] = ras_iter
    stats['ras_tol'] = ras_tol

    # -------------------------
    # EN
    # -------------------------
    print('--- EN stats ---')
    en_stats = get_EN_stats(L_new, e)
    en_stats = to_numpy_tree(en_stats)

    en_summary = {k: float(en_stats[k]) for k in comparison_keys}
    en_summary['quarter'] = quarter
    en_summary['model'] = MODEL
    en_summary['ras_iter'] = ras_iter
    en_summary['ras_tol'] = ras_tol

    # -------------------------
    # Game
    # -------------------------
    print('--- Game stats ---')
    game_rows = []

    for kappa in kappa_grid:
        theta = kappa * pbar_new
        game_stats, s_final = get_game_stats(L_new, e, theta)

        res = br_residual(L_new, e, theta, s_final)
        res_val = float(jax.device_get(res))

        game_stats_np = to_numpy_tree(game_stats)
        row = {k: float(game_stats_np[k]) for k in comparison_keys}
        row['quarter'] = quarter
        row['model'] = MODEL
        row['kappa'] = kappa
        row['NE_resid'] = res_val
        row['ras_iter'] = ras_iter
        row['ras_tol'] = ras_tol

        game_rows.append(row)

    # -------------------------
    # SAVE CSV
    # -------------------------
    print('--- Saving CSV ---')

    os.makedirs(net_stats_root, exist_ok=True)
    os.makedirs(en_stats_root, exist_ok=True)
    os.makedirs(game_stats_root, exist_ok=True)

    pd.DataFrame([stats]).to_csv(
        os.path.join(net_stats_root, f'{quarter}_stats.csv'),
        index=False,
    )

    pd.DataFrame([en_summary]).to_csv(
        os.path.join(en_stats_root, f'{quarter}_en.csv'),
        index=False,
    )

    pd.DataFrame(game_rows).to_csv(
        os.path.join(game_stats_root, f'{quarter}_game.csv'),
        index=False,
    )

    print('--- DONE ---')


if __name__ == '__main__':
    typer.run(main)