from __future__ import annotations

import gc
import hashlib
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import typer
from jax import Array, random

from app.theapp import get_EN_stats, get_game_stats, get_net_stats
from clearing_game.game_nash import br_residual
from common.utils import (
    MetricsCSV,
    cpu_mem,
    gpu_mem,
    index_by_quarter,
    load_config,
    load_reconstruction_data,
    require,
    setup_logger,
)
from graphon.graphon import (
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
    maybe_block,
    regularise,
    row_stochasticize,
    winsorise_by_row,
)
from stats.stats import strengths

jax.config.update('jax_enable_x64', True)


def has_bad_values(x: Array) -> bool:
    x_np = np.asarray(jax.device_get(x))
    return not np.all(np.isfinite(x_np))


def to_numpy_tree(tree):
    return jax.tree.map(
        lambda x: np.asarray(jax.device_get(x)),
        tree,
    )


def main(
        MODEL: str = typer.Option(...),
        runs: int = 150,
        config: str = 'config.toml',
):
    cfg = load_config(config)

    network_root = require(cfg, 'paths', 'network_root')
    ffiec_root = require(cfg, 'paths', 'ffiec_root')
    net_stats_root = require(cfg, 'paths', 'net_stats_root')
    en_stats_root = require(cfg, 'paths', 'en_stats_root')
    game_stats_root = require(cfg, 'paths', 'game_stats_root')
    logroot = require(cfg, 'paths', 'logroot')
    comparison_keys = require(cfg, 'stats', 'comparison_keys')
    kappa_grid = require(cfg, 'kappas', 'kappa_grid')

    if len(kappa_grid) == 0:
        raise ValueError('kappa_grid must be nonempty')

    path = f'{network_root}/{MODEL}'
    epath = ffiec_root
    netstats_savedir = f'{net_stats_root}/{MODEL}'
    enstats_savedir = f'{en_stats_root}/{MODEL}'
    gamestats_savedir = f'{game_stats_root}/{MODEL}'

    os.makedirs(netstats_savedir, exist_ok=True)
    os.makedirs(enstats_savedir, exist_ok=True)
    os.makedirs(gamestats_savedir, exist_ok=True)
    os.makedirs(logroot, exist_ok=True)

    Lpaths = index_by_quarter(path, 'npy')
    epaths = index_by_quarter(epath, 'csv')
    common = sorted(set(Lpaths) & set(epaths))

    logfile = os.path.join(logroot, f'{MODEL}_deter_economy.log')
    csvlogfile = os.path.join(logroot, f'{MODEL}_deter_economy.csv')
    logger = setup_logger(logfile)

    fieldnames = [
        'quarter',
        'model',
        'mode',
        'seed',
        'usv_runtime',
        'kernel_runtime',
        'calibration_runtime',
        'mc_runtime',
        'gpu_before_q',
        'cpu_before_q',
        'gpu_before_MC',
        'gpu_after_MC',
        'cpu_before_MC',
        'cpu_after_MC',
        'max_NE_resid',
        'target_d',
        'kappa_star',
        'stats_rows',
        'en_rows',
        'game_rows',
    ]
    csv_logger = MetricsCSV(csvlogfile, fieldnames)

    eps = 1e-12

    stats_rows = []
    en_rows = []
    game_rows = []

    logger.info(
        f'Starting deter economy run | model={MODEL} | '
        f'quarters={len(common)} | runs={runs}'
    )

    for q in common:
        max_ne_resid = 0.0

        gpu_before_q = gpu_mem()
        cpu_before_q = cpu_mem()

        L = jnp.asarray(np.load(Lpaths[q]))
        N = L.shape[0]
        df = load_reconstruction_data(epaths[q])
        e = jnp.asarray(df['opCF_in'].to_numpy())

        pbar = jnp.sum(L, axis=1)
        Pi = jnp.where(
            pbar[:, None] > 0,
            L / jnp.maximum(pbar[:, None], eps),
            0.0,
            )
        Pi_w = winsorise_by_row(Pi)
        Pi_tilde, _ = regularise(Pi_w)

        if has_bad_values(Pi_tilde):
            logger.warning(f'[{q}] bad values in Pi_tilde')

        usv_start = time.perf_counter()
        U, S, V = get_svd(Pi_tilde)
        U.block_until_ready()
        S.block_until_ready()
        V.block_until_ready()
        usv_runtime = time.perf_counter() - usv_start

        kernel_start = time.perf_counter()
        gmodel = fit_final_model(
            Pi_tilde,
            U,
            V,
            S,
            r_coord=1,
            rank=12,
            alpha=1e-3,
        )
        jax.tree.map(maybe_block, gmodel)
        kernel_runtime = time.perf_counter() - kernel_start

        seed = int(hashlib.md5(f'{MODEL}-{q}'.encode()).hexdigest()[:8], 16)
        key = random.PRNGKey(seed)

        logger.info(
            f'[{q}] start | model={MODEL} | N={N} | runs={runs} | seed={seed}'
        )

        calibration_start = time.perf_counter()

        target_d = density_offdiag_positive(L)
        target_d_val = float(jax.device_get(target_d))

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
        weights = row_stochasticize(sample, method='softmax')

        gc.collect()

        key, skk = random.split(key)
        key, kappa_star = bisection_kappa_for_L_density(
            skk,
            weights,
            pbar,
            target_d,
            k_lo=0.5,
            k_hi=1e5,
            tol=1e-3,
            max_iter=25,
            M=3,
        )
        kappa_star_val = float(jax.device_get(kappa_star))
        calibration_runtime = time.perf_counter() - calibration_start

        if not (0.0 <= target_d_val <= 1.0):
            logger.warning(f'[{q}] suspicious target_d={target_d_val:.6f}')

        if kappa_star_val <= 0.0:
            logger.warning(f'[{q}] suspicious kappa_star={kappa_star_val:.6f}')

        logger.info(
            f'[{q}] calibration | '
            f'target_d={target_d_val:.6f} | '
            f'kappa_star={kappa_star_val:.6f}'
        )

        gpu_before_MC = gpu_mem()
        cpu_before_MC = cpu_mem()
        mc_start = time.perf_counter()

        for i in range(runs):
            t0 = time.perf_counter()

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
            weights = row_stochasticize(sample, method='softmax')

            key, skd = random.split(key)
            Pi_new = drichlet_rows_gamma(skd, weights, kappa_star)
            Pi_new = clean_Pi(Pi_new)

            sample_runtime = time.perf_counter() - t0

            if has_bad_values(Pi_new):
                logger.warning(f'[{q}] draw={i + 1}/{runs} | bad_values in Pi_new')

            L_new = compose_L_from_Pi_pbar(Pi_new, pbar)
            L_new.block_until_ready()

            if has_bad_values(L_new):
                logger.warning(f'[{q}] draw={i + 1}/{runs} | bad_values in L_new')

            draw_density = float(jax.device_get(density_offdiag_positive(L_new)))

            r, c = strengths(L_new)
            r_pos = r[r > 0]
            c_pos = c[c > 0]

            k_hill_r = int(np.floor(np.sqrt(int(r_pos.shape[0]))))
            k_hill_c = int(np.floor(np.sqrt(int(c_pos.shape[0]))))

            logger.debug(
                f'[{q}] draw={i + 1}/{runs} | '
                f'k_hill_r={k_hill_r} | k_hill_c={k_hill_c} | '
                f'density={draw_density:.6f}'
            )

            t1 = time.perf_counter()
            stats = get_net_stats(
                L_new,
                r_pos,
                c_pos,
                k_hill_r=k_hill_r,
                k_hill_c=k_hill_c,
            )
            jax.tree.map(lambda x: x.block_until_ready(), stats)
            net_runtime = time.perf_counter() - t1

            stats_py = {k: float(v) for k, v in stats.items()}
            stats_py['quarter'] = q
            stats_py['model'] = MODEL
            stats_py['draw_id'] = i
            stats_py['mode'] = 'deter_economy'
            stats_rows.append(stats_py)

            en_start = time.perf_counter()
            en_stats = get_EN_stats(L_new, e)
            jax.tree.map(lambda x: x.block_until_ready(), en_stats)
            en_runtime = time.perf_counter() - en_start

            en_stats_np = to_numpy_tree(en_stats)
            en_summary = {k: float(en_stats_np[k]) for k in comparison_keys}
            en_summary['quarter'] = q
            en_summary['model'] = MODEL
            en_summary['draw_id'] = i
            en_summary['mode'] = 'deter_economy'
            en_rows.append(en_summary)

            t3 = time.perf_counter()
            for kappa in kappa_grid:
                theta = kappa * pbar
                game_stats, s_final = get_game_stats(L_new, e, theta)
                jax.tree.map(lambda x: x.block_until_ready(), game_stats)
                s_final.block_until_ready()

                game_stats_np = to_numpy_tree(game_stats)
                game_summary = {k: float(game_stats_np[k]) for k in comparison_keys}
                game_summary['quarter'] = q
                game_summary['model'] = MODEL
                game_summary['draw_id'] = i
                game_summary['mode'] = 'deter_economy'
                game_summary['kappa'] = kappa

                res = br_residual(L_new, e, theta, s_final)
                res_val = float(jax.device_get(res))
                game_summary['NE_resid'] = res_val
                game_rows.append(game_summary)

                if not np.isfinite(res_val):
                    logger.warning(
                        f'[{q}] draw={i + 1}/{runs} | '
                        f'kappa={kappa} | nonfinite NE_resid={res_val}'
                    )

                logger.debug(
                    f'[{q}] draw={i + 1}/{runs} | '
                    f'kappa={kappa} | NE_resid={res_val:.3e}'
                )

                max_ne_resid = max(max_ne_resid, abs(res_val))

            game_runtime = time.perf_counter() - t3
            draw_runtime = time.perf_counter() - t0

            del (
                L_new,
                Pi_new,
                sample,
                weights,
                stats,
                stats_py,
                en_stats,
                en_stats_np,
                en_summary,
                game_stats,
                game_stats_np,
                game_summary,
                s_final,
                res,
                r,
                c,
                r_pos,
                c_pos,
            )
            gc.collect()

            if (i == 0) or ((i + 1) % 10 == 0) or (i == runs - 1):
                logger.info(
                    f'[{q}] draw={i + 1}/{runs} | '
                    f'sample={sample_runtime:.3f}s | '
                    f'net={net_runtime:.3f}s | '
                    f'en={en_runtime:.3f}s | '
                    f'game={game_runtime:.3f}s | '
                    f'total={draw_runtime:.3f}s | '
                    f'density={draw_density:.4f} | '
                    f'max_NE_resid={max_ne_resid:.3e}'
                )

        mc_runtime = time.perf_counter() - mc_start
        gpu_after_MC = gpu_mem()
        cpu_after_MC = cpu_mem()

        logger.info(
            f'[{q}] done | '
            f'usv_runtime={usv_runtime:.3f}s | '
            f'kernel_runtime={kernel_runtime:.3f}s | '
            f'calibration_runtime={calibration_runtime:.3f}s | '
            f'mc_runtime={mc_runtime:.3f}s | '
            f'gpu_before_q={gpu_before_q} | '
            f'cpu_before_q={cpu_before_q} | '
            f'gpu_before_MC={gpu_before_MC} | '
            f'gpu_after_MC={gpu_after_MC} | '
            f'cpu_before_MC={cpu_before_MC} | '
            f'cpu_after_MC={cpu_after_MC} | '
            f'max_NE_resid={max_ne_resid:.3e} | '
            f'stats_rows={len(stats_rows)} | '
            f'en_rows={len(en_rows)} | '
            f'game_rows={len(game_rows)}'
        )

        csv_logger.log(
            quarter=q,
            model=MODEL,
            mode='deter_economy',
            seed=seed,
            usv_runtime=usv_runtime,
            kernel_runtime=kernel_runtime,
            calibration_runtime=calibration_runtime,
            mc_runtime=mc_runtime,
            gpu_before_q=gpu_before_q,
            cpu_before_q=cpu_before_q,
            gpu_before_MC=gpu_before_MC,
            gpu_after_MC=gpu_after_MC,
            cpu_before_MC=cpu_before_MC,
            cpu_after_MC=cpu_after_MC,
            max_NE_resid=max_ne_resid,
            target_d=target_d_val,
            kappa_star=kappa_star_val,
            stats_rows=len(stats_rows),
            en_rows=len(en_rows),
            game_rows=len(game_rows),
        )

    logger.info(
        f'Building final dataframes | '
        f'stats_rows={len(stats_rows)} | '
        f'en_rows={len(en_rows)} | '
        f'game_rows={len(game_rows)}'
    )

    net_df = pd.DataFrame(stats_rows)
    en_df = pd.DataFrame(en_rows)
    game_df = pd.DataFrame(game_rows)

    net_savepath = os.path.join(netstats_savedir, 'stats.parquet')
    en_savepath = os.path.join(enstats_savedir, 'enstats.parquet')
    game_savepath = os.path.join(gamestats_savedir, 'gamestats.parquet')

    logger.info(
        f'Writing parquet outputs | '
        f'stats={net_savepath} | '
        f'en={en_savepath} | '
        f'game={game_savepath}'
    )

    net_df.to_parquet(net_savepath, engine='pyarrow')
    en_df.to_parquet(en_savepath, engine='pyarrow')
    game_df.to_parquet(game_savepath, engine='pyarrow')

    logger.info(
        f'Output sizes | '
        f'stats={os.path.getsize(net_savepath)} | '
        f'en={os.path.getsize(en_savepath)} | '
        f'game={os.path.getsize(game_savepath)}'
    )

    logger.info('Finished successfully')
    csv_logger.close()


if __name__ == '__main__':
    typer.run(main)