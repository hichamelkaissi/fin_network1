from __future__ import annotations

import os
import time

import jax
import jax.numpy as jnp
from jax import Array
import numpy as np
import pandas as pd
import typer
from rich import print

from clearing_game.game_nash import strategic_EN_game, br_residual
from EN_mechanism.en_algo_stats import EN_stats
from common.utils import (
    load_reconstruction_data,
    index_by_quarter,
    load_config,
    require,
    MetricsCSV,
)

jax.config.update('jax_enable_x64', True)


@jax.jit
def get_game_stats(
        L: Array,
        e: Array,
        theta: Array,
        tol: float = 1e-12,
):
    p_final, s_final, pbar, M, D_hist, new_hist, T = strategic_EN_game(
        L, e, theta, tol=tol
    )
    stats = EN_stats(
        L=L,
        e=e,
        p_final=p_final,
        D_hist=D_hist,
        new_hist=new_hist,
        rounds_taken=T,
    )
    return stats, s_final


def to_numpy_tree(tree):
    return jax.tree.map(
        lambda x: np.asarray(jax.device_get(x)),
        tree,
    )


def main(
        model: str = typer.Option(...),
        config: str = 'game_main.toml'
         ):
    cfg = load_config(config)
    network_root = require(cfg, 'paths', 'network_root')
    ffiec_root = require(cfg, 'paths', 'ffiec_root')
    results_root = require(cfg, 'paths', 'results_root')
    logdir = require(cfg, 'paths', 'logdir')
    comparison_keys = require(cfg, 'stats', 'comparison_keys')
    kappa_grid = require(cfg, 'kappas', 'kappa_grid')

    path = f'{network_root}/{model}'
    epath = ffiec_root
    savedir = f'{results_root}/{model}'

    os.makedirs(savedir, exist_ok=True)
    os.makedirs(logdir, exist_ok=True)

    Lpaths = index_by_quarter(path, 'npy')
    epaths = index_by_quarter(epath, 'csv')

    common = sorted(set(Lpaths) & set(epaths))

    fieldnames = ['quarter', 'model', 'runtime', 'kappa']
    logfile = os.path.join(logdir, f'game_{model}_log.csv')
    csvlogger = MetricsCSV(logfile, fieldnames)

    rows = []

    for q in common:
        print(f'| {q} | Loading L...')
        L = jnp.asarray(np.load(Lpaths[q]))

        print(f'| {q} | Preliminary computations...')
        pbar = jnp.sum(L, axis=1)
        df = load_reconstruction_data(epaths[q])
        e = jnp.asarray(df['opCF_in'].to_numpy())

        for kappa in kappa_grid:
            theta = kappa * pbar
            print(f'{q} | Running stats...')

            t0 = time.perf_counter()
            stats, s_final = get_game_stats(L, e, theta)

            jax.tree.map(lambda x: x.block_until_ready(), stats)
            s_final.block_until_ready()

            runtime = time.perf_counter() - t0

            stats_np = to_numpy_tree(stats)
            summary = {k: float(stats_np[k]) for k in comparison_keys}
            summary['quarter'] = q
            summary['model'] = model
            summary['kappa'] = kappa
            summary['runtime'] = runtime

            res = br_residual(L, e, theta, s_final)
            summary['NE_resid'] = float(jax.device_get(res))

            csvlogger.log(
                quarter=q,
                model=model,
                runtime=runtime,
                kappa=kappa,
            )

            rows.append(summary)

    csvlogger.close()

    savepath = os.path.join(savedir, 'game_stats.parquet')
    df_out = pd.DataFrame(rows)
    try:
        df_out.to_parquet(savepath, engine='pyarrow')
    except Exception as e:
        print(f'Parquet write failed: {e}')
        csv_path = os.path.join(savedir, 'game_stats.csv')
        df_out.to_csv(csv_path, index=False)
        print(f'Wrote CSV instead: {csv_path}')


if __name__ == '__main__':
    typer.run(main)