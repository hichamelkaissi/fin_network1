from __future__ import annotations

from jax import Array
import jax
import jax.numpy as jnp
import numpy as np
import time
import os
import typer
from rich import print

from .en_algo_stats import (
    EN_rogers_veraart_fda,
    EN_stats
)
from .utils import (
    index_by_quarter,
    load_reconstruction_data,
    setup_logger
)
jax.config.update('jax_enable_x64', True)

@jax.jit
def one_pair_kernel(
        L: Array, e: Array,
        alpha: float = 1.0,
        beta: float = 1.0,
        spectral_n_iter: int = 100
):
    p_final, D_hist, new_hist, rounds_taken = EN_rogers_veraart_fda(L, e, alpha=alpha, beta=beta)
    return EN_stats(
        L, e,
        p_final, D_hist,
        new_hist, rounds_taken,
        alpha=alpha,
        beta=beta,
        spectral_n_iter=spectral_n_iter
    )
def to_numpy_tree(tree):
    return jax.tree.map(
        lambda x: np.asarray(jax.device_get(x)),
        tree
    )

def require(cfg: dict, *keys):
    cur = cfg
    path = []
    for k in keys:
        path.append(k)
        if not isinstance(cur, dict) or k not in cur:
            raise KeyError(f'Missing required config key: {".".join(path)}')
        cur = cur[k]
    return cur


def main(
        model: str = typer.Option(...),
        config: str = 'config.toml'
):
    cfg = load_config(config)

    # ---- paths from  config; Overriding allowed ----
    network_root = require(cfg, 'paths', 'networks_root')
    ffiec_root = require(cfg, 'paths', 'ffiec_root')
    results_root = require(cfg, 'paths', 'results_root')
    logdir = require(cfg, 'paths', 'logdir')

    inpath = f'{network_root}/{model}'
    epath = ffiec_root
    savedir = f'{results_root}/{model}'

    os.makedirs(savedir, exist_ok=True)
    os.makedirs(logdir, exist_ok=True)

    logfile = os.path.join(logdir, f'en_stats_{model}.log')
    logger = setup_logger(logfile)


    # ---- EN params | No override ----
    alpha = float(require(cfg, 'en', 'alpha'))
    beta = float(require(cfg, 'en', 'beta'))

    # ---- Stats keys | No override ---
    comparison_keys = require(cfg, 'stats', 'comparison_keys')

    Lpaths = index_by_quarter(inpath, 'npy')
    epaths = index_by_quarter(epath,  'csv')
    common = sorted(set(Lpaths) & set(epaths))

    logger.debug(f'Loaded config: {config}')
    logger.debug(f'alpha={alpha} | beta={beta}')
    logger.debug(f'quarters={len(common)} | model={model}')

    for q in common:

        logger.debug(f'{q} | {model}')

        L = jnp.asarray(np.load(Lpaths[q]))
        logger.debug(f'N: {L.shape[0]} | dtype: {L.dtype}')

        df = load_reconstruction_data(epaths[q])
        e = jnp.asarray(df['opCF_in'].to_numpy())

        start = time.perf_counter()

        stats = one_pair_kernel(L, e)


        jax.tree.map(
            lambda x: x.block_until_ready(),
            stats
        )

        runtime_compute = time.perf_counter() - start

        stats_np = to_numpy_tree(stats)

        summary = {k: float(stats_np[k]) for k in comparison_keys}
        summary['runtime_compute_sec'] = float(runtime_compute)

        savepath = os.path.join(savedir, f'{q}.npz')
        np.savez(savepath, **summary)

        logger.debug(
            f'{q} saved | runtime_compute_sec: {runtime_compute:.4f}'
        )


if __name__ == '__main__':
    typer.run(main)