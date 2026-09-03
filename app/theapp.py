import jax
import jax.numpy as jnp
from jax import Array
import numpy as np
from typing import Dict, Tuple
from functools import partial

from stats.stats import (
    strengths,
    sparsity,
    isolates,
    spectral_stats,
    node_level_stats,
    dist_stats,
    concentration_stats,
    correlation_stats,
    weighted_risk_stats,
)
from EN_mechanism.en_algo_stats import (
    EN_rogers_veraart_fda,
    EN_stats
)
from clearing_game.game_nash import strategic_EN_game, br_residual

@partial(jax.jit, static_argnames=('k_hill_r', 'k_hill_c'))
def get_net_stats(
        L: Array,
        r_pos: Array,
        c_pos: Array,
        k_hill_r: int,
        k_hill_c: int,
) -> Dict[str, Array]:

    r, c = strengths(L)

    return {
        'isolates': isolates(L),
        'sparsity': sparsity(L),
        **spectral_stats(L, top_eigs=1),
        **node_level_stats(r, c),
        **dist_stats(r_pos, c_pos, k_hill_r, k_hill_c),
        **concentration_stats(r, c),
        **correlation_stats(r, c),
        **weighted_risk_stats(L),
    }

@jax.jit
def get_EN_stats(
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