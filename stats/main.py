import jax
import jax.numpy as jnp
from jax import Array
import numpy as np
from typing import Dict, Tuple
from functools import partial
from tqdm import tqdm
import pandas as pd
import os
import argparse
jax.config.update('jax_enable_x64', True)

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


@partial(jax.jit, static_argnames=('k_hill_r', 'k_hill_c'))
def all_stats(
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

def main(model: str):
    path = f'/home/he266/research/data/processed/networks/{model}'

    dfs = []
    # optional: build file list so tqdm has a total
    npy_files = []
    for root, _, files in os.walk(path):
        for fname in files:
            if fname.endswith('.npy'):
                npy_files.append((root, fname))

    for root, fname in npy_files:
        quarter = os.path.splitext(fname)[0]
        print(f'Computing stats for {quarter}')

        Lpath = os.path.join(root, fname)
        L = jnp.asarray(np.load(Lpath))

        r, c = strengths(L)
        r_pos = r[r > 0]
        c_pos = c[c > 0]

        k_hill_r = int(np.floor(np.sqrt(int(r_pos.shape[0]))))
        k_hill_c = int(np.floor(np.sqrt(int(c_pos.shape[0]))))

        stats = all_stats(
            L,
            r_pos,
            c_pos,
            k_hill_r=k_hill_r,
            k_hill_c=k_hill_c,
        )

        
        print(f'Finished stats for {quarter}')
        stats_py = {k: float(v) for k, v in stats.items()}
        stats_py['quarter'] = quarter
        stats_py['file'] = fname
        dfs.append(pd.DataFrame([stats_py]))
        

    stats_df = pd.concat(dfs, ignore_index=True)

    savefolder = f'/home/he266/research/results/tables/Recon/{model}'
    os.makedirs(savefolder, exist_ok=True)
    savepath = os.path.join(savefolder, 'stats.csv')
    stats_df.to_csv(savepath, index=False)
    print(f'Saved: {savepath}')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='Stats',
        description='Compute stats for reconstructed networks'
    )
    parser.add_argument('model')
    args = parser.parse_args()
    main(args.model)