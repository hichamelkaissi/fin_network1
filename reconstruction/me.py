import pandas as pd
import numpy as np
import jax.numpy as jnp
from jax import Array
import jax
import os
from tqdm import tqdm
from typing import Dict, Tuple
import time
from datetime import datetime
import csv
import gc

from .utils import (
    get_ia_il,
    get_prior_matrix,
    load_reconstruction_data,
    gpu_memory
)

from .reconstruction import (
    RAS_max_e
)

jax.config.update('jax_enable_x64', True)

def maximum_entropy(df: pd.DataFrame, epsilon: float = 1e-10, max_iter: int = 1000) -> Tuple[Array, Dict]:
    IA, IL = get_ia_il(df)
    P, IA_tilde, IL_tilde = get_prior_matrix(IA, IL)
    l_c, l_r, rLosses, cLosses, E, it = RAS_max_e(P, IA_tilde, IL_tilde, epsilon=epsilon, max_iter=max_iter)
    return E, rLosses, cLosses, {'column_loss': l_c, 'row_loss': l_r, 'iterations': it}

def main():
    max_iter = 100_000
    eps = 1e-14
    
    logpath = '/home/he266/research/logs/runtime'
    os.makedirs(logpath, exist_ok=True)
    log_file = os.path.join(logpath, 'me_ablation01.csv')
    
    if not os.path.exists(log_file):
        with open(log_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp',
                'model',
                'seed',
                'quarter',
                'N',
                'jax_x64_enabled',
                'epsilon',
                'max_iterations',
                'iterations',
                'row_loss',
                'col_loss',
                'runtime_seconds',
                'dtype',
                'mem_before',
                'mem_after',
            ])
                
    path = '/home/he266/research/data/raw/ffiec_csv'
    dpaths = sorted(os.listdir(path))
    savefolder = '/home/he266/research/data/processed/networks/ME'
    os.makedirs(savefolder, exist_ok=True)

    Lsavefolder = '/home/he266/research/data/processed/networks/ME/losses'
    os.makedirs(Lsavefolder, exist_ok=True)

    for dpath in tqdm(dpaths):
        thepath = os.path.join(path, dpath)
        q = thepath.split('/')[-1].split('.')[0]
        df = load_reconstruction_data(thepath)
        N = df.shape[0]
        mem_before = gpu_memory()
        start = time.time()
        E, rLosses, cLosses, info = maximum_entropy(df, epsilon=eps, max_iter=max_iter)
        E.block_until_ready()
        runtime = time.time() - start
        mem_after = gpu_memory()

        with open(log_file, 'a', newline='') as f:
          writer = csv.writer(f)
          writer.writerow([
            datetime.now().isoformat(),
            'ME',
            None,
            q,
            N,
            str(jax.config.jax_enable_x64),
            float(eps),
            int(max_iter),
            int(info['iterations']),
            float(info['row_loss']),
            float(info['column_loss']),
            float(runtime),
            str(E.dtype),
            int(mem_before),
            int(mem_after),
          ])
          
        savepath = os.path.join(savefolder, f'{q}.npy')
        np.save(savepath, np.array(E.T))

        it = info['iterations']
        losses = {
            'quarter': q,
            'row_losses': np.asarray(rLosses[:it + 1]),
            'column_losses': np.asarray(cLosses[:it + 1]),
        }
        Lsavepath = os.path.join(Lsavefolder, f'{q}.npz')
        np.savez(Lsavepath, **losses)

        del E, rLosses, cLosses, info, it
        gc.collect()

if __name__ == '__main__':
  main()