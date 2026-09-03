import numpy as np
from jax import Array, random
import jax
import os
import time
from datetime import datetime
import csv
import gc

from .utils import (
    load_reconstruction_data,
    extract_arrays,
    RAS_feasibility_check,
    gpu_memory
)

from .reconstruction import (
    RAS_max_e,
    rescaled_features,
    make_type_vector,
    make_fitness,
    prob_matrix,
    build_masked_prior_until_ok,
    pick_density_by_ras_loss,
    solve_z_prob_matrix
)

jax.config.update('jax_enable_x64', True)


def main():
    save_folder = '/home/he266/research/data/processed/networks/FM'
    os.makedirs(save_folder, exist_ok=True)
    path = '/home/he266/research/data/raw/ffiec_csv'
    
    Lsavefolder = '/home/he266/research/data/processed/networks/FM/losses'
    os.makedirs(Lsavefolder, exist_ok=True)
    
    logpath = '/home/he266/research/logs/runtime'
    os.makedirs(logpath, exist_ok=True)
    logfile = os.path.join(logpath, 'fm_ablation01.csv')
    
    if not os.path.exists(logfile):
        with open(logfile, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'datetime',
                'model',
                'seed',
                'quarter',
                'N',
                'z',
                'density',
                'jax_x64_enabled',
                'epsilon',
                'max_iterations',
                'iterations',
                'row_loss',
                'col_loss',
                'RASruntime_seconds',
                'FMruntime_seconds',
                'dtype',
                'RASmem_before',
                'RASmem_after',
                'FMmem_before',
                'FMmem_after'
            ])
    dfs = []
    qs = []
    
    print('Loading data frames ...')
    for root, dirs, files in os.walk(path):
      for filename in files:
        local_path = os.path.join(root, filename)
        dfs.append(load_reconstruction_data(local_path))
        qs.append(local_path.split('/')[-1].split('.')[0])
    print('Finished loading data frames.')
    
    seed = 98642
    key = random.PRNGKey(seed)
    max_iter = 150_000
    eps = 1e-14
    for df, q in zip(dfs, qs):
        N = df.shape[0]
        IA, IL, TA, TL, cash, repo, FFP = extract_arrays(df)
        _, _, TA, TL, cash, repo, FFP, IA_tilde, IL_tilde = rescaled_features(IA, IL, TA, TL, cash, repo, FFP)
        x, l, w, n = make_type_vector(cash, TA, FFP, repo, TL, IA_tilde, IL_tilde)
        phi, psi = make_fitness(IA_tilde, IL_tilde, l, w, n)

        key, subkey = random.split(key)
        d_grid = [0.02, 0.03, 0.05, 0.07, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30,
                   0.4, 0.45, 0.55, 0.6, 0.65, 0.7]
        RASmem_before = gpu_memory()
        print(f'Starting density RAS for {q}')
        start1 = time.time()
        d_star, z_star, P_build, P0, A0, key = pick_density_by_ras_loss(
            phi, psi, x, TA, IA_tilde, IL_tilde,
            key=subkey,
            d_grid=d_grid,
            loss_tol=1e-6,          # start at 1e-6 relative; relax to 1e-5 if needed
            ras_iters_probe=800,
            build_tries_cap=500,
            tau=1e-5
            )
        RASruntime = time.time() - start1
        RASmem_after = gpu_memory()
        print(f'Finished density MC for {q}')
        
        FMmem_before = gpu_memory()
        print(f'Strating RAS for {q}')
        start2 = time.time()
        l_c, l_r, rLosses, cLosses, E, it_final = RAS_max_e(P0, IA_tilde, IL_tilde, epsilon= eps, max_iter=max_iter)
        E.block_until_ready()
        FMruntime = time.time() - start2
        FMmem_after = gpu_memory()
        print(f'Finished RAS for {q}')
        with open(logfile, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    datetime.now().isoformat(),
                    'FM',
                    seed,
                    q,
                    N,
                    z_star,
                    d_star,
                    str(jax.config.jax_enable_x64),
                    eps,
                    int(max_iter),
                    int(it_final),
                    float(l_r),
                    float(l_c),
                    RASruntime,
                    FMruntime,
                    str(E.dtype),
                    int(RASmem_before),
                    int(RASmem_after),
                    int(FMmem_before),
                    int(FMmem_after)
                ]
            )
        save_path = os.path.join(save_folder, f'{q}.npy')
        np.save(save_path, np.asarray(jax.device_get(E.T)))
        losses = {
            'quarter': q,
            'row_losses': np.asarray(jax.device_get(rLosses[:it_final + 1])),
            'column_losses': np.asarray(jax.device_get(cLosses[:it_final + 1])),
            }
        Lsavepath = os.path.join(Lsavefolder, f'{q}.npz')
        np.savez(Lsavepath, **losses)
        del E, P0, A0, P_build
        gc.collect()
        
if __name__ == '__main__':
    main()