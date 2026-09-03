import jax.numpy as jnp
from jax import random
import jax
from tqdm import tqdm
import numpy as np
import time
import os
import gc
import hashlib
from argparse import ArgumentParser

from .graphon import (
    left_coordinate,
    right_coordinate,
    fit_final_model,
    graphon_kernel
)
from .utils import (
    winsorise_by_row,
    regularise,
    get_svd,
    row_stochasticize,
    density_offdiag_positive,
    bisection_kappa_for_L_density,
    clean_Pi,
    compose_L_from_Pi_pbar,
    setup_logger,
    maybe_block,
    gpu_mem,
    cpu_mem
)

from .sample import (
    drichlet_rows_gamma
)

from .pbar_sample import (
    fit_size_function_hurdle,
    sample_pbar_hurdle
)

jax.config.update('jax_enable_x64', True)

def mem_snapshot():
    return gpu_mem(), cpu_mem()

def main(MODEL_NAME: str, mc_runs: int):
    INPUTDIR = f"/home/he266/research/data/processed/networks/{MODEL_NAME}"
    SAVEMDIR = f"/home/he266/research/data/processed/samples/{MODEL_NAME}"
    LOGDIR = "/home/he266/research/logs/runtime"
    LOGPATH = os.path.join(LOGDIR, f"{MODEL_NAME}_samples01.log")

    os.makedirs(SAVEMDIR, exist_ok=True)
    os.makedirs(LOGDIR, exist_ok=True)

    logger = setup_logger(LOGPATH)
    logger.debug(f"MODEL_NAME: {MODEL_NAME}")
    logger.debug(f"JAX x64 enabled: {jax.config.read('jax_enable_x64')}")

    eps = 1e-12

    # ------------------------------------------------------------
    # Walk quarters
    # ------------------------------------------------------------
    for root, dirs, files in os.walk(INPUTDIR):
        for fname in sorted(files):
            if not fname.endswith(".npy"):
                continue

            quarter = os.path.splitext(fname)[0]
            savedir = os.path.join(SAVEMDIR, quarter)
            os.makedirs(savedir, exist_ok=True)

            logger.debug(f"{quarter} | {MODEL_NAME} | loading")

            # ----------------------------
            # Load L
            # ----------------------------
            path = os.path.join(root, fname)
            L = jnp.asarray(np.load(path))
            N = L.shape[0]
            logger.debug(f"{quarter} | dtype={L.dtype} | N={N}")

            # Economic magnitudes (keep RAW)
            pbar = jnp.sum(L, axis=1)
            Pi = jnp.where(pbar[:, None] > 0, L / jnp.maximum(pbar[:, None], eps), 0.0)

            # ----------------------------
            # Regularise Pi for estimation stability
            # ----------------------------
            Pi_w = winsorise_by_row(Pi)
            Pi_tilde, src = regularise(Pi_w)
            logger.debug(f"{quarter} | Pi winsorised + regularised | src={src}")

            # ----------------------------
            # SVD + coords
            # ----------------------------
            logger.debug(f'GPU memory before SVD: {gpu_mem()} | {quarter}')
            logger.debug(f"CPU memory before SVD: {cpu_mem()} | {quarter}")
            usv_start = time.time()
            U, S, V = get_svd(Pi_tilde)
            U.block_until_ready(); S.block_until_ready(); V.block_until_ready()
            usv_runtime = time.time() - usv_start
            logger.debug(f"{quarter} | SVD runtime={usv_runtime:.3f}s")
            logger.debug(f'GPU memory after SVD: {gpu_mem()} | {quarter}')
            logger.debug(f'CPU memory after SVD: {cpu_mem()} | {quarter}')

            t_l, _ = left_coordinate(U, r=1)

            # ----------------------------
            # Fit graphon model
            # ----------------------------
            logger.debug(f'GPU memory before the fit: {gpu_mem()} | {quarter}')
            logger.debug(f'CPU memory before the fit: {cpu_mem()} | {quarter}')
            kernel_start = time.time()
            gmodel = fit_final_model(
                Pi_tilde, U, V, S,
                r_coord=1,
                rank=12,
                alpha=1e-3,
            )
            jax.tree.map(maybe_block, gmodel)
            kernel_runtime = time.time() - kernel_start
            logger.debug(f"{quarter} | graphon fit runtime={kernel_runtime:.3f}s")
            logger.debug(f'GPU memory after the fit: {gpu_mem()} | {quarter}')
            logger.debug(f'CPU memory after the fit: {cpu_mem()} | {quarter}')

            # ----------------------------
            # Fit pbar size model (conditional on t_l)
            # ----------------------------
            size_model = fit_size_function_hurdle(
                t_l=t_l,
                pbar=pbar,
                K=15,
                degree=3,
                ridge_alpha=1e-3,
                penalty_order=2,
                eps=eps,
                resid_winsor_q=0.999,
                n_bins=30,
                k_tail=30,
                min_tail_n=20,
            )
            tail = size_model.get("tail", None)
            if tail is not None:
                logger.debug(
                    f"{quarter} | tail_u={float(tail['u']):.4f} | "
                    f"xi={float(tail['xi']):.4f} | beta={float(tail['beta']):.4f} | "
                    f"k_tail={tail.get('k_tail','NA')} | n_exc={tail.get('n_exc','NA')}"
                )
            # ----------------------------
            # RNG seed per quarter (reproducible, different per quarter)
            # ----------------------------
            # Use a stable hash-like seed from the quarter string
            seed = int(hashlib.md5(f"{MODEL_NAME}-{quarter}".encode()).hexdigest()[:8], 16)
            key = random.PRNGKey(seed)
            logger.debug(f"{quarter} | seed={seed}")

            # ----------------------------
            # Calibrate kappa* for target density
            # ----------------------------
            target_d = density_offdiag_positive(L)
            logger.debug(f"{quarter} | target density={float(target_d):.6f}")

            # sample one pair of latent positions for calibration
            key, skx, sky = random.split(key, 3)
            Xs = jnp.sort(random.uniform(skx, (N,)))
            Ys = jnp.sort(random.uniform(sky, (N,)))

            sample = graphon_kernel(
                Xs, Ys,
                gmodel.knots_l, gmodel.knots_r,
                gmodel.theta_l, gmodel.theta_r,
                gmodel.top_s,
            )
            weights = row_stochasticize(sample, method="softmax")

            # sample pbar once for calibration target (consistent with later draws)
            key, skp = random.split(key)
            pbar_new, _, _ = sample_pbar_hurdle(
                key=skp,
                size_model=size_model,
                t_new=Xs,
                pbar_obs=pbar,
                noise_scale=1.0,
                calibrate_log_affine=True,
                cal_q1=0.01,
                cal_q2=0.9975,
                b_cap=5,
                use_evt_guardrail=True,
                p_tail_refresh=0.05,
                guardrail_mult=3,
                enforce_pos_sum=True,
                scale_cap=1.25,
            )
            gc.collect()
            key, skk = random.split(key)
            key, kappa_star = bisection_kappa_for_L_density(
                skk, weights, pbar_new, target_d,
                k_lo=0.5, k_hi=1e5,
                tol=1e-3, max_iter=25, M=3,
            )
            logger.debug(f"{quarter} | kappa_star={float(kappa_star):.6f}")

            # ----------------------------
            # Monte Carlo draws
            # ----------------------------
            logger.debug(f"{quarter} | MC draws={mc_runs}")
            logger.debug(f'GPU memory before MC {gpu_mem()} | {quarter}')
            logger.debug(f'CPU memory before MC {cpu_mem()} | {quarter}')
            for i in tqdm(range(mc_runs), desc=f"{quarter}"):

                key, skx, sky = random.split(key, 3)
                Xs = jnp.sort(random.uniform(skx, (N,)))
                Ys = jnp.sort(random.uniform(sky, (N,)))
                if i > 0 and i % 10 == 0:
                    gpu, cpu = mem_snapshot()
                    logger.debug(f"{quarter} | draw={i} | GPU={gpu}MB CPU={cpu}MB | stage=kernel/start")
                sample = graphon_kernel(
                    Xs, Ys,
                    gmodel.knots_l, gmodel.knots_r,
                    gmodel.theta_l, gmodel.theta_r,
                    gmodel.top_s,
                )
                weights = row_stochasticize(sample, method="softmax")

                if i > 0 and i % 10 == 0:
                    gpu, cpu = mem_snapshot()
                    logger.debug(f"{quarter} | draw={i} | GPU={gpu}MB CPU={cpu}MB | stage=Dirichlet/start")

                key, skd = random.split(key)
                Pi_new = drichlet_rows_gamma(skd, weights, kappa_star)
                Pi_new = clean_Pi(Pi_new)


                key, skp = random.split(key)

                pbar_new, _, _ = sample_pbar_hurdle(
                    key=skp,
                    size_model=size_model,
                    t_new=Xs,
                    pbar_obs=pbar,
                    noise_scale=1.0,
                    calibrate_log_affine=True,
                    cal_q1=0.01,
                    cal_q2=0.9975,
                    b_cap=5,
                    use_evt_guardrail=True,
                    p_tail_refresh=0.05,
                    guardrail_mult=3,
                    enforce_pos_sum=True,
                    scale_cap=1.25,
                )

                L_new = compose_L_from_Pi_pbar(Pi_new, pbar_new)
                L_new.block_until_ready()

                # optional density log
                if i > 0 and i % 10 == 0:
                    dens = density_offdiag_positive(L_new)
                    logger.debug(f"{quarter} | draw={i} | dens={float(dens):.6f}")
                if i > 0 and i % 20 == 0:
                    row_err = jnp.max(jnp.abs(jnp.sum(L_new, axis=1) - pbar_new))
                    logger.debug(f"{quarter} | draw={i} | max_row_sum_error={float(row_err):.3e}")

                savepath = os.path.join(savedir, f"draw{i}.npz")
                np.savez_compressed(
                    savepath,
                    Pi=np.array(Pi_new, dtype=np.float32),
                    pbar=np.array(pbar_new, dtype=np.float64),
                )


                # housekeeping
                del L_new, Pi_new, pbar_new
                if i > 0 and (i % 25) == 0:
                    gc.collect()
                    logger.debug(f'{quarter} | draw: {i} | gc.collect()')

            logger.debug(f'GPU memory after MC {gpu_mem()} | {quarter}')
            logger.debug(f'CPU memory after MC {cpu_mem()} | {quarter}')

            # clear memory between quarters
            del L, Pi, Pi_w, Pi_tilde, U, S, V, gmodel, size_model
            gc.collect()
            jax.clear_caches()
            logger.debug(f"{quarter} | DONE ({mc_runs} draws)")

if __name__ == "__main__":
    parser = ArgumentParser(
        prog="Graphon Sampling Pipeline",
        description="This pipeline samples networks for each quarter."
    )

    parser.add_argument(
        "-m", "--model",
        required=True,
        choices=["FM", "ME"],
        help="Network reconstruction model to use (FM or ME)."
    )

    parser.add_argument(
        "-n", "--mc-runs",
        type=int,
        default=200,
        help="Number of Monte Carlo draws per quarter (default=200)."
    )

    args = parser.parse_args()
    main(MODEL_NAME=args.model, mc_runs=args.mc_runs)