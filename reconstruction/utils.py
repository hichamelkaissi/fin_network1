import pandas as pd
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax, Array, random
from typing import Tuple, Any
from tqdm import tqdm
import subprocess

def load_reconstruction_data(path: str) -> pd.DataFrame:

  df = pd.read_csv(path)
  df = df[(df['IA'] > 0) | (df['IL'] > 0)].reset_index(drop=True)


  return df

def get_ia_il(df: pd.DataFrame) -> Tuple[Array, Array]:
  IA = jnp.asarray(df['IA'].to_numpy())
  IL = jnp.asarray(df['IL'].to_numpy())

  return IA, IL

@jax.jit
def get_prior_matrix(IA: Array, IL: Array) -> Tuple[Array, Array]:
    sum_IA, sum_IL = jnp.sum(IA), jnp.sum(IL)
    k = jnp.sqrt(sum_IA / sum_IL)
    IL_tilde = IL * k
    IA_tilde = IA / k
    coef = IL / jnp.sum(IL)
    P = jnp.outer(IA_tilde, coef)
    P = P.at[jnp.arange(P.shape[0]), jnp.arange(P.shape[0])].set(0)

    return P, IA_tilde, IL_tilde

def scale_row(P: Array, IA_hat: Array, eps: float = 1e-12) -> Array:
  R = jnp.sum(P, axis=1)

  r = jnp.where(R > 0, IA_hat / jnp.maximum(R, eps), 0)

  return r[:, None] * P

def scale_col(P: Array, IL: Array, eps: float = 1e-12) -> Array:
  col_sum = jnp.sum(P, axis=0)

  c = jnp.where(col_sum > 0, IL / jnp.maximum(col_sum, eps), 0)

  return P * c[None, :]

def row_loss(P: Array, IA_hat: Array) -> Array:
  R = jnp.sum(P, axis=1)
  loss = jnp.abs(R - IA_hat)
  return jnp.max(loss)


def col_loss(P: Array, IL: Array) -> Array:
  C = jnp.sum(P, axis=0)
  loss = jnp.abs(C - IL)
  return jnp.max(loss)

def has_no_bad_cols(A: Array, IL: Array) -> Array:
    out_degrees = jnp.sum(A, axis=0)
    bad_cols = (IL > 0) & (out_degrees == 0)
    return jnp.logical_not(jnp.any(bad_cols))
  
def has_no_bad_row(A: Array, IA: Array) -> Array:
    in_degrees = jnp.sum(A, axis=1)
    bad_rows = (IA > 0) & (in_degrees == 0)
    return jnp.logical_not(jnp.any(bad_rows))
  
def logit(p: Array, eps: float = 1e-12) -> Array:
    p = jnp.clip(p, eps, 1.0 - eps)
    return jnp.log(p) - jnp.log1p(-p)  # numerically nicer than log(p/(1-p))

def sigma(x: Array) -> Array:
    # stable sigmoid
    return jax.nn.sigmoid(x)
  
def solve_z(
    phi: Array,
    psi: Array,
    d_target: float,
    z_lo: float = 1e-16,
    z_hi: float = 1e6,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> Array:
    for _ in range(max_iter):
        z_mid = 0.5 * (z_lo + z_hi)
        val = f(z_mid, phi, psi, d_target)
        z_lo = jnp.where(val < 0, z_mid, z_lo)
        z_hi = jnp.where(val > 0, z_mid, z_hi)

        if float(jnp.abs(z_lo - z_hi)) < tol:

            break
    return 0.5 * (z_lo + z_hi)
  
jax.jit
def f(
    z: Array,
    phi: Array,
    psi: Array,
    d_target: Array,
    eps: float = 1e-12,
) -> Array:
    X = z * jnp.outer(phi, psi)
    p = X / (1.0 + X)
    n = p.shape[0]
    p = p.at[jnp.arange(n), jnp.arange(n)].set(0.0)

    mean_p = jnp.sum(p) / jnp.maximum(n * (n - 1), 1)
    return mean_p - d_target

def mc_success_rate(
    p: Array,
    IA_tilde: Array,
    IL: Array,
    key: Array,
    R: int = 200,
) -> Tuple[float, Array]:
    N = p.shape[0]
    good = 0
    for r in tqdm(range(R)):
        key, subkey = random.split(key)
        A = random.bernoulli(subkey, p=p, shape=(N, N))
        A = A.at[jnp.arange(N), jnp.arange(N)].set(False)

        ok = jnp.logical_and(has_no_bad_row(A, IA_tilde),
                             has_no_bad_cols(A, IL))
        good += int(ok)
    return good / R, key
  
@jax.jit
def RAS_feasibility_check(
    P: Array,
    IA_tilde: Array,
    IL: Array,
    eps: float = 0.0,
) -> tuple[Array, Array, Array]:
    row_sum = jnp.sum(P, axis=1)
    col_sum = jnp.sum(P, axis=0)

    ok_rows = jnp.all((IA_tilde <= 0) | (row_sum > eps))
    ok_cols = jnp.all((IL <= 0) | (col_sum > eps))

    ok = ok_rows & ok_cols
    return ok, ok_rows, ok_cols
  
def extract_arrays(
    df: pd.DataFrame,
    dtype: Any = jnp.float64,
) -> Tuple[Array, Array, Array, Array, Array, Array, Array]:

    required = [
        "IA", "IL", "TotalAssets", "TotalLiabilities",
        "CashAndBalances", "Repos", "FedFundsPurchased"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}")

    def col(name: str) -> Array:
        x = pd.to_numeric(df[name], errors="coerce").to_numpy()
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        return jnp.asarray(x, dtype=dtype)

    IA   = col("IA")
    IL   = col("IL")
    TA   = col("TotalAssets")
    TL   = col("TotalLiabilities")
    cash = col("CashAndBalances")
    repo = col("Repos")
    FFP  = col("FedFundsPurchased")

    n = IA.shape[0]
    for nm, arr in [("IL", IL), ("TA", TA), ("TL", TL),
                    ("cash", cash), ("repo", repo), ("FFP", FFP)]:
        if arr.shape != (n,):
            raise ValueError(f"{nm} has shape {arr.shape}, expected {(n,)}")

    return IA, IL, TA, TL, cash, repo, FFP

def gpu_memory():
  result = subprocess.run(
    ['nvidia-smi', '--query-gpu=memory.used',
    '--format=csv,noheader,nounits'],
    capture_output=True, text=True
  )
  return int(result.stdout.strip())


