import pandas as pd
import jax.numpy as jnp
from jax import Array, lax
import os
import re
from pathlib import Path
import logging
from logging.handlers import RotatingFileHandler
import sys

QUARTER_RE = re.compile(r"^(\d{4}Q[1-4])\.(npy|csv)$")

def index_by_quarter(root: str, ext: str) -> dict[str, Path]:
    """
    Walk `root` and return { '2020Q2': Path(...) } for files ending in `ext`.
    """
    ext = ext.lstrip('.')
    out: dict[str, Path] = {}
    rootp = Path(root)

    for dirpath, _, filenames in os.walk(rootp):
        for name in filenames:
            m = QUARTER_RE.match(name)
            if not m:
                continue
            quarter, found_ext = m.group(1), m.group(2)
            if found_ext != ext:
                continue
            # If duplicates exist, last one wins; adjust if you want to error instead.
            out[quarter] = Path(dirpath) / name
    return out

def spectral_radius(
        L: Array,
        beta: float = 1.0,
        n_iter: int = 50,
        eps: float = 1e-12
) -> Array:
    pbar = jnp.sum(L, axis=1)
    inv_pbar = jnp.where(pbar > 0.0, 1.0 / pbar, 0.0)

    def M_mul(x):
        return beta * (L.T @ (x * inv_pbar))

    v = jnp.ones((L.shape[0],), dtype=L.dtype)
    v = v / jnp.maximum(jnp.linalg.norm(v), eps)

    def body(i, v):
        w = M_mul(v)
        nrm = jnp.maximum(jnp.linalg.norm(w), eps)
        return w / nrm

    v = lax.fori_loop(0, n_iter, body, v)
    w = M_mul(v)
    rho = (v @ w) / jnp.maximum(v @ v, eps)
    return rho