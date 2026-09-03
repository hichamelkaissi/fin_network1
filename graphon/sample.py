from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array, random

@jax.jit
def sample_berounlli(key: Array, Weights: Array, kappa: int = 1) -> Array:
    bin_A = random.bernoulli(key, kappa * Weights)
    return bin_A

@jax.jit
def drichlet_rows_gamma(key: Array, W: Array, kappa: float = 200.0, eps: float = 1e-12) -> Array:
    alpha = jnp.maximum(kappa * W, eps)
    G = random.gamma(key, alpha, shape=alpha.shape)
    return G / jnp.maximum(G.sum(axis=1, keepdims=True), eps)