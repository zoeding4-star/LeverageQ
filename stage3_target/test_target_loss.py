#!/usr/bin/env python3
"""CPU algebra checks for target-weighted adjoint matching."""

import jax
import jax.numpy as jnp
import numpy as np

from agents.qam_target_region import desired_target_edit, target_residual
from stage3_target.target_weights import SCREEN_MASKS, target_weights


def main():
    key = jax.random.PRNGKey(7)
    vf, vs, p = jax.random.normal(key, (3, 10, 4, 6))
    sigma = jnp.linspace(0.4, 2.0, 10).reshape(10, 1, 1)
    ones = jnp.ones((10, 1, 1))

    old = (vf - vs) * 2.0 / sigma + sigma * p
    new = target_residual(vf, vs, sigma, p, ones)
    np.testing.assert_allclose(new, old, rtol=1e-6, atol=1e-6)

    old_grad = jax.grad(lambda z: jnp.square((z - vs) * 2 / sigma + sigma * p).sum())(vf)
    new_grad = jax.grad(lambda z: jnp.square(target_residual(z, vs, sigma, p, ones)).sum())(vf)
    np.testing.assert_allclose(new_grad, old_grad, rtol=1e-6, atol=1e-6)

    zeros = jnp.zeros((10, 1, 1))
    zero_resid = target_residual(vf, vs, sigma, p, zeros)
    assert float(jnp.square(zero_resid).sum()) > 0
    zero_grad = jax.grad(
        lambda z: jnp.square(target_residual(z, vs, sigma, p, zeros)).sum()
    )(vf)
    assert float(jnp.linalg.norm(zero_grad)) > 0

    for scale in (0.0, 0.5, 1.0):
        w = jnp.full((10, 1, 1), scale)
        optimum = vs + desired_target_edit(sigma, p, w)
        resid = target_residual(optimum, vs, sigma, p, w)
        np.testing.assert_allclose(resid, 0.0, rtol=1e-5, atol=1e-5)

    for name in SCREEN_MASKS:
        w = target_weights(name)
        assert w.shape == (10,)
        assert np.all((w == 0.0) | (w == 1.0))
    np.testing.assert_array_equal(target_weights("early50") + target_weights("late50"), 1.0)
    np.testing.assert_array_equal(target_weights("late25") + target_weights("remove_late25"), 1.0)
    print("target-weight algebra checks passed")


if __name__ == "__main__":
    main()
