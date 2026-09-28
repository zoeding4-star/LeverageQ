#!/usr/bin/env python3
"""Small CPU create/loss/snapshot test for QAMTargetRegionAgent."""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np

from agents.qam_target_region import QAMTargetRegionAgent, get_config


def tiny_config(mask):
    config = get_config()
    config.actor_hidden_dims = (16, 16)
    config.value_hidden_dims = (16, 16)
    config.num_qs = 2
    config.flow_steps = 4
    config.horizon_length = 3
    config.action_chunking = True
    config.inv_temp = 1.0
    config.target_mask_name = mask
    return config


def main():
    batch_size, horizon, obs_dim, action_dim = 4, 3, 6, 2
    obs = jnp.zeros((batch_size, obs_dim), dtype=jnp.float32)
    action_seq = jnp.zeros((batch_size, horizon, action_dim), dtype=jnp.float32)
    batch = {
        "observations": obs,
        "actions": action_seq,
        "valid": jnp.ones((batch_size, horizon), dtype=jnp.float32),
    }
    rng = jax.random.PRNGKey(11)

    for mask in ("all", "noq", "early50", "late50"):
        agent = QAMTargetRegionAgent.create(
            0, obs[:1], action_seq[:1, 0], tiny_config(mask)
        )
        loss, info = agent.actor_loss(batch, agent.network.params, rng)
        assert np.isfinite(float(loss)), mask
        assert np.isfinite(float(info["target_fit_error"])), mask
        snap = agent.diagnostic_snapshot(batch, rng)
        assert snap["target_fit_error"].shape == (4, batch_size)
        assert snap["target_w"].shape == (4,)
        sampled = agent.sample_actions(obs[0], rng)
        assert sampled.shape[-1] == horizon * action_dim
        print(mask, "loss", float(loss), "sample_shape", sampled.shape)
    print("target agent CPU smoke passed")


if __name__ == "__main__":
    main()
