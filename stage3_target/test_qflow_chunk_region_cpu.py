#!/usr/bin/env python3
"""CPU smoke test for action-chunked QFlow temporal target weighting."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp
import numpy as np

from agents.qflow_chunk_region import QFlowChunkRegionAgent, get_config, time_weight


def tiny_config(schedule):
    config = get_config()
    config.actor_hidden_dims = (16, 16)
    config.value_hidden_dims = (16, 16)
    config.num_qs = 2
    config.flow_steps = 4
    config.horizon_length = 3
    config.best_of_n = 1
    config.target_schedule = schedule
    return config


def main():
    grid = jnp.asarray([[0.0], [0.25], [0.5], [0.75], [0.99]])
    np.testing.assert_array_equal(time_weight(grid, "noq"), 0.0)
    np.testing.assert_array_equal(time_weight(grid, "all"), 1.0)
    np.testing.assert_array_equal(
        time_weight(grid, "early50").ravel(), [1, 1, 0, 0, 0]
    )
    np.testing.assert_array_equal(
        time_weight(grid, "late50").ravel(), [0, 0, 1, 1, 1]
    )

    batch_size, horizon, obs_dim, action_dim = 4, 3, 6, 2
    obs = jnp.zeros((batch_size, obs_dim), dtype=jnp.float32)
    action_seq = jnp.zeros((batch_size, horizon, action_dim), dtype=jnp.float32)
    batch = {
        "observations": obs,
        "next_observations": jnp.zeros((batch_size, horizon, obs_dim)),
        "actions": action_seq,
        "rewards": jnp.zeros((batch_size, horizon)),
        "masks": jnp.ones((batch_size, horizon)),
        "valid": jnp.ones((batch_size, horizon)),
    }
    rng = jax.random.PRNGKey(7)
    for schedule in ("all", "noq", "early50", "late50"):
        agent = QFlowChunkRegionAgent.create(
            0, obs[0], action_seq[0, 0], tiny_config(schedule)
        )
        loss, info = agent.actor_loss(batch, agent.network.params, rng)
        assert np.isfinite(float(loss)), schedule
        assert np.isfinite(float(info["steer_norm"])), schedule
        sampled = agent.sample_actions(obs[0], rng)
        assert sampled.shape == (horizon * action_dim,)
        if schedule == "all":
            _, update_info = agent.update(batch)
            assert np.isfinite(float(update_info["critic/outer_loss"]))
        print(schedule, "loss", float(loss), "sample_shape", sampled.shape)
    print("QFlow action-chunk region CPU smoke passed")


if __name__ == "__main__":
    main()
