#!/usr/bin/env python3
"""CPU tests for the deterministic inner-gradient ODE core."""

from __future__ import annotations

import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.param_grad_ode import (  # noqa: E402
    critic_action_gradient,
    euler_ode_rollout,
    full_discrete_ode_adjoint,
    low_pass_terminal_gradient,
)


def test_discrete_adjoint_matches_linear_system():
    matrix = jnp.asarray([[0.15, -0.20], [0.05, -0.10]], dtype=jnp.float32)
    flow_steps = 5
    h = 1.0 / flow_steps

    def vector_field(obs, x, t):
        del obs, t
        return x @ matrix.T

    observations = jnp.zeros((3, 1), dtype=jnp.float32)
    initial = jnp.asarray([[0.3, -0.7], [1.0, 0.2], [-0.4, 0.5]])
    states = euler_ode_rollout(vector_field, observations, initial, flow_steps)
    terminal = jnp.asarray([[0.5, -0.3], [0.1, 0.8], [-0.2, 0.6]])
    actual = full_discrete_ode_adjoint(vector_field, observations, states, terminal)

    step_jacobian = jnp.eye(2) + h * matrix
    expected = [terminal]
    gradient = terminal
    for _ in reversed(range(flow_steps)):
        gradient = gradient @ step_jacobian
        expected.append(gradient)
    expected = jnp.stack(list(reversed(expected)), axis=0)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_terminal_low_pass_reduces_high_frequency_gradient():
    frequency = 20.0
    amplitude = 0.08

    def clean_critic(obs, actions):
        del obs
        return -0.5 * jnp.sum(jnp.square(actions), axis=-1)

    def noisy_critic(obs, actions):
        return clean_critic(obs, actions) + amplitude * jnp.sin(frequency * actions[..., 0])

    observations = jnp.zeros((256, 1), dtype=jnp.float32)
    actions = jnp.linspace(-0.9, 0.9, 512, dtype=jnp.float32).reshape((256, 2))
    clean = critic_action_gradient(clean_critic, observations, actions)
    raw = critic_action_gradient(noisy_critic, observations, actions)
    noise = 0.12 * jax.random.normal(jax.random.PRNGKey(0), (64, 256, 2))
    filtered = low_pass_terminal_gradient(noisy_critic, observations, actions, noise)

    raw_error = jnp.mean(jnp.linalg.norm(raw - clean, axis=-1))
    filtered_error = jnp.mean(jnp.linalg.norm(filtered - clean, axis=-1))
    assert float(filtered_error) < 0.35 * float(raw_error), (raw_error, filtered_error)


if __name__ == "__main__":
    test_discrete_adjoint_matches_linear_system()
    test_terminal_low_pass_reduces_high_frequency_gradient()
    print("param-grad ODE tests passed")
