"""Deterministic ODE utilities for parameterized inner-Q gradients.

The key object is the gradient of the terminal value pulled back through a
deterministic flow.  For the Euler step

    x[k + 1] = x[k] + h * v(s, x[k], t[k]),

the exact discrete adjoint is

    g[k] = (d x[k + 1] / d x[k])**T g[k + 1].

This module deliberately does not implement policy guidance or a time
schedule.  It is the small, independently testable core used to establish
that a parameterized gradient field can reproduce the full ODE adjoint.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp

from utils.networks import FourierFeatures, MLP


Array = jax.Array
VectorField = Callable[[Array, Array, Array], Array]
ScalarCritic = Callable[[Array, Array], Array]


class InnerGradientField(nn.Module):
    """Parameterization of ``grad_x Q_inner(s, x, t)``."""

    action_dim: int
    hidden_dims: Sequence[int] = (256, 256, 256)
    time_embed_dim: int = 32
    layer_norm: bool = True

    @nn.compact
    def __call__(self, observations: Array, actions: Array, times: Array) -> Array:
        time_features = FourierFeatures(self.time_embed_dim)(times)
        inputs = jnp.concatenate((observations, actions, time_features), axis=-1)
        return MLP(
            (*self.hidden_dims, self.action_dim),
            activate_final=False,
            layer_norm=self.layer_norm,
        )(inputs)


def euler_ode_rollout(
    vector_field: VectorField,
    observations: Array,
    initial_actions: Array,
    flow_steps: int,
) -> Array:
    """Roll out a deterministic Euler flow and return ``[x_0, ..., x_K]``."""

    h = 1.0 / flow_steps
    x = initial_actions
    states = [x]
    for step in range(flow_steps):
        t = jnp.full(x.shape[:-1] + (1,), step / flow_steps, dtype=x.dtype)
        x = x + h * vector_field(observations, x, t)
        states.append(x)
    return jnp.stack(states, axis=0)


def full_discrete_ode_adjoint(
    vector_field: VectorField,
    observations: Array,
    states: Array,
    terminal_gradient: Array,
) -> Array:
    """Pull a terminal gradient through every Euler step with exact VJPs.

    Args:
        vector_field: Callable ``v(s, x, t)`` used by the forward ODE.
        observations: Batch of observations with shape ``(B, S)``.
        states: Euler states with shape ``(K + 1, B, A)``.
        terminal_gradient: ``dQ/dx_K`` with shape ``(B, A)``.

    Returns:
        Adjoint states ``[g_0, ..., g_K]`` with shape ``(K + 1, B, A)``.
        The VJP is taken through the *same* Euler map used in the rollout,
        avoiding the mismatched forward/backward dynamics of an SDE adjoint.
    """

    flow_steps = states.shape[0] - 1
    h = 1.0 / flow_steps
    gradient = terminal_gradient
    reverse_gradients = [gradient]

    for step in reversed(range(flow_steps)):
        x = states[step]
        t = jnp.full(x.shape[:-1] + (1,), step / flow_steps, dtype=x.dtype)

        def euler_step(x_in: Array) -> Array:
            return x_in + h * vector_field(observations, x_in, t)

        _, pullback = jax.vjp(euler_step, x)
        gradient = pullback(gradient)[0]
        reverse_gradients.append(gradient)

    return jnp.stack(list(reversed(reverse_gradients)), axis=0)


def critic_action_gradient(
    critic: ScalarCritic,
    observations: Array,
    actions: Array,
) -> Array:
    """Return the batch action-gradient of a scalar-per-sample critic."""

    return jax.grad(lambda a: jnp.sum(critic(observations, a)))(actions)


def low_pass_terminal_gradient(
    critic: ScalarCritic,
    observations: Array,
    actions: Array,
    perturbations: Array,
) -> Array:
    """Apply the method's only low-pass filter at the terminal critic.

    ``perturbations`` has shape ``(N, B, A)`` and is expected to include its
    desired scale.  Antithetic ``+eps`` and ``-eps`` evaluations suppress
    high-frequency critic-gradient artifacts without any additional temporal
    filtering.
    """

    def one_pair(eps: Array) -> Array:
        positive = critic_action_gradient(critic, observations, actions + eps)
        negative = critic_action_gradient(critic, observations, actions - eps)
        return 0.5 * (positive + negative)

    return jnp.mean(jax.vmap(one_pair)(perturbations), axis=0)


def gradient_metrics(prediction: Array, target: Array, eps: float = 1e-8) -> dict[str, Array]:
    """Cosine, relative RMSE and relative L2 error for gradient fields."""

    pred_flat = prediction.reshape((-1, prediction.shape[-1]))
    target_flat = target.reshape((-1, target.shape[-1]))
    pred_norm = jnp.linalg.norm(pred_flat, axis=-1)
    target_norm = jnp.linalg.norm(target_flat, axis=-1)
    cosine = jnp.sum(pred_flat * target_flat, axis=-1) / (
        pred_norm * target_norm + eps
    )
    error = pred_flat - target_flat
    relative_rmse = jnp.sqrt(jnp.mean(jnp.square(error))) / (
        jnp.sqrt(jnp.mean(jnp.square(target_flat))) + eps
    )
    relative_l2 = jnp.mean(jnp.linalg.norm(error, axis=-1) / (target_norm + eps))
    return {
        "cosine": jnp.mean(cosine),
        "relative_rmse": relative_rmse,
        "relative_l2": relative_l2,
    }
