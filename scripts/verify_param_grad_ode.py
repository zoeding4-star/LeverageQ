#!/usr/bin/env python3
"""Verify a parameterized gradient field against a full ODE adjoint.

This is a controlled CPU experiment, not a policy-return benchmark.  It tests
the two prerequisites requested before policy scheduling is introduced:

1. G_eta(s, x, t) reproduces a full reverse-VJP ODE adjoint on held-out paths.
2. One terminal action-space low-pass reduces sensitivity to a deliberately
   high-frequency critic-gradient perturbation.

Example:
    JAX_PLATFORMS=cpu python scripts/verify_param_grad_ode.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import optax

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.param_grad_ode import (  # noqa: E402
    InnerGradientField,
    critic_action_gradient,
    euler_ode_rollout,
    full_discrete_ode_adjoint,
    gradient_metrics,
    low_pass_terminal_gradient,
)


OBS_DIM = 3
ACTION_DIM = 4
FLOW_STEPS = 10

_FLOW_X = jnp.asarray(
    [
        [0.12, -0.08, 0.00, 0.03],
        [0.05, -0.10, 0.06, 0.00],
        [0.00, 0.04, 0.09, -0.07],
        [-0.03, 0.00, 0.08, -0.11],
    ],
    dtype=jnp.float32,
)
_FLOW_S = jnp.asarray(
    [
        [0.20, -0.05, 0.10],
        [-0.08, 0.15, 0.04],
        [0.05, 0.06, -0.12],
        [0.10, -0.10, 0.08],
    ],
    dtype=jnp.float32,
)
_GOAL = jnp.asarray(
    [
        [0.50, -0.20, 0.10],
        [-0.10, 0.45, 0.15],
        [0.20, 0.10, -0.40],
        [-0.25, 0.15, 0.35],
    ],
    dtype=jnp.float32,
)


def reference_flow(observations, actions, times):
    """Stable nonlinear reference flow used by the controlled experiment."""

    drive = observations @ _FLOW_S.T
    coupled = actions @ _FLOW_X.T
    periodic = 0.06 * jnp.sin(2.0 * jnp.pi * times) * actions
    return 0.55 * jnp.tanh(coupled + drive) + periodic


def clean_critic(observations, actions):
    goal = jnp.tanh(observations @ _GOAL.T)
    error = actions - goal
    smooth_shape = 0.06 * jnp.cos(1.7 * actions[..., 1])
    return -0.5 * jnp.sum(jnp.square(error), axis=-1) + smooth_shape


def noisy_critic(observations, actions):
    phase = actions[..., 0] + 0.35 * actions[..., 2]
    return clean_critic(observations, actions) + 0.075 * jnp.sin(18.0 * phase)


def make_data(key, sample_count, low_pass_sigma, low_pass_samples):
    obs_key, action_key, noise_key = jax.random.split(key, 3)
    observations = jax.random.uniform(
        obs_key, (sample_count, OBS_DIM), minval=-1.0, maxval=1.0
    )
    initial_actions = 0.8 * jax.random.normal(action_key, (sample_count, ACTION_DIM))
    states = euler_ode_rollout(
        reference_flow, observations, initial_actions, FLOW_STEPS
    )
    terminal_actions = states[-1]

    clean_terminal = critic_action_gradient(clean_critic, observations, terminal_actions)
    noisy_terminal = critic_action_gradient(noisy_critic, observations, terminal_actions)
    perturbations = low_pass_sigma * jax.random.normal(
        noise_key, (low_pass_samples, sample_count, ACTION_DIM)
    )
    filtered_terminal = low_pass_terminal_gradient(
        noisy_critic, observations, terminal_actions, perturbations
    )

    clean_adjoint = full_discrete_ode_adjoint(
        reference_flow, observations, states, clean_terminal
    )
    noisy_adjoint = full_discrete_ode_adjoint(
        reference_flow, observations, states, noisy_terminal
    )
    filtered_adjoint = full_discrete_ode_adjoint(
        reference_flow, observations, states, filtered_terminal
    )

    time_grid = jnp.arange(FLOW_STEPS + 1, dtype=jnp.float32) / FLOW_STEPS
    times = jnp.broadcast_to(
        time_grid[:, None, None], (FLOW_STEPS + 1, sample_count, 1)
    )
    repeated_observations = jnp.broadcast_to(
        observations[None], (FLOW_STEPS + 1, sample_count, OBS_DIM)
    )
    return {
        "observations": repeated_observations.reshape((-1, OBS_DIM)),
        "actions": states.reshape((-1, ACTION_DIM)),
        "times": times.reshape((-1, 1)),
        "target": filtered_adjoint.reshape((-1, ACTION_DIM)),
        "clean_adjoint": clean_adjoint,
        "noisy_adjoint": noisy_adjoint,
        "filtered_adjoint": filtered_adjoint,
    }


def train_model(key, train_data, steps, batch_size, learning_rate):
    model = InnerGradientField(
        action_dim=ACTION_DIM,
        hidden_dims=(128, 128, 128),
        time_embed_dim=32,
        layer_norm=False,
    )
    init_key, train_key = jax.random.split(key)
    params = model.init(
        init_key,
        train_data["observations"][:1],
        train_data["actions"][:1],
        train_data["times"][:1],
    )["params"]
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)

    @jax.jit
    def update(params, opt_state, observations, actions, times, targets):
        def loss_fn(current_params):
            prediction = model.apply(
                {"params": current_params}, observations, actions, times
            )
            target_scale = jnp.sqrt(jnp.mean(jnp.square(targets))) + 1e-6
            return jnp.mean(jnp.square((prediction - targets) / target_scale))

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    data_size = train_data["target"].shape[0]
    loss = jnp.asarray(jnp.nan)
    for step in range(steps):
        step_key = jax.random.fold_in(train_key, step)
        indices = jax.random.randint(step_key, (batch_size,), 0, data_size)
        params, opt_state, loss = update(
            params,
            opt_state,
            train_data["observations"][indices],
            train_data["actions"][indices],
            train_data["times"][indices],
            train_data["target"][indices],
        )
    return model, params, float(loss)


def as_float_dict(metrics):
    return {key: float(value) for key, value in metrics.items()}


def run(args):
    root_key = jax.random.PRNGKey(args.seed)
    train_key, test_key, model_key = jax.random.split(root_key, 3)
    train_data = make_data(
        train_key, args.train_samples, args.low_pass_sigma, args.low_pass_samples
    )
    test_data = make_data(
        test_key, args.test_samples, args.low_pass_sigma, args.low_pass_samples
    )
    model, params, final_loss = train_model(
        model_key,
        train_data,
        args.steps,
        args.batch_size,
        args.learning_rate,
    )
    prediction = model.apply(
        {"params": params},
        test_data["observations"],
        test_data["actions"],
        test_data["times"],
    ).reshape(test_data["filtered_adjoint"].shape)

    reproduction = as_float_dict(
        gradient_metrics(prediction, test_data["filtered_adjoint"])
    )
    raw_sensitivity = as_float_dict(
        gradient_metrics(test_data["noisy_adjoint"], test_data["clean_adjoint"])
    )
    filtered_sensitivity = as_float_dict(
        gradient_metrics(test_data["filtered_adjoint"], test_data["clean_adjoint"])
    )
    parameterized_sensitivity = as_float_dict(
        gradient_metrics(prediction, test_data["clean_adjoint"])
    )

    raw_error = raw_sensitivity["relative_rmse"]
    filtered_error = filtered_sensitivity["relative_rmse"]
    parameterized_error = parameterized_sensitivity["relative_rmse"]
    result = {
        "config": {
            "seed": args.seed,
            "flow_steps": FLOW_STEPS,
            "train_samples": args.train_samples,
            "test_samples": args.test_samples,
            "train_steps": args.steps,
            "low_pass_sigma": args.low_pass_sigma,
            "low_pass_samples": args.low_pass_samples,
        },
        "final_normalized_train_loss": final_loss,
        "adjoint_reproduction": reproduction,
        "sensitivity_to_high_frequency_terminal_error": {
            "raw_full_adjoint": raw_sensitivity,
            "filtered_full_adjoint": filtered_sensitivity,
            "parameterized_filtered_adjoint": parameterized_sensitivity,
            "filtered_error_over_raw": filtered_error / raw_error,
            "parameterized_error_over_raw": parameterized_error / raw_error,
        },
    }

    if args.assert_thresholds:
        assert reproduction["cosine"] >= 0.97, result
        assert reproduction["relative_rmse"] <= 0.25, result
        assert filtered_error <= 0.55 * raw_error, result
        assert parameterized_error <= 0.75 * raw_error, result

    output = json.dumps(result, indent=2, sort_keys=True)
    print(output)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output + "\n", encoding="utf-8")
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--train-samples", type=int, default=2048)
    parser.add_argument("--test-samples", type=int, default=512)
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--low-pass-sigma", type=float, default=0.12)
    parser.add_argument("--low-pass-samples", type=int, default=16)
    parser.add_argument("--output", type=str, default="")
    parser.add_argument(
        "--no-assert-thresholds",
        dest="assert_thresholds",
        action="store_false",
    )
    parser.set_defaults(assert_thresholds=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
