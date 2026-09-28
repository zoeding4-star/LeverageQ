"""Action-chunked QFlow with flow-time weights inside the velocity target.

This module adapts the advanced QFlow implementation in the sibling qflow
checkout to the QAM training stack.  The intervention is

    v_target(t) = u_t + w(t) / lambda * grad_x V(s, x_t, t),

so w=0 remains directly supervised by the behavior-flow target u_t.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import flax.linen as nn
import jax
import jax.numpy as jnp

import utils.networks as qam_networks
from utils.networks import MLP, ensemblize


class _TimeConditionedCritic(nn.Module):
    """QAM-network-compatible implementation of V(s, x_t, t)."""

    hidden_dims: tuple
    layer_norm: bool = True
    num_ensembles: int = 2
    encoder: nn.Module | None = None

    def setup(self):
        mlp_class = MLP
        if self.num_ensembles > 1:
            mlp_class = ensemblize(mlp_class, self.num_ensembles)
        self.value_net = mlp_class(
            (*self.hidden_dims, 1), activate_final=False, layer_norm=self.layer_norm
        )

    def __call__(self, observations, actions, times):
        if self.encoder is not None:
            observations = self.encoder(observations)
        inputs = jnp.concatenate([observations, actions, times], axis=-1)
        return self.value_net(inputs).squeeze(-1)


# qflow_qamsetting imports this symbol from utils.networks.  Install only the
# missing compatibility symbol; existing QAM network classes stay untouched.
if not hasattr(qam_networks, "TimeConditionedCritic"):
    qam_networks.TimeConditionedCritic = _TimeConditionedCritic

_BASE_PATH = Path(__file__).resolve().parents[2] / "qflow" / "agents" / "qflow_qamsetting.py"
if not _BASE_PATH.exists():
    raise ImportError(f"advanced QFlow source not found: {_BASE_PATH}")
_SPEC = importlib.util.spec_from_file_location("_advanced_qflow_qamsetting", _BASE_PATH)
_BASE_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_BASE_MODULE)
_BaseQFlowAgent = _BASE_MODULE.QFlowAgent


SCHEDULES = ("all", "noq", "early50", "late50", "late25", "remove_late25")


def time_weight(t, schedule: str):
    """Continuous-time Q-guidance multiplier with QAM-compatible boundaries."""
    name = str(schedule).strip().lower().replace("-", "_")
    name = {"no_q": "noq", "early": "early50", "late": "late50"}.get(name, name)
    if name == "all":
        return jnp.ones_like(t)
    if name == "noq":
        return jnp.zeros_like(t)
    if name == "early50":
        return (t < 0.5).astype(t.dtype)
    if name == "late50":
        return (t >= 0.5).astype(t.dtype)
    if name == "late25":
        return (t >= 0.7).astype(t.dtype)
    if name == "remove_late25":
        return (t < 0.7).astype(t.dtype)
    raise ValueError(f"unknown QFlow schedule {schedule!r}; expected one of {SCHEDULES}")


class QFlowChunkRegionAgent(_BaseQFlowAgent):
    """Advanced QFlow with target-level temporal ablations."""

    def actor_loss(self, batch, grad_params, rng):
        batch_actions = self._flatten_actions(batch["actions"])
        batch_size, action_dim = batch_actions.shape
        _, x_rng, t_rng = jax.random.split(rng, 3)

        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        x_1 = batch_actions
        t = jax.random.uniform(t_rng, (batch_size, 1), minval=0.0, maxval=1.0)
        x_t = (1.0 - t) * x_0 + t * x_1
        u_t = x_1 - x_0
        observations = batch["observations"]

        pred_vel = self.network.select("actor_bc_flow")(
            observations, x_t, t, is_encoded=True, params=grad_params
        )
        t_features = self._get_time_features(t)

        def value_sum(actions):
            values = self.network.select("tc_critic")(
                observations, actions, t_features
            )
            return values.mean(axis=0).sum()

        grad_v = jax.lax.stop_gradient(jax.grad(value_sum)(x_t))
        weight = time_weight(t, self.config["target_schedule"])
        beta = 1.0 / (self.config["guidance_lambda"] + 1e-8)
        steer = weight * beta * grad_v
        v_target = u_t + steer
        error = jnp.mean((pred_vel - jax.lax.stop_gradient(v_target)) ** 2, axis=-1)
        valid = batch["valid"][..., -1] if "valid" in batch else jnp.ones_like(error)
        total_loss = jnp.mean(error * valid)

        grad_norm = jnp.linalg.norm(grad_v, axis=-1)
        steer_norm = jnp.linalg.norm(steer, axis=-1)
        base_norm = jnp.linalg.norm(u_t, axis=-1)
        info = {
            "total_loss": total_loss,
            "target_weight_mean": weight.mean(),
            "critic_grad_norm": grad_norm.mean(),
            "steer_norm": steer_norm.mean(),
            "steer_over_base": (steer_norm / (base_norm + 1e-8)).mean(),
            "anchor_error": (
                jnp.linalg.norm(pred_vel - u_t, axis=-1) * (1.0 - weight[..., 0])
            ).mean(),
        }
        for idx in range(4):
            lo = idx / 4.0
            hi = (idx + 1) / 4.0
            in_bin = ((t[..., 0] >= lo) & (t[..., 0] < hi)).astype(error.dtype)
            denom = in_bin.sum() + 1e-8
            info[f"loss_t{idx}"] = (error * in_bin).sum() / denom
            info[f"steer_t{idx}"] = (steer_norm * in_bin).sum() / denom
        return total_loss, info

    @staticmethod
    def _update(agent, batch):
        """One optimizer step (the upstream prototype accidentally did two)."""
        new_rng, rng = jax.random.split(agent.rng)

        def loss_fn(grad_params):
            return agent.total_loss(batch, grad_params, rng=rng)

        new_network, info = agent.network.apply_loss_fn(loss_fn=loss_fn)
        agent.target_update(new_network, "critic")
        agent.target_update(new_network, "tc_critic")
        return agent.replace(network=new_network, rng=new_rng), info

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        schedule = str(config.get("target_schedule", "all")).lower().replace("-", "_")
        # Validate before JIT tracing and record the intended action-chunk setup.
        time_weight(jnp.asarray([[0.0]]), schedule)
        config["target_schedule"] = schedule
        if not config["action_chunking"]:
            raise ValueError("QFlowChunkRegionAgent requires action_chunking=True")
        print(
            f"[qflow-chunk-region] schedule={schedule} "
            f"horizon={config['horizon_length']} flow_steps={config['flow_steps']}",
            flush=True,
        )
        return super().create(seed, ex_observations, ex_actions, config)


def get_config():
    config = _BASE_MODULE.get_config()
    config.agent_name = "qflow_chunk_region"
    config.action_chunking = True
    config.target_schedule = "all"
    return config
