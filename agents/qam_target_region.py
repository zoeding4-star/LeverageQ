"""QAM with flow-time weights inside the adjoint-matching target.

Unlike the Stage-3 loss weighting, zero target weight remains supervised:
it explicitly anchors actor_fast to actor_slow at that flow time.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from agents.qam import QAMAgent, get_config as qam_get_config
from agents.qam_diag import QAMDiagAgent
from stage3_target.target_weights import describe_target_mask


def target_residual(v_fast, v_slow, sigmas, adjs, weights):
    """AM residual whose pointwise optimum scales the Q edit by weights."""
    return (v_fast - v_slow) * 2.0 / sigmas + weights * sigmas * adjs


def desired_target_edit(sigmas, adjs, weights):
    """The exact velocity edit that zeros target_residual."""
    return -weights * jnp.square(sigmas) * adjs / 2.0


class QAMTargetRegionAgent(QAMDiagAgent):
    """QAM target intervention with behavior anchoring at zero-weight times."""

    # Bypass QAMRegionAgent's all-zero -> slow evaluation shortcut. Under the
    # target loss, actor_fast is trained to equal actor_slow when every w is 0.
    sample_actions = QAMAgent.sample_actions

    def actor_loss(self, batch, grad_params, rng):
        if self.config["fql_alpha"] != 0.0 or self.config["edit_scale"] != 0.0:
            raise ValueError("qam_target_region screen requires fql_alpha=edit_scale=0")
        if self.config["residual"]:
            raise ValueError("qam_target_region screen is defined for residual=False")

        if self.config["action_chunking"]:
            batch_actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))
        else:
            batch_actions = batch["actions"][..., 0, :]

        batch_size, action_dim = batch_actions.shape
        # Keep the original QAM key split so target=All preserves its random
        # streams as well as its algebra. edit_rng is intentionally unused.
        rng, x_rng, t_rng, adj_rng, _edit_rng = jax.random.split(rng, 5)

        # Behavior flow matching is unchanged.
        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        x_1 = batch_actions
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1.0 - t) * x_0 + t * x_1
        vel = x_1 - x_0
        pred = self.network.select("actor_slow")(
            batch["observations"], x_t, t, params=grad_params
        )
        flow_loss = jnp.mean(
            jnp.square(pred - vel).mean(axis=-1) * batch["valid"][..., -1]
        )

        # Full QAM path and adjoint construction are unchanged.
        xs, adjs, ts, pre_adj_info = self.adj_matching(batch["observations"], adj_rng)
        h = 1.0 / self.config["flow_steps"]
        sigmas = jnp.sqrt(2.0 * (1.0 - ts + h) / (ts + h))
        observations = jnp.repeat(
            batch["observations"][None], self.config["flow_steps"], axis=0
        )
        v_fast = self.network.select("actor_fast")(observations, xs, ts, params=grad_params)
        actor_slow = self.network.select(
            "target_actor_slow" if self.config["target_actor"] else "actor_slow"
        )
        v_slow = actor_slow(observations, xs, ts)

        w = jnp.asarray(self.config["target_weights"], dtype=v_fast.dtype)
        w = w.reshape((-1, 1, 1))
        resid = target_residual(v_fast, v_slow, sigmas, adjs, w)
        ell = jnp.sum(jnp.square(resid), axis=-1)
        target_loss = jnp.mean(jnp.sum(ell, axis=0))

        desired = desired_target_edit(sigmas, adjs, w)
        actual = v_fast - v_slow
        desired_norm = jnp.linalg.norm(desired, axis=-1)
        actual_norm = jnp.linalg.norm(actual, axis=-1)
        fit_norm = jnp.linalg.norm(actual - desired, axis=-1)
        cosine = jnp.sum(actual * desired, axis=-1) / (
            actual_norm * desired_norm + 1e-8
        )

        active = (w[..., 0] > 0).astype(actual_norm.dtype)
        active_denom = jnp.sum(active) * actual_norm.shape[-1] + 1e-8
        info = {
            "flow_loss": flow_loss,
            "fast_loss": target_loss,
            "adj_loss": target_loss,
            "target_weight_sum": jnp.sum(w),
            "target_anchor_error": jnp.mean(actual_norm * (1.0 - w[..., 0])),
            "target_fit_error": jnp.mean(fit_norm),
            "target_edit_ratio": jnp.sum(
                (actual_norm / (desired_norm + 1e-8)) * active
            ) / active_denom,
            "target_edit_cosine": jnp.sum(cosine * active) / active_denom,
            **pre_adj_info,
        }
        return flow_loss + target_loss, info

    @jax.jit
    def diagnostic_snapshot(self, batch, rng):
        snap = QAMDiagAgent.diagnostic_snapshot(self, batch, rng)
        x = snap["x"]
        p = snap["p"]
        t_grid = snap["t"]
        obs = jnp.repeat(batch["observations"][None], x.shape[0], axis=0)
        ts = jnp.broadcast_to(t_grid[:, None, None], x.shape[:-1] + (1,))
        actor_slow = self.network.select(
            "target_actor_slow" if self.config["target_actor"] else "actor_slow"
        )
        v_fast = self.network.select("actor_fast")(obs, x, ts)
        v_slow = actor_slow(obs, x, ts)
        h = 1.0 / self.config["flow_steps"]
        sigma = jnp.sqrt(2.0 * (1.0 - ts + h) / (ts + h))
        w_vec = jnp.asarray(self.config["target_weights"], dtype=x.dtype)
        w = w_vec.reshape((-1, 1, 1))
        desired = desired_target_edit(sigma, p, w)
        actual = v_fast - v_slow
        actual_norm = jnp.linalg.norm(actual, axis=-1)
        desired_norm = jnp.linalg.norm(desired, axis=-1)
        fit_norm = jnp.linalg.norm(actual - desired, axis=-1)
        cosine = jnp.sum(actual * desired, axis=-1) / (
            actual_norm * desired_norm + 1e-8
        )
        resid = target_residual(v_fast, v_slow, sigma, p, w)
        ell = jnp.sum(jnp.square(resid), axis=-1)
        return {
            **snap,
            "target_w": w_vec,
            "target_ell": ell,
            "target_actual_edit_norm": actual_norm,
            "target_desired_edit_norm": desired_norm,
            "target_fit_error": fit_norm,
            "target_edit_cosine": cosine,
            "target_anchor_error": actual_norm * (1.0 - w[..., 0]),
        }

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        spec = describe_target_mask(
            str(config.get("target_mask_name", "all")),
            flow_steps=int(config["flow_steps"]),
        )
        config["target_mask_name"] = spec["name"]
        config["target_weights"] = spec["weights"]
        config["target_n_active"] = spec["n_active"]
        config["target_weight_sum"] = spec["weight_sum"]
        print("[target-weight]", spec["summary"], flush=True)
        try:
            import wandb

            if wandb.run is not None:
                wandb.config.update(
                    {
                        "target_weight_location": "inside_residual_target",
                        "target_mask_name": spec["name"],
                        "target_weights": list(spec["weights"]),
                        "target_active_steps": spec["active_steps"],
                    },
                    allow_val_change=True,
                )
        except Exception as exc:
            print("[target-weight] wandb.config.update skipped:", exc, flush=True)
        return super().create(seed, ex_observations, ex_actions, config)


def get_config():
    config = qam_get_config()
    config.agent_name = "qam_target_region"
    config.target_mask_name = "all"
    config.target_weights = (1.0,) * 10
    config.target_n_active = 10
    config.target_weight_sum = 10.0
    return config
