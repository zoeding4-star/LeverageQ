"""QAM with flow-time adjoint-matching weights (Stage 3).

Official files are not modified: `agents/qam.py`, `main.py`.
This agent inherits QAMDiagAgent so critic / FM / TD / solver stay identical.
Only the AM loss aggregation changes:

    L_actor = L_FM + sum_k w_k L_AM,k

The full flow and all adjoints are still computed; w_k is applied at the loss.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from agents.qam import get_config as qam_get_config
from agents.qam_diag import QAMDiagAgent
from stage3.am_weights import describe_am_mask


def _no_q_from_config(config) -> bool:
    w = config.get("am_weights", None)
    if w is None:
        return False
    return float(sum(float(x) for x in w)) == 0.0


class QAMRegionAgent(QAMDiagAgent):
    """QAM whose adjoint-matching loss is weighted in normalized flow time."""

    def actor_loss(self, batch, grad_params, rng):
        if self.config["action_chunking"]:
            batch_actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))
        else:
            batch_actions = batch["actions"][..., 0, :]

        batch_size, action_dim = batch_actions.shape
        rng, x_rng, t_rng, adj_rng, edit_rng = jax.random.split(rng, 5)

        ## BC flow-matching loss. Unmasked.
        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        x_1 = batch_actions
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0

        pred = self.network.select("actor_slow")(batch["observations"], x_t, t, params=grad_params)
        flow_loss = jnp.mean(jnp.square(pred - vel).mean(axis=-1) * batch["valid"][..., -1])
        actor_loss = flow_loss

        info = {}
        total_fast_loss = 0
        actor_slow = self.network.select(
            "target_actor_slow" if self.config["target_actor"] else "actor_slow"
        )

        ## Adjoint-matching: full forward/backward, then weight the loss.
        xs, adjs, ts, pre_adj_info = self.adj_matching(batch["observations"], adj_rng)
        h = 1 / self.config["flow_steps"]
        sigmas = jnp.sqrt(2 * (1 - ts + h) / (ts + h))

        observations = jnp.repeat(batch["observations"][None], self.config["flow_steps"], axis=0)
        vf_fine = self.network.select("actor_fast")(observations, xs, ts, params=grad_params)
        vf_base = actor_slow(observations, xs, ts)

        if self.config["residual"]:
            adj_resid = vf_fine * 2 / sigmas + sigmas * adjs
        else:
            adj_resid = (vf_fine - vf_base) * 2 / sigmas + sigmas * adjs

        ell = jnp.sum(jnp.square(adj_resid), axis=-1)  # (T, B)
        w = jnp.asarray(self.config["am_weights"], dtype=ell.dtype)
        w = w.reshape((-1,) + (1,) * (ell.ndim - 1))
        adj_loss = jnp.mean(jnp.sum(w * ell, axis=0))
        adj_loss_unweighted = jnp.mean(jnp.sum(ell, axis=0))

        info["adj_loss"] = adj_loss
        info["adj_loss_unweighted"] = adj_loss_unweighted
        info.update(pre_adj_info)
        total_fast_loss += adj_loss

        # Per-timestep logs (batch-mean).
        L_t = jnp.mean(ell, axis=1)
        w_t = w.reshape(-1)
        wL_t = w_t * L_t
        d_dv = 2.0 * adj_resid * (2.0 / sigmas)
        g_vf = w.reshape((-1, 1, 1)) * d_dv
        g_vf_norm = jnp.mean(jnp.linalg.norm(g_vf, axis=-1), axis=1)
        p_norm_t = jnp.mean(jnp.linalg.norm(adjs, axis=-1), axis=1)
        rel_edit_t = jnp.mean(
            jnp.linalg.norm(vf_fine - vf_base, axis=-1)
            / (jnp.linalg.norm(vf_base, axis=-1) + 1e-8),
            axis=1,
        )
        info["am_weight_sum"] = jnp.sum(w_t)
        info["am_n_active"] = jnp.sum(w_t > 0)
        T = int(self.config["flow_steps"])
        t_grid = jnp.arange(T) / float(T)
        regions = {
            "early": t_grid < 0.5,
            "middle": (t_grid >= 0.3) & (t_grid < 0.7),
            "late": t_grid >= 0.5,
            "active": w_t > 0,
        }
        p_tot = jnp.sum(p_norm_t) + 1e-8
        for rname, mask in regions.items():
            info[f"F_signal_{rname}"] = jnp.sum(p_norm_t * mask.astype(p_norm_t.dtype)) / p_tot
        for k in range(T):
            pfx = f"t{k:02d}"
            info[f"{pfx}/t"] = t_grid[k]
            info[f"{pfx}/w"] = w_t[k]
            info[f"{pfx}/L"] = L_t[k]
            info[f"{pfx}/wL"] = wL_t[k]
            info[f"{pfx}/G_vf"] = g_vf_norm[k]
            info[f"{pfx}/p_norm"] = p_norm_t[k]
            info[f"{pfx}/rel_edit"] = rel_edit_t[k]

        if self.config["fql_alpha"] > 0.0:
            edit_base_rng, edit_rng = jax.random.split(edit_rng, 2)
            fql_noises = jax.random.normal(edit_base_rng, (batch_size, action_dim))
            flow_actions = self.compute_flow_actions(
                batch["observations"],
                fql_noises,
                model="slow,fast" if self.config["residual"] else "fast",
            )
            os_actions = self.network.select("one_step_actor")(
                batch["observations"], fql_noises, params=grad_params
            )
            fql_distill_loss = jnp.mean((flow_actions - os_actions) ** 2)
            os_actions = jnp.clip(os_actions, -1, 1)
            fql_qs = self.network.select("critic")(batch["observations"], actions=os_actions)
            fql_q = jnp.mean(fql_qs, axis=0)
            fql_q_loss = -fql_q.mean()
            info["fql_distill_loss"] = fql_distill_loss
            info["fql_q_loss"] = fql_q_loss
            actor_loss += fql_q_loss + fql_distill_loss * self.config["fql_alpha"]

        if self.config["edit_scale"] > 0.0:
            edit_base_rng, edit_rng = jax.random.split(edit_rng, 2)
            flow_actions = self.compute_flow_actions(
                batch["observations"],
                jax.random.normal(edit_base_rng, (batch_size, action_dim)),
                model="slow,fast" if self.config["residual"] else "fast",
            )
            edit_dist = self.network.select("edit_actor")(
                jnp.concatenate((batch["observations"], flow_actions), axis=-1),
                params=grad_params,
            )
            edit = edit_dist.sample(seed=edit_rng)
            edit_log_probs = edit_dist.log_prob(edit)
            edited_actions = flow_actions + edit * self.config["edit_scale"]
            edited_actions = jnp.clip(edited_actions, -1, 1)
            qs = self.network.select("critic")(batch["observations"], actions=edited_actions)
            q = jnp.mean(qs, axis=0)
            edit_q_loss = -q.mean()
            edit_entropy_loss = (edit_log_probs * self.network.select("edit_alpha")()).mean()
            alpha = self.network.select("edit_alpha")(params=grad_params)
            entropy = -jax.lax.stop_gradient(edit_log_probs).mean()
            edit_alpha_loss = (alpha * (entropy - self.config["edit_target_entropy"])).mean()
            actor_loss += edit_q_loss + edit_entropy_loss + edit_alpha_loss
            info["edit_q_loss"] = edit_q_loss
            info["edit_entropy_loss"] = edit_entropy_loss
            info["edit_alpha_loss"] = edit_alpha_loss
            info["edit_entropy"] = entropy
            info["edit_alpha"] = alpha

        return actor_loss + total_fast_loss, {
            "flow_loss": flow_loss,
            "fast_loss": total_fast_loss,
            **info,
        }

    @jax.jit
    def sample_actions(self, observations, rng):
        """Same solver as QAM. No-Q (all w_k=0) uses the behavior slow flow."""
        rng, edit_rng = jax.random.split(rng)

        action_dim = self.config["action_dim"] * (
            self.config["horizon_length"] if self.config["action_chunking"] else 1
        )
        noises = jax.random.normal(
            rng,
            (
                *observations.shape[: -len(self.config["ob_dims"])],
                self.config["best_of_n"],
                action_dim,
            ),
        )
        observations = jnp.repeat(observations[..., None, :], self.config["best_of_n"], axis=-2)
        no_q = _no_q_from_config(self.config)

        if self.config["fql_alpha"] > 0.0:
            actions = self.network.select("one_step_actor")(observations, noises)
            actions = jnp.clip(actions, -1, 1)
        else:
            if self.config["inv_temp"] == 0.0 or no_q:
                actions = self.compute_flow_actions(observations, noises, model="slow")
            else:
                actions = self.compute_flow_actions(
                    observations,
                    noises,
                    model="slow,fast" if self.config["residual"] else "fast",
                )
            if self.config["edit_scale"] > 0.0:
                edit_dist = self.network.select("edit_actor")(
                    jnp.concatenate((observations, actions), axis=-1)
                )
                actions = actions + edit_dist.sample(seed=edit_rng) * self.config["edit_scale"]
            actions = jnp.clip(actions, -1, 1)

        q = self.network.select("critic")(observations, actions).mean(axis=0)
        indices = jnp.argmax(q, axis=-1)
        bshape = indices.shape
        indices = indices.reshape(-1)
        bsize = len(indices)
        actions = jnp.reshape(actions, (-1, self.config["best_of_n"], action_dim))[
            jnp.arange(bsize), indices, :
        ].reshape(bshape + (action_dim,))
        return actions

    @jax.jit
    def diagnostic_snapshot(self, batch, rng):
        snap = QAMDiagAgent.diagnostic_snapshot(self, batch, rng)
        w = jnp.asarray(self.config["am_weights"], dtype=jnp.float32)
        ell = snap["ell"]
        p_norm = snap["p_norm"]
        L_t = jnp.mean(ell, axis=1)
        p_mean = jnp.mean(p_norm, axis=1)
        denom = jnp.sum(p_mean) + 1e-8
        t_grid = snap["t"]
        masks = {
            "early": t_grid < 0.5,
            "middle": (t_grid >= 0.3) & (t_grid < 0.7),
            "late": t_grid >= 0.5,
            "active": w > 0,
        }

        def fsig(mask):
            return jnp.sum(p_mean * mask.astype(p_mean.dtype)) / denom

        extra = {
            "am_w": w,
            "am_L_t": L_t,
            "am_wL_t": w * L_t,
            "F_signal_early": fsig(masks["early"]),
            "F_signal_middle": fsig(masks["middle"]),
            "F_signal_late": fsig(masks["late"]),
            "F_signal_active": fsig(masks["active"]),
            "am_weight_sum": jnp.sum(w),
        }
        return {**snap, **extra}

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        T = int(config["flow_steps"])
        spec = describe_am_mask(
            str(config.get("am_mask_name", "all")),
            flow_steps=T,
            eps=float(config.get("am_schedule_eps", 1e-3)),
            budget_normalize=config.get("am_budget_normalize", False),
        )
        config["am_mask_name"] = spec["name"]
        config["am_weights"] = spec["weights"]
        config["am_n_active"] = spec["n_active"]
        config["am_weight_sum"] = spec["weight_sum"]
        config["am_active_t_lo"] = spec["t_lo"]
        config["am_active_t_hi"] = spec["t_hi"]
        print("[stage3]", spec["summary"], flush=True)
        try:
            import wandb

            if wandb.run is not None:
                wandb.config.update(
                    {
                        "am_mask_id": spec["id"],
                        "am_mask_name": spec["name"],
                        "am_weights": list(spec["weights"]),
                        "am_n_active": spec["n_active"],
                        "am_weight_sum": spec["weight_sum"],
                        "am_active_steps": spec["active_steps"],
                        "am_t_lo": spec["t_lo"],
                        "am_t_hi": spec["t_hi"],
                    },
                    allow_val_change=True,
                )
        except Exception as exc:
            print("[stage3] wandb.config.update skipped:", exc, flush=True)
        return super().create(seed, ex_observations, ex_actions, config)


def get_config():
    config = qam_get_config()
    config.agent_name = "qam_region"
    config.am_mask_name = "all"
    config.am_budget_normalize = False
    config.am_schedule_eps = 1e-3
    config.am_weights = (1.0,) * 10
    config.am_n_active = 10
    config.am_weight_sum = 10.0
    config.am_active_t_lo = 0.0
    config.am_active_t_hi = 0.9
    return config
