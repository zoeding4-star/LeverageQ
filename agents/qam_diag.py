"""Instrumentation wrapper around official QAM.

Training loss / update is identical to `QAMAgent`. This file only adds a
separate `diagnostic_snapshot` that records flow-time raw signals.
Do not import this from `agents/__init__.py`; `main_diag.py` registers it.
"""

import jax
import jax.numpy as jnp

from agents.qam import QAMAgent
from agents.qam import get_config as qam_get_config


def _unit_dir_agreement(g, eps=1e-8):
    """g: (M, B, A) -> R_grad, U_grad, g_norm, ghat with shapes (B,), (B,), (M, B), (M, B, A)."""
    g_norm = jnp.linalg.norm(g, axis=-1, keepdims=True)
    ghat = g / (g_norm + eps)
    mean_dir = jnp.mean(ghat, axis=0)
    r_grad = jnp.linalg.norm(mean_dir, axis=-1)
    u_grad = 1.0 - r_grad
    return r_grad, u_grad, g_norm.squeeze(-1), ghat


def _min_l2_and_knn(x, support, k=5, eps=1e-8):
    """x, support: (B, A). Min L2 and mean of k nearest in the current minibatch."""
    d = jnp.linalg.norm(x[:, None, :] - support[None, :, :], axis=-1)
    d_sorted = jnp.sort(d, axis=1)
    k = min(k, d.shape[1])
    return d.min(axis=1), d_sorted[:, :k].mean(axis=1)


class QAMDiagAgent(QAMAgent):
    """Official QAM + flow-time diagnostics. Do not change actor/critic losses here."""

    def _critic_q_and_action_grad(self, obs, actions):
        """Return ensemble Q and per-critic action gradients at `actions`.

        Matches QAM adjoint init: target critic if `use_target_grad`, clip if `clip_adj`.
        Q shape (M, B); grad shape (M, B, A).
        """
        critic_name = "target_critic" if self.config["use_target_grad"] else "critic"
        a = jnp.clip(actions, -1.0, 1.0) if self.config["clip_adj"] else actions
        q = self.network.select(critic_name)(obs, a)

        def grad_from_weights(w):
            def scalar(act):
                qq = self.network.select(critic_name)(obs, act)
                return (qq * w[:, None]).sum()

            return jax.grad(scalar)(a)

        num_qs = int(self.config["num_qs"])
        g = jax.vmap(grad_from_weights)(jnp.eye(num_qs, dtype=a.dtype))
        return q, g

    @jax.jit
    def diagnostic_snapshot(self, batch, rng):
        """Raw signals for one training batch. Not used by `update` / gradients."""
        if self.config["action_chunking"]:
            batch_actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))
        else:
            batch_actions = batch["actions"][..., 0, :]

        obs = batch["observations"]
        flow_steps = int(self.config["flow_steps"])
        action_dim = self.config["action_dim"] * (
            self.config["horizon_length"] if self.config["action_chunking"] else 1
        )
        h = 1.0 / flow_steps
        eps = 1e-8

        rng, flow_rng, noise_rng = jax.random.split(rng, 3)
        x = jax.random.normal(flow_rng, shape=obs.shape[:-1] + (action_dim,))
        actor_slow = self.network.select(
            "target_actor_slow" if self.config["target_actor"] else "actor_slow"
        )

        xs = []
        vs_fast = []
        vs_slow = []
        vs_int = []
        ts = []
        keys = jax.random.split(flow_rng, flow_steps)
        for i in range(flow_steps):
            t = (i / flow_steps) * jnp.ones_like(x[..., 0:1])
            sigma = jnp.sqrt(2.0 * (1.0 - t + h) / (t + h))
            v_fast = self.network.select("actor_fast")(obs, x, t)
            v_slow = actor_slow(obs, x, t)
            if self.config["residual"]:
                v_field = v_fast + v_slow
            else:
                v_field = v_fast
            xs.append(x)
            ts.append(t)
            vs_fast.append(v_fast)
            vs_slow.append(v_slow)
            if i != flow_steps - 1:
                v_int = 2.0 * v_field - x / (t + h)
                noise = jax.random.normal(keys[i], x.shape)
                x = x + h * v_int + jnp.sqrt(h) * sigma * noise
            else:
                v_int = v_slow
                x = x + h * v_slow
            vs_int.append(v_int)

        x_term = x
        xs_mid = jnp.stack(xs, axis=0)
        ts_mid = jnp.stack(ts, axis=0)
        vf_fast = jnp.stack(vs_fast, axis=0)
        vf_slow = jnp.stack(vs_slow, axis=0)
        vf_int = jnp.stack(vs_int, axis=0)

        critic_name = "target_critic" if self.config["use_target_grad"] else "critic"
        a_term = jnp.clip(x_term, -1.0, 1.0) if self.config["clip_adj"] else x_term
        if self.config["clip_adj"]:
            grad_fn = jax.grad(
                lambda y: self.network.select(critic_name)(obs, jnp.clip(y, -1.0, 1.0))
                .mean(axis=0)
                .sum(),
            )
        else:
            grad_fn = jax.grad(
                lambda y: self.network.select(critic_name)(obs, y).mean(axis=0).sum(),
            )
        adj = -grad_fn(x_term) * self.config["inv_temp"]

        adjs_rev = []
        adj_t = adj
        for i in reversed(range(flow_steps)):
            t = (i / flow_steps) * jnp.ones_like(x_term[..., 0:1])

            def fn(xi, t=t):
                return 2.0 * actor_slow(obs, xi, t + h) - xi / (t + h)

            vjp = jax.vjp(fn, xs[i])[1](adj_t)[0]
            adj_t = adj_t + h * vjp
            adjs_rev.append(adj_t)
        p = jnp.stack(list(reversed(adjs_rev)), axis=0)

        sigmas = jnp.sqrt(2.0 * (1.0 - ts_mid + h) / (ts_mid + h))
        if self.config["residual"]:
            f_theta = vf_fast + vf_slow
            f_beta = vf_slow
            adj_resid = vf_fast * 2.0 / sigmas + sigmas * p
        else:
            f_theta = vf_fast
            f_beta = vf_slow
            adj_resid = (vf_fast - vf_slow) * 2.0 / sigmas + sigmas * p
        ell = jnp.sum(jnp.square(adj_resid), axis=-1)

        x_norm = jnp.linalg.norm(xs_mid, axis=-1)
        p_norm = jnp.linalg.norm(p, axis=-1)
        v_fast_norm = jnp.linalg.norm(vf_fast, axis=-1)
        v_slow_norm = jnp.linalg.norm(vf_slow, axis=-1)
        v_int_norm = jnp.linalg.norm(vf_int, axis=-1)
        f_theta_norm = jnp.linalg.norm(f_theta, axis=-1)
        f_beta_norm = jnp.linalg.norm(f_beta, axis=-1)
        vf_ratio = f_theta_norm / (f_beta_norm + eps)
        vf_rel_edit = jnp.linalg.norm(f_theta - f_beta, axis=-1) / (f_beta_norm + eps)

        def qg_one(xt):
            return self._critic_q_and_action_grad(obs, xt)

        q_xt, g_xt = jax.vmap(qg_one)(xs_mid)
        q_term, g_term = qg_one(x_term)

        def agree_t(g):
            r, u, gn, _ = _unit_dir_agreement(g)
            return r, u, gn.mean(axis=0)

        r_grad_t, u_grad_t, g_xt_norm_mean = jax.vmap(agree_t)(g_xt)
        r_grad_term, u_grad_term, g_term_norm, _ = _unit_dir_agreement(g_term)

        q_xt_mean = q_xt.mean(axis=1)
        q_xt_std = q_xt.std(axis=1)
        q_term_mean = q_term.mean(axis=0)
        q_term_std = q_term.std(axis=0)

        p_term = p[-1]
        p_term_u = p_term / (jnp.linalg.norm(p_term, axis=-1, keepdims=True) + eps)
        p_u = p / (p_norm[..., None] + eps)
        cos_p_pterm = jnp.sum(p_u * p_term_u[None], axis=-1)

        g_mean_xt = g_xt.mean(axis=1)
        g_mean_u = g_mean_xt / (jnp.linalg.norm(g_mean_xt, axis=-1, keepdims=True) + eps)
        cos_p_gmean = jnp.sum(p_u * g_mean_u, axis=-1)

        noises = jax.random.normal(noise_rng, (obs.shape[0], action_dim))
        a_base = self.compute_flow_actions(obs, noises, model="slow")
        if self.config["residual"]:
            a_rl = self.compute_flow_actions(obs, noises, model="slow,fast")
        else:
            a_rl = self.compute_flow_actions(obs, noises, model="fast")
        policy_drift = jnp.linalg.norm(a_rl - a_base, axis=-1)

        a_rl_sup_min, a_rl_sup_knn = _min_l2_and_knn(a_rl, batch_actions)
        a_base_sup_min, a_base_sup_knn = _min_l2_and_knn(a_base, batch_actions)
        x_flat = xs_mid.reshape((-1, action_dim))
        x_sup_min, x_sup_knn = _min_l2_and_knn(x_flat, batch_actions)
        t_size = xs_mid.shape[0]
        x_support_min = x_sup_min.reshape((t_size, -1))
        x_support_knn = x_sup_knn.reshape((t_size, -1))

        t_grid = jnp.arange(flow_steps) / float(flow_steps)

        return {
            "t": t_grid,
            "x": xs_mid,
            "p": p,
            "x_term": x_term,
            "x_norm": x_norm,
            "p_norm": p_norm,
            "v_fast_norm": v_fast_norm,
            "v_slow_norm": v_slow_norm,
            "v_int_norm": v_int_norm,
            "ell": ell,
            "vf_ratio": vf_ratio,
            "vf_rel_edit": vf_rel_edit,
            "q_xt": q_xt,
            "g_xt": g_xt,
            "q_xt_mean": q_xt_mean,
            "q_xt_std": q_xt_std,
            "R_grad_t": r_grad_t,
            "U_grad_t": u_grad_t,
            "g_xt_norm_mean": g_xt_norm_mean,
            "q_term": q_term,
            "g_term": g_term,
            "q_term_mean": q_term_mean,
            "q_term_std": q_term_std,
            "R_grad_term": r_grad_term,
            "U_grad_term": u_grad_term,
            "g_term_norm": g_term_norm,
            "cos_p_pterm": cos_p_pterm,
            "cos_p_gmean": cos_p_gmean,
            "A_rl": a_rl,
            "A_base": a_base,
            "policy_drift": policy_drift,
            "A_rl_support_min": a_rl_sup_min,
            "A_rl_support_knn5": a_rl_sup_knn,
            "A_base_support_min": a_base_sup_min,
            "A_base_support_knn5": a_base_sup_knn,
            "x_support_min": x_support_min,
            "x_support_knn5": x_support_knn,
            "dataset_actions": batch_actions,
        }


def get_config():
    config = qam_get_config()
    config.agent_name = "qam_diag"
    return config
