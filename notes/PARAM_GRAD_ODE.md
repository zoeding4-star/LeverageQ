# Parameterized inner-Q gradient: minimal validation

This experiment intentionally keeps only one low-pass operation: antithetic
Gaussian smoothing of the terminal critic action-gradient.  There is no
flow-time moving average and no policy schedule yet.

For the matched deterministic Euler flow

\[
x_{k+1}=x_k+h v_{\bar\theta}(s,x_k,t_k),
\]

the full discrete adjoint target is computed by exact VJPs through the same
step map:

\[
g_k=\left(\frac{\partial x_{k+1}}{\partial x_k}\right)^\top g_{k+1}.
\]

`InnerGradientField(s, x, t)` is then trained directly against all adjoint
states.  Before connecting it to policy improvement, the held-out validation
must establish:

1. cosine similarity to the filtered full adjoint >= 0.97;
2. relative RMSE to the filtered full adjoint <= 0.25;
3. filtered-adjoint sensitivity to high-frequency terminal critic error <= 55%
   of the unfiltered full-adjoint sensitivity;
4. parameterized prediction sensitivity <= 75% of the unfiltered sensitivity.

Run the deterministic CPU check with:

```bash
JAX_PLATFORMS=cpu /mnt/zoe/conda-envs/qam/bin/python \
  scripts/verify_param_grad_ode.py \
  --output notes/figures/param_grad_ode/verification.json
```

The next stage should replace the controlled reference flow and critic with a
frozen/EMA QAM `actor_slow` and target-critic checkpoint.  A schedule should
only be introduced after the four validation conditions above pass there as
well.

## Controlled verification result

Three independent seeds (`7, 8, 9`) passed all four conditions with the
default configuration (2,048 training paths, 512 held-out paths, 10 Euler
steps, and 2,500 gradient updates):

| Metric | Three-seed mean |
| --- | ---: |
| Held-out cosine to filtered full adjoint | 0.9889 |
| Held-out relative RMSE to filtered full adjoint | 0.1206 |
| Filtered sensitivity / raw-adjoint sensitivity | 0.1892 |
| Parameterized sensitivity / raw-adjoint sensitivity | 0.0473 |

This establishes the mechanism in a controlled nonlinear ODE, not yet policy
improvement on OGBench.  The apparent extra denoising from 18.9% to 4.7% is
the regression network averaging the remaining Monte-Carlo filter noise; it
must be re-measured on a trained QAM checkpoint rather than assumed to carry
over to environment return.
