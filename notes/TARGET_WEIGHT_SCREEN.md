# QAM target-weight screen (cube-double-task2)

Date: 2026-09-28

This screen puts the flow-time multiplier inside the adjoint-matching target:

\[
L=\mathbb E\sum_k\left\|
\frac{2}{\sigma_k}(v_{\rm fast}-v_{\rm slow})
+w_k\sigma_kp_k
\right\|^2.
\]

Thus the pointwise optimum is

\[
v_{\rm fast}^*=v_{\rm slow}-w_k\frac{\sigma_k^2}{2}p_k.
\]

Zero weight remains supervised and anchors the fast field to the behavior
field. The original QAM and Stage-3 files are unchanged.

## Two-hour screen

- Environment: `cube-double-play-singletask-task2-v0`
- Seed: 10001
- 400k offline updates, no online phase
- Evaluation every 50k with 100 episodes
- Masks: All, No-Q, Early-50, Late-50, Late-25, Remove-Late-25
- W&B project: `qam-target-region`
- Local output: `exp/qam-target-region/screen-400k/`

The dynamic tmux scheduler checks GPUs 0 through 7 and starts work only when
a card has no compute process and uses less than 200 MiB. It never kills or
preempts another process. All six smoke runs must pass before full runs begin.

```bash
tmux attach -t qam-target-master
```

At launch time all eight GPUs were occupied by other users, so the scheduler
entered its non-GPU waiting state.
