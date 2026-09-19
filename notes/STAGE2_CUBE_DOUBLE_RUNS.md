# Stage 2 cube-double-task2: official repro + raw-signal logging

Official QAM files were not edited. Seed 10001. Idle GPUs 1 and 4 only.

## Official QAM (unmodified main.py)

- W&B: https://wandb.ai/nightingale-314-/qam-reproduce/runs/5vwx45ky
- save_dir: exp/qam-reproduce/reproduce-double/cube-double-play-singletask-task2-v0/c0d825d199bdba96005896fa57f0354270c0b96d560fc4a0971f995928e6258f
- config: inv_temp=1.0, fql_alpha=0, edit_scale=0, horizon=5, chunking, 1M+500k
- success: 0.00 @50k, 0.24 @250k, 0.56 @500k, 0.86 @1M, 1.00 @1.5M
- official pkl 12-seed mean: 0.79 @1M (range 0.64-0.98), ~1.00 @1.5M
- this seed sits inside the official offline band and matches the climbing shape

Smoke (200 steps) W&B: https://wandb.ai/nightingale-314-/qam-reproduce/runs/tw0ljj74

## Diagnostic run (main_diag.py / qam_diag.py, same optimizer)

- W&B: https://wandb.ai/nightingale-314-/qam-reproduce/runs/pns0es9p
- save_dir: exp/qam-reproduce/reproduce-double-diag/cube-double-play-singletask-task2-v0/1f520b7ec2467ef5884d30911178cf05c4247513d51accf32425865ede1bec85
- 150 npz dumps (every 10k) + diag.csv; plots in save_dir/plots and notes/figures/stage2_cube_double/
- success also reaches 1.0 online; first version uses the existing 10 critics only

## Evidence from raw signals (no reliability score yet)

Latest dump is end of training (step 1.5M). Mean + sample distribution:

- Fig A: E[||p_t||] rises with t (0.01 -> 7.8) but q10-q90 and Var(||p|| | t) explode after t~0.6. Median stays below the mean (heavy tail).
- Fig B: E[R_grad] is only weakly increasing (~0.73 to ~0.80). The cloud spans ~0.45-0.95 at every t. Time is a weak predictor of critic-gradient agreement.
- Fig C: ||f_theta|| / ||f_beta|| stays ~1. Relative edit ||f_theta-f_beta||/||f_beta|| grows toward late t (mean ~0.10 to ~0.21) with a wide upper tail. QAM edits more near the end of the flow, but many samples barely move.
- Heatmaps: ||p_t|| and ell concentrate at large t and grow with training. Q(s,x_t) increases over training for all t. Support distance of x_t is largest at early t (noise) and smallest near t=0.9. cos(p_t, p_term) is high only at the last step.
- Policy drift ||A_RL-A_base|| stays ~0.5 throughout training with huge batch spread. Terminal U_grad stays ~0.20.

Interpretation for later Q-weighting: flow time is related to adjoint scale and where the field is edited, but t alone does not pin down whether Q gradients agree on a given sample. Do not collapse these channels into one score yet.
