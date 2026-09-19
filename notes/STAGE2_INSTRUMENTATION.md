# Stage 2：只记录 raw signal（不合成 reliability score）

官方训练文件保持不动：`agents/qam.py`、`main.py`、`experiments/reproduce.py`。

新增（本分支）：

| 文件 | 作用 |
| --- | --- |
| `agents/qam_diag.py` | 继承 `QAMAgent`；**loss 与官方相同**；另加 `diagnostic_snapshot` |
| `main_diag.py` | 训练循环与 `main.py` 相同，额外按 interval 记诊断 |
| `diag/io.py` | 把 snapshot 收成标量、写 npz |
| `scripts/plot_qam_flow_diagnostics.py` | 图 A/B/C + 后续分析用的配套图（mean + sample distribution） |
| `scripts/launch_cube_double_official.sh` | 官方 QAM cube-double：先 smoke 再 1M+500k |
| `scripts/launch_cube_double_diag.sh` | 诊断 run（同样官方超参） |

QAM 已有 `num_qs=10` 的 critic ensemble。第一版可靠性信号 **只用这 10 个 Q**，不再训练新模型。

## 每个 batch、每个 flow 时间 \(t_k\) 保存什么

Flow / adjoint（QAM 一阶量）：

- \(x_k\)：adjoint-matching 积分器在 \(t_k=k/T\) 的状态（未含 terminal）
- \(p_k^{\mathrm{QAM}}\)：反向 adjoint（代码里的 `adjs`）
- \(\|x_k\|,\ \|p_k\|,\ \|v_k\|\)
- \(\ell_k\)：该步 adjoint-matching 被加进 `adj_loss` 的 per-sample 项  
  residual=False（官方 cube 默认）：\(\ell_k=\|(v_{\mathrm{fast}}-v_{\mathrm{slow}})\cdot 2/\sigma_k + \sigma_k p_k\|_2^2\)

Terminal critic（执行动作 \(A=x_T\)）：

- \(Q_i(s,A),\ i=1,\ldots,10\)
- \(g_i=\nabla_A Q_i\)
- \(\bar Q,\ \mathrm{Std}_i(Q_i)\)
- \(R_{\mathrm{grad}}=\bigl\|\frac1M\sum_i g_i/\|g_i\|\bigr\|\)，\(U_{\mathrm{grad}}=1-R_{\mathrm{grad}}\)

为了能画 **\(t\) vs \(R_{\mathrm{grad}}\)**（Q 只定义在动作上，而 \(x_k\) 与动作同空间），对每个 \(x_k\) 再算一遍 \(Q_i(s,\mathrm{clip}(x_k))\) 和 \(\nabla_{x}Q_i\)。这不是改算法，只是问：假如在这个 flow 位置用 Q，ensemble 方向是否一致。

Policy drift：

- \(\|A_{\mathrm{RL}}-A_{\mathrm{base}}\|\)，其中 \(A_{\mathrm{base}}\) 只用 `actor_slow` 积分，\(A_{\mathrm{RL}}\) 用 eval 同款 `compute_flow_actions`（cube 上 `residual=False` 故为 `fast`）
- dataset-support proxy：当前 minibatch 动作上的最小 L2，以及 5-NN 均值（对 \(A_{\mathrm{RL}},A_{\mathrm{base}},x_k\)）

**不**在这一阶段把上述量合成一个 reliability scalar。先把 raw 曲线和分布画出来。

## 存储策略

全量 \(x_k\) 每个 step 会到几百 GB。因此：

- 每个 `log_interval`（默认 5k）：只写 **batch 上的 mean/std/q10/q50/q90** 到 W&B + `diag.csv`
- 每个 `dump_interval`（默认 10k）：子采样 `dump_samples`（默认 64）条，把 \(x_k,p_k,Q_i,\nabla Q_i\) 存成 `diag_dumps/step_XXXXXX.npz`（向量用 float16）

## 必看图

- **图 A** \(t\) vs \(\|p_t\|\)：QAM 一阶信号沿 flow 怎么变
- **图 B** \(t\) vs \(R_{\mathrm{grad}}\)：critic gradient 方向一致性是否随时间变
- **图 C** \(t\) vs \(\|f_\theta\|/(\|f_\beta\|+\epsilon)\)：QAM 实际在哪些 flow 位置改 policy

三张图都是 **mean curve + sample distribution**。若 \(t\mapsto \mathbb{E}[C_t]\) 有趋势，但 \(\mathrm{Var}(C_t\mid t)\) 很大：说明 flow time 与 policy-improvement 结构有关，但 **time alone 可能不够决定何时用 Q**。这是第一块 evidence，不是失败。

配套图（同一脚本）：\(\ell_t,\|x_t\|,\|v_t\|,\bar Q(s,x_t),\mathrm{Std}(Q_i),\cos(p_t,p_T)\)、support 距离、\(\|A_{\mathrm{RL}}-A_{\mathrm{base}}\|\) 随训练步、以及 step×t heatmap。

## 跑法

官方复现（不动算法）：

```bash
tmux new-session -d -s qam-double
# GPU 必须 IDLE
bash /mnt/zoe/projects/qam/scripts/launch_cube_double_official.sh 1 10001
```

诊断（另卡，另进程）：

```bash
bash /mnt/zoe/projects/qam/scripts/launch_cube_double_diag.sh 4 10001
```

画图：

```bash
/mnt/zoe/conda-envs/qam/bin/python scripts/plot_qam_flow_diagnostics.py \
  --run_dir <save_dir>
```
