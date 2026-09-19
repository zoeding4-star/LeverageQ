# Stage 3：Flow-region Policy Improvement Ablation

日期：2026-09-19  
任务：`cube-double-play-singletask-task2-v0`（与 Stage 2 相同；cube-triple 的 offline success≈0，区分不了 mask）  
官方超参：`inv_temp=1.0`，`fql_alpha=0`，`edit_scale=0`，`horizon_length=5`，`action_chunking=True`，1M offline + 500k online  
W&amp;B：entity `nightingale-314-`，project `qam-reproduce`，group `stage3-region`  
代码：新分支 `stage3-flow-region-ablation`。**不改** `agents/qam.py` / `main.py`。

## 这一阶段验证什么

不是「early/middle/late 一定有用」，而是：

> QAM 的 policy improvement 对 Flow 中不同时间区域的依赖是什么？

三个问题：

1. **Region capacity**：No-Q vs Early-50 vs Middle-40 vs Late-50 vs All  
2. **Guidance onset**：All / Late-75 / Late-50 / Late-25 是否单调，或存在 interior optimum  
3. **Hard vs smooth**（第一轮只做 natural / standard scaling）：Uniform vs Early-heavy vs Late-heavy

成功标准不是「找到最好的 schedule」，而是搞清楚：region effect、onset effect、以及 time-only 是否已经不够（seed / state 异质性），后者才导向 Stage 4/5。

## 时间定义

$$
t\in[0,1],\qquad t_k=k/T,\quad k=0,\ldots,T-1
$$

$t=0$ 是 source / initial noise，$t\to 1$ 是 terminal action。$T=10$ 时：

$$
\{0,0.1,\ldots,0.9\}
$$

报告里写 **normalized $t$ + active step count**，不写「第 3 步」。

## 实现原则

$$
L_{\mathrm{actor}}=L_{\mathrm{FM}}+\sum_{k=0}^{T-1} w_k\,L_{\mathrm{AM},k}
$$

只改 $w_k$。完整计算 flow 与全部 adjoint，最后才 mask loss。不改 critic / FM / TD / dataset / target critic / sampling solver。

例外：No-Q（$w_k=0$）在 eval 时用 behavior `actor_slow`，否则 untrained `actor_fast` 不是 behavior baseline。训练时 adjoint 仍完整计算，只是不进入 loss。

文件：

| 路径 | 作用 |
| --- | --- |
| `stage3/am_weights.py` | $t_k$ 与 $w_k$ |
| `agents/qam_region.py` | 加权 AM loss；继承 `QAMDiagAgent` |
| `main_stage3.py` | 注册 `qam_region`，训练循环沿用 `main_diag.py` |
| `scripts/launch_stage3.sh` | 单 run；只占用当前 idle GPU |

## 第一轮矩阵（natural / standard scaling）

| ID | 方法 | $w$ | 目的 |
| --- | --- | --- | --- |
| M0 | No-Q | 全 0 | behavior baseline |
| M1 | All | 全 1 | 原始 QAM |
| M2 | Early-50 | `[1,1,1,1,1,0,0,0,0,0]` | early capacity |
| M3 | Middle-40 | `[0,0,0,1,1,1,1,0,0,0]` | middle capacity |
| M4 | Late-50 | `[0,0,0,0,0,1,1,1,1,1]` | late capacity |
| M5 | Late-75 | `[0,0,1,…,1]`（8 active） | onset |
| M6 | Late-25 | `[0,…,0,1,1,1]`（3 active） | onset |
| M7 | Early-heavy | $w\propto 1-t+\epsilon$，mean 1 | reverse smooth |
| M8 | Late-heavy | $w\propto t+\epsilon$，mean 1 | intuitive smooth |

Hard mask **不**重新归一化（3A natural）。Smooth schedule 归一化到 $\frac1T\sum w_k=1$，与 All 总预算相同。

Equal-budget（3B）已实现 `--agent.am_budget_normalize=True`，例如 Late-50 → `[0,0,0,0,0,2,2,2,2,2]`。**等 M2/M4 出现明显差异再跑** `M2_norm` / `M4_norm`，避免一上来 $w=2$ 改变优化动力学。

## Stage-2 先验：$F_{\mathrm{signal}}$

Stage 2 终局 $\mathbb{E}\|p_t\|$（`notes/figures/stage2_cube_double/latest_t_moments.csv`）：

$$
F_{\mathrm{signal}}(S)=\frac{\sum_{t\in S}\|p_t\|}{\sum_t\|p_t\|}
$$

| region | active $k$ | $F_{\mathrm{signal}}$ |
| --- | --- | --- |
| Early-50 | 0–4 | ≈ 0.059 |
| Middle-40 | 3–6 | ≈ 0.191 |
| Late-50 | 5–9 | ≈ 0.941 |
| Late-25 | 7–9 | ≈ 0.799 |

Late-only 即使 step 少，也可能保留绝大多数一阶 AM signal。因此 Late≈All **不能**直接解释成 early 完全没作用；更准确是 QAM 的 first-order signal 已经高度集中在 late flow。这就是为什么 3A 和 3B 必须分开。

## 记录的量

每个 log_interval：

- `eval/success` 整条 learning curve  
- `critic/q_mean`, `critic/q_max/min`  
- 每个 $t_k$：`w`, `L`（unweighted $\ell_t$）, `wL`, `G_vf=\|\nabla_{v_t}(w_t L_t)\|$, `p_norm`, `rel_edit`  
- `F_signal_{early,middle,late,active}`  
- 诊断 npz（每 20k，32 samples）：与 Stage 2 相同的 raw $x_t,p_t,Q,\nabla Q$

`G_t` 第一版是 velocity-space 范数，不是 $\|\nabla_\theta(w_t L_t)\|$（后者每个 $t$ 一次全参反向，编译/显存都重）。解释结果时不要把 $G_{\mathrm{vf}}$ 当成参数更新贡献的严格值。

## Seeds

- Debug：每 mask 1 seed（10001），smoke 200 steps 确认无 NaN、weight 正确、loss 有限  
- Screening：3 seeds `10001/20002/30003`，9 configs × 3 = 27 full runs  
- Final 5–8 seeds：只对真正有差异的方法再加，现在不加

## 怎么跑 / 怎么看

允许的 GPU：**只 1 / 4 / 5**（当前 idle）。0/2/3/6/7 上有别人的 Blender，不占用。

```bash
# CPU 单测
/mnt/zoe/conda-envs/qam/bin/python stage3/test_am_weights.py

# 在 tmux 里：先 9 个 smoke，通过后再 27 个 full
bash /mnt/zoe/projects/qam/scripts/launch_stage3_tmux.sh all

tmux attach -t qam-s3-master
tmux attach -t qam-s3-smoke-g1   # GPU 1
tmux attach -t qam-s3-g1         # full 队列
```

W&amp;B：https://wandb.ai/nightingale-314-/qam-reproduce

画图：

```bash
/mnt/zoe/conda-envs/qam/bin/python scripts/plot_stage3_region.py
```

## 四种可能的 onset 结果

1. 越早越好：All > 75% > 50% > 25% → 更多 flow-wide Q supervision 有利  
2. 越晚越好：early guidance 可能有害  
3. 中间最好：onset 存在 interior optimum  
4. 全部差不多：timestep allocation 可能不是这个任务的 bottleneck，就不要在这上面强行做 scheduler

## 下一步（现在不要做）

若 Stage 3 证明 where 会影响 performance → Stage 4 counterfactual importance / BPTT reliability → Stage 5 adaptive $w(C,R)$。
