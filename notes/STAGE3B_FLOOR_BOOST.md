# Stage 3B：floor + boost（QAM，cube-double-task4）

日期：2026-09-20  
分支：`stage3b-floor-boost-c2t4`（从 `stage3-flow-region-ablation` 分出）  
**不改** `agents/qam.py` / `main.py`。旧 Stage 3 结果和脚本保留。  
W&B：独立项目 [`qam-stage3b`](https://wandb.ai/nightingale-314-/qam-stage3b)（不和以前的 `qam-reproduce` 混在一起）。  
group：`smoke` / `full` 分开排。  
W&B 的 eval 只上传 `success`、`episode.return`、`episode.final_reward`。默认 Area 图会把 `total.timesteps`（上万）和 success（0–1）叠在一起变成色块；本地 `eval.csv` 仍保留全部列。

## 为什么换任务 / 为什么用这篇论文的配置

上一轮 Stage 3 在 `cube-double-task2` + 官方 QAM（`inv_temp=1`）上，All / Late-50 / Late-75 / Late-25 终局都到了 1.0，硬 0 mask 又把 Early 打成 No-Q。区分度不够，而且「某一段为 0」本身就会把 `actor_fast` 的 late 场写崩。

对照论文 Table 2 / `exp_data/qam-exp-data.pkl` 的官方 QAM（12 seed）：

| 任务 | 配置 | @1M offline | @1.5M online | 适不适合这次消融 |
| --- | --- | --- | --- | --- |
| cube-triple-task2 | `inv_temp=3`（Table 5） | **0.007**（0–0.04） | 0.67，方差极大 | 否。offline 地板，Stage 1 已复现 |
| cube-double-task2 | `inv_temp=1` | 0.79（0.64–0.98） | ≈1.0 | 否。上一轮已经饱和 |
| **cube-double-task4** | **同一套 cube-double QAM** | **0.215（0.14–0.34）** | ≈1.0 | **是。offline 有中段空间** |

论文（Appendix E）：manipulation 调参用 **task 2 和 task 4**；同一 domain 共用超参。cube-double 的 QAM 是：

$$
\tau=1,\quad \alpha_{\mathrm{FQL}}=0,\quad \sigma_a=0,\quad h=5,\quad \text{chunking},\quad K=10,\quad \rho=0.5,\quad T=10
$$

Table 4 公共项：batch 256，Adam \(3\times 10^{-4}\)，\(\gamma=0.99\)，clip 1，offline \(10^6\) + online \(5\times 10^5\)。  
本地已有 `cube-double-play-v0.npz`，不下载别人的数据。

**不用 cube-triple**：官方 QAM 在 1M 几乎是 0；online 种子 10001/20002/30003 曾是 1.00 / 0.02 / 0.08，1 seed 消融不可信。  
**不用乱改 \(\tau\)**：论文 Figure 3 写明温度是最敏感的超参；这次只改 \(w_k\)。

## 这一轮在问什么

硬 0/1 已经证明「late 为 0 会毁 policy」。现在问的是：

> 在 **始终有弱 AM** 的前提下，把某一段加强，会不会改变 cube-double-task4 上的 offline 学习曲线？

$$
L_{\mathrm{actor}}=L_{\mathrm{FM}}+\sum_k w_k L_{\mathrm{AM},k},\qquad t_k=k/T
$$

只改 \(w_k\)。不改 critic / FM / TD / 数据 / solver。No-Q 仍用 `actor_slow` 做 eval。

## 权重（自然尺度，不 equal-budget）

弱限制 \(\underline{w}=0.25\)。除 No-Q 外 \(\min_k w_k>0\)。

| ID | 方法 | \(w\) | 目的 |
| --- | --- | --- | --- |
| B0 | No-Q | 全 0 | behavior |
| B1 | All | 全 1 | 官方 QAM |
| B2 | Const-0.5 | 全 0.5 | 均匀中等强度；看 AM 损失尺度 |
| B3 | Early peak=1 | early=1，其余=0.25 | 前段 boost 1 |
| B4 | Mid peak=1 | middle=1，其余=0.25 | 中段 boost 1 |
| B5 | Late peak=1 | late=1，其余=0.25 | 后段 boost 1 |
| B6 | Early +0.25 | early=0.50，其余=0.25 | 前段弱 boost |
| B7 | Mid +0.25 | middle=0.50，其余=0.25 | 中段弱 boost |
| B8 | Late +0.25 | late=0.50，其余=0.25 | 后段弱 boost |
| B9 | Increasing | 0.25 → 1.0 | 逐渐增大 |
| B10 | Decreasing | 1.0 → 0.25 | 逐渐减小 |

early: \(t<0.5\)（k=0–4）；middle: \(0.3\le t<0.7\)（k=3–6）；late: \(t\ge 0.5\)（k=5–9）。  
「0.25 boost」按 **加在 floor 上** 实现：peak = 0.25+0.25=0.50，避免 peak=floor 退化成常数。

Const-0.5 是新对照：如果它和 All 接近，说明这一任务上 AM 对全局尺度不敏感，差异应来自 **形状** 而不是 \(\sum w\)。

## 训练 / 机器

- 任务：`cube-double-play-singletask-task4-v0`
- 种子：默认 1 个（10001）。`STAGE3_ALL_SEEDS=1` 才加 20002/30003
- GPU：**只 1 / 4 / 5**。0/2/3/6/7 上有别人的 Blender，不占用
- 旧 tmux `qam-s3-*` 不碰；本轮是 `qam-s3b-*`
- 结果目录：`exp/qam-reproduce/stage3b-floor-boost/`，不覆盖 Stage 3

```bash
/mnt/zoe/conda-envs/qam/bin/python stage3/test_am_weights.py
bash /mnt/zoe/projects/qam/scripts/launch_stage3b_tmux.sh all
```

11 个 full run，3 卡各约 4/4/3 个，按上一轮 ~2.7h/run 估计约 10–12 小时。

## 怎么读结果（先看 offline）

官方 QAM 在 task4 上 @1M 只有 ~0.22，online 才拉到 1。所以：

1. **主指标**：offline 曲线（250k / 500k / 1M），不要只看 1.5M
2. All vs Const-0.5：尺度 vs 形状
3. Late peak=1 vs Early peak=1（都有 floor）：floor 能否救 Early
4. peak=1 vs +0.25：加强幅度够不够
5. Increasing vs Decreasing

若 11 条 offline 曲线仍然重叠，再考虑换 domain，而不是先改 \(\tau\)。
