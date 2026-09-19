# Stage 1 简要总结：官方 QAM 在 cube-triple-task2 上的复现

日期：2026-09-18  
代码：`/mnt/zoe/projects/qam` @ `2726d76`（官方 `qsm-bugfix-0509`，未改 `agents/qam.py` / `main.py`）  
任务：`cube-triple-play-singletask-task2-v0`  
配置（与 README / `experiments/reproduce.py` 的 QAM 一致）：`inv_temp=3.0`，`fql_alpha=0`，`edit_scale=0`，`horizon_length=5`，`action_chunking=True`，`sparse=False`，`num_qs=10`，`rho=0.5`，offline 1M + online 500k，eval 每 50k × 50 episodes。  
种子：`10001 / 20002 / 30003`（官方 12-seed 协议的前 3 个）。  
对照：`exp_data/qam-exp-data.pkl` 键 `('cube-triple-play-singletask-task2-v0', 'QAM')`，形状 `(30, 12)`，仅 success。

## 结论

**离线算法复现是成功的。** 不能用 cube-triple 的 offline success 做 `w_k` / 何时用 Q 的消融：官方自己在 1M 时几乎是 0。

| 节点 | 官方 12-seed 均值 | 我们 3-seed 均值 | 备注 |
| --- | --- | --- | --- |
| offline @ 1M | **0.0067**（种子范围 0.00–0.04） | **0.0067**（0.00 / 0.00 / 0.02） | 均值曲线 MAE（20 个 offline 点）**0.0038** |
| online @ 1.5M | **0.668**（种子含 1.00 和 0.00） | **0.367**（1.00 / 0.02 / 0.08） | 高方差；3 个种子不够估计 online 均值 |

训练侧三个种子都健康：`flow_loss` 下降到 ~0.18，`q_mean` ~ -268，无 NaN。W&B eval 里大色块来自稀疏 log + MuJoCo `qpos/qvel/control` 被当成标量画面积，不是崩溃。

## 为什么 offline≈0 仍算复现成功

官方 cube-triple-task2 的 QAM **本来就是 offline 地板、online 才拉起来**。success 在 1M 之前几乎全是 0，1.05M–1.5M 才分化。三个种子的 online 结果（一个满分、两个接近 0）也落在官方 12-seed 的支撑里（官方 final 含 `1.00, 0.98, …, 0.04, 0.00`）。

因此：Stage 1 验证的是 **实现 + 数据 + 优化器与官方一致**，不是「这个任务上 QAM 已经学会了」。在 cube-triple 上改 adjoint 权重 / 分段用 Q，offline success 区分不了方法。

## 下一步为什么换 cube-double-task2

官方同一 pkl、同一方法：

- `cube-double-play-singletask-task2-v0` + QAM，`inv_temp=1.0`
- offline success 从 ~0 爬到 **@1M ≈ 0.79**（12-seed 范围 0.64–0.98），online 接近 1.0

这是「离线算法是否在改 policy」的正确尺子。Stage 2 的 flow-time 诊断也放在这个任务上。

## 冻结的实现细节（后续不要动）

- conda：`/mnt/zoe/conda-envs/qam`，JAX 0.6.2 cuda12，ogbench 1.1.0
- 数据：`HOME=/mnt/zoe/home` → `~/.ogbench/data/*.npz`（官方 `main.py` 不用 `OGBENCH_DATASET_DIR`）
- GPU：只占用 idle 卡（本机通常 1/4/5）；启动前 `scripts/run_on_idle_gpu.sh` 检查 `used<200MiB` 且无 compute proc
- W&B：project `qam-reproduce`，账号 nightingale-314
