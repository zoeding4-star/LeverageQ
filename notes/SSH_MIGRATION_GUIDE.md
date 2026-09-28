# QAM / Q-Flow SSH 迁移说明

GitHub 只保存代码和脚本；数据集、`exp/`、W&B 日志、模型 checkpoint 和密钥放在 SSH 机器上。

## 克隆代码

```bash
mkdir -p /mnt/zoe/projects
cd /mnt/zoe/projects
git clone -b target-weight-screen git@github.com:zoeding4-star/LeverageQ.git qam
git clone git@github.com:zoeding4-star/FLowtimeQ.git qflow
```

## 配置 GitHub SSH

```bash
ssh-keygen -t ed25519 -C "你的 GitHub 邮箱"
cat ~/.ssh/id_ed25519.pub
ssh -T git@github.com
```

把公钥添加到 GitHub 的 `Settings → SSH and GPG keys`。

## 安装环境

```bash
cd /mnt/zoe/projects/qam
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cd /mnt/zoe/projects/qflow
pip install -r requirements.txt
```

如果服务器已有 QAM 环境，可先验证：

```bash
/mnt/zoe/conda-envs/qam/bin/python -c \
  'import jax, flax, ogbench, wandb; print(jax.devices())'
```

设置配置：

```bash
export OGBENCH_DATASET_DIR=/mnt/zoe/datasets/ogbench
export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_DEVICE_ORDER=PCI_BUS_ID
```

## 传输数据集

不要把数据集提交到 GitHub。假设旧 SSH 主机名是 `OLD_SSH_HOST`：

```bash
mkdir -p /mnt/zoe/datasets/ogbench
rsync -av --progress \
  OLD_SSH_HOST:/mnt/zoe/datasets/ogbench/cube-double-play-v0.npz \
  /mnt/zoe/datasets/ogbench/
```

检查 GPU 和 JAX：

```bash
nvidia-smi
python -c 'import jax; print(jax.devices())'
```

## QAM target-weight screen

先跑 smoke：

```bash
cd /mnt/zoe/projects/qam
GPU=0 bash scripts/launch_target_screen.sh 0 all smoke
```

通过后运行：

```bash
bash scripts/dispatch_target_screen.sh
```

或者手动分配三张空闲 GPU：

```bash
GPU=0 bash scripts/launch_target_screen_worker.sh 0 0 3 smoke
GPU=1 bash scripts/launch_target_screen_worker.sh 1 1 3 smoke
GPU=2 bash scripts/launch_target_screen_worker.sh 2 2 3 smoke
```

smoke 全部通过后，把 `smoke` 改成 `full`。full 是 400k offline steps、0 online steps、每 50k steps evaluation。

监控：

```bash
watch -n 5 nvidia-smi
tmux ls
tail -f exp/logs/target-worker-gpu0-shard0-full.log
find exp/qam-target-region -name token.tk -print
```

## QAM Stage 3B

需要重跑时：

```bash
cd /mnt/zoe/projects/qam
bash scripts/launch_stage3b_tmux.sh all
```

## Q-Flow

```bash
cd /mnt/zoe/projects/qflow
python stage3/test_guidance_weights.py
bash run/launch_tmux.sh smoke
bash run/launch_tmux.sh after-smoke
```

## 结果与密钥

训练结果在 `exp/` 中，已被 Git 忽略。需要迁移结果时单独复制：

```bash
rsync -av --progress OLD_SSH_HOST:/mnt/zoe/projects/qam/exp/ /mnt/zoe/projects/qam/exp/
rsync -av --progress OLD_SSH_HOST:/mnt/zoe/projects/qflow/exp/ /mnt/zoe/projects/qflow/exp/
```

不要复制或提交 `.env`、W&B token、SSH 私钥；在新 SSH 上单独执行 `wandb login`。
