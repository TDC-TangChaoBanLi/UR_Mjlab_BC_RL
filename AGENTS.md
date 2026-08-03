

当前是一个基于 UV 的 Python 项目，请务必在 UV 虚拟环境中运行 python 命令：
```bash
source .venv/bin/activate
```

然后运行 Python 脚本：
```bash
uv run python_script.py
```

本项目主要由 LeRobot-Mujoco 仿真数据采集模块 `src/ur_mjlab_bc_rl/data` 、 LeRobot 仿真环境模块 `src/ur_mjlab_bc_rl/lerobot_env` (已经写为 LeRobot 环境插件包) 、 LeRobot 策略模型模块 `src/ur_mjlab_bc_rl/lerobot_policy` (已经写为 LeRobot 策略模型插件包) 、 Mujoco 仿真环境模块 `src/ur_mjlab_bc_rl/simulate` 、 PyTorch 模型骨干模块 `src/ur_mjlab_bc_rl/models` 构成。


关于 LeRobot 的相关资料，请参考 [LeRobot 文档](https://huggingface.co/docs/lerobot/index)

