"""LeRobot 策略插件。

导入此模块会触发 @PreTrainedConfig.register_subclass("bimft") 注册。
"""
from . import bimft  # noqa: F401 — 注册 PreTrainedConfig("bimft")
