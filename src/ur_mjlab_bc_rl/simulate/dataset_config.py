"""数据集配置加载 — 解析 dataset_*.yaml 中的多速率记录配置。

支持嵌套 recode_scale，各数据源以不同频率采样，打包进同一 LeRobot 帧。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_CONFIG_ROOT = Path(__file__).resolve().parents[3] / "configs"


@dataclass
class DataSource:
    """单个数据源的规格。"""
    name: str                              # "state.joint.position"
    dim_per_sub: int                       # 每子采样维度 (14 for 14 joints)
    num_subs: int                          # 子采样数 (scale=3, 即 90Hz)
    sub_indices: list[int]                 # 在 max_scale 网格中的子采样索引 [0,1,2]
    source_type: str                       # "joint_pos" | "sensor.force" | "action" | ...
    source_names: list[str]                # LeRobot displaying names (expanded, 对齐最后一维)
    read_names: list[str] = field(default_factory=list)  # MuJoCo 原始名称（传感器用）


@dataclass
class DatasetConfig:
    """数据集记录配置。"""
    recode_hz: float = 30.0
    max_scale: int = 1
    sources: list[DataSource] = field(default_factory=list)
    camera_config_file: str = ""
    depth_range: tuple[float, float] = (0.1, 2.0)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DatasetConfig":
        """从 YAML 加载。path 相对于项目根目录。"""
        full = Path(path)
        if not full.is_absolute():
            full = _CONFIG_ROOT.parent / full
        with open(full) as f:
            raw = yaml.safe_load(f)

        recode_hz = float(raw["recode_hz"])
        sources: list[DataSource] = []

        # ── 遍历 state 树 ──
        state = raw.get("state", {})
        cls._walk_tree(state, "state", recode_hz, sources)

        # ── 遍历 action 树 ──
        action = raw.get("action", {})
        cls._walk_tree(action, "action", recode_hz, sources)

        max_scale = max((s.num_subs for s in sources), default=1)

        camera_file = ""
        depth_range = (0.1, 2.0)
        state_cam = state.get("camera", {})
        if isinstance(state_cam, dict):
            camera_file = state_cam.get("scene_config_file", "")
            dr = state_cam.get("depth_range")
            if dr and len(dr) == 2:
                depth_range = (float(dr[0]), float(dr[1]))

        return cls(
            recode_hz=recode_hz,
            max_scale=max_scale,
            sources=sources,
            camera_config_file=camera_file,
            depth_range=depth_range,
        )

    @classmethod
    def _walk_tree(
        cls, node: dict, prefix: str, base_hz: float,
        sources: list[DataSource], parent_scale: int = 1,
    ) -> None:
        """递归遍历 YAML 树，收集叶子节点。"""
        recode_scale = node.get("recode_scale", 1) if isinstance(node, dict) else 1
        scale = parent_scale * recode_scale

        # 叶子判定：包含 joint_names / sensor_names / site_names / scene_config_file
        if isinstance(node, dict) and "joint_names" in node:
            names = node["joint_names"]
            dim = len(names)  # joint position: 1 per joint
            leaf_scale = scale * node.get("recode_scale", 1)
            cls._add_source(sources, prefix, dim, leaf_scale,
                            "joint_pos", names)
            return

        if isinstance(node, dict) and "sensor_names" in node:
            names = node["sensor_names"]
            sensor_type = prefix.rsplit(".", 1)[-1]  # "force"/"torque"/"gyro"/"accelerometer"
            dim_map = {"force": 3, "torque": 3, "gyro": 3, "accelerometer": 3}
            dim_per_sensor = dim_map.get(sensor_type, 1)
            dim = len(names) * dim_per_sensor
            leaf_scale = scale * node.get("recode_scale", 1)
            # 展开 names：sensor_x, sensor_y, sensor_z
            comp_suffixes = {3: ["_x", "_y", "_z"], 4: ["_w", "_x", "_y", "_z"]}
            expanded = []
            for n in names:
                for s in comp_suffixes.get(dim_per_sensor, [f"_{i}" for i in range(dim_per_sensor)]):
                    expanded.append(n + s)
            cls._add_source(sources, prefix, dim, leaf_scale,
                            f"sensor.{sensor_type}", expanded, read_names=names)
            return

        if isinstance(node, dict) and "site_names" in node:
            names = node["site_names"]
            suffix = prefix.rsplit(".", 1)[-1]  # "position" / "euler" / "quat"
            dim_map = {"position": 3, "euler": 3, "quat": 4}
            dim_per = dim_map.get(suffix, 3)
            dim = len(names) * dim_per
            leaf_scale = scale * node.get("recode_scale", 1)
            comp_suffixes = {3: ["_x", "_y", "_z"], 4: ["_w", "_x", "_y", "_z"]}
            expanded = []
            for n in names:
                for s in comp_suffixes.get(dim_per, [f"_{i}" for i in range(dim_per)]):
                    expanded.append(n + s)
            cls._add_source(sources, prefix, dim, leaf_scale,
                            f"site_{suffix}", expanded, read_names=names)
            return

        if isinstance(node, dict) and "scene_config_file" in node:
            # camera — 无数据维度，仅标记存在
            return

        # 递归子节点
        if isinstance(node, dict):
            for key, val in node.items():
                if key in ("recode_scale", "joint_names", "sensor_names",
                           "site_names", "scene_config_file", "frame_site", "type"):
                    continue
                sub_prefix = f"{prefix}.{key}" if prefix else key
                cls._walk_tree(val, sub_prefix, base_hz, sources, scale)

    @staticmethod
    def _add_source(
        sources: list, name: str, dim: int, scale: int,
        source_type: str, names: list[str], read_names: list[str] | None = None,
    ) -> None:
        sources.append(DataSource(
            name=name, dim_per_sub=dim, num_subs=scale,
            sub_indices=list(range(scale)),
            source_type=source_type,
            source_names=names,
            read_names=read_names if read_names is not None else names,
        ))

    @property
    def sample_interval_s(self) -> float:
        """子采样间隔（秒）：1 / (recode_hz * max_scale)。"""
        return 1.0 / (self.recode_hz * self.max_scale)

    @property
    def frame_interval_s(self) -> float:
        """帧间隔（秒）：1 / recode_hz。"""
        return 1.0 / self.recode_hz

    @property
    def state_dim(self) -> int:
        """所有 state 叶子节点的总维度（各源 num_subs × dim_per_sub 之和）。"""
        return sum(s.dim_per_sub * s.num_subs
                   for s in self.sources
                   if s.name.startswith("state."))

    @property
    def action_dim(self) -> int:
        """所有 action 叶子节点的总维度。"""
        return sum(s.dim_per_sub * s.num_subs
                   for s in self.sources
                   if s.name.startswith("action."))

    def validate(self, model) -> list[str]:
        """验证所有名称在 MuJoCo 模型中存在，返回 WARN 列表。"""
        import mujoco
        warnings: list[str] = []
        for src in self.sources:
            names = src.read_names if src.read_names else src.source_names
            for name in names:
                try:
                    if src.source_type == "joint_pos":
                        _ = model.joint(name).id
                    elif src.source_type.startswith("sensor."):
                        _ = model.sensor(name).id
                    elif src.source_type.startswith("site_"):
                        _ = model.site(name).id
                except Exception:
                    warnings.append(
                        f"DatasetConfig: '{name}' (source={src.name}) not found in MuJoCo model"
                    )
        return warnings
