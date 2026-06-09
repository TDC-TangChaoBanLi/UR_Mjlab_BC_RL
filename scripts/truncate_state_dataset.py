#!/usr/bin/env python3
"""截断数据集的 observation.state：保留前 7 维，删除后 7 维（last_action）。

用法:
  python scripts/truncate_state_dataset.py outputs/datasets/expert/pick_place/20260606_193958
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


def truncate_state_column(
    table: pa.Table,
    column: str = "observation.state",
    keep_dims: int = 7,
) -> pa.Table:
    """截断 fixed_size_list 列，保留前 keep_dims 个元素。"""
    col = table.column(column)
    col_type = col.type

    if not pa.types.is_fixed_size_list(col_type):
        raise TypeError(f"列 {column} 类型为 {col_type}，期望 fixed_size_list")

    old_size = col_type.list_size
    if old_size <= keep_dims:
        print(f"  {column}: {old_size}d，已 ≤ {keep_dims}，跳过")
        return table

    # FixedSizeList → List（截断）→ FixedSizeList
    col_as_list = col.cast(pa.list_(col_type.value_type))  # → list<float>
    sliced = pc.list_slice(col_as_list, 0, keep_dims)        # truncate
    new_col = sliced.cast(pa.list_(col_type.value_type, keep_dims))  # → fixed_size_list<float>[7]

    # 替换列
    new_table = table.set_column(
        table.schema.get_field_index(column),
        pa.field(column, new_col.type),
        new_col,
    )
    return new_table


def process_dataset(dataset_path: Path, keep_dims: int = 7) -> None:
    """处理完整 LeRobot 数据集。"""
    if not dataset_path.exists():
        raise FileNotFoundError(f"数据集不存在: {dataset_path}")

    data_dir = dataset_path / "data"
    if not data_dir.exists():
        raise FileNotFoundError(f"数据目录不存在: {data_dir}")

    parquets = sorted(data_dir.rglob("*.parquet"))
    if not parquets:
        raise FileNotFoundError(f"没有找到 parquet 文件: {data_dir}")

    print(f"数据集: {dataset_path}")
    print(f"Parquet 文件: {len(parquets)} 个")
    print(f"保留状态维度: {keep_dims}")

    # ── 1. 截断每个 parquet 文件 ──
    for p in parquets:
        rel = p.relative_to(dataset_path)
        print(f"  处理: {rel} ...", end=" ", flush=True)

        table = pq.read_table(str(p))
        try:
            new_table = truncate_state_column(table, keep_dims=keep_dims)
        except TypeError as e:
            print(f"跳过 ({e})")
            continue

        # 保留原始 schema metadata（huggingface json）
        new_meta = table.schema.metadata
        new_table = new_table.replace_schema_metadata(new_meta)

        pq.write_table(new_table, str(p))
        print("OK")

    # ── 2. 更新 meta/info.json ──
    info_path = dataset_path / "meta" / "info.json"
    if info_path.exists():
        with open(info_path) as f:
            info = json.load(f)

        features = info.get("features", {})
        state_feat = features.get("observation.state", {})
        if isinstance(state_feat, dict):
            old_shape = state_feat.get("shape", [])
            state_feat["shape"] = [keep_dims]
            print(f"\n  info.json: observation.state shape {old_shape} → [{keep_dims}]")

        with open(info_path, "w") as f:
            json.dump(info, f, indent=2)

    # ── 3. 截断 meta/stats.json（保留前 keep_dims 维的统计）──
    stats_path = dataset_path / "meta" / "stats.json"
    if stats_path.exists():
        with open(stats_path) as f:
            stats = json.load(f)

        for key in list(stats.keys()):
            if "state" in key:
                val = stats[key]
                if isinstance(val, dict):
                    for stat_name in list(val.keys()):
                        arr = val[stat_name]
                        if isinstance(arr, list):
                            val[stat_name] = arr[:keep_dims]
                    print(f"  stats.json: {key} 截断到 {keep_dims} 维")

        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)

    print("\n完成！")


def main():
    parser = argparse.ArgumentParser(description="截断 LeRobot 数据集的 observation.state 维度")
    parser.add_argument("dataset", type=str, help="LeRobot 数据集根目录")
    parser.add_argument("--keep-dims", type=int, default=7, help="保留的状态维度（默认 7）")
    args = parser.parse_args()

    process_dataset(Path(args.dataset), keep_dims=args.keep_dims)


if __name__ == "__main__":
    main()
