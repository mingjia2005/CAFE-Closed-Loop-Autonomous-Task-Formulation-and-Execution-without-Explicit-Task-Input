# -*- coding: utf-8 -*-
"""
对象“正常状态/位置”的规范文件（可由用户继续扩充）。
用于对比实时观测，发现异常（脏、开着、放错位置等）。

字段含义：
- expected_state: 默认正常状态（如餐具是干净的、柜门是关闭的、开关是关闭的）
- preferred_zone: 更适合/预期所在的大区（示例用中文名称，与 KDE 区域命名一致）
- preferred_container_classes: 更适合放置的容器类别（Cabinet/Drawer/Shelf/...），用于收纳打分
- ACCEPTABLE_SURFACES: 针对物体类型，可接受的“表面类”放置位置（例如：Tablet 在 Table/CounterTop 上可接受）
- FLOOR_OK_TYPES: 少数允许放在地上的可拾取类型（一般为空）。

注意：这是“知识库”文件，不直接依赖 AI2-THOR，对项目通用。
"""
from __future__ import annotations
from typing import Dict, Any, List, Set

NORMS: Dict[str, Dict[str, Any]] = {
    # —— 餐具 ——
    "Plate": {
        "expected_state": {"isDirty": False},
        "preferred_zone": "碗碟子区",
        "preferred_container_classes": ["Cabinet", "Shelf"],
    },
    "Bowl": {
        "expected_state": {"isDirty": False},
        "preferred_zone": "碗碟子区",
        "preferred_container_classes": ["Cabinet", "Shelf"],
    },
    "Cup": {
        "expected_state": {"isDirty": False},
        "preferred_zone": "碗碟子区",
        "preferred_container_classes": ["Cabinet", "Shelf"],
    },
    "Mug": {
        "expected_state": {"isDirty": False},
        "preferred_zone": "碗碟子区",
        "preferred_container_classes": ["Cabinet", "Shelf"],
    },

    # —— 炊具 ——
    "Pot": {
        "expected_state": {"isDirty": False},
        "preferred_zone": "炊具子区",
        "preferred_container_classes": ["Cabinet", "Shelf"],
    },
    "Pan": {
        "expected_state": {"isDirty": False},
        "preferred_zone": "炊具子区",
        "preferred_container_classes": ["Cabinet", "Shelf"],
    },

    # —— 食材/成品 ——
    "Apple": {"expected_state": {}, "preferred_zone": "冷藏子区", "preferred_container_classes": ["Appliance", "Cabinet"]},
    "Potato": {"expected_state": {}, "preferred_zone": "室温/干燥子区", "preferred_container_classes": ["Cabinet", "Shelf"]},
    "Tomato": {"expected_state": {}, "preferred_zone": "冷藏子区", "preferred_container_classes": ["Appliance", "Cabinet"]},
    "Lettuce": {"expected_state": {}, "preferred_zone": "冷藏子区", "preferred_container_classes": ["Appliance", "Cabinet"]},
    "Bread": {"expected_state": {}, "preferred_zone": "室温/干燥子区", "preferred_container_classes": ["Cabinet", "Shelf"]},
    "Cereal": {"expected_state": {}, "preferred_zone": "室温/干燥子区", "preferred_container_classes": ["Cabinet", "Shelf"]},
    "Egg": {"expected_state": {}, "preferred_zone": "冷藏子区", "preferred_container_classes": ["Appliance", "Cabinet"]},

    # —— 清洁工具 ——
    "Sponge": {"expected_state": {}, "preferred_zone": "清洁用品区", "preferred_container_classes": ["Cabinet", "CounterTop"]},
    "SoapBottle": {"expected_state": {}, "preferred_zone": "Wash Area", "preferred_container_classes": ["CounterTop", "Cabinet"]},
}

# —— 可接受表面（Whitelist） ——
# 注意：键为对象类型（AI2-THOR objectType），值为“支持/表面类”类型名（如 Table、DiningTable、CounterTop、Desk、Shelf、Sofa 等）
ACCEPTABLE_SURFACES: Dict[str, List[str]] = {
    # 电子/日常小物
    "Tablet": ["Table", "DiningTable", "CoffeeTable", "CounterTop", "Desk", "SideTable", "Sofa"],
    "Laptop": ["Desk", "Table", "CounterTop"],
    "Phone": ["Table", "CounterTop", "Desk", "SideTable", "Sofa"],
    "Book": ["Table", "DiningTable", "CoffeeTable", "CounterTop", "Desk", "Shelf", "Sofa"],
    "RemoteControl": ["Table", "CoffeeTable", "SideTable", "Sofa"],
    # 餐具默认可以临时在台面/桌面/搁板
    "Cup": ["CounterTop", "Table", "DiningTable", "CoffeeTable", "Shelf"],
    "Mug": ["CounterTop", "Table", "DiningTable", "CoffeeTable", "Shelf"],
    "Bowl": ["CounterTop", "Table", "DiningTable", "CoffeeTable", "Shelf"],
    "Plate": ["CounterTop", "Table", "DiningTable", "CoffeeTable", "Shelf"],
}

# —— 允许放地上的可拾取类型（通常为空；如将来扩展“鞋/地垫”等） ——
FLOOR_OK_TYPES: Set[str] = set()

__all__ = ["NORMS", "ACCEPTABLE_SURFACES", "FLOOR_OK_TYPES"]

