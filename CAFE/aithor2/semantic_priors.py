# -*- coding: utf-8 -*-
"""
语义先验与区域推断
- 提供常见物体 -> 区域 的先验概率（可按需扩充/修改）
- 根据场景中锚点对象（如 Sink / Stove / DiningTable 等）自动生成区域边界（粗略方框）
- 在不知道具体柜体标签时，利用先验+距离，给新发现物体赋予一个最可能的区域标签

使用约定：
- semantic_map 为 main_with_depth.py 中维护的字典对象
- 这里不依赖有空格的包路径，文件置于仓库根目录，便于 import semantic_priors
"""
from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import math

# 物体类型 -> 区域名称 -> 概率（0~1）
REGION_PRIORS: Dict[str, Dict[str, float]] = {
    # 餐具器皿
    "Plate": {"SinkArea": 0.45, "CountertopArea": 0.25, "DiningArea": 0.20, "Storage_Plates": 0.10},
    "Bowl": {"SinkArea": 0.45, "CountertopArea": 0.25, "DiningArea": 0.15, "Storage_Bowls": 0.15},
    "Cup":  {"SinkArea": 0.50, "CountertopArea": 0.20, "DiningArea": 0.15, "Storage_Cups": 0.15},
    "Mug":  {"SinkArea": 0.50, "CountertopArea": 0.20, "DiningArea": 0.15, "Storage_Cups": 0.15},
    # 锅具
    "Pot":  {"StoveArea": 0.55, "CountertopArea": 0.25, "Storage_PansPots": 0.20},
    "Pan":  {"StoveArea": 0.55, "CountertopArea": 0.25, "Storage_PansPots": 0.20},
    # 刀叉勺
    "Knife": {"CountertopArea": 0.50, "DiningArea": 0.10, "Storage_Utensils": 0.40},
    "Fork":  {"DiningArea": 0.40, "CountertopArea": 0.20, "Storage_Utensils": 0.40},
    "Spoon": {"DiningArea": 0.40, "CountertopArea": 0.20, "Storage_Utensils": 0.40},
    # 清洁用品
    "Sponge": {"SinkArea": 0.80, "CountertopArea": 0.20},
    "SoapBottle": {"SinkArea": 0.80, "CountertopArea": 0.20},
}
# 区域名称（中文）
ZONE_NAMES = [
    "食材处理区","洗涤子区","切配子区","备餐/混合子区","烹饪区","食物储存区","冷藏子区","室温/干燥子区",
    "碗碟/炊具收纳区","碗碟子区","炊具子区","清洁用品区",
    "娱乐/社交中心区","休闲阅读区","通道区",
    "用餐核心区","备餐/服务区",
    "睡眠区","衣物储存/更衣区","工作/学习区",
    "Wash Area","淋浴/沐浴区","如厕区",
    "洗衣/杂物区","入口/玄关区",
]

# 核心物体 -> 功能区（打分）
CORE_OBJECT_ZONE_SCORES = {
    "Sink": [("洗涤子区", 10), ("Wash Area", 9)],
    "SinkBasin": [("洗涤子区", 10)],
    "Faucet": [("洗涤子区", 9)],
    "StoveBurner": [("烹饪区", 10)],
    "Stove": [("烹饪区", 10)],
    "Range": [("烹饪区", 10)],
    "Microwave": [("烹饪区", 9)],
    "Fridge": [("冷藏子区", 10)],
    "Refrigerator": [("冷藏子区", 10)],
    "CounterTop": [("切配子区", 9), ("备餐/混合子区", 9), ("烹饪区", 7)],
    "DiningTable": [("用餐核心区", 10)],
    "Table": [("用餐核心区", 8)],
    "CoffeeTable": [("娱乐/社交中心区", 8)],
    "Sofa": [("娱乐/社交中心区", 10)],
    "Bed": [("睡眠区", 10)],
    "Toilet": [("如厕区", 10)],
    "ShowerDoor": [("淋浴/沐浴区", 10)],
    "Bathtub": [("淋浴/沐浴区", 10)],
    "WashingMachine": [("洗衣/杂物区", 10)],
    "UpperCabinet": [("碗碟/炊具收纳区", 8), ("碗碟子区", 7)],
    "LowerCabinet": [("碗碟/炊具收纳区", 8), ("炊具子区", 7)],
    "Cabinet": [("碗碟/炊具收纳区", 8)],
    "Drawer": [("碗碟/炊具收纳区", 8)],
    "Shelf": [("碗碟/炊具收纳区", 7), ("休闲阅读区", 7)],
    "Door": [("入口/玄关区", 8)],
}

# KDE 参数
_SIGMA_M = 0.8  # 高斯核标准差（米）
_MAX_RADIUS_SIGMA = 3.0

def _sm_w2m(semantic_map: dict, x: float, z: float) -> tuple[int, int]:
    b = semantic_map.get("bounds", [0, 0, 0, 0])
    res = float(semantic_map.get("resolution", 0.1) or 0.1)
    u = int((x - b[0]) / res)
    v = int((z - b[2]) / res)
    u = max(0, min(int(semantic_map.get("width", 1)) - 1, u))
    v = max(0, min(int(semantic_map.get("height", 1)) - 1, v))
    return u, v

def _sm_m2w(semantic_map: dict, u: int, v: int) -> tuple[float, float]:
    b = semantic_map.get("bounds", [0, 0, 0, 0])
    res = float(semantic_map.get("resolution", 0.1) or 0.1)
    x = b[0] + (u + 0.5) * res
    z = b[2] + (v + 0.5) * res
    return x, z


# 区域锚点：区域名称 -> 在场景中用于定位该区域的对象类型（任一匹配即可）
REGION_ANCHORS: Dict[str, List[str]] = {
    "SinkArea": ["Sink", "SinkBasin", "Faucet"],
    "StoveArea": ["Stove", "StoveBurner", "Range"],
    "DiningArea": ["DiningTable", "Table", "CoffeeTable"],
    "CountertopArea": ["CounterTop", "KitchenCounter"],
    # 储物类区域常以柜体为锚点（此处仅提供类别名称，具体柜体由运行时学习/标注）
    "Storage_Plates": ["UpperCabinet", "Cabinet"],
    "Storage_Bowls": ["UpperCabinet", "Cabinet"],
    "Storage_Cups": ["UpperCabinet", "Cabinet"],
    "Storage_Utensils": ["Drawer", "Cabinet"],
    "Storage_PansPots": ["LowerCabinet", "Cabinet"],
}

# 生成区域边界时，默认半尺寸（米）。可按需微调。
DEFAULT_REGION_HALF_EXTENTS: Dict[str, Tuple[float, float]] = {
    "SinkArea": (0.8, 0.8),
    "StoveArea": (0.9, 0.9),
    "DiningArea": (1.2, 1.0),
    "CountertopArea": (2.0, 1.0),
    "Storage_Plates": (0.8, 0.8),
    "Storage_Bowls": (0.8, 0.8),
    "Storage_Cups": (0.8, 0.8),
    "Storage_Utensils": (1.0, 0.8),
    "Storage_PansPots": (1.0, 0.8),
}


def _obj_pos(obj: dict) -> Tuple[float, float]:
    p = obj.get("position", {})
    return float(p.get("x", 0.0)), float(p.get("z", 0.0))


def _find_anchors(event) -> Dict[str, List[Tuple[float, float]]]:
    """返回每个区域的锚点位置（可有多个）"""
    res: Dict[str, List[Tuple[float, float]]] = {k: [] for k in REGION_ANCHORS}
    objs = event.metadata.get("objects", []) or []
    for o in objs:
        t = (o.get("objectType") or "").title()
        for region, types in REGION_ANCHORS.items():
            if any(tt.lower() in t.lower() for tt in types):
                res[region].append(_obj_pos(o))
    return res


def ensure_semantic_areas(event, semantic_map: dict) -> None:
    """基于探索到的“核心物体”用 KDE 生成功能区热力图，并阈值化为矩形边界。
    结果写回 semantic_map['areas']，名称为中文功能区名；并缓存 sigma 信息。
    """
    areas = semantic_map.setdefault("areas", {})
    width = int(semantic_map.get("width", 200))
    height = int(semantic_map.get("height", 200))
    res = float(semantic_map.get("resolution", 0.1) or 0.1)
    sigma_px = max(2, int(round(_SIGMA_M / res)))
    max_r = int(_MAX_RADIUS_SIGMA * sigma_px)
    # 初始化热力图
    heatmaps = {zn: [[0.0 for _ in range(height)] for _ in range(width)] for zn in ZONE_NAMES}

    # 仅使用“已探索/已发现”的对象来生成热力（随着探索动态变化）
    discovered = (semantic_map.get("objects", {}) or {}).values()
    for o in discovered:
        t = (o.get("type") or o.get("objectType") or "").title()
        p = o.get("position") or {}
        x, z = float(p.get("x", 0.0)), float(p.get("z", 0.0))
        u, v = _sm_w2m(semantic_map, x, z)
        # 找到对应的区域分值
        scores: List[Tuple[str, float]] = []
        for key, lst in CORE_OBJECT_ZONE_SCORES.items():
            if key.lower() in t.lower():
                scores.extend(lst)
        if not scores:
            continue
        for zone_name, score in scores:
            hm = heatmaps.get(zone_name)
            if hm is None:
                continue
            for du in range(-max_r, max_r + 1):
                uu = u + du
                if uu < 0 or uu >= width:
                    continue
                for dv in range(-max_r, max_r + 1):
                    vv = v + dv
                    if vv < 0 or vv >= height:
                        continue
                    d2 = du * du + dv * dv
                    val = float(score) * math.exp(-d2 / (2.0 * sigma_px * sigma_px))
                    hm[uu][vv] += val

    # 根据阈值（0.5*max）生成矩形区域，并保存稀疏热力点用于可视化
    semantic_map["zone_heatmaps"] = {"meta": {"sigma_px": sigma_px}, "zones": {}}
    for zn, hm in heatmaps.items():
        m = 0.0
        for uu in range(width):
            col = hm[uu]
            for vv in range(height):
                if col[vv] > m:
                    m = col[vv]
        if m <= 0.0:
            continue
        th = 0.5 * m
        sparse_th = 0.35 * m
        step = 2
        umin, vmin, umax, vmax = width, height, -1, -1
        sparse = []
        for uu in range(0, width, step):
            col = hm[uu]
            for vv in range(0, height, step):
                val = col[vv]
                if val >= th:
                    if uu < umin:
                        umin = uu
                    if uu > umax:
                        umax = uu
                    if vv < vmin:
                        vmin = vv
                    if vv > vmax:
                        vmax = vv
                if val >= sparse_th:
                    sparse.append((uu, vv, val / m))
        if umax < umin or vmax < vmin:
            continue
        min_x, min_z = _sm_m2w(semantic_map, umin, vmin)
        max_x, max_z = _sm_m2w(semantic_map, umax, vmax)
        area_id = f"area::{zn}"
        areas[area_id] = {
            "id": area_id,
            "name": zn,
            "center": {"x": (min_x + max_x) / 2.0, "z": (min_z + max_z) / 2.0},
            "boundary": {
                "min_x": min_x, "max_x": max_x,
                "min_z": min_z, "max_z": max_z,
            }
        }
        semantic_map["zone_heatmaps"]["zones"][zn] = {"max": m, "cells": sparse}


def get_region_priors(obj_type: str) -> List[Tuple[str, float]]:
    d = REGION_PRIORS.get(obj_type, {})
    if not d:
        return []
    # 归一化
    s = sum(d.values())
    if s <= 0:
        return []
    return sorted([(k, v / s) for k, v in d.items()], key=lambda x: -x[1])


def infer_region_for_object(obj_type: str, pos: Tuple[float, float], semantic_map: dict) -> Optional[str]:
    """优先：若已有功能区边界，则直接判断点是否落在某个区域矩形内；否则回退到旧的先验+距离。"""
    areas: Dict[str, dict] = semantic_map.get("areas", {})
    if areas:
        x, z = pos
        for aid, a in areas.items():
            b = a.get("boundary", {})
            min_x, max_x = float(b.get("min_x", 0.0)), float(b.get("max_x", 0.0))
            min_z, max_z = float(b.get("min_z", 0.0)), float(b.get("max_z", 0.0))
            if (min_x <= x <= max_x) and (min_z <= z <= max_z):
                return aid
    # 回退：旧逻辑（先验 / 距离）
    priors = get_region_priors(obj_type)
    if not priors:
        return None
    areas2: Dict[str, dict] = semantic_map.get("areas", {})
    if not areas2:
        return None
    best_id, best_score = None, -1.0
    x, z = pos
    name_to_area = {}
    for aid, a in areas2.items():
        name_to_area.setdefault(a.get("name"), aid)
    for region_name, prior in priors:
        aid = name_to_area.get(region_name)
        if not aid:
            continue
        center = areas2[aid].get("center", {})
        cx, cz = float(center.get("x", 0.0)), float(center.get("z", 0.0))
        dist = math.hypot(x - cx, z - cz)
        score = float(prior) / (1.0 + dist)
        if score > best_score:
            best_score, best_id = score, aid
    return best_id


__all__ = [
    "REGION_PRIORS",
    "REGION_ANCHORS",
    "ZONE_NAMES",
    "CORE_OBJECT_ZONE_SCORES",
    "ensure_semantic_areas",
    "get_region_priors",
    "infer_region_for_object",
]

