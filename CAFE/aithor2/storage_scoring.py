# -*- coding: utf-8 -*-
"""
Bayesian-like scoring for storage placement.
- LLM provides semantic features; this module scores candidate containers.
- Keep simple, transparent, tunable.

Object features (O): {TypeGroup, Function?, Material?, PreferredZone}
Container features (C): {TypeClass, LocationZone, ObservedContents[TypeGroup...], Openness}

Score(C,O) = Prior(C) * (w1*Compatibility + w2*ContentsMatch + w3*ZoneMatch)
"""
from __future__ import annotations
from typing import Dict, List, Optional

# ---- Type grouping and container classes ----
OBJ_TYPE_TO_GROUP: Dict[str, str] = {
    # Dishware
    "Plate": "Dishware", "Bowl": "Dishware", "Cup": "Dishware", "Mug": "Dishware",
    # Cookware
    "Pot": "Cookware", "Pan": "Cookware",
    # Utensils
    "Knife": "Utensil", "Fork": "Utensil", "Spoon": "Utensil", "Spatula": "Utensil",
    # Cleaning
    "Sponge": "CleaningTool", "SoapBottle": "CleaningTool",
    # Food (cold vs room-temp will be decided by zone/prior)
    "Apple": "Food", "Potato": "Food", "Tomato": "Food", "Bread": "Food",
    "Cereal": "Food", "Egg": "Food", "Lettuce": "Food",
}

def obj_group(obj_type: str) -> str:
    return OBJ_TYPE_TO_GROUP.get(obj_type, "Other")

CONTAINER_TYPE_CLASS: Dict[str, str] = {
    "UpperCabinet": "Cabinet", "LowerCabinet": "Cabinet", "Cabinet": "Cabinet",
    "Drawer": "Drawer", "Dresser": "Drawer",
    "CounterTop": "CounterTop", "Table": "Table", "DiningTable": "Table", "CoffeeTable": "Table",
    "Shelf": "Shelf", "ShelvingUnit": "Shelf",
    "Fridge": "Appliance", "Microwave": "Appliance",
    "Sink": "Sink", "SinkBasin": "Sink",
}

def container_class(t: str) -> str:
    for k, v in CONTAINER_TYPE_CLASS.items():
        if k.lower() in t.lower():
            return v
    return "Other"

# ---- Zone mapping ----
REGION_TO_ZONE: Dict[str, str] = {
    "SinkArea": "Cleaning",
    "StoveArea": "Cooking",
    "DiningArea": "Dining",
    "CountertopArea": "Prep",
    # Storage_* 继承其名称用于标签，可视为对应上方任务区；这里默认中性
}

def region_to_zone(region_name: Optional[str]) -> Optional[str]:
    if not region_name:
        return None
    name = str(region_name)
    low = name.lower()
    # English-style region names
    for k, v in REGION_TO_ZONE.items():
        if low.startswith(k.lower()):
            return v
    if low.startswith("storage_"):
        return "Storage"
    # Chinese aliases -> canonical zones
    if any(s in name for s in ("洗涤", "洗漱", "清洁")):
        return "Cleaning"
    if any(s in name for s in ("切配", "备餐", "处理")):
        return "Prep"
    if "烹饪" in name:
        return "Cooking"
    if any(s in name for s in ("用餐", "餐厅")):
        return "Dining"
    if any(s in name for s in ("收纳", "储存", "碗碟", "炊具")):
        return "Storage"
    if "冷藏" in name:
        return "Storage"  # treat cold storage as storage zone for scoring
    return None

# ---- Prior(C) ----
PRIOR_BY_CLASS: Dict[str, float] = {
    "Cabinet": 0.95,
    "Drawer": 0.80,
    "Shelf": 0.70,
    "CounterTop": 0.55,
    "Table": 0.50,
    "Appliance": 0.30,
    "Sink": 0.10,
    "Other": 0.20,
}

# ---- Compatibility matrix (TypeGroup x ContainerClass) -> [0..1] ----
COMPAT: Dict[str, Dict[str, float]] = {
    "Dishware": {"Cabinet": 1.0, "Drawer": 0.7, "Shelf": 0.8, "CounterTop": 0.5, "Table": 0.4},
    "Cookware": {"Cabinet": 0.9, "Drawer": 0.5, "Shelf": 0.7, "CounterTop": 0.5},
    "Utensil":  {"Drawer": 1.0, "Cabinet": 0.7, "CounterTop": 0.4},
    "CleaningTool": {"Cabinet": 0.8, "Shelf": 0.7, "CounterTop": 0.4},
    "Food":     {"Appliance": 1.0, "Cabinet": 0.6, "Shelf": 0.4, "CounterTop": 0.3, "Table": 0.2},
    "Other":    {"Cabinet": 0.6, "Drawer": 0.5, "Shelf": 0.5, "CounterTop": 0.4, "Table": 0.3},
}

# ---- Weights ----
W1_COMPAT, W2_CONTENTS, W3_ZONE = 0.5, 0.3, 0.2


def prior_score(container_type_class: str, loc_zone: Optional[str]) -> float:
    base = PRIOR_BY_CLASS.get(container_type_class, PRIOR_BY_CLASS["Other"])
    # small boost if in a storage-like or relevant zone
    if loc_zone in ("Dining", "Prep", "Cooking", "Cleaning", "Storage"):
        base = min(1.0, base + 0.05)
    return base


def compatibility_score(obj_group_name: str, container_type_class: str) -> float:
    row = COMPAT.get(obj_group_name, COMPAT["Other"])
    return row.get(container_type_class, 0.3)


def contents_match_score(obj_group_name: str, observed_contents: List[str]) -> float:
    if not observed_contents:
        return 0.5  # neutral when unknown
    same = sum(1 for g in observed_contents if g == obj_group_name)
    return same / float(len(observed_contents))


def zone_match_score(pref_zone: Optional[str], loc_zone: Optional[str]) -> float:
    if not pref_zone or not loc_zone:
        return 0.5
    if pref_zone == loc_zone:
        return 1.0
    # near-related zones
    NEAR = {("Prep", "Cooking"), ("Cooking", "Prep"), ("Cleaning", "Prep"), ("Prep", "Cleaning")}
    if (pref_zone, loc_zone) in NEAR:
        return 0.7
    return 0.2


def score_container(obj_group_name: str, pref_zone: Optional[str], container_type_class: str,
                    loc_zone: Optional[str], observed_contents: List[str]) -> float:
    prior = prior_score(container_type_class, loc_zone)
    compat = compatibility_score(obj_group_name, container_type_class)
    cont = contents_match_score(obj_group_name, observed_contents)
    zm = zone_match_score(pref_zone, loc_zone)
    return prior * (W1_COMPAT * compat + W2_CONTENTS * cont + W3_ZONE * zm)

__all__ = [
    "obj_group", "container_class", "region_to_zone", "score_container",
]

