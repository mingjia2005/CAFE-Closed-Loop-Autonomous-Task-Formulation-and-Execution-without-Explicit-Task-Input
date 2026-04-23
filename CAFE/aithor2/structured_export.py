# -*- coding: utf-8 -*-
"""
实时对象快照（结构化导出）
将当前帧的对象信息导出为严格的统一格式：
- 名称 name: 对象类型（objectType），并保留原始 AI2-THOR objectId
- 状态 state: isDirty/isOpen/isToggledOn 等
- 可交互性 interact: pickupable/openable/toggleable/receptacle
- 类别 category: 由大模型贴标签；此处预留字段（llm_category）并给出启发式 group_hint
- 位置 position: 3D 坐标与区域（若已推断）

输出文件：semantic_maps/realtime_objects_structured.json
"""
from __future__ import annotations
import json, os, time
from typing import Dict, Any

try:
    import storage_scoring as SS
except Exception:
    SS = None

OUTPUT_PATH = "semantic_maps/realtime_objects_structured.json"


def _get_explored_objects_from_semantic_map(semantic_map: Dict[str, Any]) -> list:
    """从语义地图中获取已探索的对象信息，而不是使用仿真平台的全知信息"""
    explored_objects = []

    try:
        # 从语义地图的objects字段获取已探索的对象
        objects = semantic_map.get("objects", {}) or {}

        for obj_id, obj_info in objects.items():
            # 获取状态信息（从语义地图的state字段）
            state_info = obj_info.get("state", {})



            # 构造类似AI2-THOR event.metadata.objects格式的对象
            obj = {
                "objectId": obj_id,
                "objectType": obj_info.get("type", "Unknown"),
                "position": obj_info.get("position", {}),
                "visible": True,  # 语义地图中的对象都认为是"已知"的

                # 可交互性信息
                "pickupable": state_info.get("pickupable", False),
                "openable": state_info.get("openable", False),
                "receptacle": state_info.get("receptacle", False),
                "toggleable": state_info.get("toggleable", False),
                "dirtyable": state_info.get("dirtyable", False),

                # 状态信息（这是关键！）
                "isDirty": state_info.get("isDirty", False),
                "isOpen": state_info.get("isOpen", False),
                "isToggledOn": state_info.get("isToggledOn", False),
                "isFilledWithLiquid": state_info.get("isFilledWithLiquid", False),
                "isFilled": state_info.get("isFilled", False),
                "isOn": state_info.get("isOn", False),
                "isBroken": state_info.get("isBroken", False),

                # 其他属性
                "temperature": state_info.get("temperature", "RoomTemp"),
            }
            explored_objects.append(obj)



    except Exception as e:
        print(f"⚠ 从语义地图获取对象失败: {e}")
        # 如果语义地图格式不对，返回空列表（严格模式）
        explored_objects = []

    return explored_objects


def _group_hint(t: str) -> str | None:
    try:
        if SS is None:
            return None
        return SS.obj_group(t)
    except Exception:
        return None


def update_structured_realtime_json(event, semantic_map: Dict[str, Any]) -> None:
    """将已探索的对象导出为结构化 JSON（不使用仿真平台的全知信息）。
    每帧可调用；内部会做小幅节流（只写入当内容变化时）。
    注意：event参数保留以兼容现有调用，但不再使用其中的全知信息。
    """
    # 只使用已探索的对象，不使用仿真平台的全知信息
    objs = _get_explored_objects_from_semantic_map(semantic_map) or []
    nodes = []
    areas = semantic_map.get("areas", {}) or {}
    # 反查 areaId -> name
    aid_to_name = {aid: (ainfo.get("name") or aid) for aid, ainfo in areas.items()}

    for o in objs:
        oid = o.get("objectId")
        t = o.get("objectType") or ""
        p = o.get("position") or {}
        state = {
            "isDirty": o.get("isDirty", False),
            "isOpen": o.get("isOpen", False),
            "isToggledOn": o.get("isToggledOn", o.get("isOn", False)),
            "isFilledWithLiquid": o.get("isFilledWithLiquid", False),
        }
        inter = {
            "pickupable": o.get("pickupable", o.get("canPickup", o.get("isPickupable", False))),
            "openable": o.get("openable", o.get("canOpen", False)),
            "toggleable": o.get("toggleable", o.get("canToggle", False)),
            "receptacle": o.get("receptacle", o.get("canFillWithLiquid", False)),
        }
        # 已推断区域（若有）
        rid = None
        try:
            rid = (semantic_map.get("objects", {}).get(oid, {}) or {}).get("regionId")
        except Exception:
            pass
        area_name = aid_to_name.get(rid) if rid else None

        node = {
            "名称": {"type": t, "id": oid},
            "状态": state,
            "可交互性": inter,
            "类别": {
                "llm_category": None,  # 预留由LLM填充
                "group_hint": _group_hint(t),
            },
            "位置": {
                "x": float(p.get("x", 0.0)),
                "y": float(p.get("y", 0.0)),
                "z": float(p.get("z", 0.0)),
                "区域Id": rid,
                "区域": area_name,
            },
        }
        nodes.append(node)

    # 尝试读取轻量化LLM检测到的变化信息
    detected_changes = None
    changes_file = "semantic_maps/detected_changes.json"
    if os.path.exists(changes_file):
        try:
            with open(changes_file, 'r', encoding='utf-8') as f:
                detected_changes = json.load(f)
        except Exception:
            pass

    out = {
        "session_id": semantic_map.get("session_id") or "RealtimeSession",
        "updated_at": time.time(),
        "objects": nodes,
    }

    # 如果有检测到的变化，添加到输出中
    if detected_changes:
        out["detected_changes"] = {
            "score": detected_changes.get("score", 0),
            "reason": detected_changes.get("reason", ""),
            "changes": detected_changes.get("changes", []),
            "detected_at": detected_changes.get("detected_at", time.time())
        }

    try:
        os.makedirs("semantic_maps", exist_ok=True)
        # 只在内容变化时写入（简单比较长度和首对象id）
        write = True
        if os.path.exists(OUTPUT_PATH):
            try:
                with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
                    old = json.load(f)
                if isinstance(old, dict) and isinstance(old.get("objects"), list):
                    if len(old["objects"]) == len(nodes) and nodes:
                        oid0 = nodes[0].get("名称", {}).get("id")
                        oid0_old = (old["objects"][0].get("名称", {}).get("id") if old["objects"] else None)
                        if oid0 == oid0_old:
                            write = True  # 仍然更新时间戳与区域
            except Exception:
                write = True
        if write:
            with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

__all__ = ["update_structured_realtime_json", "OUTPUT_PATH"]

