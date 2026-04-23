"""
探索/语义地图 IO 模块：从 main_with_depth.py 抽离
- 探索进度 JSON 初始化/更新/快照
- 语义地图导出（PNG + JSON）
- 容器标签与重规划触发器加载
"""
from __future__ import annotations
import os
import json
import time
import datetime
import cv2
import numpy as np
from typing import Dict, Any, Optional

# 全局路径常量
EXPLORATION_JSON_PATH = "semantic_maps/exploration_progress.json"
SESSION_TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
# 正式的 realtime 文件（仅在确认触发时由候选文件覆盖/复制）
REALTIME_EXPLORATION_JSON = f"semantic_maps/realtime_exploration_{SESSION_TIMESTAMP}.json"
# 实时更新写入候选文件，避免无触发时覆盖正式文件
REALTIME_CANDIDATE_JSON = "semantic_maps/realtime_exploration_candidate.json"

# 关系推断常量（从主脚本复制）
SURFACE_LIKE = {"Floor", "CounterTop", "TableTop", "Table", "Desk", "Shelf", "ShelvingUnit", "Stool", "Sofa"}
CONTAINER_LIKE = {"Drawer", "Cabinet", "Fridge", "Microwave", "Bowl", "Mug", "Pan", "Pot", "Cup", "Sink", "GarbageCan"}


def init_exploration_json(scene_id: Optional[str] = None) -> Dict[str, Any]:
    """初始化渐进式探索JSON文件"""
    # 如果文件已存在，加载现有数据
    if os.path.exists(EXPLORATION_JSON_PATH):
        try:
            with open(EXPLORATION_JSON_PATH, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
            print(f"📖 加载现有探索数据: {len(existing_data.get('nodes', []))} 个对象")
            return existing_data
        except Exception as e:
            print(f"⚠ 加载现有探索数据失败: {e}")

    # 创建新的探索数据结构
    if scene_id is None:
        ts_id = time.strftime("%Y%m%d_%H%M")
        scene_id = f"FloorPlan_{ts_id}"

    exploration_data = {
        "scene_id": scene_id,
        "exploration_start_time": time.time(),
        "last_update_time": time.time(),
        "nodes": [],
        "edges": [],
        "exploration_stats": {
            "total_objects_discovered": 0,
            "exploration_frames": 0
        }
    }

    # 添加Floor节点（总是存在）
    exploration_data["nodes"].append({
        "id": "Floor|+00.00|+00.00|+00.00",
        "label": "Floor",
        "category": "Surface",
        "attributes": {
            "position_3d": {"x": 0.0, "y": 0.0, "z": 0.0},
            "discovered_frame": 0,
            "discovery_time": time.time()
        }
    })

    # 保存初始文件
    try:
        os.makedirs("semantic_maps", exist_ok=True)
        with open(EXPLORATION_JSON_PATH, 'w', encoding='utf-8') as f:
            json.dump(exploration_data, f, ensure_ascii=False, indent=2)
        print(f"🆕 创建新的探索JSON文件: {EXPLORATION_JSON_PATH}")
    except Exception as e:
        print(f"❌ 创建探索JSON文件失败: {e}")

    return exploration_data


def update_exploration_from_semantic_map(semantic_map: Dict[str, Any], frame_idx: int) -> None:
    """基于semantic_map直接更新JSON文件，处理增加、删除、移动"""
    if not semantic_map.get("objects"):
        return

    # 加载或创建候选 JSON 数据（写入候选文件，避免无触发时覆盖正式 realtime 文件）
    try:
        with open(REALTIME_CANDIDATE_JSON, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except:
        data = {
            "session_id": f"RealtimeSession_{SESSION_TIMESTAMP}",
            "last_update_time": time.time(),
            "nodes": [{"id": "Floor", "label": "Floor", "category": "Surface",
                      "attributes": {"position_3d": {"x": 0.0, "y": 0.0, "z": 0.0}}}],
            "edges": []
        }

    # 当前semantic_map中的对象ID集合
    current_objects = set(semantic_map["objects"].keys())

    # JSON中现有的对象ID集合（排除Floor）
    existing_nodes = {node["attributes"].get("original_id"): node
                     for node in data["nodes"] if node["id"] != "Floor" and "original_id" in node.get("attributes", {})}
    existing_object_ids = set(existing_nodes.keys())

    changes = 0

    # 1. 删除不再存在的对象
    for obj_id in existing_object_ids - current_objects:
        data["nodes"] = [n for n in data["nodes"] if n["attributes"].get("original_id") != obj_id]
        changes += 1

    # 2. 添加新对象或更新现有对象
    for oid, obj_info in semantic_map["objects"].items():
        pos = obj_info["position"]
        obj_type = obj_info["type"]

        if oid in existing_object_ids:
            # 更新现有对象位置（如果变化超过阈值）
            node = existing_nodes[oid]
            old_pos = node["attributes"]["position_3d"]
            distance = ((pos['x'] - old_pos['x'])**2 + (pos['y'] - old_pos['y'])**2 + (pos['z'] - old_pos['z'])**2)**0.5

            if distance > 0.1:  # 10cm阈值
                node["attributes"]["position_3d"] = pos
                changes += 1
            # 同步状态（即使位置未变也可更新状态）
            try:
                node["attributes"]["state"] = semantic_map["objects"].get(oid, {}).get("state", {})
            except Exception:
                pass
        else:
            # 添加新对象
            new_node = {
                "id": f"{obj_type}_{len(data['nodes'])}",
                "label": obj_type,
                "category": "Furniture" if obj_type in SURFACE_LIKE else "Object",
                "attributes": {
                    "position_3d": pos,
                    "discovered_frame": frame_idx,
                    "original_id": oid,  # 这是AI2-THOR的原生对象ID
                    "ai2thor_id": oid,   # 明确标记为AI2-THOR ID
                    "state": semantic_map["objects"].get(oid, {}).get("state", {})
                }
            }
            data["nodes"].append(new_node)
            changes += 1

    # 3. 保存候选文件（仅在有变化时）
    if changes > 0:
        data["last_update_time"] = time.time()
        try:
            os.makedirs("semantic_maps", exist_ok=True)
            with open(REALTIME_CANDIDATE_JSON, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"❌ 候选 JSON 更新失败: {e}")


def promote_candidate_to_realtime() -> bool:
    """将候选的 realtime JSON 推送为正式的 realtime 文件。
    仅在用户/监控决定触发三LLM理解时调用，避免自动覆盖历史文件。"""
    try:
        if not os.path.exists(REALTIME_CANDIDATE_JSON):
            return False
        # 读取候选并写入正式文件名（带会话时间戳）
        with open(REALTIME_CANDIDATE_JSON, 'r', encoding='utf-8') as f:
            data = json.load(f)
        os.makedirs(os.path.dirname(REALTIME_EXPLORATION_JSON), exist_ok=True)
        with open(REALTIME_EXPLORATION_JSON, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"⚠️ 推送候选到正式 realtime 文件失败: {e}")
        return False


def create_exploration_snapshot(event, frame_idx: int, semantic_map: Dict[str, Any], infer_relationships_func) -> Optional[str]:
    """按M键时创建探索快照，保存到JSON文件"""
    # 加载现有数据或创建新数据
    try:
        with open(EXPLORATION_JSON_PATH, 'r', encoding='utf-8') as f:
            exploration_data = json.load(f)
    except Exception:
        exploration_data = init_exploration_json()

    # 获取当前已知对象的标准化ID集合（避免重复）
    existing_standard_ids = set()
    for node in exploration_data["nodes"]:
        existing_standard_ids.add(node["id"])

    # 统计新发现的对象
    new_discoveries = 0

    # 遍历语义地图中的对象（这些都是通过探索发现的）
    for oid, obj_info in semantic_map.get("objects", {}).items():
        # 生成标准化的对象ID
        pos = obj_info["position"]
        standard_id = f"{obj_info['type']}|{pos['x']:+06.2f}|{pos['y']:+06.2f}|{pos['z']:+06.2f}"

        # 只添加新发现的对象（基于标准化ID去重）
        if standard_id not in existing_standard_ids:
            new_discoveries += 1

            # 添加新节点
            new_node = {
                "id": standard_id,
                "label": obj_info["type"],
                "category": "Furniture" if obj_info["type"] in SURFACE_LIKE else "Object",
                "attributes": {
                    "position_3d": pos,
                    "discovered_frame": frame_idx,
                    "discovery_time": time.time(),
                    "original_id": oid
                }
            }

            # 如果有关系信息，添加到属性中
            if "relation" in obj_info:
                new_node["attributes"]["relation"] = obj_info["relation"]

            exploration_data["nodes"].append(new_node)
            existing_standard_ids.add(standard_id)  # 更新集合

    # 更新关系边（基于当前已知对象）
    exploration_data["edges"] = []
    rels = infer_relationships_func(event)
    rel_map = {"on": "ON", "inside": "INSIDE", "under": "UNDER", "near": "NEAR"}

    for node in exploration_data["nodes"]:
        original_id = node["attributes"].get("original_id")
        if original_id and original_id in rels:
            rel_info = rels[original_id]
            support_id = rel_info.get("supportId")

            # 查找支撑对象的标准化ID
            support_standard_id = None
            if support_id == "Floor":
                support_standard_id = "Floor|+00.00|+00.00|+00.00"
            else:
                for support_node in exploration_data["nodes"]:
                    if support_node["attributes"].get("original_id") == support_id:
                        support_standard_id = support_node["id"]
                        break

            if support_standard_id:
                exploration_data["edges"].append({
                    "source": node["id"],
                    "target": support_standard_id,
                    "relationship": rel_map.get(rel_info.get("relation"), "RELATED")
                })

    # 更新统计信息
    exploration_data["last_update_time"] = time.time()
    exploration_data["exploration_stats"]["total_objects_discovered"] = len(exploration_data["nodes"]) - 1  # 减去Floor
    exploration_data["exploration_stats"]["exploration_frames"] = frame_idx

    # 保存更新后的数据
    try:
        with open(EXPLORATION_JSON_PATH, 'w', encoding='utf-8') as f:
            json.dump(exploration_data, f, ensure_ascii=False, indent=2)

        if new_discoveries > 0:
            print(f"📷 探索快照已保存: +{new_discoveries} 个新对象，总计 {len(exploration_data['nodes'])} 个节点")
        else:
            print(f"📸 探索快照已保存: 无新对象，总计 {len(exploration_data['nodes'])} 个节点")

        return EXPLORATION_JSON_PATH
    except Exception as e:
        print(f"❌ 探索快照保存失败: {e}")
        return None


def export_semantic_map(semantic_map_img: np.ndarray, semantic_map: Dict[str, Any], prefix: str = "semmap") -> None:
    """导出语义地图：PNG + JSON（对象抽象信息）"""
    ts = time.strftime("%Y%m%d_%H%M%S")
    json_path = f"semantic_maps/{prefix}_{ts}.json"
    png_path = f"semantic_maps/{prefix}_{ts}.png"

    data = {
        "resolution_m_per_cell": semantic_map["resolution"],
        "bounds": {
            "minX": semantic_map["bounds"][0],
            "maxX": semantic_map["bounds"][1],
            "minZ": semantic_map["bounds"][2],
            "maxZ": semantic_map["bounds"][3],
        },
        "objects": semantic_map["objects"],
    }
    try:
        os.makedirs("semantic_maps", exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        cv2.imwrite(png_path, semantic_map_img)
        print(f"✓ 语义地图已保存: {png_path}, {json_path}")
    except Exception as e:
        print(f"⚠ 语义地图保存失败: {e}")


def load_container_labels(semantic_map: Dict[str, Any], path: str = "semantic_maps/container_labels.json") -> None:
    """从JSON加载柜体/容器的语义标签，写入 semantic_map['container_labels']"""
    try:
        if not os.path.exists(path):
            return
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        out: dict[str, dict] = {}
        for oid, val in (data or {}).items():
            if isinstance(val, dict):
                labels = val.get("labels", [])
                confidence = val.get("confidence", 1.0)
            elif isinstance(val, list):
                labels = val
                confidence = 1.0
            else:
                continue
            if labels:
                out[oid] = {"labels": labels, "confidence": confidence}
        semantic_map["container_labels"] = out
        print(f"🏷️ 已加载容器标签: {len(out)} 个对象")
    except Exception as e:
        print(f"⚠ 加载容器标签失败: {e}")


def load_replan_triggers(semantic_map: Dict[str, Any], path: str = "semantic_maps/replan_triggers.json") -> None:
    """加载二次规划触发器配置（不硬编码）"""
    try:
        cfg = {}
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                cfg = json.load(f) or {}
        semantic_map["replan_triggers"] = cfg
        print(f"🧭 已加载二次规划触发配置: when_discovered={len(cfg.get('when_discovered_objectTypes', []))} 项")
    except Exception as e:
        print(f"⚠ 加载二次规划触发配置失败: {e}")
