# -*- coding: utf-8 -*-
"""
场景状态管理器
管理固定的实时状态文件，用于轻量化LLM监控
"""
import json
import os
import time
from typing import Dict, Any, Optional

# 固定的文件路径
CURRENT_STATE_FILE = "semantic_maps/current_scene_state.json"
BASELINE_15S_AGO_FILE = "semantic_maps/baseline_15s_ago.json"

class SceneStateManager:
    """场景状态管理器"""
    
    def __init__(self):
        self.ensure_directories()
    
    def ensure_directories(self):
        """确保目录存在"""
        os.makedirs("semantic_maps", exist_ok=True)
    
    def initialize_on_startup(self):
        """程序启动时初始化 - 清空实时状态文件"""
        try:
            # 清空当前状态文件
            if os.path.exists(CURRENT_STATE_FILE):
                os.remove(CURRENT_STATE_FILE)
                print("🔄 已清空实时状态文件")
            
            # 清空基线文件
            if os.path.exists(BASELINE_15S_AGO_FILE):
                os.remove(BASELINE_15S_AGO_FILE)
                print("🔄 已清空基线状态文件")
                
        except Exception as e:
            print(f"⚠️ 初始化状态文件失败: {e}")
    
    def update_current_state(self, semantic_map: Dict[str, Any]):
        """更新当前状态文件（从语义地图）"""
        try:
            # 从语义地图提取对象信息
            objects = self._extract_objects_from_semantic_map(semantic_map)
            
            # 构建状态数据
            state_data = {
                "session_id": semantic_map.get("session_id", "RealtimeSession"),
                "updated_at": time.time(),
                "objects": objects
            }
            
            # 保存到固定文件
            with open(CURRENT_STATE_FILE, 'w', encoding='utf-8') as f:
                json.dump(state_data, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            print(f"⚠️ 更新当前状态失败: {e}")
    
    def _extract_objects_from_semantic_map(self, semantic_map: Dict[str, Any]) -> list:
        """从语义地图提取对象信息"""
        objects = []
        
        try:
            # 从语义地图的objects字段获取对象
            semantic_objects = semantic_map.get("objects", {}) or {}
            
            for obj_id, obj_info in semantic_objects.items():
                # 获取状态信息
                state_info = obj_info.get("state", {})
                position = obj_info.get("position", {})
                
                # 构造对象数据（使用中文字段名，与现有系统兼容）
                obj_data = {
                    "名称": {
                        "type": obj_info.get("type", "Unknown"),
                        "id": obj_id
                    },
                    "状态": {
                        "isDirty": state_info.get("isDirty", False),
                        "isOpen": state_info.get("isOpen", False),
                        "isToggledOn": state_info.get("isToggledOn", False),
                        "isFilledWithLiquid": state_info.get("isFilledWithLiquid", False),
                    },
                    "可交互性": {
                        "pickupable": state_info.get("pickupable", False),
                        "openable": state_info.get("openable", False),
                        "toggleable": state_info.get("toggleable", False),
                        "receptacle": state_info.get("receptacle", False),
                    },
                    "类别": {
                        "llm_category": None,
                        "group_hint": obj_info.get("group_hint", None),
                    },
                    "位置": {
                        "x": float(position.get("x", 0.0)),
                        "y": float(position.get("y", 0.0)),
                        "z": float(position.get("z", 0.0)),
                        "区域Id": obj_info.get("regionId"),
                        "区域": obj_info.get("region_name"),
                    },
                }
                objects.append(obj_data)
                
        except Exception as e:
            print(f"⚠️ 从语义地图提取对象失败: {e}")
            
        return objects
    
    def save_current_as_baseline(self):
        """将当前状态保存为基线（15秒前的快照）"""
        try:
            if os.path.exists(CURRENT_STATE_FILE):
                # 读取当前状态
                with open(CURRENT_STATE_FILE, 'r', encoding='utf-8') as f:
                    current_data = json.load(f)
                
                # 保存为基线
                with open(BASELINE_15S_AGO_FILE, 'w', encoding='utf-8') as f:
                    json.dump(current_data, f, ensure_ascii=False, indent=2)
                    
                return True
        except Exception as e:
            print(f"⚠️ 保存基线快照失败: {e}")
            return False
    
    def get_current_state(self) -> Optional[Dict[str, Any]]:
        """获取当前状态"""
        try:
            if os.path.exists(CURRENT_STATE_FILE):
                with open(CURRENT_STATE_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"⚠️ 读取当前状态失败: {e}")
        return None
    
    def get_baseline_state(self) -> Optional[Dict[str, Any]]:
        """获取基线状态（15秒前）"""
        try:
            if os.path.exists(BASELINE_15S_AGO_FILE):
                with open(BASELINE_15S_AGO_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"⚠️ 读取基线状态失败: {e}")
        return None
    
    def has_current_state(self) -> bool:
        """检查是否有当前状态文件"""
        return os.path.exists(CURRENT_STATE_FILE)
    
    def has_baseline_state(self) -> bool:
        """检查是否有基线状态文件"""
        return os.path.exists(BASELINE_15S_AGO_FILE)

# 全局实例
scene_state_manager = SceneStateManager()
