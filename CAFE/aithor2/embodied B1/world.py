#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
世界模型处理模块
用于加载、验证和管理结构化世界模型数据
"""

import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import re

logger = logging.getLogger(__name__)

class WorldModel:
    """
    世界模型类
    处理具身智能体多模态感知模块的结构化世界模型数据
    """
    
    def __init__(self, world_model_path: str = None):
        """
        初始化世界模型
        
        Args:
            world_model_path (str): 世界模型JSON文件路径
        """
        self.world_model_path = world_model_path
        self.data = {}
        self.validation_errors = []
        self.is_loaded = False
        
        if world_model_path:
            self.load_from_file(world_model_path)
    
    def load_from_file(self, file_path: str) -> bool:
        """
        从JSON文件加载世界模型
        
        Args:
            file_path (str): 文件路径
            
        Returns:
            bool: 加载是否成功
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            
            # 验证数据格式
            if self._validate_structure():
                self.is_loaded = True
                self.world_model_path = file_path
                logger.info(f"✅ 世界模型加载成功: {file_path}")
                return True
            else:
                logger.error(f"❌ 世界模型格式验证失败: {self.validation_errors}")
                return False
                
        except FileNotFoundError:
            logger.error(f"❌ 世界模型文件未找到: {file_path}")
            return False
        except json.JSONDecodeError as e:
            logger.error(f"❌ 世界模型JSON格式错误: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"❌ 加载世界模型时发生错误: {str(e)}")
            return False
    
    def load_from_dict(self, data: Dict[str, Any]) -> bool:
        """
        从字典数据加载世界模型
        
        Args:
            data (Dict[str, Any]): 世界模型数据
            
        Returns:
            bool: 加载是否成功
        """
        try:
            self.data = data.copy()
            
            if self._validate_structure():
                self.is_loaded = True
                logger.info("✅ 世界模型从字典数据加载成功")
                return True
            else:
                logger.error(f"❌ 世界模型格式验证失败: {self.validation_errors}")
                return False
                
        except Exception as e:
            logger.error(f"❌ 从字典加载世界模型时发生错误: {str(e)}")
            return False
    
    def _validate_structure(self) -> bool:
        """
        验证世界模型数据结构
        
        Returns:
            bool: 验证是否通过
        """
        self.validation_errors = []
        
        # 检查必需字段
        required_fields = ["scene_id", "scene_name", "objects"]
        for field in required_fields:
            if field not in self.data:
                self.validation_errors.append(f"缺少必需字段: {field}")
        
        if self.validation_errors:
            return False
        
        # 验证场景信息
        if not self._validate_scene_info():
            return False
        
        # 验证对象列表
        if not self._validate_objects():
            return False
        
        # 验证区域信息（如果存在）
        if "areas" in self.data and not self._validate_areas():
            return False
        
        # 验证声音信息（如果存在）
        if "sounds" in self.data and not self._validate_sounds():
            return False
        
        # 验证关系信息（如果存在）
        if "relationships" in self.data and not self._validate_relationships():
            return False
        
        return len(self.validation_errors) == 0
    
    def _validate_scene_info(self) -> bool:
        """验证场景信息"""
        scene_id = self.data.get("scene_id", "")
        scene_name = self.data.get("scene_name", "")
        
        if not scene_id or not isinstance(scene_id, str):
            self.validation_errors.append("scene_id 必须是非空字符串")
        
        if not scene_name or not isinstance(scene_name, str):
            self.validation_errors.append("scene_name 必须是非空字符串")
        
        return len([e for e in self.validation_errors if "scene" in e]) == 0
    
    def _validate_objects(self) -> bool:
        """验证对象列表"""
        objects = self.data.get("objects", [])
        
        if not isinstance(objects, list):
            self.validation_errors.append("objects 必须是列表")
            return False
        
        if len(objects) == 0:
            logger.warning("⚠️ 世界模型中没有对象")
            return True
        
        for i, obj in enumerate(objects):
            if not self._validate_object(obj, i):
                return False
        
        return True
    
    def _validate_object(self, obj: Dict[str, Any], index: int) -> bool:
        """验证单个对象"""
        # 检查必需字段
        required_fields = ["id", "class_name", "position"]
        for field in required_fields:
            if field not in obj:
                self.validation_errors.append(f"对象 {index} 缺少必需字段: {field}")
                return False
        
        # 验证ID格式
        obj_id = obj.get("id", "")
        if not self._is_valid_id(obj_id):
            self.validation_errors.append(f"对象 {index} ID格式无效: {obj_id}")
            return False
        
        # 验证位置信息
        position = obj.get("position", {})
        if not self._validate_position(position, f"对象 {index}"):
            return False
        
        # 验证属性列表（如果存在）
        if "properties" in obj and not self._validate_properties(obj["properties"], f"对象 {index}"):
            return False
        
        # 验证关联工具（如果存在）
        if "associated_tools" in obj and not self._validate_associated_tools(obj["associated_tools"], f"对象 {index}"):
            return False
        
        return True
    
    def _validate_position(self, position: Dict[str, Any], context: str) -> bool:
        """验证位置信息"""
        required_coords = ["x", "y", "z"]
        for coord in required_coords:
            if coord not in position:
                self.validation_errors.append(f"{context} 位置缺少坐标: {coord}")
                return False
            
            coord_value = position[coord]
            if not isinstance(coord_value, (int, float)):
                self.validation_errors.append(f"{context} 坐标 {coord} 必须是数值: {coord_value}")
                return False
        
        return True
    
    def _validate_properties(self, properties: List[Dict[str, Any]], context: str) -> bool:
        """验证属性列表"""
        if not isinstance(properties, list):
            self.validation_errors.append(f"{context} properties 必须是列表")
            return False
        
        for i, prop in enumerate(properties):
            if not isinstance(prop, dict):
                self.validation_errors.append(f"{context} 属性 {i} 必须是字典")
                return False
            
            if "type" not in prop or "value" not in prop:
                self.validation_errors.append(f"{context} 属性 {i} 缺少必需字段")
                return False
        
        return True
    
    def _validate_associated_tools(self, tools: List[str], context: str) -> bool:
        """验证关联工具列表"""
        if not isinstance(tools, list):
            self.validation_errors.append(f"{context} associated_tools 必须是列表")
            return False
        
        for tool in tools:
            if not isinstance(tool, str):
                self.validation_errors.append(f"{context} 工具必须是字符串: {tool}")
                return False
        
        return True
    
    def _validate_areas(self) -> bool:
        """验证区域信息"""
        areas = self.data.get("areas", [])
        
        if not isinstance(areas, list):
            self.validation_errors.append("areas 必须是列表")
            return False
        
        for i, area in enumerate(areas):
            if not self._validate_area(area, i):
                return False
        
        return True
    
    def _validate_area(self, area: Dict[str, Any], index: int) -> bool:
        """验证单个区域"""
        required_fields = ["id", "name", "boundary"]
        for field in required_fields:
            if field not in area:
                self.validation_errors.append(f"区域 {index} 缺少必需字段: {field}")
                return False
        
        # 验证边界信息
        boundary = area.get("boundary", {})
        required_bounds = ["min_x", "max_x", "min_y", "max_y", "min_z", "max_z"]
        for bound in required_bounds:
            if bound not in boundary:
                self.validation_errors.append(f"区域 {index} 边界缺少: {bound}")
                return False
        
        return True
    
    def _validate_sounds(self) -> bool:
        """验证声音信息"""
        sounds = self.data.get("sounds", [])
        
        if not isinstance(sounds, list):
            self.validation_errors.append("sounds 必须是列表")
            return False
        
        for i, sound in enumerate(sounds):
            if not self._validate_sound(sound, i):
                return False
        
        return True
    
    def _validate_sound(self, sound: Dict[str, Any], index: int) -> bool:
        """验证单个声音"""
        required_fields = ["id", "sound_class", "location"]
        for field in required_fields:
            if field not in sound:
                self.validation_errors.append(f"声音 {index} 缺少必需字段: {field}")
                return False
        
        # 验证位置信息
        location = sound.get("location", {})
        if not self._validate_position(location, f"声音 {index}"):
            return False
        
        return True
    
    def _validate_relationships(self) -> bool:
        """验证关系信息"""
        relationships = self.data.get("relationships", [])
        
        if not isinstance(relationships, list):
            self.validation_errors.append("relationships 必须是列表")
            return False
        
        for i, rel in enumerate(relationships):
            if not self._validate_relationship(rel, i):
                return False
        
        return True
    
    def _validate_relationship(self, rel: Dict[str, Any], index: int) -> bool:
        """验证单个关系"""
        required_fields = ["subject_id", "relationship_type", "object_id"]
        for field in required_fields:
            if field not in rel:
                self.validation_errors.append(f"关系 {index} 缺少必需字段: {field}")
                return False
        
        # 验证ID格式
        subject_id = rel.get("subject_id", "")
        object_id = rel.get("object_id", "")
        
        if not self._is_valid_id(subject_id):
            self.validation_errors.append(f"关系 {index} 主体ID格式无效: {subject_id}")
            return False
        
        if not self._is_valid_id(object_id):
            self.validation_errors.append(f"关系 {index} 客体ID格式无效: {object_id}")
            return False
        
        return True
    
    def _is_valid_id(self, obj_id: str) -> bool:
        """验证ID格式"""
        if not isinstance(obj_id, str):
            return False
        
        # 检查ID格式：字母数字下划线，不能以数字开头
        pattern = r'^[a-zA-Z_][a-zA-Z0-9_]*$'
        return bool(re.match(pattern, obj_id))
    
    def get_data(self) -> Dict[str, Any]:
        """
        获取世界模型数据
        
        Returns:
            Dict[str, Any]: 世界模型数据
        """
        return self.data.copy()
    
    def get_summary(self) -> Dict[str, Any]:
        """
        获取世界模型摘要
        
        Returns:
            Dict[str, Any]: 摘要信息
        """
        if not self.is_loaded:
            return {"error": "世界模型未加载"}
        
        return {
            "scene_id": self.data.get("scene_id", ""),
            "scene_name": self.data.get("scene_name", ""),
            "object_count": len(self.data.get("objects", [])),
            "area_count": len(self.data.get("areas", [])),
            "sound_count": len(self.data.get("sounds", [])),
            "relationship_count": len(self.data.get("relationships", [])),
            "validation_status": "valid" if not self.validation_errors else "invalid",
            "validation_errors": self.validation_errors.copy()
        }
    
    def query_object(self, object_id: str) -> Optional[Dict[str, Any]]:
        """
        查询特定对象
        
        Args:
            object_id (str): 对象ID
            
        Returns:
            Optional[Dict[str, Any]]: 对象信息，如果不存在返回None
        """
        if not self.is_loaded:
            return None
        
        for obj in self.data.get("objects", []):
            if obj.get("id") == object_id:
                return obj.copy()
        
        return None
    
    def query_objects_by_class(self, class_name: str) -> List[Dict[str, Any]]:
        """
        按类别查询对象
        
        Args:
            class_name (str): 类别名称
            
        Returns:
            List[Dict[str, Any]]: 匹配的对象列表
        """
        if not self.is_loaded:
            return []
        
        matching_objects = []
        for obj in self.data.get("objects", []):
            if obj.get("class_name") == class_name:
                matching_objects.append(obj.copy())
        
        return matching_objects
    
    def query_objects_in_area(self, area_id: str) -> List[Dict[str, Any]]:
        """
        查询特定区域内的对象
        
        Args:
            area_id (str): 区域ID
            
        Returns:
            List[Dict[str, Any]]: 区域内的对象列表
        """
        if not self.is_loaded:
            return []
        
        # 获取区域边界
        area = None
        for a in self.data.get("areas", []):
            if a.get("id") == area_id:
                area = a
                break
        
        if not area:
            return []
        
        boundary = area.get("boundary", {})
        
        # 查找区域内的对象
        objects_in_area = []
        for obj in self.data.get("objects", []):
            position = obj.get("position", {})
            if self._is_position_in_boundary(position, boundary):
                objects_in_area.append(obj.copy())
        
        return objects_in_area
    
    def _is_position_in_boundary(self, position: Dict[str, Any], boundary: Dict[str, Any]) -> bool:
        """检查位置是否在边界内"""
        x, y, z = position.get("x", 0), position.get("y", 0), position.get("z", 0)
        
        min_x = boundary.get("min_x", float("-inf"))
        max_x = boundary.get("max_x", float("inf"))
        min_y = boundary.get("min_y", float("-inf"))
        max_y = boundary.get("max_y", float("inf"))
        min_z = boundary.get("min_z", float("-inf"))
        max_z = boundary.get("max_z", float("inf"))
        
        return (min_x <= x <= max_x and 
                min_y <= y <= max_y and 
                min_z <= z <= max_z)
    
    def get_validation_errors(self) -> List[str]:
        """
        获取验证错误列表
        
        Returns:
            List[str]: 验证错误列表
        """
        return self.validation_errors.copy()
    
    def is_valid(self) -> bool:
        """
        检查世界模型是否有效
        
        Returns:
            bool: 是否有效
        """
        return self.is_loaded and len(self.validation_errors) == 0
    
    def save_to_file(self, file_path: str) -> bool:
        """
        保存世界模型到文件
        
        Args:
            file_path (str): 文件路径
            
        Returns:
            bool: 保存是否成功
        """
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"💾 世界模型已保存到: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ 保存世界模型失败: {str(e)}")
            return False


def create_example_world_model() -> Dict[str, Any]:
    """
    创建示例世界模型
    
    Returns:
        Dict[str, Any]: 示例世界模型数据
    """
    return {
        "scene_id": "bathroom_001",
        "scene_name": "主浴室",
        "objects": [
            {
                "id": "toilet_001",
                "class_name": "toilet",
                "position": {"x": 2.0, "y": 0.0, "z": 1.5},
                "properties": [
                    {"type": "material", "value": "ceramic"},
                    {"type": "color", "value": "white"}
                ],
                "state": "clean",
                "is_movable": False,
                "is_interactable": True,
                "associated_tools": ["toilet_brush_001", "toilet_cleaner_001"]
            },
            {
                "id": "sink_001",
                "class_name": "sink",
                "position": {"x": 1.0, "y": 0.0, "z": 0.8},
                "properties": [
                    {"type": "material", "value": "stainless_steel"},
                    {"type": "color", "value": "silver"}
                ],
                "state": "dirty",
                "is_movable": False,
                "is_interactable": True,
                "associated_tools": ["sponge_001", "dish_soap_001"]
            },
            {
                "id": "vacuum_cleaner_001",
                "class_name": "vacuum_cleaner",
                "position": {"x": 0.5, "y": 0.0, "z": 0.0},
                "properties": [
                    {"type": "type", "value": "upright"},
                    {"type": "power", "value": "1200W"}
                ],
                "state": "available",
                "is_movable": True,
                "is_interactable": True,
                "associated_tools": []
            },
            {
                "id": "mop_001",
                "class_name": "mop",
                "position": {"x": 0.0, "y": 0.0, "z": 0.0},
                "properties": [
                    {"type": "type", "value": "sponge_mop"},
                    {"type": "material", "value": "cotton"}
                ],
                "state": "clean",
                "is_movable": True,
                "is_interactable": True,
                "associated_tools": ["bucket_001"]
            }
        ],
        "areas": [
            {
                "id": "floor_area_001",
                "name": "地板区域",
                "boundary": {
                    "min_x": 0.0, "max_x": 3.0,
                    "min_y": 0.0, "max_y": 0.0,
                    "min_z": 0.0, "max_z": 2.0
                }
            },
            {
                "id": "wall_area_001",
                "name": "墙面区域",
                "boundary": {
                    "min_x": 0.0, "max_x": 3.0,
                    "min_y": 0.0, "max_y": 2.5,
                    "min_z": 0.0, "max_z": 2.0
                }
            }
        ],
        "sounds": [
            {
                "id": "water_sound_001",
                "sound_class": "running_water",
                "location": {"x": 1.0, "y": 0.0, "z": 0.8}
            }
        ],
        "relationships": [
            {
                "subject_id": "toilet_001",
                "relationship_type": "near",
                "object_id": "sink_001"
            },
            {
                "subject_id": "vacuum_cleaner_001",
                "relationship_type": "on",
                "object_id": "floor_area_001"
            }
        ]
    }


def main():
    """测试世界模型模块"""
    print("🌍 世界模型模块测试")
    print("=" * 40)
    
    # 创建示例世界模型
    example_data = create_example_world_model()
    
    # 测试从字典加载
    world_model = WorldModel()
    if world_model.load_from_dict(example_data):
        print("✅ 从字典加载世界模型成功")
        
        # 显示摘要
        summary = world_model.get_summary()
        print(f"\n📊 世界模型摘要:")
        for key, value in summary.items():
            if key != "validation_errors":
                print(f"  {key}: {value}")
        
        # 测试查询功能
        print(f"\n🔍 查询测试:")
        toilet = world_model.query_object("toilet_001")
        if toilet:
            print(f"  马桶位置: ({toilet['position']['x']}, {toilet['position']['y']}, {toilet['position']['z']})")
        
        # 测试按类别查询
        tools = world_model.query_objects_by_class("vacuum_cleaner")
        print(f"  吸尘器数量: {len(tools)}")
        
        # 测试区域查询
        floor_objects = world_model.query_objects_in_area("floor_area_001")
        print(f"  地板区域对象数量: {len(floor_objects)}")
        
        # 保存到文件
        if world_model.save_to_file("maps/example_world_model.json"):
            print("\n💾 示例世界模型已保存到 maps/example_world_model.json")
        
    else:
        print("❌ 从字典加载世界模型失败")
        errors = world_model.get_validation_errors()
        for error in errors:
            print(f"  {error}")


if __name__ == "__main__":
    main()



