"""
俯视图（Top-Down/Floorplan View）管理模块
- 使用 AddThirdPartyCamera 添加俯视图相机
- 实时显示俯视图并标注Agent位置、目标、障碍等
- 支持实时可视化房间布局、物体位置、Agent路径
"""
import cv2
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import math


class TopDownCameraManager:
    """管理俯视图相机及显示"""
    
    def __init__(self, field_of_view: float = 90, height_offset: float = 1.5):
        """
        初始化俯视图管理器
        
        Args:
            field_of_view: 相机视野(度)，默认90度
            height_offset: 相机高度偏移(米)，默认1.5米（俯视房间）
        """
        self.field_of_view = field_of_view
        self.height_offset = height_offset
        self.camera_position = {"x": 0, "y": height_offset, "z": 0}
        self.camera_rotation = {"x": 90, "y": 0, "z": 0}  # x=90度朝下，y=0度向北
        self.topdown_image = None
        self.room_bounds = None
        self.scale_px_per_m = 100  # 每米100像素
        
    def get_add_camera_action(self, agent_pos: Dict[str, float]) -> Dict[str, Any]:
        """
        获取AddThirdPartyCamera动作参数，相机始终跟随Agent
        
        Args:
            agent_pos: Agent当前位置 {"x": float, "y": float, "z": float}
            
        Returns:
            dict: 适合 controller.step() 的动作参数
        """
        # 相机跟随Agent，但保持高空俯视
        cam_pos = {
            "x": agent_pos.get("x", 0),
            "y": self.height_offset,  # 始终在高空
            "z": agent_pos.get("z", 0)
        }
        
        action = {
            "action": "AddThirdPartyCamera",
            "position": cam_pos,
            "rotation": self.camera_rotation,
            "fieldOfView": self.field_of_view,
            "ortho": False,  # 透视投影
        }
        return action
    
    def update_topdown_camera_position(self, controller, agent_pos: Dict[str, float]):
        """
        更新俯视图相机位置（跟随Agent）
        
        Args:
            controller: AI2-THOR Controller
            agent_pos: Agent当前位置
        """
        try:
            action = self.get_add_camera_action(agent_pos)
            # 使用 SetThirdPartyCameraProperties 更新现有相机
            controller.step(
                action="SetThirdPartyCameraProperties",
                cameraId=0,  # 第一个第三方相机
                position=action["position"],
                rotation=action["rotation"],
                fieldOfView=action["fieldOfView"]
            )
        except Exception as e:
            # 如果更新失败，可能相机还未创建
            pass
    
    def render_topdown_view(self, 
                           topdown_rgb: np.ndarray,
                           event: Any,
                           agent_pos: Dict[str, float],
                           agent_rotation: float,
                           visited_positions: List[Tuple[float, float]] = None,
                           planned_path: List[Tuple[float, float]] = None,
                           target_pos: Optional[Tuple[float, float]] = None) -> np.ndarray:
        """
        在俯视图上叠加信息：Agent位置、路径、目标等
        
        Args:
            topdown_rgb: 俯视图RGB图像（1280x720或其他分辨率）
            event: AI2-THOR事件对象
            agent_pos: Agent位置
            agent_rotation: Agent的Y轴旋转角度(度)
            visited_positions: 已访问位置列表 [(x, z), ...]
            planned_path: 规划路径 [(x, z), ...]
            target_pos: 目标位置 (x, z)
            
        Returns:
            np.ndarray: 标注后的俯视图(BGR)
        """
        if topdown_rgb is None:
            return None
            
        # 转换为BGR（如果是RGB）
        if len(topdown_rgb.shape) == 3 and topdown_rgb.shape[2] == 3:
            # 假设输入是RGB
            display = cv2.cvtColor(topdown_rgb, cv2.COLOR_RGB2BGR).copy() if topdown_rgb.dtype != np.uint8 else topdown_rgb.copy()
        else:
            display = topdown_rgb.copy()
            
        h, w = display.shape[:2]
        center_x, center_y = w // 2, h // 2
        
        # 获取可见物体信息
        try:
            objs = event.metadata.get("objects", []) if event else []
            
            # 绘制可见物体（圆形标记）
            for obj in objs:
                if not obj.get("visible", False):
                    continue
                    
                obj_type = obj.get("objectType", "Unknown")
                pos = obj.get("position", {})
                
                # 简单的世界坐标到图像坐标映射
                # 假设Agent在中心，相机朝下
                rel_x = pos.get("x", 0) - agent_pos.get("x", 0)
                rel_z = pos.get("z", 0) - agent_pos.get("z", 0)
                
                # 应用旋转（根据Agent方向）
                rad = math.radians(agent_rotation)
                img_x = center_x + rel_x * self.scale_px_per_m * math.cos(rad) - rel_z * self.scale_px_per_m * math.sin(rad)
                img_y = center_y + rel_x * self.scale_px_per_m * math.sin(rad) + rel_z * self.scale_px_per_m * math.cos(rad)
                
                img_x, img_y = int(img_x), int(img_y)
                
                # 检查是否在图像范围内
                if 0 <= img_x < w and 0 <= img_y < h:
                    # 根据物体类型着色
                    color = self._get_object_color(obj_type)
                    cv2.circle(display, (img_x, img_y), 5, color, -1)
                    cv2.putText(display, obj_type[:8], (img_x + 8, img_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        except Exception as e:
            print(f"⚠ 俯视图绘制物体失败: {e}")
        
        # 绘制访问路径（灰色线）
        if visited_positions:
            points = []
            for x, z in visited_positions[-100:]:  # 仅显示最近100个点
                rel_x = x - agent_pos.get("x", 0)
                rel_z = z - agent_pos.get("z", 0)
                rad = math.radians(agent_rotation)
                img_x = center_x + rel_x * self.scale_px_per_m * math.cos(rad) - rel_z * self.scale_px_per_m * math.sin(rad)
                img_y = center_y + rel_x * self.scale_px_per_m * math.sin(rad) + rel_z * self.scale_px_per_m * math.cos(rad)
                if 0 <= img_x < w and 0 <= img_y < h:
                    points.append((int(img_x), int(img_y)))
            
            if len(points) > 1:
                for i in range(len(points) - 1):
                    cv2.line(display, points[i], points[i + 1], (200, 200, 200), 1)
        
        # 绘制规划路径（蓝色虚线）
        if planned_path:
            points = []
            for x, z in planned_path:
                rel_x = x - agent_pos.get("x", 0)
                rel_z = z - agent_pos.get("z", 0)
                rad = math.radians(agent_rotation)
                img_x = center_x + rel_x * self.scale_px_per_m * math.cos(rad) - rel_z * self.scale_px_per_m * math.sin(rad)
                img_y = center_y + rel_x * self.scale_px_per_m * math.sin(rad) + rel_z * self.scale_px_per_m * math.cos(rad)
                if 0 <= img_x < w and 0 <= img_y < h:
                    points.append((int(img_x), int(img_y)))
            
            if len(points) > 1:
                for i in range(len(points) - 1):
                    cv2.line(display, points[i], points[i + 1], (255, 0, 0), 2)
        
        # 绘制目标位置（红色圆形）
        if target_pos:
            rel_x = target_pos[0] - agent_pos.get("x", 0)
            rel_z = target_pos[1] - agent_pos.get("z", 0)
            rad = math.radians(agent_rotation)
            img_x = center_x + rel_x * self.scale_px_per_m * math.cos(rad) - rel_z * self.scale_px_per_m * math.sin(rad)
            img_y = center_y + rel_x * self.scale_px_per_m * math.sin(rad) + rel_z * self.scale_px_per_m * math.cos(rad)
            img_x, img_y = int(img_x), int(img_y)
            
            if 0 <= img_x < w and 0 <= img_y < h:
                cv2.circle(display, (img_x, img_y), 8, (0, 0, 255), 2)
                cv2.circle(display, (img_x, img_y), 3, (0, 0, 255), -1)
        
        # 绘制Agent（绿色三角形表示朝向）
        agent_x, agent_z = center_x, center_y
        arrow_len = 20
        rad = math.radians(agent_rotation)
        arrow_end_x = int(agent_x + arrow_len * math.cos(rad))
        arrow_end_y = int(agent_y + arrow_len * math.sin(rad))
        
        cv2.circle(display, (agent_x, int(agent_y)), 10, (0, 255, 0), 2)
        cv2.arrowedLine(display, (agent_x, int(agent_y)), (arrow_end_x, arrow_end_y), (0, 255, 0), 2, tipLength=0.3)
        
        # 添加标题和刻度
        cv2.putText(display, "TopDown View (Bird's Eye)", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 绘制刻度尺（10米）
        scale_px = int(10 * self.scale_px_per_m)
        cv2.line(display, (w - 80, h - 30), (w - 80 + scale_px, h - 30), (255, 255, 255), 2)
        cv2.putText(display, "10m", (w - 90, h - 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # 显示Agent信息
        agent_y_coord = agent_pos.get("y", 0)
        info_text = f"Agent: ({agent_pos.get('x', 0):.2f}, {agent_pos.get('z', 0):.2f}), Y={agent_y_coord:.2f}m, Rot={agent_rotation:.0f}°"
        cv2.putText(display, info_text, (10, h - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return display
    
    @staticmethod
    def _get_object_color(obj_type: str) -> Tuple[int, int, int]:
        """根据物体类型返回BGR颜色"""
        # 物体类型到BGR颜色的映射
        color_map = {
            "Sofa": (0, 255, 255),  # 黄
            "Table": (255, 165, 0),  # 蓝
            "Chair": (0, 165, 255),  # 橙
            "Lamp": (0, 255, 0),     # 绿
            "Door": (255, 0, 0),     # 蓝
            "Cabinet": (255, 0, 255),# 洋红
            "Bowl": (255, 255, 0),   # 青
            "Cup": (0, 0, 255),      # 红
            "Knife": (0, 128, 255),  # 橙红
            "Spatula": (0, 200, 100),# 草绿
            "Plant": (0, 255, 100),  # 绿
            "Bed": (200, 100, 55),   # 棕
        }
        
        # 精确匹配
        if obj_type in color_map:
            return color_map[obj_type]
        
        # 模糊匹配
        for key in color_map:
            if key.lower() in obj_type.lower():
                return color_map[key]
        
        # 默认灰色
        return (128, 128, 128)


def initialize_topdown_camera(controller, scene_bounds: Optional[Dict] = None) -> TopDownCameraManager:
    """
    初始化俯视图相机
    
    Args:
        controller: AI2-THOR Controller
        scene_bounds: 场景边界信息
        
    Returns:
        TopDownCameraManager: 俯视图管理器实例
    """
    manager = TopDownCameraManager()
    
    # 获取初始Agent位置以添加相机
    try:
        event = controller.last_event
        if event:
            agent_pos = event.metadata.get("agent", {}).get("position", {"x": 0, "y": 0, "z": 0})
            action = manager.get_add_camera_action(agent_pos)
            controller.step(**action)
            print("✓ 俯视图相机已添加")
        else:
            print("⚠ 无法获取初始事件，俯视图相机初始化延迟")
    except Exception as e:
        print(f"⚠ 俯视图相机添加失败: {e}")
    
    return manager
