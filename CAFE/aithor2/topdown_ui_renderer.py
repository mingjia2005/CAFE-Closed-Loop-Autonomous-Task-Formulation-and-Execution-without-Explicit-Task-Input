"""
俯视图实时显示UI - 增强版本
显示房间的鸟瞰图并标注Agent位置、物体、路径等信息
"""

import cv2
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import math


class TopDownUIRenderer:
    """俯视图UI渲染器 - 提供增强的UI标注和显示"""
    
    def __init__(self, scene_bounds: Optional[Dict] = None):
        """
        初始化UI渲染器
        
        Args:
            scene_bounds: 场景边界 {"x_min", "x_max", "z_min", "z_max"}
        """
        self.scene_bounds = scene_bounds or {
            "x_min": -5.0, "x_max": 5.0,
            "z_min": -5.0, "z_max": 5.0
        }
        self.pixels_per_meter = 50  # 调整分辨率
        
    def render_topdown_with_annotations(self,
                                       topdown_rgb: np.ndarray,
                                       event: Any,
                                       agent_pos: Dict[str, float],
                                       agent_rotation: float,
                                       visited_path: List[Tuple[float, float]] = None,
                                       planned_path: List[Tuple[float, float]] = None,
                                       target_pos: Optional[Tuple[float, float]] = None) -> np.ndarray:
        """
        在俯视图上渲染所有标注信息
        
        Args:
            topdown_rgb: RGB俯视图
            event: AI2-THOR事件
            agent_pos: Agent位置
            agent_rotation: Agent旋转角
            visited_path: 访问路径
            planned_path: 规划路径
            target_pos: 目标位置
            
        Returns:
            np.ndarray: 标注后的俯视图(BGR)
        """
        # 转换为BGR
        if topdown_rgb is None:
            return None
            
        try:
            if len(topdown_rgb.shape) == 3 and topdown_rgb.shape[2] == 3:
                display = cv2.cvtColor(topdown_rgb, cv2.COLOR_RGB2BGR).copy()
            else:
                display = topdown_rgb.copy()
        except Exception:
            display = topdown_rgb.copy()
            
        h, w = display.shape[:2]
        center_x = w // 2
        center_y = h // 2
        
        # 绘制背景信息面板
        cv2.rectangle(display, (0, 0), (w, 40), (20, 20, 20), -1)
        cv2.rectangle(display, (0, h-50), (w, h), (20, 20, 20), -1)
        
        # 标题
        cv2.putText(display, "TOPDOWN VIEW: Agent Position, Objects & Paths", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        try:
            # 绘制可见物体
            objs = event.metadata.get("objects", [])
            obj_count = 0
            
            for obj in objs:
                if not obj.get("visible", False):
                    continue
                    
                obj_type = obj.get("objectType", "Unknown")
                pos = obj.get("position", {})
                
                # 相对位置
                rel_x = pos.get("x", 0) - agent_pos.get("x", 0)
                rel_z = pos.get("z", 0) - agent_pos.get("z", 0)
                
                # 旋转坐标（基于Agent方向）
                rad = math.radians(agent_rotation)
                cos_a = math.cos(rad)
                sin_a = math.sin(rad)
                
                img_x = center_x + (rel_x * cos_a - rel_z * sin_a) * self.pixels_per_meter
                img_y = center_y + (rel_x * sin_a + rel_z * cos_a) * self.pixels_per_meter
                
                img_x, img_y = int(img_x), int(img_y)
                
                # 检查范围
                if not (0 <= img_x < w and 0 <= img_y < h):
                    continue
                
                # 物体着色
                color = self._get_object_color(obj_type)
                
                # 绘制物体
                size = 6
                if obj.get("pickupable"):
                    cv2.circle(display, (img_x, img_y), size + 2, color, 2)  # 可拾取的物体用空心圆
                else:
                    cv2.circle(display, (img_x, img_y), size, color, -1)  # 不可拾取的物体用实心圆
                
                # 物体标签（缩写）
                label = obj_type[:4]
                if obj.get("isDirty"):
                    label += "*"
                if obj.get("isOpen"):
                    label += "o"
                    
                cv2.putText(display, label, (img_x + 8, img_y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
                
                obj_count += 1
        except Exception as e:
            print(f"⚠ 绘制物体失败: {e}")
        
        # 绘制访问路径（浅灰色）
        if visited_path and len(visited_path) > 1:
            try:
                points = []
                for x, z in visited_path[-50:]:  # 最近50个点
                    rel_x = x - agent_pos.get("x", 0)
                    rel_z = z - agent_pos.get("z", 0)
                    
                    rad = math.radians(agent_rotation)
                    cos_a = math.cos(rad)
                    sin_a = math.sin(rad)
                    
                    px = center_x + (rel_x * cos_a - rel_z * sin_a) * self.pixels_per_meter
                    py = center_y + (rel_x * sin_a + rel_z * cos_a) * self.pixels_per_meter
                    
                    if 0 <= px < w and 0 <= py < h:
                        points.append((int(px), int(py)))
                
                if len(points) > 1:
                    for i in range(len(points) - 1):
                        cv2.line(display, points[i], points[i+1], (150, 150, 150), 1)
            except Exception:
                pass
        
        # 绘制规划路径（蓝色箭头）
        if planned_path and len(planned_path) > 1:
            try:
                points = []
                for x, z in planned_path:
                    rel_x = x - agent_pos.get("x", 0)
                    rel_z = z - agent_pos.get("z", 0)
                    
                    rad = math.radians(agent_rotation)
                    cos_a = math.cos(rad)
                    sin_a = math.sin(rad)
                    
                    px = center_x + (rel_x * cos_a - rel_z * sin_a) * self.pixels_per_meter
                    py = center_y + (rel_x * sin_a + rel_z * cos_a) * self.pixels_per_meter
                    
                    if 0 <= px < w and 0 <= py < h:
                        points.append((int(px), int(py)))
                
                if len(points) > 1:
                    for i in range(len(points) - 1):
                        cv2.arrowedLine(display, points[i], points[i+1], (255, 100, 0), 2, tipLength=0.1)
            except Exception:
                pass
        
        # 绘制目标位置（红色十字）
        if target_pos:
            try:
                rel_x = target_pos[0] - agent_pos.get("x", 0)
                rel_z = target_pos[1] - agent_pos.get("z", 0)
                
                rad = math.radians(agent_rotation)
                cos_a = math.cos(rad)
                sin_a = math.sin(rad)
                
                px = center_x + (rel_x * cos_a - rel_z * sin_a) * self.pixels_per_meter
                py = center_y + (rel_x * sin_a + rel_z * cos_a) * self.pixels_per_meter
                px, py = int(px), int(py)
                
                if 0 <= px < w and 0 <= py < h:
                    size = 15
                    cv2.line(display, (px - size, py), (px + size, py), (0, 0, 255), 2)
                    cv2.line(display, (px, py - size), (px, py + size), (0, 0, 255), 2)
                    cv2.circle(display, (px, py), size + 2, (0, 0, 255), 1)
            except Exception:
                pass
        
        # 绘制Agent（中心绿色三角形）
        try:
            arrow_len = 25
            rad = math.radians(agent_rotation)
            arrow_x = int(center_x + arrow_len * math.cos(rad))
            arrow_y = int(center_y + arrow_len * math.sin(rad))
            
            # Agent本体
            cv2.circle(display, (center_x, center_y), 8, (0, 255, 0), 2)
            # 朝向箭头
            cv2.arrowedLine(display, (center_x, center_y), (arrow_x, arrow_y), (0, 255, 0), 2, tipLength=0.3)
        except Exception:
            pass
        
        # 底部信息面板
        info_y = h - 45
        agent_x = agent_pos.get("x", 0)
        agent_z = agent_pos.get("z", 0)
        agent_y = agent_pos.get("y", 0)
        
        info_text = f"Agent: X={agent_x:.2f} Z={agent_z:.2f} Y={agent_y:.2f}m | Rotation={agent_rotation:.0f}° | Objects={obj_count}"
        cv2.putText(display, info_text, (10, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # 刻度尺（代表10米）
        scale_px = int(10 * self.pixels_per_meter)
        scale_x_start = w - 120
        scale_y = info_y + 15
        
        cv2.line(display, (scale_x_start, scale_y), (scale_x_start + scale_px, scale_y), (255, 255, 255), 2)
        cv2.putText(display, "10m", (scale_x_start + 2, scale_y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return display
    
    @staticmethod
    def _get_object_color(obj_type: str) -> Tuple[int, int, int]:
        """根据物体类型返回BGR颜色"""
        colors = {
            "Sofa": (255, 200, 0),      # 青
            "Chair": (255, 100, 50),    # 蓝
            "Table": (100, 200, 255),   # 橙
            "Cabinet": (200, 100, 255), # 洋红
            "Drawer": (255, 150, 100),  # 浅蓝
            "Bed": (150, 100, 200),     # 紫
            "Door": (100, 255, 200),    # 青绿
            "Fridge": (100, 100, 255),  # 红
            "Microwave": (100, 150, 255),  # 黄橙
            "Cup": (255, 0, 100),       # 洋红
            "Bowl": (255, 100, 0),      # 蓝绿
            "Plate": (100, 255, 100),   # 绿
            "Knife": (200, 200, 200),   # 灰
            "Spatula": (150, 255, 100), # 亮绿
            "Plant": (100, 255, 100),   # 绿
            "Lamp": (200, 255, 100),    # 亮绿
            "Rug": (150, 150, 100),     # 棕
        }
        
        if obj_type in colors:
            return colors[obj_type]
        
        # 模糊匹配
        for key in colors:
            if key.lower() in obj_type.lower():
                return colors[key]
        
        # 默认白色
        return (180, 180, 180)


def create_topdown_visualization_window(title: str = "TopDown View - Bird's Eye"):
    """创建俯视图可视化窗口"""
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, 800, 800)  # 设置默认大小
    return title


def display_topdown_annotation(topdown_image: np.ndarray,
                               event: Any = None,
                               agent_pos: Dict = None,
                               agent_rotation: float = 0.0,
                               visited_path: List = None,
                               planned_path: List = None,
                               target_pos: Tuple = None,
                               window_name: str = "TopDown View - Bird's Eye"):
    """
    显示带有标注的俯视图
    
    Args:
        topdown_image: 俯视图RGB图像
        event: AI2-THOR事件
        agent_pos: Agent位置
        agent_rotation: Agent旋转
        visited_path: 访问路径
        planned_path: 规划路径
        target_pos: 目标位置
        window_name: 窗口名称
    """
    if topdown_image is None:
        return
    
    try:
        renderer = TopDownUIRenderer()
        
        if event and agent_pos:
            annotated = renderer.render_topdown_with_annotations(
                topdown_image,
                event,
                agent_pos,
                agent_rotation,
                visited_path=visited_path,
                planned_path=planned_path,
                target_pos=target_pos
            )
        else:
            # 降级处理：仅显示原始图像
            if len(topdown_image.shape) == 3 and topdown_image.shape[2] == 3:
                annotated = cv2.cvtColor(topdown_image, cv2.COLOR_RGB2BGR)
            else:
                annotated = topdown_image.copy()
            
            cv2.putText(annotated, "TopDown View", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if annotated is not None:
            cv2.imshow(window_name, annotated)
    except Exception as e:
        print(f"⚠ 俯视图标注显示失败: {e}")
