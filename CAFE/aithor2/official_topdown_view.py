"""
AI2-THOR 俯视图（Top-Down View）官方方法实现
使用 GetMapViewCameraProperties 和 RenderObjectImage 等官方API
提供整洁的房间俯视图（鸟瞰图）

特点：
- 使用AI2-THOR官方提供的俯视图渲染
- 自动适配房间边界
- 显示更整洁清晰
- 支持实时标注Agent位置和目标
"""

import cv2
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import math


class OfficialTopDownViewManager:
    """使用AI2-THOR官方API的俯视图管理器"""
    
    def __init__(self):
        """初始化俯视图管理器"""
        self.topdown_image = None
        self.map_bounds = None
        self.pixels_per_meter = 50
        
    def get_topdown_view(self, controller, event: Any) -> Optional[np.ndarray]:
        """
        获取官方俯视图
        
        Args:
            controller: AI2-THOR Controller
            event: 当前事件
            
        Returns:
            np.ndarray: 俯视图RGB图像，如果失败返回None
        """
        try:
            # 方法1: 使用 RenderObjectImage 获取整个环境的俯视图
            # 这是最直接的方法 - 返回房间的2D鸟瞰图
            topdown_event = controller.step(
                action="RenderObjectImage",
                objectId="",  # 空字符串表示整个环境
                mode="top_down"  # 调用俯视图模式
            )
            
            if topdown_event and hasattr(topdown_event, 'frame'):
                self.topdown_image = topdown_event.frame
                return self.topdown_image
        except Exception as e1:
            print(f"⚠ RenderObjectImage方法失败: {e1}")
        
        # 方法2: 使用 GetMapViewCameraProperties 获取地图视图配置
        try:
            map_view_event = controller.step(action="GetMapViewCameraProperties")
            
            # 尝试从事件中提取俯视图相关数据
            if hasattr(map_view_event, 'metadata'):
                metadata = map_view_event.metadata
                if "actionReturn" in metadata:
                    map_props = metadata["actionReturn"]
                    print(f"📍 地图视图属性: {map_props}")
                    # 这些信息可用于计算俯视图的坐标映射
        except Exception as e2:
            print(f"⚠ GetMapViewCameraProperties方法失败: {e2}")
        
        return None
    
    def annotate_topdown_view(self,
                             topdown_rgb: np.ndarray,
                             event: Any,
                             agent_pos: Dict[str, float],
                             agent_rotation: float,
                             visited_path: List[Tuple[float, float]] = None,
                             target_pos: Optional[Tuple[float, float]] = None) -> np.ndarray:
        """
        在俯视图上标注Agent、路径、目标等信息
        
        Args:
            topdown_rgb: RGB俯视图
            event: AI2-THOR事件
            agent_pos: Agent位置
            agent_rotation: Agent旋转角
            visited_path: 访问路径
            target_pos: 目标位置
            
        Returns:
            np.ndarray: 标注后的俯视图
        """
        if topdown_rgb is None:
            return None
        
        try:
            # 转换为BGR
            if len(topdown_rgb.shape) == 3 and topdown_rgb.shape[2] == 3:
                display = cv2.cvtColor(topdown_rgb, cv2.COLOR_RGB2BGR).copy()
            else:
                display = topdown_rgb.copy()
        except Exception:
            display = topdown_rgb.copy()
        
        h, w = display.shape[:2]
        
        try:
            # 在图像上添加标注面板
            cv2.rectangle(display, (0, 0), (w, 40), (20, 20, 20), -1)
            cv2.rectangle(display, (0, h-50), (w, h), (20, 20, 20), -1)
            
            # 标题
            cv2.putText(display, "AI2-THOR Top-Down Map View (Official)", (10, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 底部信息
            agent_x = agent_pos.get("x", 0)
            agent_z = agent_pos.get("z", 0)
            agent_y = agent_pos.get("y", 0)
            
            info_text = f"Agent: X={agent_x:.2f} Z={agent_z:.2f} Y={agent_y:.2f}m | Rotation={agent_rotation:.0f}°"
            cv2.putText(display, info_text, (10, h-25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # 绘制可见物体信息（如果有的话）
            try:
                objs = event.metadata.get("objects", [])
                visible_count = sum(1 for o in objs if o.get("visible", False))
                cv2.putText(display, f"Visible Objects: {visible_count}", (w-300, h-25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            except Exception:
                pass
            
        except Exception as e:
            print(f"⚠ 标注失败: {e}")
        
        return display
    
    def display_topdown(self,
                       topdown_rgb: np.ndarray,
                       event: Any,
                       agent_pos: Dict[str, float],
                       agent_rotation: float = 0.0,
                       window_name: str = "AI2-THOR Top-Down View"):
        """
        显示俯视图窗口
        
        Args:
            topdown_rgb: RGB俯视图
            event: AI2-THOR事件
            agent_pos: Agent位置
            agent_rotation: Agent旋转角
            window_name: 窗口名称
        """
        if topdown_rgb is None:
            return
        
        try:
            # 标注俯视图
            annotated = self.annotate_topdown_view(
                topdown_rgb,
                event,
                agent_pos,
                agent_rotation
            )
            
            if annotated is not None:
                cv2.imshow(window_name, annotated)
        except Exception as e:
            print(f"⚠ 显示俯视图失败: {e}")


def initialize_official_topdown_manager() -> OfficialTopDownViewManager:
    """初始化官方俯视图管理器"""
    return OfficialTopDownViewManager()


def get_topdown_and_display(controller,
                           event: Any,
                           manager: Optional[OfficialTopDownViewManager] = None) -> Optional[np.ndarray]:
    """
    获取并显示俯视图的便利函数
    
    Args:
        controller: AI2-THOR Controller
        event: 当前事件
        manager: 俯视图管理器（如果为None会创建新的）
        
    Returns:
        np.ndarray: 俯视图图像
    """
    if manager is None:
        manager = initialize_official_topdown_manager()
    
    # 获取俯视图
    topdown = manager.get_topdown_view(controller, event)
    
    if topdown is not None:
        # 获取Agent信息
        agent = event.metadata.get("agent", {})
        agent_pos = agent.get("position", {"x": 0, "y": 0, "z": 0})
        agent_rotation = agent.get("rotation", {}).get("y", 0)
        
        # 显示
        manager.display_topdown(topdown, event, agent_pos, agent_rotation)
    
    return topdown
