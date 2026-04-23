#!/usr/bin/env python3
"""
AI2-THOR 5.0.0 俯视图管理器 - 使用正确的API
基于GetMapViewCameraProperties和AddThirdPartyCamera的官方方法
"""

import cv2
import numpy as np
from typing import Optional, Tuple, Dict, Any


class TopDownViewManager:
    """
    AI2-THOR 5.0俯视图管理器
    
    使用官方API（GetMapViewCameraProperties + AddThirdPartyCamera）
    提供俯视图的获取、注解和显示功能
    
    用法:
        manager = TopDownViewManager()
        
        # 初始化俯视视角（每个场景仅需一次）
        controller.step(action="Pass")
        manager.setup_topdown_camera(controller)
        
        # 获取俯视图像
        event = controller.step(action="MoveAhead")
        topdown_image = manager.get_topdown_image(event)
        
        # 显示俯视图像
        manager.display_topdown(topdown_image, "Room TopDown View")
    """
    
    def __init__(self):
        """初始化俯视图管理器"""
        self.topdown_camera_initialized = False
        self.camera_properties = None
        self.topdown_image = None
    
    def setup_topdown_camera(self, controller) -> Dict[str, Any]:
        """
        设置俯视相机（每个场景仅需调用一次）
        
        Args:
            controller: AI2-THOR Controller实例
            
        Returns:
            dict: 相机属性（位置、旋转、正交大小等）
        """
        try:
            # 1. 先执行一个Pass确保场景已初始化
            event = controller.step(action="Pass")
            
            # 2. 获取MapView相机属性
            mapview_event = controller.step(action="GetMapViewCameraProperties")
            self.camera_properties = mapview_event.metadata.get("actionReturn", {})
            
            if not self.camera_properties:
                raise ValueError("GetMapViewCameraProperties返回空值")
            
            # 3. 使用MapView参数添加第三方俯视相机
            position = self.camera_properties.get('position')
            rotation = self.camera_properties.get('rotation')
            orthographic_size = self.camera_properties.get('orthographicSize', 3.0)
            
            controller.step(
                action="AddThirdPartyCamera",
                position=position,
                rotation=rotation,
                orthographic=True,
                orthographicSize=orthographic_size
            )
            
            self.topdown_camera_initialized = True
            
            return {
                "status": "initialized",
                "position": position,
                "rotation": rotation,
                "orthographicSize": orthographic_size
            }
            
        except Exception as e:
            raise RuntimeError(f"俯视相机设置失败: {e}")
    
    def get_topdown_image(self, event) -> Optional[np.ndarray]:
        """
        从事件中提取俯视图像
        
        Args:
            event: AI2-THOR step()返回的事件对象
            
        Returns:
            numpy.ndarray: 俯视图像（RGB格式），形状为(H, W, 3)
                          如果无法获取则返回None
        """
        try:
            if not hasattr(event, 'third_party_camera_frames'):
                return None
            
            frames = event.third_party_camera_frames
            if frames is None or len(frames) == 0:
                return None
            
            # 第一个第三方相机就是我们的俯视相机
            self.topdown_image = frames[0]
            return self.topdown_image
            
        except Exception as e:
            print(f"❌ 获取俯视图像失败: {e}")
            return None
    
    def annotate_topdown_image(
        self,
        topdown_image: np.ndarray,
        agent_position: Optional[Dict[str, float]] = None,
        agent_rotation: Optional[float] = None,
        target_objects: Optional[list] = None
    ) -> np.ndarray:
        """
        在俯视图上添加标注
        
        Args:
            topdown_image: 原始俯视图像
            agent_position: agent的(x, z)位置 (相对于camera z轴方向)
            agent_rotation: agent的旋转角度（度数）
            target_objects: 目标物体的位置列表
            
        Returns:
            numpy.ndarray: 标注后的图像
        """
        annotated = topdown_image.copy()
        H, W = annotated.shape[:2]
        
        # 绘制中心标记（表示相机的俯视中心）
        center = (W // 2, H // 2)
        cv2.circle(annotated, center, 5, (0, 255, 0), -1)
        cv2.circle(annotated, center, 15, (0, 255, 0), 2)
        
        # 如果有agent位置信息，标记agent
        if agent_position is not None:
            try:
                # 需要转换世界坐标到图像坐标
                px = int(W // 2 + agent_position.get('x', 0) * 50)
                py = int(H // 2 + agent_position.get('z', 0) * 50)
                
                # 确保在图像范围内
                px = max(0, min(W - 1, px))
                py = max(0, min(H - 1, py))
                
                # 绘制agent位置
                cv2.circle(annotated, (px, py), 8, (0, 0, 255), -1)
                cv2.circle(annotated, (px, py), 12, (0, 0, 255), 2)
                
                # 如果有旋转信息，绘制方向箭头
                if agent_rotation is not None:
                    angle_rad = np.radians(agent_rotation)
                    arrow_len = 20
                    end_x = int(px + arrow_len * np.cos(angle_rad))
                    end_y = int(py + arrow_len * np.sin(angle_rad))
                    cv2.arrowedLine(annotated, (px, py), (end_x, end_y), 
                                    (0, 0, 255), 2, tipLength=0.3)
            except Exception as e:
                print(f"⚠️  标注agent位置失败: {e}")
        
        # 标注目标物体
        if target_objects is not None:
            for i, obj in enumerate(target_objects):
                try:
                    ox = int(W // 2 + obj.get('x', 0) * 50)
                    oy = int(H // 2 + obj.get('z', 0) * 50)
                    
                    ox = max(0, min(W - 1, ox))
                    oy = max(0, min(H - 1, oy))
                    
                    # 绘制物体位置
                    cv2.circle(annotated, (ox, oy), 6, (255, 0, 0), -1)
                    
                    # 添加物体标签
                    obj_name = obj.get('name', f'Obj{i}')
                    cv2.putText(annotated, obj_name, (ox + 10, oy - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
                except Exception as e:
                    pass
        
        # 添加文字信息面板
        info_text = [
            "TopDown View",
            f"Size: {H}x{W}",
        ]
        
        y_offset = 25
        for i, text in enumerate(info_text):
            cv2.putText(annotated, text, (10, y_offset + i * 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                       cv2.LINE_AA)
        
        return annotated
    
    def display_topdown(
        self,
        topdown_image: np.ndarray,
        window_name: str = "TopDown View",
        annotate: bool = False
    ) -> bool:
        """
        显示俯视图像
        
        Args:
            topdown_image: 俯视图像
            window_name: 窗口名称
            annotate: 是否添加标注
            
        Returns:
            bool: 是否按下ESC（True = 退出）
        """
        display_image = topdown_image.copy()
        
        if annotate:
            display_image = self.annotate_topdown_image(display_image)
        
        # 将RGB转换为BGR以供OpenCV显示
        display_image_bgr = cv2.cvtColor(display_image, cv2.COLOR_RGB2BGR)
        
        cv2.imshow(window_name, display_image_bgr)
        key = cv2.waitKey(1) & 0xFF
        
        return key == 27  # ESC键
    
    def save_topdown_image(
        self,
        topdown_image: np.ndarray,
        filename: str,
        annotate: bool = True
    ) -> bool:
        """
        保存俯视图像
        
        Args:
            topdown_image: 俯视图像
            filename: 保存文件名
            annotate: 是否保存标注版本
            
        Returns:
            bool: 是否成功保存
        """
        try:
            save_image = topdown_image.copy()
            
            if annotate:
                save_image = self.annotate_topdown_image(save_image)
            
            # OpenCV需要BGR格式
            save_image_bgr = cv2.cvtColor(save_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(filename, save_image_bgr)
            
            return True
        except Exception as e:
            print(f"❌ 保存图像失败: {e}")
            return False


def initialize_topdown_manager(controller) -> TopDownViewManager:
    """
    便利函数：初始化俯视图管理器
    
    Args:
        controller: AI2-THOR Controller实例
        
    Returns:
        TopDownViewManager: 已初始化的管理器
    """
    manager = TopDownViewManager()
    manager.setup_topdown_camera(controller)
    return manager


def get_topdown_and_display(controller, window_name: str = "TopDown") -> np.ndarray:
    """
    便利函数：一行代码获取并显示俯视图
    
    Args:
        controller: AI2-THOR Controller实例
        window_name: 窗口名称
        
    Returns:
        numpy.ndarray: 俯视图像
    """
    manager = TopDownViewManager()
    manager.setup_topdown_camera(controller)
    
    event = controller.step(action="Pass")
    topdown_img = manager.get_topdown_image(event)
    
    if topdown_img is not None:
        manager.display_topdown(topdown_img, window_name)
    
    return topdown_img
