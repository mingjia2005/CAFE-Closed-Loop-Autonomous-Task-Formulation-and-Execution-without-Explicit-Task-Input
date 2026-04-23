"""
显示管理模块：从 main_with_depth.py 抽离
- 处理多视图GUI显示
- 管理图像保存
- 处理OpenCV窗口事件
- 支持俯视图（Top-Down View）显示
"""
from __future__ import annotations
import cv2
import numpy as np
import time
import os
from typing import Any, Dict, Optional, Tuple


def display_and_save_images(event, save_image: bool = False, 
                           detection_mode: str = 'gt',
                           save_captures: bool = False,
                           image_counter: int = 0,
                           detect_objects_func=None,
                           detect_objects_from_segmentation_func=None) -> int:
    """
    显示多视图GUI并可选保存图像
    
    Args:
        event: AI2-THOR事件
        save_image: 是否保存图像
        detection_mode: 检测模式 ('gt' 或 'yolo')
        save_captures: 是否启用保存功能
        image_counter: 图像计数器
        detect_objects_func: YOLO检测函数
        detect_objects_from_segmentation_func: GT分割检测函数
    
    Returns:
        int: 更新后的图像计数器
    """
    # 1. 获取图像数据
    rgb_frame = event.frame
    depth_frame = event.depth_frame
    
    if rgb_frame is None or depth_frame is None:
        print("⚠ 警告: RGB或深度图像为空")
        return image_counter
    
    # 转换RGB为BGR用于OpenCV显示
    bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
    
    # 深度图可视化
    depth_normalized = cv2.normalize(depth_frame, None, 0, 255, cv2.NORM_MINMAX)
    depth_colored = cv2.applyColorMap(depth_normalized.astype(np.uint8), cv2.COLORMAP_JET)
    
    # 获取实例分割图像
    instance_frame = event.instance_segmentation_frame
    if instance_frame is None:
        print("⚠ 警告: 分割图像为空")
        instance_colored = np.zeros_like(bgr_frame)
    else:
        instance_colored = cv2.applyColorMap(instance_frame, cv2.COLORMAP_HSV)
    
    # 2. 物体检测：根据 DETECTION_MODE 选择 GT 分割或 YOLO
    if detection_mode == 'gt' and detect_objects_from_segmentation_func:
        det_frame_bgr, detections = detect_objects_from_segmentation_func(event, rgb_frame)
    elif detect_objects_func:
        det_frame_bgr, detections = detect_objects_func(rgb_frame)
    else:
        det_frame_bgr = bgr_frame.copy()
        detections = []
    
    # 3. 组合显示
    height, width = bgr_frame.shape[:2]
    display_width = width // 2
    display_height = height // 2
    
    rgb_small = cv2.resize(bgr_frame, (display_width, display_height))
    depth_small = cv2.resize(depth_colored, (display_width, display_height))
    instance_small = cv2.resize(instance_colored, (display_width, display_height))
    det_small = cv2.resize(det_frame_bgr, (display_width, display_height))
    
    # 组合所有图像
    top_row = np.hstack([rgb_small, depth_small])
    bottom_row = np.hstack([instance_small, det_small])
    combined = np.vstack([top_row, bottom_row])
    
    # 添加标签
    cv2.putText(combined, "RGB", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(combined, "Depth", (display_width + 10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(combined, "Instance Segmentation", (10, display_height + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(combined, f"Detection ({detection_mode})", (display_width + 10, display_height + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # 在右下角添加检测信息
    if detections:
        detection_info = f"Detected: {len(detections)} objects"
        cv2.putText(combined, detection_info, (display_width + 10, display_height + 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # 显示前3个检测结果
        for i, det in enumerate(detections[:3]):
            det_text = f"{det['class']}: {det['confidence']:.2f}"
            cv2.putText(combined, det_text, (display_width + 10, display_height + 75 + i * 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # 显示GUI窗口
    cv2.imshow('AI2-THOR Multi-View', combined)
    
    # 尝试显示第三视角窗口
    try:
        tp_frames = getattr(event, 'third_party_camera_frames', None)
        if tp_frames is None:
            tp_frames = getattr(event, 'third_party_frames', None)
        if tp_frames is None:
            tp_frames = getattr(event, 'third_party_images', None)
        if tp_frames is not None and len(tp_frames) > 0:
            tp0 = tp_frames[0]
            if tp0 is not None:
                tp_bgr = cv2.cvtColor(tp0, cv2.COLOR_RGB2BGR)
                cv2.imshow('Third-Person View', tp_bgr)
    except Exception:
        pass
    
    # 4. 保存图像（如果需要）
    if save_captures and save_image:
        image_counter = _save_all_images(
            bgr_frame, depth_frame, depth_colored, instance_colored, 
            det_frame_bgr, detections, event, detection_mode, image_counter
        )
    
    # 处理OpenCV窗口事件
    cv2.waitKey(1)
    
    return image_counter


def _save_all_images(bgr_frame, depth_frame, depth_colored, instance_colored, 
                    det_frame_bgr, detections, event, detection_mode, image_counter):
    """保存所有类型的图像和数据"""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # 保存RGB图像
    rgb_filename = f"captured_images/rgb_{image_counter:04d}_{timestamp}.jpg"
    cv2.imwrite(rgb_filename, bgr_frame)
    
    # 保存深度图像（原始数据和可视化版本）
    depth_filename = f"captured_depth/depth_{image_counter:04d}_{timestamp}.png"
    depth_vis_filename = f"captured_depth/depth_vis_{image_counter:04d}_{timestamp}.jpg"
    np.save(depth_filename.replace('.png', '.npy'), depth_frame)  # 保存原始深度数据
    cv2.imwrite(depth_vis_filename, depth_colored)
    
    # 保存实例分割图像
    seg_filename = f"captured_segmentation/seg_{image_counter:04d}_{timestamp}.png"
    cv2.imwrite(seg_filename, instance_colored)
    
    # 保存检测结果
    det_filename = f"captured_images/{detection_mode}_detect_{image_counter:04d}_{timestamp}.jpg"
    cv2.imwrite(det_filename, det_frame_bgr)
    
    # 保存第三视角图像（若可用）
    try:
        tp_frames = getattr(event, 'third_party_camera_frames', None)
        if tp_frames is None:
            tp_frames = getattr(event, 'third_party_frames', None)
        if tp_frames is None:
            tp_frames = getattr(event, 'third_party_images', None)
        if tp_frames is not None and len(tp_frames) > 0 and tp_frames[0] is not None:
            tp_bgr = cv2.cvtColor(tp_frames[0], cv2.COLOR_RGB2BGR)
            tp_filename = f"captured_images/third_person_{image_counter:04d}_{timestamp}.jpg"
            cv2.imwrite(tp_filename, tp_bgr)
    except Exception:
        pass
    
    # 保存检测信息到文本文件
    if detections:
        detection_txt = f"captured_images/detections_{image_counter:04d}_{timestamp}.txt"
        with open(detection_txt, 'w') as f:
            f.write(f"Detection Results ({detection_mode}) - {timestamp}\n")
            f.write(f"Total objects detected: {len(detections)}\n\n")
            for i, det in enumerate(detections):
                f.write(f"Object {i+1}:\n")
                f.write(f"  Class: {det['class']}\n")
                f.write(f"  Confidence: {det['confidence']:.3f}\n")
                f.write(f"  Bounding Box: {det['bbox']}\n\n")
    
    print(f"图像已保存: RGB={rgb_filename}, Depth={depth_vis_filename}, Segmentation={seg_filename}, Detection={det_filename}")
    if detections:
        print(f"检测结果已保存: {detection_txt}")
    
    return image_counter + 1


def check_window_closed() -> bool:
    """检查OpenCV窗口是否被关闭"""
    try:
        if cv2.getWindowProperty('AI2-THOR Multi-View', cv2.WND_PROP_VISIBLE) < 1:
            return True
    except:
        # 如果窗口属性检查失败，继续运行
        pass
    return False


def cleanup_display():
    """清理显示资源"""
    cv2.destroyAllWindows()


def display_topdown_view(topdown_image: np.ndarray,
                        event: Any = None,
                        topdown_manager: Any = None,
                        visited_positions: list = None,
                        planned_path: list = None,
                        target_pos: tuple = None) -> None:
    """
    显示俯视图（鸟瞰图）窗口
    
    Args:
        topdown_image: 俯视图RGB图像
        event: AI2-THOR事件对象（用于获取物体信息）
        topdown_manager: TopDownCameraManager实例
        visited_positions: 访问过的位置列表
        planned_path: 规划路径
        target_pos: 目标位置
    """
    if topdown_image is None:
        return
    
    try:
        # 如果有管理器，使用其渲染函数
        if topdown_manager and event:
            agent = event.metadata.get("agent", {})
            agent_pos = agent.get("position", {"x": 0, "y": 0, "z": 0})
            agent_rotation = agent.get("rotation", {}).get("y", 0)
            
            display = topdown_manager.render_topdown_view(
                topdown_image,
                event,
                agent_pos,
                agent_rotation,
                visited_positions=visited_positions,
                planned_path=planned_path,
                target_pos=target_pos
            )
        else:
            # 直接显示，添加基础标题
            display = cv2.cvtColor(topdown_image, cv2.COLOR_RGB2BGR) if len(topdown_image.shape) == 3 and topdown_image.shape[2] == 3 else topdown_image.copy()
            cv2.putText(display, "TopDown View (Bird's Eye View)", (10, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 显示窗口
        if display is not None:
            cv2.imshow('TopDown View - Bird Eye', display)
    except Exception as e:
        print(f"⚠ 显示俯视图失败: {e}")


def display_combined_multi_view(rgb_frame: np.ndarray,
                               depth_frame: np.ndarray,
                               instance_frame: np.ndarray,
                               topdown_frame: np.ndarray,
                               detection_frame: np.ndarray = None,
                               detection_info: dict = None) -> None:
    """
    显示结合了俯视图的多视图GUI
    
    Args:
        rgb_frame: RGB图像
        depth_frame: 深度图像
        instance_frame: 实例分割图像
        topdown_frame: 俯视图
        detection_frame: 检测结果图像
        detection_info: 检测信息
    """
    try:
        # 转换颜色空间
        if rgb_frame is None:
            return
            
        bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR) if len(rgb_frame.shape) == 3 else rgb_frame
        
        # 处理深度图
        if depth_frame is not None:
            depth_normalized = cv2.normalize(depth_frame, None, 0, 255, cv2.NORM_MINMAX)
            depth_colored = cv2.applyColorMap(depth_normalized.astype(np.uint8), cv2.COLORMAP_JET)
        else:
            depth_colored = bgr_frame.copy()
        
        # 处理分割图
        if instance_frame is not None:
            instance_colored = cv2.applyColorMap(instance_frame, cv2.COLORMAP_HSV)
        else:
            instance_colored = bgr_frame.copy()
        
        # 确定检测帧
        if detection_frame is None:
            detection_frame = bgr_frame.copy()
        
        # 调整大小进行网格显示
        h, w = bgr_frame.shape[:2]
        display_w = w // 2
        display_h = h // 2
        
        rgb_small = cv2.resize(bgr_frame, (display_w, display_h))
        depth_small = cv2.resize(depth_colored, (display_w, display_h))
        instance_small = cv2.resize(instance_colored, (display_w, display_h))
        det_small = cv2.resize(detection_frame, (display_w, display_h))
        
        # 如果有俯视图，调整为5视图布局
        if topdown_frame is not None:
            topdown_small = cv2.resize(topdown_frame, (display_w, display_h))
            
            # 2x3 网格：RGB, Depth, Topdown
            #         Instance, Detection, (空)
            top_row = np.hstack([rgb_small, depth_small, topdown_small])
            bottom_row_left = np.hstack([instance_small, det_small])
            bottom_row_right = np.zeros((display_h, display_w, 3), dtype=np.uint8) + 50
            bottom_row = np.hstack([bottom_row_left, bottom_row_right])
            
            combined = np.vstack([top_row, bottom_row])
            
            # 添加标签
            cv2.putText(combined, "RGB", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            cv2.putText(combined, "Depth", (display_w + 10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            cv2.putText(combined, "TopDown", (2*display_w + 10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            cv2.putText(combined, "Instance Seg", (10, display_h + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            cv2.putText(combined, "Detection", (display_w + 10, display_h + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        else:
            # 原始的 2x2 网格
            top_row = np.hstack([rgb_small, depth_small])
            bottom_row = np.hstack([instance_small, det_small])
            combined = np.vstack([top_row, bottom_row])
            
            cv2.putText(combined, "RGB", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            cv2.putText(combined, "Depth", (display_w + 10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            cv2.putText(combined, "Instance Seg", (10, display_h + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            cv2.putText(combined, "Detection", (display_w + 10, display_h + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        
        # 显示检测信息
        if detection_info:
            y_offset = 50
            cv2.putText(combined, f"Detected: {detection_info.get('count', 0)} objects",
                       (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        cv2.imshow('AI2-THOR Multi-View with TopDown', combined)
    except Exception as e:
        print(f"⚠ 显示多视图失败: {e}")
