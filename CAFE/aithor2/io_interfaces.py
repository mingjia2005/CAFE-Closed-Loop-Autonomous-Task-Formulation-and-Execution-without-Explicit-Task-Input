#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
输入/传感器接口（可扩展为实机）

目标：把 RGB / Depth / 相机内参 / 位姿 的数据流抽象出来，
默认使用 AI2-THOR 事件作为数据源，同时预留“外接摄像头 + 深度相机”的接口。

使用方式：
- 在仿真里：调用 thor_event_to_frame(event) 获取 FrameData
- 在实机里：配置 ExternalCameraConfig，并周期性调用 read_external_frame(cfg)

注意：本模块尽量不引入额外依赖，RealSense/ZED 等深度相机的具体实现仅保留占位接口。
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import numpy as np


@dataclass
class FrameData:
    """统一的单帧数据结构"""
    rgb: Optional[np.ndarray] = None          # HxWx3, uint8, RGB
    depth: Optional[np.ndarray] = None        # HxW, float32/uint16, 单位米（建议）
    intrinsics: Optional[Dict[str, float]] = None  # fx, fy, cx, cy（可选）
    pose: Optional[Dict[str, Dict[str, float]]] = None  # {position:{x,y,z}, rotation:{x,y,z}}
    extra: Optional[Dict[str, Any]] = None    # 数据源特定的附加信息（如 AI2-THOR event）


def thor_event_to_frame(event) -> FrameData:
    """将 AI2-THOR 的 event 转为统一的 FrameData。
    仅做字段拷贝，不做重采样/单位换算。
    """
    rgb = getattr(event, 'frame', None)
    depth = getattr(event, 'depth_frame', None)
    agent = (event.metadata or {}).get('agent', {}) if hasattr(event, 'metadata') else {}
    intr = None
    # 如果有相机内参（示例保留占位；AI2-THOR 可从 event.camera之类派生）
    # intr = {"fx": fx, "fy": fy, "cx": cx, "cy": cy}
    pose = {
        "position": {
            "x": float(agent.get('position', {}).get('x', 0.0)),
            "y": float(agent.get('position', {}).get('y', 0.0)),
            "z": float(agent.get('position', {}).get('z', 0.0)),
        },
        "rotation": {
            "x": float(agent.get('rotation', {}).get('x', 0.0)),
            "y": float(agent.get('rotation', {}).get('y', 0.0)),
            "z": float(agent.get('rotation', {}).get('z', 0.0)),
        },
    }
    return FrameData(rgb=rgb, depth=depth, intrinsics=intr, pose=pose, extra={"event": event})


@dataclass
class ExternalCameraConfig:
    """外接摄像头/深度相机配置（占位实现）"""
    rgb_index: int = 0         # OpenCV 摄像头编号，或视频路径（字符串）
    depth_type: str = "realsense"  # realsense/zed/azure_kinect/none
    depth_params: Optional[Dict[str, Any]] = None
    # 可扩展：同步时间戳、硬件触发、校准文件路径等


class _ExternalCameraState:
    def __init__(self):
        self.rgb_cap = None
        self.depth_dev = None


_external_state = _ExternalCameraState()


def open_external(cfg: ExternalCameraConfig) -> bool:
    """打开外部设备（占位）。实际实现可使用 cv2.VideoCapture + pyrealsense2/pyzed 等。
    当前实现为最小占位：仅尝试打开 RGB 相机；深度返回 None。
    """
    try:
        import cv2
        if _external_state.rgb_cap is None:
            _external_state.rgb_cap = cv2.VideoCapture(cfg.rgb_index)
        return True
    except Exception:
        return False


def read_external_frame(cfg: ExternalCameraConfig) -> Optional[FrameData]:
    """读取一帧实机数据（占位）。
    - RGB: 从 OpenCV 摄像头读取并转换为 RGB
    - Depth: 暂未实现（返回 None），后续可接入 RealSense/ZED SDK
    """
    try:
        import cv2
        if _external_state.rgb_cap is None:
            ok = open_external(cfg)
            if not ok:
                return None
        ok, bgr = _external_state.rgb_cap.read()
        if not ok:
            return None
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        # 深度相机占位：此处可接 realsense/zed 的读取逻辑
        depth = None
        return FrameData(rgb=rgb, depth=depth, intrinsics=None, pose=None, extra=None)
    except Exception:
        return None


def close_external():
    try:
        if _external_state.rgb_cap is not None:
            _external_state.rgb_cap.release()
            _external_state.rgb_cap = None
    except Exception:
        pass
    # 深度设备关闭逻辑（略）
    try:
        _external_state.depth_dev = None
    except Exception:
        pass

