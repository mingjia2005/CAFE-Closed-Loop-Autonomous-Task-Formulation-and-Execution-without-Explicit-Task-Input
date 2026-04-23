#!/usr/bin/env python3
"""
AI2-THOR 多视图实时显示版本 + YOLO检测
- 实时显示RGB、深度图、实例分割图
- 集成YOLO物体检测功能
- GUI窗口显示所有视觉信息
- 支持键盘控制和图像保存
"""

try:
    from pynput import keyboard
    PYNPUT_AVAILABLE = True
except Exception as _pynput_import_error:
    PYNPUT_AVAILABLE = False

    class _FallbackKey:
        esc = "esc"
        page_up = "page_up"
        page_down = "page_down"
        up = "up"
        down = "down"

    class _FallbackListener:
        def __init__(self, on_press=None, on_release=None):
            self.on_press = on_press
            self.on_release = on_release

        def start(self):
            return None

        def stop(self):
            return None

    class _FallbackKeyboard:
        Key = _FallbackKey
        Listener = _FallbackListener

    keyboard = _FallbackKeyboard()
    print(f"⚠ 未检测到 pynput（{_pynput_import_error}），将使用 OpenCV 窗口按键作为降级输入。")

import cv2
import numpy as np
import time
import os
from typing import Dict, Any, List, Tuple, Optional

import json
import math
import random

import hashlib

# 🛠 在导入 ai2thor 之前，清理可能遗留的 commit 相关环境变量，避免绑定到不存在的构建
os.environ.pop("AI2THOR_COMMIT_ID", None)
os.environ.pop("AI2THOR_VERSION", None)

from ai2thor.controller import Controller

# Pillow for Unicode text rendering in OpenCV images
try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False


# 语义先验与区域推断
import semantic_priors as SP


# 收纳评分模型（贝叶斯打分）
import storage_scoring as SS
import structured_export as SEXP
import object_norms as ON



# 直接导入YOLO
from ultralytics import YOLO
from io_interfaces import ExternalCameraConfig, read_external_frame

# 导入点云处理库
import open3d as o3d

import pointcloud_utils as PCU
import exploration_io as EIO
import input_handler as IH
import display_manager as DM
import main_loop as ML
import lightweight_llm_monitor as LLM_MONITOR
import scene_state_manager as SSM
import autonomous_navigation as AN
import viewpoint_navigation as VN  # 新增：视角感知导航
import frontier_fullmap_navigation as FFN
import known_map_navigator as KMN
import pointnav_navigator as PNN

# 俯视图（鸟瞰图）管理 - 使用官方API
from topdown_view import TopDownViewManager

# 全局变量
yolo_model = None

# 检测模式：'gt' 使用仿真器分割；'yolo' 使用YOLO
DETECTION_MODE = 'gt'
# 场景切换：1=当前厨房场景（保持不变），2=卫生间场景（并制造混乱），3=客厅场景（并制造混乱）
SCENE_SWITCH = 1

# 场景预设：仅通过 SCENE_SWITCH 选择，不改其他主流程
_SCENE_PRESETS = {
    1: {'scene': 'FloorPlan10', 'initial_chaos_level': 'off'},
    2: {'scene': 'FloorPlan401', 'initial_chaos_level': 'heavy'},
    3: {'scene': 'FloorPlan201', 'initial_chaos_level': 'heavy'},
}
_active_scene_cfg = _SCENE_PRESETS.get(SCENE_SWITCH, _SCENE_PRESETS[1])
SCENE_NAME = _active_scene_cfg['scene']

# 启动时自动制造混乱（off/light/medium/heavy）
INITIAL_CHAOS_LEVEL = _active_scene_cfg['initial_chaos_level']
# 安全模式：不重定位已有物体，避免物体掉出可见范围/穿模导致“消失”
CHAOS_SAFE_MODE = True

# 输入数据流模式：'thor'（仿真器）或 'external'（外接摄像头/深度相机）
INPUT_MODE = 'thor'

# 命令分解显示回调函数
_command_breakdown_callback = None

# 人工指令学习系统
PREFERENCE_LEARNING_FILE = "semantic_maps/preference_learning.json"
_preference_scores = {}  # 格式: {object_type: {target_type: score}}
_correction_history = []  # 修正事件历史
LEARNING_DECAY_RATE = 0.999  # 遗忘衰减率
LEARNING_THRESHOLD = 3.0  # 学习阈值，超过此分数差异才会改变默认偏好
# 外接相机配置（占位实现，可按需修改）
EXTERNAL_CFG = ExternalCameraConfig(rgb_index=0, depth_type='none')

# 点云可视化相关全局变量
vis = None
pcd_vis = None
# 执行放缓与第三视角配置
SLOW_EXECUTION = True           # 是否放慢执行以便观察
ACTION_DELAY_SEC = 0.1          # 每个动作后的延迟时间（秒）
THIRD_PERSON_ENABLED = False    # 🆕 禁用第三视角相机
THIRD_PERSON_CAMERA_ID = 0      # 第三视角相机ID（首个）
TOPDOWN_CAMERA_ID = 1           # 俯视图相机ID（第二个）

# 俯视图相机配置
TOPDOWN_CAMERA_ENABLED = True   # 启用俯视图相机
topdown_manager = None          # 俯视图管理器（在控制器初始化后创建）

# 热力图显示控制
SHOW_HEATMAPS = True            # 是否显示热力图区域（True=显示全部，False=只显示物体）

# 运行时输出控制：是否保存 captured_* 目录与文件
SAVE_CAPTURES = False  # 默认不保存，避免自动生成 captured_* 目录/文件
CAPTURE_DIRS = ["captured_images", "captured_depth", "captured_segmentation", "captured_pointclouds"]

# 图例滚动控制（鼠标滚轮）
legend_scroll_offset = 0
legend_mouse_callback_set = False
LEGEND_ROW_H = 26

def _sleep_if_slow():
    if SLOW_EXECUTION and ACTION_DELAY_SEC > 0:
        try:
            time.sleep(ACTION_DELAY_SEC)
        except Exception:
            pass

vis_thread = None
vis_running = False

# SLAM建图相关全局变量
global_point_cloud = None  # 全局累积点云
agent_positions = []       # Agent历史位置
current_position = None    # 当前Agent位置
current_rotation = None    # 当前Agent旋转
frame_count = 0           # 帧计数器
mapping_enabled = True    # 是否启用建图功能

# 点云生成函数
# generate_point_cloud, save_point_cloud, visualize_point_cloud 已迁移到 pointcloud_utils.py

def transform_point_cloud_to_global(point_cloud, agent_position, agent_rotation):
    """
    将点云从相机坐标系转换到全局坐标系

    Args:
        point_cloud: 相机坐标系下的点云
        agent_position: Agent位置 {'x': float, 'y': float, 'z': float}
        agent_rotation: Agent旋转 {'x': float, 'y': float, 'z': float}

    Returns:
        o3d.geometry.PointCloud: 全局坐标系下的点云
    """
    # 复制点云
    global_pcd = o3d.geometry.PointCloud(point_cloud)

    # AI2-THOR坐标系转换：
    # 相机坐标系：X右，Y上，Z前
    # 全局坐标系：X右，Y上，Z前

    # 1. 首先绕Y轴旋转（Agent的朝向）
    rotation_y = np.radians(agent_rotation['y'])
    R_y = np.array([
        [np.cos(rotation_y), 0, np.sin(rotation_y)],
        [0, 1, 0],
        [-np.sin(rotation_y), 0, np.cos(rotation_y)]
    ])

    # 应用旋转
    global_pcd.rotate(R_y, center=(0, 0, 0))

    # 应用平移
    translation = np.array([agent_position['x'], agent_position['y'], agent_position['z']])
    global_pcd.translate(translation)

    return global_pcd

def update_global_map(current_pcd, agent_position, agent_rotation):
    """
    更新全局地图

    Args:
        current_pcd: 当前帧的点云
        agent_position: Agent位置
        agent_rotation: Agent旋转
    """
    global global_point_cloud, agent_positions, current_position, current_rotation, frame_count

    # 更新当前位置信息
    current_position = agent_position
    current_rotation = agent_rotation
    frame_count += 1

    # 将当前点云转换到全局坐标系
    global_pcd = transform_point_cloud_to_global(current_pcd, agent_position, agent_rotation)

    # 如果是第一帧，直接设置为全局点云
    if global_point_cloud is None:
        global_point_cloud = global_pcd
        print(f"🗺️ 初始化全局地图，点数: {len(global_point_cloud.points)}")
    else:
        # 合并点云
        global_point_cloud += global_pcd

        # 每50帧进行一次下采样，避免点云过于密集
        if frame_count % 50 == 0:
            global_point_cloud = global_point_cloud.voxel_down_sample(voxel_size=0.02)
            print(f"🔄 地图下采样完成，当前点数: {len(global_point_cloud.points)}")

    # 记录Agent位置
    agent_positions.append({
        'position': agent_position.copy(),
        'rotation': agent_rotation.copy(),
        'frame': frame_count
    })

    # 限制历史位置记录数量
    if len(agent_positions) > 1000:
        agent_positions = agent_positions[-500:]  # 保留最近500个位置
# ===================== 2D 语义地图（基于仿真器）====================
# 不依赖SLAM/ROS，直接使用AI2-THOR的位姿与对象元数据
sem_map_initialized = False
semantic_map = {
    "resolution": 0.10,   # 每个像素表示的米数（可调）
    "bounds": None,       # (minX, maxX, minZ, maxZ)
    "width": None,
    "height": None,
    "objects": {},        # objectId -> {type, position{x,y,z}, frame}
    "agent_path": [],
    "planned_path": [],   # 规划路径：[(x,z), ...]
    "task_points": [],    # 任务目标点：[{x,z,label}]
    "container_labels": {},  # receptacleObjectId -> {labels: [..], confidence?: float}
    "container_contents": {}, # receptacleObjectId -> [TypeGroup, ...]（观测/学习累积）

    # 语义角色与偏好（基础知识 + 探索更新）
    "roles": {            # 角色名称 -> 实际容器objectId列表（探索时填充）
        "cup_cabinet": [],
        "utensil_cabinet": [],
        "bowl_cabinet": [],
        "plate_cabinet": [],
        "pan_pot_cabinet": [],
        "countertop": [],  # 可放置的台面/桌面
    },
    "preferences": {      # 物体类别 -> 期望的角色优先级列表
        "Cup": ["cup_cabinet", "countertop"],
        "Mug": ["cup_cabinet", "countertop"],
        "Bowl": ["bowl_cabinet", "countertop"],
        "Plate": ["plate_cabinet", "countertop"],
        "Knife": ["utensil_cabinet", "countertop"],
        "Spatula": ["utensil_cabinet", "countertop"],
        "Fork": ["utensil_cabinet", "countertop"],
        "Spoon": ["utensil_cabinet", "countertop"],
        "Pan": ["pan_pot_cabinet", "countertop"],
        "Pot": ["pan_pot_cabinet", "countertop"],
    }
}
semantic_map_img = None

# 类别到颜色（BGR）的稳定映射
_type_color_map = {}

def get_type_color(name: str):
    if name in _type_color_map:
        return _type_color_map[name]
    # 基于名称的MD5生成稳定颜色，并避免过暗/过亮
    h = hashlib.md5(name.encode('utf-8')).hexdigest()
    r = int(h[0:2], 16)
    g = int(h[2:4], 16)
    b = int(h[4:6], 16)
    def adjust(v):
        return 60 + (v % 170)  # 60..229
    color = (adjust(b), adjust(g), adjust(r))  # OpenCV使用BGR
    _type_color_map[name] = color
    return color


def draw_unicode_text(img: np.ndarray, text: str, pos: Tuple[int, int], font_size: int = 16, color=(25, 25, 25), font_path: Optional[str] = None):
    """在 numpy(BGR) 图像上使用 PIL 绘制 Unicode 文本。若 Pillow 不可用，回退到 cv2.putText（可能不支持中文）。"""
    x, y = pos
    if not PIL_AVAILABLE:
        # 回退：用 cv2 绘制（仅 ASCII 可靠）
        try:
            cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, max(0.4, font_size / 24), color, 1, cv2.LINE_AA)
        except Exception:
            pass
        return

    try:
        # Convert BGR numpy to PIL Image (RGB)
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)

        # Try to load a font that supports CJK; fall back to default
        font = None
        if font_path:
            try:
                font = ImageFont.truetype(font_path, font_size)
            except Exception:
                font = None
        if font is None:
            # Common font candidates
            candidates = [
                "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
                "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.otf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/usr/share/fonts/truetype/arphic/ukai.ttc",
            ]
            for p in candidates:
                try:
                    font = ImageFont.truetype(p, font_size)
                    break
                except Exception:
                    font = None
            if font is None:
                font = ImageFont.load_default()

        # Draw text (PIL uses top-left origin)
        draw.text((x, y - int(font_size * 0.8)), text, font=font, fill=tuple(color[::-1]))

        # Convert back to BGR numpy and copy into original image
        out = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        img[:, :] = out
    except Exception:
        # 最后回退
        try:
            cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, max(0.4, font_size / 24), color, 1, cv2.LINE_AA)
        except Exception:
            pass


def init_semantic_map(controller, resolution=0.10, margin=0.5):
    """使用可达区域估计XZ范围并初始化2D语义地图网格"""
    global sem_map_initialized, semantic_map
    try:
        ev = controller.step(action="GetReachablePositions")
        positions = ev.metadata.get("actionReturn", []) or []
        if positions:
            xs = [p["x"] for p in positions]
            zs = [p["z"] for p in positions]
            minX, maxX = min(xs) - margin, max(xs) + margin
            minZ, maxZ = min(zs) - margin, max(zs) + margin
        else:
            # 兜底范围
            minX, maxX, minZ, maxZ = -5.0, 5.0, -5.0, 5.0
    except Exception as e:
        print(f"⚠ 语义地图：GetReachablePositions 失败，使用默认范围: {e}")
        minX, maxX, minZ, maxZ = -5.0, 5.0, -5.0, 5.0

    w = int(math.ceil((maxX - minX) / resolution))
    h = int(math.ceil((maxZ - minZ) / resolution))
    w = max(w, 10)
    h = max(h, 10)

    semantic_map.update({
        "resolution": resolution,
        "bounds": (minX, maxX, minZ, maxZ),
        "width": w,
        "height": h,
    })
    sem_map_initialized = True
    os.makedirs("semantic_maps", exist_ok=True)
    print(f"🗺️ 语义地图初始化: size={w}x{h}, res={resolution}m/px, X[{minX:.2f},{maxX:.2f}] Z[{minZ:.2f},{maxZ:.2f}]")

# ---------------- 关系推断（方案C依赖）----------------
SURFACE_LIKE = {"Floor", "CounterTop", "TableTop", "Table", "Desk", "Shelf", "ShelvingUnit", "Stool", "Sofa"}
CONTAINER_LIKE = {"Drawer", "Cabinet", "Fridge", "Microwave", "Bowl", "Mug", "Pan", "Pot", "Cup", "Sink", "GarbageCan"}


def _get_obj_map(event):
    objs = event.metadata.get("objects", [])
    return {o.get("objectId"): o for o in objs}


def _get_aabb(o):
    aabb = o.get("axisAlignedBoundingBox") or {}
    c = aabb.get("center") or {"x": 0.0, "y": 0.0, "z": 0.0}
    s = aabb.get("size") or {"x": 0.0, "y": 0.0, "z": 0.0}
    return c, s


def _xz_overlap(c1, s1, c2, s2, margin=0.02):
    # 轴对齐矩形在XZ的投影是否重叠
    def rng(c, s):
        return (c - s / 2 - margin, c + s / 2 + margin)
    rx1 = rng(c1["x"], s1["x"]) ; rz1 = rng(c1["z"], s1["z"])
    rx2 = rng(c2["x"], s2["x"]) ; rz2 = rng(c2["z"], s2["z"])
    return not (rx1[1] < rx2[0] or rx2[1] < rx1[0] or rz1[1] < rz2[0] or rz2[1] < rz1[0])


def _infer_relation_single(o, obj_map):
    # type name not used directly here
    # 1) 元数据承载信息优先
    if REL_USE_METADATA_RELATIONSHIPS:
        parents = o.get("parentReceptacles") or []
        if parents:
            sid = parents[0]
            sup = obj_map.get(sid)
            stype = (sup.get("objectType") if sup else None) or "Unknown"
            if any(k in stype for k in CONTAINER_LIKE):
                return {"supportId": sid, "supportType": stype, "relation": "inside"}
            return {"supportId": sid, "supportType": stype, "relation": "on"}

    # 2) 几何回退：找最近的表面
    c1, s1 = _get_aabb(o)
    best = None
    for sid, so in obj_map.items():
        if sid == o.get("objectId"): continue
        stype = (so.get("objectType") or so.get("name") or "").title()
        if stype not in SURFACE_LIKE and stype not in CONTAINER_LIKE:
            continue
        c2, s2 = _get_aabb(so)
        # 上表面y
        top2 = c2["y"] + s2["y"] / 2
        bottom2 = c2["y"] - s2["y"] / 2
        # 物体底/顶
        bottom1 = c1["y"] - s1["y"] / 2
        top1 = c1["y"] + s1["y"] / 2
        if _xz_overlap(c1, s1, c2, s2):
            # 距离阈值
            if abs(bottom1 - top2) < 0.06:
                best = (sid, stype, "on"); break
            if top1 < bottom2 - 0.02:
                best = (sid, stype, "under"); break
            if any(k in stype for k in CONTAINER_LIKE):
                if bottom1 > bottom2 and top1 < top2:
                    best = (sid, stype, "inside"); break
        # near（XZ距离）
        dx = abs(c1["x"] - c2["x"]) ; dz = abs(c1["z"] - c2["z"]) ; d = (dx*dx + dz*dz) ** 0.5
        if d < 0.3:
            best = (sid, stype, "near")
    if best:
        sid, stype, rel = best
        return {"supportId": sid, "supportType": stype, "relation": rel}

    # 3) 兜底：on floor/unknown
    y = (o.get("position") or {}).get("y", 0.0)
    if y < 0.12:
        return {"supportId": "Floor", "supportType": "Floor", "relation": "on"}
    return {"supportId": None, "supportType": None, "relation": "unknown"}


def infer_relationships(event):
    obj_map = _get_obj_map(event)
    rels = {}
    for oid, o in obj_map.items():
        rels[oid] = _infer_relation_single(o, obj_map)
    return rels


def world_to_map(x, z):
    """世界坐标(X,Z) -> 语义图像素(u,v)，v轴向上显示（图像坐标原点左上）"""
    b = semantic_map["bounds"]
    res = semantic_map["resolution"]
    u = int((x - b[0]) / res)
    v = int((z - b[2]) / res)
    u = max(0, min(semantic_map["width"] - 1, u))
    v = max(0, min(semantic_map["height"] - 1, v))
    # 翻转v便于显示（上为正）
    return u, semantic_map["height"] - 1 - v


def _update_obstacle_map(event, semantic_map: Dict[str, Any]):
    """更新障碍物地图信息（用于自主导航避障）"""
    try:
        # 初始化障碍物地图
        if "obstacles" not in semantic_map:
            semantic_map["obstacles"] = {}

        # 从不可见对象中提取障碍物
        obstacles = semantic_map["obstacles"]

        # 清空旧的障碍物记录（保留最近100帧的数据）
        current_frame = event.metadata.get("currentFrame", 0)
        obstacles_to_remove = [oid for oid, info in obstacles.items()
                              if current_frame - info.get("last_seen", 0) > 100]
        for oid in obstacles_to_remove:
            del obstacles[oid]

        # 添加不可见的对象作为潜在障碍物
        for obj in event.metadata.get("objects", []):
            oid = obj.get("objectId")
            otype = obj.get("objectType") or obj.get("name") or "Object"

            # 不可见但存在的对象可能是障碍物
            if not obj.get("visible", False) and obj.get("objectType"):
                pos = obj.get("position", {})
                x, z = pos.get("x", 0.0), pos.get("z", 0.0)

                obstacles[oid] = {
                    "type": otype,
                    "position": {"x": x, "z": z},
                    "last_seen": current_frame,
                    "is_obstacle": True,
                }

        # 从深度图检测障碍物（可选，用于更精确的避障）
        if event.depth_frame is not None:
            _detect_obstacles_from_depth(event, semantic_map)

    except Exception as e:
        print(f"⚠ 障碍物地图更新异常: {e}")


def _detect_obstacles_from_depth(event, semantic_map: Dict[str, Any]):
    """从深度图检测障碍物"""
    try:
        depth_frame = event.depth_frame
        if depth_frame is None:
            return

        # 获取Agent位置和方向
        agent = event.metadata.get("agent", {})
        agent_pos = agent.get("position", {"x": 0.0, "z": 0.0})
        agent_rot = agent.get("rotation", {"y": 0.0})

        # 简单的深度阈值检测：距离Agent太近的物体视为障碍物
        depth_threshold = 0.5  # 0.5米内的物体

        # 获取深度图中心区域（Agent前方）
        h, w = depth_frame.shape
        center_region = depth_frame[h//3:2*h//3, w//3:2*w//3]

        # 检测前方障碍物
        if np.any(center_region < depth_threshold):
            if "front_obstacle" not in semantic_map.get("obstacles", {}):
                semantic_map.setdefault("obstacles", {})["front_obstacle"] = {
                    "type": "detected_obstacle",
                    "position": agent_pos,
                    "distance": float(np.min(center_region)),
                    "is_obstacle": True,
                }
    except Exception as e:
        print(f"⚠ 深度图障碍物检测失败: {e}")


def update_semantic_map(event, frame_idx: int):
    """从仿真器元数据更新对象与Agent轨迹，并渲染2D语义图窗口"""
    global SHOW_HEATMAPS
    # 
    #     ( SP )
    try:
        SP.ensure_semantic_areas(event, semantic_map)
    except Exception:
        pass

    global semantic_map_img
    if not sem_map_initialized:
        return

    # 逐帧刷新3-LLM上层任务状态（基于实时环境信息，而非原子动作）
    try:
        _refresh_llm_task_board_live(event)
    except Exception:
        pass

    h, w = semantic_map["height"], semantic_map["width"]
    img = np.full((h, w, 3), 255, dtype=np.uint8)  # 白底

    # 🆕 更新障碍物信息（用于自主导航避障）
    try:
        _update_obstacle_map(event, semantic_map)
    except Exception as e:
        print(f"⚠ 障碍物地图更新失败: {e}")

    # 绘制语义区域（多色虚线框，可关闭）
    area_names_present = set()
    try:
        areas = semantic_map.get("areas", {}) or {}
        render_opts = semantic_map.setdefault("render_options", {})
        show_boxes = bool(render_opts.get("show_area_boxes", False))
        def _draw_dashed_rect(img_, p1, p2, color=(150,150,150), dash=6, gap=6, thickness=1):
            u1, v1 = p1; u2, v2 = p2
            # 顶/底边
            for u in range(min(u1,u2), max(u1,u2), dash+gap):
                cv2.line(img_, (u, v1), (min(u+dash, max(u1,u2)), v1), color, thickness)
                cv2.line(img_, (u, v2), (min(u+dash, max(u1,u2)), v2), color, thickness)
            # 左/右边
            for v in range(min(v1,v2), max(v1,v2), dash+gap):
                cv2.line(img_, (u1, v), (u1, min(v+dash, max(v1,v2))), color, thickness)
                cv2.line(img_, (u2, v), (u2, min(v+dash, max(v1,v2))), color, thickness)
        if show_boxes:
            for aid, a in areas.items():
                b = a.get("boundary", {})
                name = str(a.get("name", "Area"))
                try:
                    u1, v1 = world_to_map(float(b.get("min_x",0.0)), float(b.get("min_z",0.0)))
                    u2, v2 = world_to_map(float(b.get("max_x",0.0)), float(b.get("max_z",0.0)))
                    color = get_type_color(name)
                    _draw_dashed_rect(img, (u1,v1), (u2,v2), color=color)
                    area_names_present.add(name)
                except Exception:
                    continue
    except Exception:
        pass
    # 叠加功能区热力图（半透明覆盖）- 根据 SHOW_HEATMAPS 模式控制
    if SHOW_HEATMAPS:
        try:
            zhm = (semantic_map.get("zone_heatmaps", {}) or {}).get("zones", {})
            if zhm:
                overlay = img.copy()
                for zn, data in zhm.items():
                    color = get_type_color(zn)
                    cells = (data or {}).get("cells", [])
                    for uu, vv, val in cells:
                        if 0 <= uu < w and 0 <= vv < h:
                            # 注意：显示坐标系v轴翻转（world_to_map里做了翻转），这里同样翻转
                            vv_disp = h - 1 - int(vv)
                            alpha = max(0.12, min(0.4, 0.12 + 0.28 * float(val)))
                            b, g, r = color
                            pb, pg, pr = overlay[vv_disp, uu]
                            overlay[vv_disp, uu] = (
                                int((1-alpha)*pb + alpha*b),
                                int((1-alpha)*pg + alpha*g),
                                int((1-alpha)*pr + alpha*r),
                            )
                    try:
                        area_names_present.add(zn)
                    except Exception:
                        pass
                img = overlay
        except Exception:
            pass

        pass


    # Agent 轨迹
    agent = event.metadata.get("agent", {})
    apos = agent.get("position", {"x": 0.0, "z": 0.0})
    ax, az = apos.get("x", 0.0), apos.get("z", 0.0)
    semantic_map["agent_path"].append((ax, az))
    if len(semantic_map["agent_path"]) > 2000:
        semantic_map["agent_path"] = semantic_map["agent_path"][-1000:]
    # 停用基于启发式的“角色归类”，改为标签驱动（由LLM/用户提供标签）
    try:
        pass
    except Exception:
        pass


    # 画轨迹
    for i in range(1, len(semantic_map["agent_path"])):
        u1, v1 = world_to_map(*semantic_map["agent_path"][i - 1])
        u2, v2 = world_to_map(*semantic_map["agent_path"][i])
        cv2.line(img, (u1, v1), (u2, v2), (200, 200, 200), 1)



    # 更新可见对象（可改为不过滤visible，取全量）
    for obj in event.metadata.get("objects", []):
        if not obj.get("visible", False):
            continue
        oid = obj.get("objectId")
        otype = obj.get("objectType") or obj.get("name") or "Object"
        pos = obj.get("position", {})
        x, y, z = pos.get("x", 0.0), pos.get("y", 0.0), pos.get("z", 0.0)
        # 同步关键状态，供LLM发现“脏/干净/可开/已开/可切换/已开启”等
        state = {
            "dirtyable": obj.get("dirtyable", False),
            "isDirty": obj.get("isDirty", False),
            "openable": obj.get("openable", False),
            "isOpen": obj.get("isOpen", False),
            "toggleable": obj.get("toggleable", obj.get("canToggle", False)),
            "isToggledOn": obj.get("isToggledOn", obj.get("isOn", False)),
        }
        is_new = oid not in semantic_map["objects"]
        semantic_map["objects"][oid] = {
            "type": otype,
            "position": {"x": x, "y": y, "z": z},
            "frame": frame_idx,
            "state": state,
        }
        #   SP 
        try:
            rid = SP.infer_region_for_object(str(otype), (float(x), float(z)), semantic_map)
            if rid:
                semantic_map["objects"][oid]["regionId"] = rid
        except Exception:
            pass

        if is_new:
            _progress(f"")
            _on_object_discovered(event, otype, oid)

    # —— 方案C：类型着色 + 高度茎图 + 底座标记（on/inside/under/near） ——
    rels = infer_relationships(event)
    types_present = set()
    # 更新容器内容观测（基于关系 inside）
    try:
        cont_map = semantic_map.setdefault("container_contents", {})
        for _oid, _rel in rels.items():
            if not isinstance(_rel, dict):
                continue
            if _rel.get("relation") == "inside":
                sup = _rel.get("supportId")
                if not sup:
                    continue
                oinfo = semantic_map.get("objects", {}).get(_oid, {})
                otype = oinfo.get("type") or ""
                grp = SS.obj_group(str(otype))
                if grp and grp != "Other":
                    lst = cont_map.setdefault(sup, [])
                    if grp not in lst:
                        lst.append(grp)
    except Exception:
        pass

    for oid, info in semantic_map["objects"].items():
        # 记录出现类别
        types_present.add(info["type"])
        # 写回关系到对象条目，便于导出
        if oid in rels:
            info["relation"] = rels[oid]
        # 坐标
        u, v = world_to_map(info["position"]["x"], info["position"]["z"])
        color = get_type_color(info["type"])  # BGR
        # 位置点
        cv2.circle(img, (u, v), 3, color, -1)
        # 高度茎图（长度按 y 和 resolution 映射，夹在 [2,12] 像素）
        y_m = float(info["position"].get("y", 0.0))
        stem_cells = int(round(y_m / max(semantic_map["resolution"], 1e-6)))
        stem_cells = max(2, min(12, stem_cells))
    # 绘制规划路径（蓝色）
    try:
        if semantic_map.get("planned_path"):
            pts = semantic_map["planned_path"]
            for i in range(1, len(pts)):
                u1, v1 = world_to_map(*pts[i - 1])
                u2, v2 = world_to_map(*pts[i])
                cv2.line(img, (u1, v1), (u2, v2), (255, 0, 0), 1, cv2.LINE_AA)
    except Exception:
        pass

    # 绘制任务目标点（红色 + 标签）
    try:
        for tp in semantic_map.get("task_points", [])[-30:]:
            u, v = world_to_map(float(tp.get("x", 0.0)), float(tp.get("z", 0.0)))
            cv2.circle(img, (u, v), 4, (0, 0, 255), -1)
            label = str(tp.get("label", "target"))
            cv2.putText(img, label, (u + 5, v - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 180), 1, cv2.LINE_AA)
    except Exception:
        pass

        cv2.line(img, (u, v), (u, max(0, v - stem_cells)), color, 1, cv2.LINE_8)
        # 底座标记（关系编码）
        rel = (rels.get(oid) or {}).get("relation", "unknown")
        base_color = {
            "on": (0, 200, 0),        # 绿
            "inside": (255, 0, 0),    # 蓝（BGR）
            "under": (0, 165, 255),   # 橙
            "near": (160, 160, 160),
            "unknown": (200, 200, 200),
        }.get(rel, (200, 200, 200))
        cv2.line(img, (u - 3, min(h - 1, v + 1)), (u + 3, min(h - 1, v + 1)), base_color, 1, cv2.LINE_8)

    # 基于语义地图更新JSON文件
    try:
        EIO.update_exploration_from_semantic_map(semantic_map, frame_idx)
    except Exception:
        pass  # 静默处理错误，不影响主循环
    try:
        SEXP.update_structured_realtime_json(event, semantic_map)
    except Exception:
        pass

    # 更新固定的实时状态文件（用于轻量化监控）
    try:
        SSM.scene_state_manager.update_current_state(semantic_map)
    except Exception:
        pass  # 静默处理错误，不影响主循环

    # 绘制Agent当前点（绿色）
    uA, vA = world_to_map(ax, az)
    cv2.circle(img, (uA, vA), 4, (0, 255, 0), -1)

    # 放大显示尺寸（长边~800px，保留像素风格）
    long_side = max(h, w)
    scale = max(3, int(800 / long_side)) if long_side > 0 else 4
    map_vis = cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)

    # 生成右侧图例面板（颜色-类别对照）
    # 生成右侧面板：左侧 Areas（宽），右侧 Objects（窄），统一字体与行高
    area_w = 420
    obj_w = 260
    panel_w = area_w + obj_w
    panel = np.full((map_vis.shape[0], panel_w, 3), 245, dtype=np.uint8)

    # 字体与排版
    font = cv2.FONT_HERSHEY_SIMPLEX
    title_scale = 0.8
    item_scale = 0.6
    item_thickness = 1
    row_h = LEGEND_ROW_H

    # Prepare sorted lists
    areas_list = sorted(area_names_present)
    types_list = sorted(types_present)

    # Compute content height based on max rows
    content_rows = max(len(areas_list), len(types_list))
    content_height = max(1, content_rows) * row_h + 16

    # 顶部预留3-LLM上层任务进度区
    task_panel_h = 170

    global legend_scroll_offset, legend_mouse_callback_set
    visible_content_h = max(1, panel.shape[0] - task_panel_h - 44)
    max_offset = max(0, content_height - visible_content_h)
    legend_scroll_offset = max(0, min(legend_scroll_offset, max_offset))
    start_y = task_panel_h + 40 - legend_scroll_offset

    # 3-LLM上层任务实时进度区
    task_ui = _llm_task_progress_for_ui(max_items=5)
    total_tasks = int(task_ui.get('total', 0))
    done_tasks = int(task_ui.get('done', 0))
    in_progress_tasks = int(task_ui.get('in_progress', 0))
    progress_ratio = (done_tasks / total_tasks) if total_tasks > 0 else 0.0

    cv2.rectangle(panel, (8, 8), (panel_w - 24, task_panel_h - 12), (236, 239, 242), -1)
    cv2.rectangle(panel, (8, 8), (panel_w - 24, task_panel_h - 12), (170, 170, 170), 1)

    cv2.putText(panel, "3-LLM Macro Task Progress", (16, 30), font, 0.62, (35, 35, 35), 2)
    state_text = str(task_ui.get('state', 'idle'))
    cv2.putText(panel, f"State: {state_text}", (16, 52), font, 0.5, (70, 70, 70), 1)

    summary_text = f"Done {done_tasks}/{total_tasks} | In Progress {in_progress_tasks}"
    cv2.putText(panel, summary_text, (16, 72), font, 0.52, (45, 45, 45), 1)

    bar_x, bar_y = 16, 80
    bar_w, bar_h = panel_w - 64, 16
    cv2.rectangle(panel, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (210, 210, 210), -1)
    fill_w = int(bar_w * max(0.0, min(1.0, progress_ratio)))
    if fill_w > 0:
        cv2.rectangle(panel, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), (60, 170, 80), -1)
    cv2.rectangle(panel, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (130, 130, 130), 1)

    top_tasks = task_ui.get('tasks') or []
    if not top_tasks:
        cv2.putText(panel, "No macro tasks", (16, 122), font, 0.52, (90, 90, 90), 1)
    else:
        for idx, task in enumerate(top_tasks[:3]):
            status = str(task.get('status', '待执行'))
            marker = "[OK]" if status == '已完成' else ("[~]" if status in ('进行中', '执行中', '待执行') else "[?]")
            issue = str(task.get('issue_description') or '')
            short_issue = issue if len(issue) <= 44 else issue[:43] + "…"
            y = 104 + idx * 20
            cv2.putText(panel, f"{marker} {short_issue}", (16, y), font, 0.47, (50, 50, 50), 1)

    # Titles
    cv2.putText(panel, "Areas", (12, task_panel_h + 24), font, title_scale, (50, 50, 50), 2)
    cv2.putText(panel, "Objects", (area_w + 12, task_panel_h + 24), font, title_scale, (50, 50, 50), 2)
    
    # 显示热力图切换模式状态（右上角）
    heatmap_mode_text = f"Heatmap: {'ON' if SHOW_HEATMAPS else 'OFF'}"
    heatmap_color = (0, 200, 0) if SHOW_HEATMAPS else (0, 0, 200)  # 绿=显示，红=隐藏
    cv2.putText(panel, heatmap_mode_text, (panel_w - 180, task_panel_h + 24), font, 0.6, heatmap_color, 2)
    cv2.putText(panel, "(Press 'h' to toggle)", (panel_w - 200, task_panel_h + 45), font, 0.5, (100, 100, 100), 1)

    # Draw rows
    sample_x = 12
    sample_w = 80
    area_text_x = sample_x + sample_w + 12
    obj_text_x = area_w + 44

    for i in range(content_rows):
        y = start_y + i * row_h
        if y < (task_panel_h + 28) or y > panel.shape[0] - 12:
            continue

        # Area column
        if i < len(areas_list):
            nm = areas_list[i]
            col = get_type_color(nm)
            for x in range(sample_x, sample_x + sample_w, 20):
                cv2.line(panel, (x, y), (min(x + 14, sample_x + sample_w), y), col, 3)
            # label
            try:
                _AREA_NAME_EN = {
                    "洗涤子区": "Washing",
                    "切配子区": "Prep",
                    "备餐/混合子区": "Serving/Mix",
                    "烹饪区": "Cooking",
                    "冷藏子区": "Fridge",
                    "室温/干燥子区": "Pantry",
                    "碗碟/炊具收纳区": "Storage",
                }
                label = _AREA_NAME_EN.get(nm, nm)
            except Exception:
                label = nm
            try:
                draw_unicode_text(panel, label, (area_text_x, y + 6), font_size=18, color=(25, 25, 25))
            except Exception:
                cv2.putText(panel, label, (area_text_x, y + 6), font, item_scale, (25, 25, 25), item_thickness, cv2.LINE_AA)

        # Objects column
        if i < len(types_list):
            t = types_list[i]
            c = get_type_color(t)
            # color swatch
            cv2.rectangle(panel, (area_w + 12, y - 12), (area_w + 12 + 24, y + 12), c, -1)
            cv2.rectangle(panel, (area_w + 12, y - 12), (area_w + 12 + 24, y + 12), (30, 30, 30), 1)
            try:
                draw_unicode_text(panel, t, (obj_text_x, y + 6), font_size=18, color=(20, 20, 20))
            except Exception:
                cv2.putText(panel, t, (obj_text_x, y + 6), font, item_scale, (20, 20, 20), item_thickness, cv2.LINE_AA)

    # Draw scrollbar on far right
    try:
        sb_x = panel_w - 18
        sb_w = 10
        sb_top = task_panel_h + 32
        sb_h = panel.shape[0] - sb_top - 16
        cv2.rectangle(panel, (sb_x - sb_w, sb_top), (sb_x, panel.shape[0] - 16), (220, 220, 220), -1)
        if content_height > 0:
            thumb_h = max(24, int(sb_h * min(1.0, max(1.0, visible_content_h / (content_height + 1)))))
            if max_offset > 0:
                thumb_top = int(sb_top + (legend_scroll_offset / max_offset) * (sb_h - thumb_h))
            else:
                thumb_top = sb_top
            cv2.rectangle(panel, (sb_x - sb_w + 2, thumb_top), (sb_x - 2, thumb_top + thumb_h), (180, 180, 180), -1)
    except Exception:
        pass

    # Mouse callback for scrolling
    try:
        if not legend_mouse_callback_set:
            def _panel_mouse_cb(ev, x, y, flags, param):
                global legend_scroll_offset
                if ev == cv2.EVENT_MOUSEWHEEL:
                    if flags > 0:
                        legend_scroll_offset = max(0, legend_scroll_offset - row_h * 3)
                    else:
                        legend_scroll_offset = legend_scroll_offset + row_h * 3

            try:
                cv2.setMouseCallback('Semantic Map 2D', _panel_mouse_cb)
                legend_mouse_callback_set = True
            except Exception:
                legend_mouse_callback_set = True
    except Exception:
        pass

    # 拼接并显示
    semantic_map_img = np.hstack([map_vis, panel])
    cv2.imshow("Semantic Map 2D", semantic_map_img)


# export_semantic_map 已迁移到 exploration_io.py


# ---------------- 场景图（关系图）导出：每帧更新的 JSON ----------------
SCENE_ID = None
# 控制导出的“平台语义”程度（尽量客观直观）
INCLUDE_ROOM_NODE = False           # 关闭房间节点与 Kitchen_1/LivingRoom_1 等标签
REL_USE_METADATA_RELATIONSHIPS = False  # 关系推断不使用 parentReceptacles，只用几何规则
INCLUDE_CATEGORY_AND_SURFACE = False     # 不导出 category/is_surface 等平台语义


def _scene_label(scene_name: str) -> str:
    if not scene_name:
        return "Room"
    try:
        if scene_name.startswith("FloorPlan"):
            num = int(''.join(ch for ch in scene_name if ch.isdigit()))
            if 1 <= num <= 30:
                return "Kitchen"
            if 201 <= num <= 230:
                return "LivingRoom"
            if 301 <= num <= 330:
                return "Bedroom"
            if 401 <= num <= 430:
                return "Bathroom"
    except Exception:
        pass
    return "Room"


def _infer_state_str(o: dict) -> str | None:
    if o.get("isFilled") is True:
        return "full"
    if o.get("isFilled") is False:
        return "empty"
    if o.get("isOn") is True:
        return "on"
    if o.get("isOn") is False:
        return "off"
    if o.get("isOpen") is True:
        return "open"
    if o.get("isOpen") is False:
        return "closed"
    return None


# 渐进式探索JSON文件路径
EXPLORATION_JSON_PATH = "semantic_maps/exploration_progress.json"

# 当前会话的实时探索JSON文件路径
import datetime
SESSION_TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
REALTIME_EXPLORATION_JSON = f"semantic_maps/realtime_exploration_{SESSION_TIMESTAMP}.json"

# init_exploration_json 已迁移到 exploration_io.py

# update_exploration_from_semantic_map 已迁移到 exploration_io.py

# create_exploration_snapshot 已迁移到 exploration_io.py



# init_point_cloud_visualizer 已迁移到 pointcloud_utils.py
# ---------------- 制造“混乱”场景的辅助函数 ----------------

def _get_first_by_type(event, type_name: str):
    objs = event.metadata.get("objects", [])
    for o in objs:
        if (o.get("objectType") or "").lower() == type_name.lower():
            return o
    return None


def _near_agent_floor_pose(event, dx=0.2, dz=0.2):
    agent = event.metadata.get("agent", {})
    pos = agent.get("position", {"x": 0.0, "y": 0.0, "z": 0.0})
    return {"x": float(pos.get("x", 0.0) + dx), "y": 0.05, "z": float(pos.get("z", 0.0) + dz)}


def chaos_drop_apple_on_floor(controller, event):
    """将一个苹果放到地上（若无苹果则尝试创建）。"""
    apple = _get_first_by_type(event, "Apple")
    if apple is None:
        try:
            p = _near_agent_floor_pose(event, dx=0.1, dz=0.1)
            controller.step(action="CreateObject", objectType="Apple", position=p, rotation={"x":0,"y":0,"z":0},
                            forceAction=True)
            event = controller.step(action="Pass")
            apple = _get_first_by_type(event, "Apple")
        except Exception as e:
            print(f"⚠ 创建 Apple 失败: {e}")
            return event
    try:
        p = _near_agent_floor_pose(event, dx=0.0, dz=0.0)
        controller.step(
            action="SetObjectPoses",
            objectPoses=[{
                "objectId": apple["objectId"],
                "position": p,
                "rotation": {"x": 0, "y": 0, "z": 0}
            }],
            placeStationary=True,
            forceAction=True
        )
        print("✓ 已将 Apple 放到地上")
    except Exception as e:
        print(f"⚠ 放置 Apple 失败: {e}")
    return controller.step(action="Pass")


def chaos_tip_chair(controller, event):
    """将一把椅子（或凳子）放倒在地面。
    做法：将 x 轴旋转约 90 度，并把物体中心放到靠近地面的高度。
    """
    chair = _get_first_by_type(event, "Chair") or _get_first_by_type(event, "Stool")
    if chair is None:
        print("⚠ 未找到 Chair/Stool，跳过")
        return event
    try:
        # 使用其 AABB 估计尺寸，辅助确定放倒后的中心高度
        aabb = chair.get("axisAlignedBoundingBox") or {}
        size = aabb.get("size") or {"x": 0.4, "y": 0.8, "z": 0.4}
        # 放倒后，中心略高于地面一些，避免穿插
        p = _near_agent_floor_pose(event, dx=-0.2, dz=0.0)
        p["y"] = max(0.05, 0.5 * min(float(size.get("x",0.4)), float(size.get("z",0.4))))
        controller.step(
            action="SetObjectPoses",
            objectPoses=[{
                "objectId": chair["objectId"],
                "position": p,
                "rotation": {"x": 90, "y": 0, "z": 0}
            }],
            placeStationary=True,
            forceAction=True
        )
        print("✓ 已将 Chair/Stool 放倒")
    except Exception as e:
        print(f"⚠ 放倒椅子失败: {e}")
    return controller.step(action="Pass")

# ---------------- 批量设置碗的脏/净状态 ----------------

def set_all_bowls_dirty(controller, dirty: bool = True):
    """
    遍历当前场景，将所有 Bowl 的脏/净状态设置为指定值。
    - dirty=True  -> DirtyObject（仅对当前非脏的碗执行）
    - dirty=False -> CleanObject（仅对当前为脏的碗执行）
    """
    try:
        ev = getattr(controller, 'last_event', None)
        if ev is None:
            ev = controller.step(action="Pass")
        objs = (ev.metadata or {}).get("objects", [])
        count = 0
        for o in objs:
            if (o.get("objectType") == "Bowl") and o.get("dirtyable", False):
                is_dirty = bool(o.get("isDirty", False))
                oid = o.get("objectId")
                if dirty and not is_dirty:
                    r = controller.step(action="DirtyObject", objectId=oid, forceAction=True)
                    if r.metadata.get("lastActionSuccess", False):
                        count += 1
                elif (not dirty) and is_dirty:
                    r = controller.step(action="CleanObject", objectId=oid, forceAction=True)
                    if r.metadata.get("lastActionSuccess", False):
                        count += 1
        print(f"✓ 已更新 {count} 个碗的脏净状态 -> {'Dirty' if dirty else 'Clean'}")
    except Exception as e:
        print(f"⚠ 设置碗脏净状态失败: {e}")


def apply_initial_chaos(controller, level: str = 'medium'):
    """启动时自动制造混乱，不依赖按键。"""
    level = str(level or 'off').strip().lower()
    if level in ('off', 'none', '0', 'false'):
        print("ℹ️ 已关闭初始化混乱")
        return

    cfg = {
        'light': {'open_n': 2, 'toggle_n': 1, 'dirty_ratio': 0.25},
        'medium': {'open_n': 4, 'toggle_n': 2, 'dirty_ratio': 0.45},
        'heavy': {'open_n': 7, 'toggle_n': 3, 'dirty_ratio': 0.70},
    }
    plan = cfg.get(level, cfg['medium'])

    try:
        event = controller.step(action="Pass")
        objs = event.metadata.get("objects", []) or []

        def _sample_items(items, n):
            if not items:
                return []
            n = max(0, min(int(n), len(items)))
            if n <= 0:
                return []
            return random.sample(items, n)

        openables = [o for o in objs if o.get('openable') and (o.get('objectType') in ('Cabinet', 'Drawer', 'Microwave'))]
        toggles_primary = [o for o in objs if o.get('toggleable') and o.get('objectType') in ('StoveKnob', 'Faucet')]
        toggles_fallback = [o for o in objs if o.get('toggleable') and o.get('objectType') in ('CoffeeMachine', 'Toaster', 'Microwave')]
        dirtyables = [o for o in objs if o.get('dirtyable')]
        opened_count = 0
        for o in _sample_items(openables, plan['open_n']):
            oid = o.get('objectId')
            if not oid or o.get('isOpen'):
                continue
            ev = controller.step(action='OpenObject', objectId=oid, forceAction=True)
            if ev.metadata.get('lastActionSuccess', False):
                opened_count += 1

        toggled_count = 0
        toggle_targets = _sample_items(toggles_primary, min(len(toggles_primary), plan['toggle_n']))
        if len(toggle_targets) < plan['toggle_n']:
            remain = plan['toggle_n'] - len(toggle_targets)
            used_ids = {x.get('objectId') for x in toggle_targets}
            fallback_pool = [x for x in toggles_fallback if x.get('objectId') not in used_ids]
            toggle_targets += _sample_items(fallback_pool, remain)
        for o in toggle_targets:
            oid = o.get('objectId')
            if not oid:
                continue
            ev = controller.step(action='ToggleObjectOn', objectId=oid, forceAction=True)
            if ev.metadata.get('lastActionSuccess', False):
                toggled_count += 1

        dirty_count = 0
        dirty_target_n = int(len(dirtyables) * float(plan['dirty_ratio']))
        for o in _sample_items(dirtyables, dirty_target_n):
            if o.get('isDirty'):
                continue
            oid = o.get('objectId')
            if not oid:
                continue
            ev = controller.step(action='DirtyObject', objectId=oid, forceAction=True)
            if ev.metadata.get('lastActionSuccess', False):
                dirty_count += 1

        scattered_count = 0
        stove_mess_count = 0

        # 安全模式下，不移动已有物体，避免出现“物体消失”问题
        if not CHAOS_SAFE_MODE:
            pass

        print(
            f"🔥 初始化混乱({level})完成: "
            f"opened={opened_count}, toggled_on={toggled_count}, dirty={dirty_count}, "
            f"scattered={scattered_count}, stove_mess={stove_mess_count}, safe_mode={CHAOS_SAFE_MODE}"
        )
    except Exception as e:
        print(f"⚠ 初始化混乱失败: {e}")

# ---------------- LLM 场景理解与“动作执行队列”集成（按 M 触发） ----------------
import threading, importlib.util
import re

# 计划执行相关的全局变量（由主循环消费，避免多线程同时调用 controller.step）
planned_actions: list[dict] = []
# 是否在调用LLM前打印本地候选任务（仅展示，不入列）。默认为False以避免“看起来像已规划”。
LOCAL_TASK_PREVIEW_BEFORE_LLM = False
executing_plan: bool = False
_plan_lock = threading.Lock()
_llm_nav_restore_state: dict | None = None
_llm_task_flow_active: bool = False
_llm_understanding_in_progress: bool = False
_llm_postcheck_done: bool = False
_llm_task_board_lock = threading.Lock()
_llm_task_board = {
    'state': 'idle',
    'generated_tasks': [],
    'verification': [],
    'last_updated': 0.0,
}

# 3-LLM任务执行追踪（每次理解新建一个JSON，执行后回填已执行动作）
_llm_trace_lock = threading.Lock()
_llm_trace_file_path: str | None = None
_llm_trace_data: dict | None = None
_llm_last_started_step_ref = None


def _write_llm_trace_file_locked():
    """在已持有 _llm_trace_lock 时写盘。"""
    global _llm_trace_file_path, _llm_trace_data
    if not _llm_trace_file_path or not isinstance(_llm_trace_data, dict):
        return
    try:
        os.makedirs('semantic_maps', exist_ok=True)
        with open(_llm_trace_file_path, 'w', encoding='utf-8') as f:
            json.dump(_llm_trace_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"⚠ 写入3-LLM任务追踪文件失败: {e}")


def _init_llm_execution_trace(results: dict, actions: list[dict]):
    """每次3-LLM理解后，创建新的任务追踪JSON文件。"""
    global _llm_trace_file_path, _llm_trace_data, _llm_last_started_step_ref
    ts = time.strftime('%Y%m%d_%H%M%S')
    file_path = os.path.join('semantic_maps', f'three_llm_task_trace_{ts}.json')
    trace_data = {
        'trace_id': f'three_llm_task_trace_{ts}',
        'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'state': 'planned',
        'source_results_summary': {
            'has_detailed_steps': bool((results or {}).get('detailed_steps')),
            'has_macro_tasks': bool((results or {}).get('llma_macro_tasks_raw') or (results or {}).get('candidate_macro_tasks')),
        },
        'planned_actions': [dict(a) for a in (actions or [])],
        'executed_actions': [],
        'finished_at': None,
    }

    with _llm_trace_lock:
        _llm_trace_file_path = file_path
        _llm_trace_data = trace_data
        _llm_last_started_step_ref = None
        _write_llm_trace_file_locked()

    print(f"🧾 已创建3-LLM任务追踪文件: {file_path}")


def _record_llm_executed_step_started(step: dict):
    """记录已开始执行的步骤（用于执行完成后写回JSON）。"""
    global _llm_last_started_step_ref
    if not isinstance(step, dict):
        return

    step_ref = id(step)
    with _llm_trace_lock:
        if not isinstance(_llm_trace_data, dict):
            return
        if _llm_last_started_step_ref == step_ref:
            return
        _llm_last_started_step_ref = step_ref
        _llm_trace_data['state'] = 'executing'
        _llm_trace_data.setdefault('executed_actions', []).append({
            'started_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'action': dict(step),
        })


def _finalize_llm_execution_trace():
    """任务执行完成后，将已执行动作写回追踪JSON。"""
    global _llm_last_started_step_ref
    with _llm_trace_lock:
        if not isinstance(_llm_trace_data, dict):
            return
        _llm_trace_data['state'] = 'finished'
        _llm_trace_data['finished_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
        _write_llm_trace_file_locked()
        _llm_last_started_step_ref = None



def _extract_json_array_from_text(text: str) -> list[dict]:
    """从普通文本或```json代码块中提取首个JSON数组。"""
    if not text:
        return []
    raw = str(text).strip()
    blocks = re.findall(r"```json\s*([\s\S]*?)\s*```", raw, flags=re.IGNORECASE)
    candidates = blocks + [raw]
    for c in candidates:
        try:
            data = json.loads(c)
            if isinstance(data, list):
                return [x for x in data if isinstance(x, dict)]
        except Exception:
            pass
    return []


def _extract_macro_tasks_from_results(results: dict) -> list[dict]:
    """抽取3-LLM上层任务（不是原子动作）。"""
    if not isinstance(results, dict):
        return []

    raw_tasks = []
    for k in ('llma_macro_tasks_raw', 'scene_description'):
        raw_tasks.extend(_extract_json_array_from_text(results.get(k) or ''))

    out = []
    seen = set()
    for i, t in enumerate(raw_tasks):
        issue = str(t.get('issue_description') or t.get('issue') or '').strip()
        implied = str(t.get('implied_action') or t.get('action') or '').strip()
        primary = str(t.get('primary_object_id') or t.get('object_id') or '').strip()
        target = str(t.get('target_receptacle_id') or t.get('target_id') or '').strip()
        key = (issue, implied, primary, target)
        if key in seen:
            continue
        seen.add(key)
        out.append({
            'id': f'macro_{len(out)+1}',
            'issue_description': issue or f'任务{len(out)+1}',
            'implied_action': implied,
            'primary_object_id': primary,
            'target_receptacle_id': target,
            'status': '待执行',
        })

    # 兜底：若未提取到结构化任务，使用候选文本任务
    if not out:
        cands = results.get('candidate_macro_tasks') or []
        for c in cands:
            text = str(c).strip()
            if not text:
                continue
            out.append({
                'id': f'macro_{len(out)+1}',
                'issue_description': text,
                'implied_action': '',
                'primary_object_id': '',
                'target_receptacle_id': '',
                'status': '待执行',
            })
    return out


def _update_llm_task_board_generated(results: dict):
    tasks = _extract_macro_tasks_from_results(results)
    with _llm_task_board_lock:
        _llm_task_board['state'] = 'generated'
        _llm_task_board['generated_tasks'] = tasks
        _llm_task_board['verification'] = []
        _llm_task_board['last_updated'] = time.time()


def _refresh_llm_task_board_live(event):
    """基于实时环境信息逐帧刷新3-LLM上层任务状态。"""
    if event is None:
        return

    with _llm_task_board_lock:
        generated = list(_llm_task_board.get('generated_tasks') or [])
        current_state = str(_llm_task_board.get('state', 'idle'))

    if not generated:
        return

    updated_generated = []
    verification = []
    done_count = 0

    for t in generated:
        ok, reason = _check_macro_task_completion(t, event)
        prev_status = str(t.get('status', '待执行'))

        if ok is True:
            status = '已完成'
            done_count += 1
        elif prev_status == '已完成':
            status = '已完成'
            done_count += 1
        elif ok is False:
            status = '进行中'
        else:
            status = '待人工确认'

        nt = dict(t)
        nt['status'] = status
        updated_generated.append(nt)
        verification.append({
            'task_id': t.get('id'),
            'status': status,
            'reason': reason,
            'implied_action': t.get('implied_action', ''),
            'issue_description': t.get('issue_description', ''),
        })

    total = len(updated_generated)
    if total > 0 and done_count >= total:
        next_state = 'live_verified'
    elif current_state in ('generated', 'executing', 'live_verified', 'verified'):
        next_state = 'executing'
    else:
        next_state = current_state

    with _llm_task_board_lock:
        _llm_task_board['state'] = next_state
        _llm_task_board['generated_tasks'] = updated_generated
        _llm_task_board['verification'] = verification
        _llm_task_board['last_updated'] = time.time()


def _llm_task_progress_for_ui(max_items: int = 6) -> dict:
    """供UI展示的3-LLM上层任务实时进度摘要。"""
    with _llm_task_board_lock:
        state = str(_llm_task_board.get('state', 'idle'))
        tasks = list(_llm_task_board.get('generated_tasks') or [])
        updated_at = float(_llm_task_board.get('last_updated', 0.0) or 0.0)

    total = len(tasks)
    done = sum(1 for t in tasks if str(t.get('status', '')) == '已完成')
    in_progress = sum(1 for t in tasks if str(t.get('status', '')) in ('进行中', '执行中'))
    pending = max(0, total - done - in_progress)

    return {
        'state': state,
        'total': total,
        'done': done,
        'in_progress': in_progress,
        'pending': pending,
        'tasks': tasks[:max_items],
        'last_updated': updated_at,
    }


def _build_object_map(event) -> dict:
    objs = (event.metadata or {}).get('objects', []) if event is not None else []
    out = {}
    for o in objs or []:
        oid = o.get('objectId')
        if oid:
            out[oid] = o
    return out


def _check_macro_task_completion(task: dict, event) -> tuple[bool | None, str]:
    """返回(是否完成, 说明)；None表示无法判定。"""
    obj_map = _build_object_map(event)
    action = str(task.get('implied_action') or '').lower()
    primary = str(task.get('primary_object_id') or '')
    target = str(task.get('target_receptacle_id') or '')
    obj = obj_map.get(primary) if primary else None

    if not action:
        return None, '任务无结构化动作，无法自动判定'

    if action == 'cleanobject':
        if not obj:
            return None, '目标对象不可见/未找到'
        is_dirty = bool(obj.get('isDirty', False))
        return (not is_dirty), ('已清洁' if not is_dirty else '仍为脏污')

    if action == 'dirtyobject':
        if not obj:
            return None, '目标对象不可见/未找到'
        is_dirty = bool(obj.get('isDirty', False))
        return is_dirty, ('已变脏' if is_dirty else '仍为干净')

    if action == 'openobject':
        if not obj:
            return None, '目标对象不可见/未找到'
        is_open = bool(obj.get('isOpen', False))
        return is_open, ('已打开' if is_open else '未打开')

    if action == 'closeobject':
        if not obj:
            return None, '目标对象不可见/未找到'
        is_open = bool(obj.get('isOpen', False))
        return (not is_open), ('已关闭' if not is_open else '仍打开')

    if action == 'toggleobjecton':
        if not obj:
            return None, '目标对象不可见/未找到'
        on = bool(obj.get('isToggledOn', obj.get('isOn', False)))
        return on, ('已开启' if on else '未开启')

    if action == 'toggleobjectoff':
        if not obj:
            return None, '目标对象不可见/未找到'
        on = bool(obj.get('isToggledOn', obj.get('isOn', False)))
        return (not on), ('已关闭' if not on else '仍开启')

    if action == 'putobject':
        if not obj:
            return None, '目标对象不可见/未找到'
        parents = obj.get('parentReceptacles') or []
        on_floor = bool(obj.get('isOnFloor', False))
        if target:
            done = any(target in str(p) for p in parents)
            return done, ('已放入目标容器' if done else '未在目标容器中')
        # 没有明确目标时，判定是否已离地并有容器归属
        done = (not on_floor) and len(parents) > 0
        return done, ('已离地并放入容器' if done else '仍在地面或无容器归属')

    return None, f'暂不支持自动验收动作: {action}'


def _run_llm_task_postcheck(event):
    """在3-LLM任务执行后做一次全局复检，并写回任务看板。"""
    with _llm_task_board_lock:
        generated = list(_llm_task_board.get('generated_tasks') or [])
    if not generated:
        return

    verification = []
    updated_generated = []
    done_count = 0
    for t in generated:
        ok, reason = _check_macro_task_completion(t, event)
        if ok is True:
            status = '已完成'
            done_count += 1
        elif ok is False:
            status = '未完成'
        else:
            status = '待人工确认'
        nt = dict(t)
        nt['status'] = status
        updated_generated.append(nt)
        verification.append({
            'task_id': t.get('id'),
            'status': status,
            'reason': reason,
            'implied_action': t.get('implied_action', ''),
            'issue_description': t.get('issue_description', ''),
        })

    with _llm_task_board_lock:
        _llm_task_board['state'] = 'verified'
        _llm_task_board['generated_tasks'] = updated_generated
        _llm_task_board['verification'] = verification
        _llm_task_board['last_updated'] = time.time()

    print(f"🧾 3-LLM任务验收完成: {done_count}/{len(generated)} 已完成")


def _llm_task_board_for_ui(max_items: int = 24) -> dict:
    with _llm_task_board_lock:
        return {
            'state': _llm_task_board.get('state', 'idle'),
            'generated_tasks': list((_llm_task_board.get('generated_tasks') or [])[:max_items]),
            'verification': list((_llm_task_board.get('verification') or [])[:max_items]),
            'last_updated': _llm_task_board.get('last_updated', 0.0),
        }


def _capture_and_disable_navigation_for_llm():
    """在触发三LLM理解时，记录当前导航状态并统一关闭导航。"""
    global _llm_nav_restore_state, _llm_task_flow_active

    # 已经处于“LLM理解→任务执行→恢复导航”的流程中，避免重复覆盖恢复快照
    if _llm_task_flow_active and _llm_nav_restore_state is not None:
        return

    explorer_obj = globals().get('explorer')
    pointnav_obj = globals().get('pointnav_nav')
    known_map_obj = globals().get('known_map_nav')
    frontier_obj = globals().get('frontier_nav')

    state = {
        'explorer': bool(getattr(explorer_obj, 'is_enabled', False)) if explorer_obj is not None else False,
        'pointnav': bool(getattr(pointnav_obj, 'is_enabled', False)) if pointnav_obj is not None else False,
        'known_map': bool(getattr(known_map_obj, 'is_enabled', False)) if known_map_obj is not None else False,
        'frontier': bool(getattr(frontier_obj, 'is_enabled', False)) if frontier_obj is not None else False,
    }

    _llm_nav_restore_state = state
    _llm_task_flow_active = True

    disabled_names = []
    try:
        if state['explorer'] and explorer_obj is not None:
            explorer_obj.disable()
            disabled_names.append('自主探索')
    except Exception as e:
        print(f"⚠ 关闭自主探索失败: {e}")

    try:
        if state['pointnav'] and pointnav_obj is not None:
            pointnav_obj.disable()
            disabled_names.append('PointNav导航')
    except Exception as e:
        print(f"⚠ 关闭PointNav导航失败: {e}")

    try:
        if state['known_map'] and known_map_obj is not None:
            known_map_obj.disable()
            disabled_names.append('已知地图导航')
    except Exception as e:
        print(f"⚠ 关闭已知地图导航失败: {e}")

    try:
        if state['frontier'] and frontier_obj is not None:
            frontier_obj.disable()
            disabled_names.append('Frontier导航')
    except Exception as e:
        print(f"⚠ 关闭Frontier导航失败: {e}")

    # 按需静默：避免在3-LLM理解期间反复输出导航停止/关闭提示


def _auto_restore_navigation_after_llm_tasks(controller):
    """三LLM理解完成且任务队列清空后，恢复触发前的导航状态。"""
    global _llm_nav_restore_state, _llm_task_flow_active, _llm_postcheck_done

    if not _llm_task_flow_active or _llm_nav_restore_state is None:
        return

    # 理解阶段未完成时，不恢复导航
    if _llm_understanding_in_progress:
        return

    with _plan_lock:
        has_pending_plan = executing_plan or len(planned_actions) > 0
    if has_pending_plan:
        return

    # 任务执行结束后做一次全局验收（上帝视角基于当前event metadata）
    if not _llm_postcheck_done:
        try:
            post_ev = controller.step(action="Pass")
            _run_llm_task_postcheck(post_ev)
        except Exception as e:
            print(f"⚠ 3-LLM任务验收失败: {e}")
        _llm_postcheck_done = True

    # 执行完成后回填“已执行任务”到追踪JSON
    try:
        _finalize_llm_execution_trace()
    except Exception as e:
        print(f"⚠ 3-LLM任务追踪文件收尾失败: {e}")

    explorer_obj = globals().get('explorer')
    pointnav_obj = globals().get('pointnav_nav')
    known_map_obj = globals().get('known_map_nav')
    frontier_obj = globals().get('frontier_nav')

    restored_names = []
    try:
        if _llm_nav_restore_state.get('explorer') and explorer_obj is not None and not explorer_obj.is_enabled:
            explorer_obj.enable()
            restored_names.append('自主探索')
    except Exception as e:
        print(f"⚠ 恢复自主探索失败: {e}")

    try:
        if _llm_nav_restore_state.get('pointnav') and pointnav_obj is not None and not pointnav_obj.is_enabled:
            if len(getattr(pointnav_obj, 'known_points', [])) == 0:
                if pointnav_obj.initialize_map(controller):
                    pointnav_obj.enable()
                    restored_names.append('PointNav导航')
            else:
                pointnav_obj.enable()
                restored_names.append('PointNav导航')
    except Exception as e:
        print(f"⚠ 恢复PointNav导航失败: {e}")

    try:
        if _llm_nav_restore_state.get('known_map') and known_map_obj is not None and not known_map_obj.is_enabled:
            if len(getattr(known_map_obj, 'known_points', [])) == 0:
                if known_map_obj.initialize_map(controller):
                    known_map_obj.enable()
                    restored_names.append('已知地图导航')
            else:
                known_map_obj.enable()
                restored_names.append('已知地图导航')
    except Exception as e:
        print(f"⚠ 恢复已知地图导航失败: {e}")

    try:
        if _llm_nav_restore_state.get('frontier') and frontier_obj is not None and not frontier_obj.is_enabled:
            frontier_obj.enable()
            restored_names.append('Frontier导航')
    except Exception as e:
        print(f"⚠ 恢复Frontier导航失败: {e}")

    if restored_names:
        print(f"▶ 三LLM任务执行完成：已自动恢复导航 -> {', '.join(restored_names)}")
    else:
        print("▶ 三LLM任务执行完成：无需恢复导航（触发前未启用）")

    _llm_nav_restore_state = None
    _llm_task_flow_active = False


def _load_module_from_path(mod_name: str, path: str):
    try:
        spec = importlib.util.spec_from_file_location(mod_name, path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        print(f"⚠ 模块加载失败 {mod_name} @ {path}: {e}")
        return None


def _parse_detailed_steps(text: str) -> list[dict]:
    """将 LLM 生成的详细步骤解析为 [{'action': 'MoveAhead', 'params': {..}}, ...]"""
    actions = []
    if not text:
        return actions
    for line in str(text).splitlines():
        m = re.search(r"\b([A-Za-z_]+)\s*\(([^)]*)\)\s*$", line.strip())
        if not m:
            continue
        name, args = m.group(1), m.group(2).strip()
        params = {}
        if args:
            # 逗号分隔的 k=v
            for kv in args.split(','):
                if '=' not in kv:
                    continue
                k, v = kv.split('=', 1)
                k = k.strip()
                v = v.strip()
                # 去掉引号，尝试转数字
                if v.startswith(('"', "'")) and v.endswith(('"', "'")):
                    v = v[1:-1]
                else:
                    try:
                        if '.' in v:
                            v = float(v)
                        else:
                            v = int(v)
                    except Exception:
                        pass
                params[k] = v
        actions.append({'action': name, 'params': params})
    return actions


def _enqueue_plan_from_results(results: dict):
    global planned_actions, executing_plan, _action_retry_count, _llm_postcheck_done
    steps_text = (results or {}).get('detailed_steps') or ''
    actions = _parse_detailed_steps(steps_text)
    # 安全过滤：只保留清洁/收纳相关动作，禁止打开电器/开启设备
    actions = _sanitize_planned_actions(actions)
    _init_llm_execution_trace(results, actions)
    _update_llm_task_board_generated(results)
    _llm_postcheck_done = False
    with _plan_lock:
        planned_actions = actions[:]  # 覆盖旧计划
        executing_plan = len(planned_actions) > 0
    # 清理旧的重试计数
    _action_retry_count.clear()
    print(f"▶ 已接收并入列 {len(actions)} 个动作，将在主循环中依次执行。")
    if actions:
        print("  例如：", actions[0])



def _sanitize_planned_actions(actions: list[dict]) -> list[dict]:
    """
    按你的要求：凡是 AI2-THOR 允许的动作都可以做，这里不再做严格过滤。
    仅做最基本的结构化清洗，确保每项包含 action 与 params 字段。
    同时保留可选的元信息：source（例如 problem_agent）。
    """
    if not actions:
        return []
    out = []
    for a in actions:
        if not isinstance(a, dict):
            continue
        name = str(a.get('action', '')).strip()
        if not name:
            continue
        params = a.get('params') or {}
        if not isinstance(params, dict):
            params = {}
        step = {'action': name, 'params': params}
        if isinstance(a, dict) and a.get('source'):
            step['source'] = a.get('source')
        out.append(step)
    return out


def start_llm_scene_understanding(scene_graph_path: str = EXPLORATION_JSON_PATH, event=None):
    """
    按下 M 后：
    - 优先使用“结构化实时对象”文件（包含区域/分组提示）：semantic_maps/realtime_objects_structured.json
    - 否则回退到探索快照 semantic_maps/exploration_progress.json
    - 读取源JSON并保存为 snapshots/ 下的只读快照，交给embodied系统
    - 异步运行，不阻塞主循环
    - 分析完成后：把详细步骤解析为动作队列，交由主循环执行
    """
    global _goto_failure_counts, _llm_understanding_in_progress  # 引用全局变量
    print("[3-LLM生命周期] 开始：_llm_understanding_in_progress = True")
    print("🔄 重置导航失败记录，以适应新计划。")
    _goto_failure_counts.clear()  # 清空计数器
    _capture_and_disable_navigation_for_llm()
    _llm_understanding_in_progress = True

    try:
        # 选择信息更完整的输入：structured -> exploration
        source_path = scene_graph_path or EXPLORATION_JSON_PATH
        try:
            import os as _os
            sp = getattr(SEXP, 'OUTPUT_PATH', 'semantic_maps/realtime_objects_structured.json')
            use_structured = _os.path.exists(sp) and (_os.path.getmtime(sp) >= _os.path.getmtime(source_path) if _os.path.exists(source_path) else True)
            if use_structured:
                source_path = sp
                print(f"🧩 使用结构化实时对象文件: {source_path}")
            else:
                print(f"🧾 使用探索快照文件: {source_path}")
        except Exception:
            pass
        if not os.path.exists(source_path):
            print(f"❌ 未找到场景图文件: {source_path}")
            return
        # 读取最新JSON并保存快照（避免分析期间文件被持续更新）；带校验与重试
        os.makedirs(os.path.join('semantic_maps', 'snapshots'), exist_ok=True)
        ts = time.strftime('%Y%m%d_%H%M%S')
        snapshot_path = os.path.join('semantic_maps', 'snapshots', f'scene_graph_snapshot_{ts}.json')
        import json as _json
        valid_json_text = None
        for _try in range(5):
            try:
                with open(source_path, 'r', encoding='utf-8') as fsrc:
                    text = fsrc.read()
                if not text or not text.strip():
                    time.sleep(0.2)
                    continue
                _json.loads(text)  # 验证JSON格式
                valid_json_text = text
                break
            except Exception:
                time.sleep(0.2)
        if valid_json_text is not None:
            try:
                with open(snapshot_path, 'w', encoding='utf-8') as fdst:
                    fdst.write(valid_json_text)
            except Exception:
                snapshot_path = source_path
        else:
            # 回退：尝试使用最近一次有效快照
            snaps_dir = os.path.join('semantic_maps', 'snapshots')
            fallback = None
            try:
                files = sorted([os.path.join(snaps_dir, f) for f in os.listdir(snaps_dir) if f.endswith('.json')], reverse=True)
                for p in files:
                    try:
                        with open(p, 'r', encoding='utf-8') as fp:
                            t = fp.read()
                        if t and t.strip():
                            _json.loads(t)
                            fallback = p
                            break
                    except Exception:
                        continue
            except Exception:
                pass
            if fallback is None:
                print("❌ 场景图快照无效且无可用历史快照，已取消本次场景理解")
                return
            snapshot_path = fallback

        # 先检查外部依赖 dashscope，缺失则静默跳过，不报错
        try:
            import importlib
            if importlib.util.find_spec('dashscope') is None:
                print("⚠ 缺少 dashscope，已跳过场景理解（按M前请先安装依赖）")
                return
        except Exception:
            print("⚠ 缺少 dashscope，已跳过场景理解")
            return

        # 智能路径检查：支持从不同工作目录运行
        possible_paths = [
            'embodied B1',  # 从 aithor2/ 目录运行
            'aithor2/embodied B1',  # 从根目录运行
            os.path.join(os.path.dirname(__file__), 'embodied B1')  # 相对于脚本位置
        ]

        emb_dir = None
        for path in possible_paths:
            test_main = os.path.join(path, 'main.py')
            if os.path.exists(test_main):
                emb_dir = path
                break

        if emb_dir is None:
            print(f"❌ 未找到 embodied 系统，尝试过的路径:")
            for path in possible_paths:
                test_main = os.path.join(path, 'main.py')
                print(f"   - {test_main} (存在: {os.path.exists(test_main)})")
            return

        main_py = os.path.join(emb_dir, 'main.py')
        cfg_py = os.path.join(emb_dir, 'config.py')
        print(f"✅ 找到 embodied 系统: {main_py}")
        # 确保本地依赖（config/prompts/mab等）可被导入
        import sys as _sys
        if emb_dir not in _sys.path:
            _sys.path.insert(0, emb_dir)

        # 热重载 prompts 等模块，避免长进程中使用旧缓存
        import importlib as _importlib
        try:
            import prompts as _pr
            # 如果已加载过则reload，否则保持import的效果
            if 'prompts' in _sys.modules:
                try:
                    _importlib.reload(_sys.modules['prompts'])
                except Exception:
                    pass
            print(f"🧠 使用提示词文件: {_pr.__file__}")
        except Exception:
            # 尝试清理后由子模块重新导入
            if 'prompts' in _sys.modules:
                try:
                    del _sys.modules['prompts']
                except Exception:
                    pass
        for _mn in ('prompts_new', 'mab', 'api', 'config'):
            if _mn in _sys.modules:
                try:
                    _importlib.reload(_sys.modules[_mn])
                except Exception:
                    try:
                        del _sys.modules[_mn]
                    except Exception:
                        pass
        if 'embodied_b1_main' in _sys.modules:
            try:
                del _sys.modules['embodied_b1_main']
            except Exception:
                pass

        mod = _load_module_from_path('embodied_b1_main', main_py)
        if mod is None:
            print("❌ embodied 系统加载失败")
            return
        api_key = os.environ.get('DASHSCOPE_API_KEY')
        if os.path.exists(cfg_py):
            cfg_mod = _load_module_from_path('embodied_b1_config', cfg_py)
            if cfg_mod and getattr(cfg_mod, 'DASHSCOPE_API_KEY', None):
                api_key = getattr(cfg_mod, 'DASHSCOPE_API_KEY')
        if not api_key:
            print("⚠ 未检测到 DASHSCOPE_API_KEY，可能无法成功调用外部LLM API")

        # 可选：本地多任务候选（仅用于展示，不入列）
        if event is not None and LOCAL_TASK_PREVIEW_BEFORE_LLM:
            print_A_candidates(event, limit=12)

        print(f"🚀 启动三个LLM协作系统 V2（使用渐进式探索数据）：{os.path.basename(snapshot_path)}")
        system = mod.ThreeLLMSystemV2(api_key or "", snapshot_path, extra_room_paths=None)

        results = system.execute_planning()
        out_path = os.path.join('semantic_maps', f'three_llm_results_{ts}.json')
        system.save_results(results, output_path=out_path)
        print(f"✅ LLM 场景理解完成，结果已保存到: {out_path}")

        # —— 将详细步骤解析并入列，交给主线程执行 ——
        _enqueue_plan_from_results(results)

        # 通知轻量化监控系统：三LLM理解已完成
        try:
            monitor = LLM_MONITOR.get_monitor_instance()
            if monitor:
                monitor.mark_three_llm_completed()
        except Exception:
            pass

    except Exception as e:
        print(f"❌ LLM 场景理解运行失败: {e}")

        # 即使失败也要重置三LLM运行状态
        try:
            monitor = LLM_MONITOR.get_monitor_instance()
            if monitor:
                monitor.three_llm_running = False
        except Exception:
            pass
    finally:
        print("[3-LLM生命周期] 结束：_llm_understanding_in_progress = False")
        _llm_understanding_in_progress = False


def trigger_llm_scene_understanding_async(event=None):
    global _llm_understanding_in_progress, executing_plan
    
    # 重复触发保护：3-LLM理解已在进行中，拒绝新触发
    if _llm_understanding_in_progress:
        print("⚠ 3-LLM场景理解已在进行中，拒绝重复触发。请稍候...")
        return
    
    # 执行阶段保护：3-LLM计划正在执行中，拒绝新触发
    if executing_plan:
        print("⚠ 3-LLM计划正在执行中，拒绝触发新的理解。请稍候...")
        return
    
    _capture_and_disable_navigation_for_llm()
    t = threading.Thread(target=start_llm_scene_understanding, args=(EXPLORATION_JSON_PATH, event), daemon=True)
    t.start()
    print("⏳ 已启动场景理解（后台运行），请稍候...")


# —— 主循环消费计划动作 ——

def _resolve_object_id_from_hint(event, hint: str | None) -> str | None:
    """
    改进的对象ID解析函数，支持多种匹配策略：
    1. 直接objectId匹配
    2. 从JSON映射中查找（Apple_1 -> 原始AI2-THOR ID）
    3. objectType精确匹配
    4. objectType模糊匹配
    5. 可见对象优先，不可见对象作为备选
    """
    if not hint:
        return None
    objs = event.metadata.get("objects", []) or []
    h = str(hint).lower()

    # 1) 直接匹配 objectId
    for o in objs:
        if o.get("objectId") == hint:
            return o.get("objectId")

    # 2) 从JSON映射中查找（新增）
    # 尝试从最新的实时探索JSON文件中找到映射
    try:
        import glob
        json_files = glob.glob("semantic_maps/realtime_exploration_*.json")
        if json_files:
            latest_json = max(json_files, key=os.path.getctime)
            with open(latest_json, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 查找匹配的节点
            for node in data.get("nodes", []):
                if node.get("id") == hint:
                    # 找到匹配的节点，获取AI2-THOR原生ID
                    ai2thor_id = node.get("attributes", {}).get("ai2thor_id") or node.get("attributes", {}).get("original_id")
                    if ai2thor_id:
                        # 验证AI2-THOR ID是否在当前场景中存在
                        for o in objs:
                            if o.get("objectId") == ai2thor_id:
                                print(f"✅ 成功映射: {hint} -> {ai2thor_id}")
                                return ai2thor_id

                        # 如果直接ID不存在，尝试通过类型和位置匹配
                        node_pos = node.get("attributes", {}).get("position_3d", {})
                        node_type = node.get("label", "").lower()

                        best_match = None
                        min_distance = float('inf')

                        for o in objs:
                            obj_type = (o.get("objectType") or "").lower()
                            if obj_type == node_type:
                                obj_pos = o.get("position", {})
                                if obj_pos:
                                    distance = ((node_pos.get('x', 0) - obj_pos.get('x', 0))**2 +
                                              (node_pos.get('z', 0) - obj_pos.get('z', 0))**2)**0.5
                                    if distance < min_distance:
                                        min_distance = distance
                                        best_match = o.get("objectId")

                        if best_match and min_distance < 1.0:  # 1米内认为是同一物体
                            print(f"✅ 位置匹配: {hint} -> {best_match} (距离: {min_distance:.2f}m)")
                            return best_match
    except Exception:
        pass  # 静默处理JSON解析错误

    # 3) 精确匹配 objectType（优先可见对象）
    for o in objs:
        obj_type = (o.get("objectType") or "").lower()
        if obj_type == h:
            if o.get("visible", False):
                return o.get("objectId")

    # 4) 精确匹配 objectType（不可见对象作为备选）
    for o in objs:
        obj_type = (o.get("objectType") or "").lower()
        if obj_type == h:
            return o.get("objectId")

    # 4) 模糊匹配 objectType（优先可见对象）
    for o in objs:
        obj_type = (o.get("objectType") or "").lower()
        if h in obj_type or obj_type in h:
            if o.get("visible", False):
                return o.get("objectId")

    # 5) 模糊匹配 objectType（不可见对象作为备选）
    for o in objs:
        obj_type = (o.get("objectType") or "").lower()
        if h in obj_type or obj_type in h:
            return o.get("objectId")

    # 6) 最后尝试匹配对象名称（如果有的话）
    for o in objs:
        obj_name = (o.get("name") or "").lower()
        if obj_name and (h in obj_name or obj_name in h):
            return o.get("objectId")

    return None



# === LLM 决策：在动作失败时由LLM裁决“改策略或跳过” ===
import os as _os
import json as _json

def _build_failure_context(event, name: str, params: dict, err_msg: str) -> dict:
    """收集足够的上下文发给LLM（或记录日志）。保持小而关键，避免过大对象。"""
    try:
        agent = event.metadata.get('agent', {}) if event else {}
        inv = event.metadata.get('inventoryObjects', []) if event else []
        visible = []
        for o in (event.metadata.get('objects', []) if event else [])[:120]:
            if o.get('visible'):
                visible.append({'id': o.get('objectId'), 'type': o.get('objectType')})
        ctx = {
            'action': name,
            'params': params,
            'error': err_msg,
            'agent': {
                'pos': agent.get('position'),
                'rot': agent.get('rotation')
            },
            'inventory': [x.get('objectId') for x in inv],
            'visible_sample': visible[:50],
        }
    except Exception:
        ctx = {'action': name, 'params': params, 'error': err_msg}
    return ctx


PROBLEM_LLM_TRIGGER_COUNT = 3
_problem_error_counts = {}


def _build_problem_error_signature(name: str, params: dict, err_msg: str) -> tuple[str, str, str]:
    """构建“同类错误”签名，用于3次触发问题LLM。"""
    msg = str(err_msg or '').strip().lower()
    if 'no valid positions to place object found' in msg:
        msg_key = 'no_valid_place_position'
    elif 'object not found' in msg:
        msg_key = 'object_not_found'
    elif 'not reachable' in msg or 'cannot be reached' in msg:
        msg_key = 'target_not_reachable'
    elif 'navmesh' in msg and ('fail' in msg or '无法' in msg):
        msg_key = 'navmesh_plan_fail'
    else:
        msg_key = msg[:96]

    target = str(params.get('receptacleObjectId') or params.get('objectId') or '')
    return str(name or ''), msg_key, target


def _should_trigger_problem_llm(name: str, params: dict, err_msg: str) -> tuple[bool, int, tuple[str, str, str]]:
    """返回(是否触发, 当前计数, 签名)。达到阈值时触发。"""
    sig = _build_problem_error_signature(name, params, err_msg)
    cnt = int(_problem_error_counts.get(sig, 0)) + 1
    _problem_error_counts[sig] = cnt
    return cnt >= PROBLEM_LLM_TRIGGER_COUNT, cnt, sig


def _build_local_problem_recovery_decision(name: str, params: dict, event, err_msg: str) -> dict | None:
    """Problem-LLM 不可用时的本地兜底修复策略。"""
    low = str(err_msg or '').lower()
    if name == 'PutObject' and 'no valid positions to place object found' in low:
        src = params.get('objectId')
        dst = params.get('receptacleObjectId')
        safe_dst = _find_nearest_safe_receptacle(event)
        if src and safe_dst and safe_dst != dst:
            return {
                'decision': 'replace_steps',
                'steps': [
                    {'action': 'PutObject', 'params': {'objectId': src, 'receptacleObjectId': safe_dst}}
                ],
                'source': 'problem_agent_local',
            }
    return None


def _try_call_llm_for_decision(prompt: str) -> dict | None:
    """
    可选调用外部LLM：
    - 若未配置API或未安装SDK，返回None（改由本地兜底策略处理）。
    - 预期LLM返回一个JSON对象，例如：
      {"decision": "skip"} 或 {"decision": "replace_steps", "steps": [{"action": "GoTo", "params": {...}}]}
    """
    # 1) 优先尝试 DashScope（与 embodied B1/main.py 一致的真实API调用方式）
    try:
        ds_key = _os.environ.get('DASHSCOPE_API_KEY')
        if not ds_key:
            # 参考 start_llm_scene_understanding 的做法，从 embodied B1/config.py 读取
            import os
            cfg_py = os.path.join('embodied B1', 'config.py')
            if os.path.exists(cfg_py):
                cfg_mod = _load_module_from_path('embodied_b1_config_for_decision', cfg_py)
                if cfg_mod and getattr(cfg_mod, 'DASHSCOPE_API_KEY', None):
                    ds_key = getattr(cfg_mod, 'DASHSCOPE_API_KEY')
        if ds_key:
            try:
                import dashscope  # type: ignore
                from dashscope import Generation  # type: ignore
                dashscope.api_key = ds_key
                # 这里不设置单独的 system prompt，直接用我们构造的严格JSON提示词
                resp = Generation.call(
                    model="deepseek-v3",
                    prompt=prompt,
                    result_format='message',
                    max_tokens=800,
                    temperature=0.2,
                    top_p=0.8,
                )
                if getattr(resp, 'status_code', None) == 200:
                    txt = resp.output.choices[0].message.content.strip()
                    print(f"🤖 DashScope响应: {txt[:200]}...")  # 显示前200字符
                else:
                    txt = str(getattr(resp, 'message', ''))
                    print(f"⚠ DashScope错误响应: {txt}")
                try:
                    result = _json.loads(txt)
                    print(f"✅ JSON解析成功")
                    return result
                except Exception as e:
                    print(f"⚠ JSON解析失败: {e}")
                    print(f"🔍 原始响应内容: {txt}")

                    # 尝试多种JSON提取策略
                    import re as _re

                    # 策略1: 提取完整的JSON对象（支持嵌套）
                    def extract_complete_json(text):
                        """提取完整的JSON对象，支持嵌套结构"""
                        start_idx = text.find('{')
                        if start_idx == -1:
                            return None

                        brace_count = 0
                        for i, char in enumerate(text[start_idx:], start_idx):
                            if char == '{':
                                brace_count += 1
                            elif char == '}':
                                brace_count -= 1
                                if brace_count == 0:
                                    return text[start_idx:i+1]
                        return None

                    complete_json = extract_complete_json(txt or "")
                    if complete_json:
                        try:
                            result = _json.loads(complete_json.strip())
                            print(f"✅ 成功提取JSON (策略1): {complete_json[:100]}...")
                            return result
                        except Exception as e1:
                            print(f"⚠ 策略1解析失败: {e1}")

                    # 策略1b: 简单的正则匹配（备用）
                    json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
                    matches = _re.findall(json_pattern, txt or "", _re.DOTALL)

                    for match in matches:
                        try:
                            result = _json.loads(match.strip())
                            print(f"✅ 成功提取JSON (策略1b): {match[:100]}...")
                            return result
                        except:
                            continue

                    # 策略2: 寻找包含steps的JSON结构
                    steps_pattern = r'\{[^{}]*"steps"[^{}]*\[[^\]]*\][^{}]*\}'
                    match = _re.search(steps_pattern, txt or "", _re.DOTALL)
                    if match:
                        try:
                            result = _json.loads(match.group(0).strip())
                            print(f"✅ 成功提取JSON (策略2): {match.group(0)[:100]}...")
                            return result
                        except:
                            pass

                    # 策略3: 最宽泛的JSON提取
                    m = _re.search(r"\{[\s\S]*\}", txt or "")
                    if m:
                        try:
                            result = _json.loads(m.group(0))
                            print(f"✅ 成功提取JSON (策略3): {m.group(0)[:100]}...")
                            return result
                        except Exception as e2:
                            print(f"❌ 策略3也失败: {e2}")

                    print(f"❌ 所有JSON提取策略都失败")
            except Exception as _e:
                print(f"⚠ DashScope 决策调用失败: {_e}")
    except Exception as _e:
        print(f"⚠ 读取 DashScope 配置失败: {_e}")

    # 2) 回退到 OpenAI（如果已安装并设置 OPENAI_API_KEY）
    try:
        api_key = _os.environ.get('OPENAI_API_KEY')
        if api_key:
            import openai  # type: ignore
            client = openai.OpenAI(api_key=api_key)
            sys = "你是家务机器人执行监督者。只输出严格的JSON，不要输出其它文字。"
            msg = [
                {"role": "system", "content": sys},
                {"role": "user", "content": prompt},
            ]
            resp = client.chat.completions.create(model=_os.environ.get('OPENAI_MODEL','gpt-4o-mini'), messages=msg, temperature=0.2)
            txt = resp.choices[0].message.content.strip()
            # 尝试解析JSON
            try:
                return _json.loads(txt)
            except Exception:
                # 从文本中提取首个JSON对象
                import re as _re
                m = _re.search(r"\{[\s\S]*\}", txt)
                if m:
                    return _json.loads(m.group(0))
                return None
    except Exception as _e:
        print(f"⚠ LLM决策调用失败: {_e}")
    return None



def _call_problem_llm(prompt: str) -> dict | None:
    """
    独立的问题求解 LLM 通道（与现有“场景理解/命令解析”用的通道分离）。
    首选读取 PROBLEM_* 环境变量；若未配置，则回退到现有 embodied 配置方式（embodied B1/config.py 或通用环境变量）。
    支持两类提供商：
      - DashScope: PROBLEM_DASHSCOPE_API_KEY, PROBLEM_DASHSCOPE_MODEL(默认 deepseek-v3)
      - OpenAI:    PROBLEM_OPENAI_API_KEY, PROBLEM_OPENAI_MODEL(默认 gpt-4o-mini)
    约定：严格返回 JSON 字符串；若响应中掺杂其它文本，将尝试提取首个 JSON 对象。
    """
    import os as _os
    # 0) 收集可用密钥（优先 PROBLEM_*，否则回退 embodied B1/config.py 与通用环境变量）
    ds_key = _os.environ.get('PROBLEM_DASHSCOPE_API_KEY') or None
    ds_model = _os.environ.get('PROBLEM_DASHSCOPE_MODEL') or 'deepseek-v3'
    oa_key = _os.environ.get('PROBLEM_OPENAI_API_KEY') or None
    oa_model = _os.environ.get('PROBLEM_OPENAI_MODEL') or 'gpt-4o-mini'
    if not (ds_key or oa_key):
        # 尝试从 embodied B1/config.py 读取
        try:
            import importlib.util as _importlib_util
            import sys as _sys
            _p = _os.path.join(_os.getcwd(), 'embodied B1', 'config.py')
            if _os.path.exists(_p):
                spec = _importlib_util.spec_from_file_location('emb_config', _p)
                emb = _importlib_util.module_from_spec(spec)
                assert spec and spec.loader
                spec.loader.exec_module(emb)  # type: ignore
                # 这两个变量名基于用户现有配置文件
                ds_key = getattr(emb, 'DASHSCOPE_API_KEY', None) or _os.environ.get('DASHSCOPE_API_KEY') or ds_key
                oa_key = getattr(emb, 'OPENAI_API_KEY', None) or _os.environ.get('OPENAI_API_KEY') or oa_key
            else:
                # 再尝试通用环境变量
                ds_key = _os.environ.get('DASHSCOPE_API_KEY') or ds_key
                oa_key = _os.environ.get('OPENAI_API_KEY') or oa_key
        except Exception as _e:
            print(f"ℹ️ Problem-LLM 回退读取 embodied 配置失败: {_e}")

    # 额外：参考 embodied B1/api.py 的配置读取方式
    try:
        if not ds_key:
            import importlib.util as _importlib_util
            _ap = _os.path.join(_os.getcwd(), 'embodied B1', 'api.py')
            if _os.path.exists(_ap):
                spec2 = _importlib_util.spec_from_file_location('emb_api_cfg', _ap)
                api_mod = _importlib_util.module_from_spec(spec2)
                assert spec2 and spec2.loader
                spec2.loader.exec_module(api_mod)  # type: ignore
                try:
                    cfg = api_mod.DashScopeConfig()  # may raise if no key
                    if getattr(cfg, 'validate_config', lambda: True)():
                        ds_key = cfg.api_key
                        # 若 api.py 指定了模型，也可采用
                        try:
                            ds_model = cfg.get_model() or ds_model
                        except Exception:
                            pass
                except Exception:
                    pass
    except Exception as _e:
        print(f"ℹ️ Problem-LLM 读取 embodied api.py 配置失败: {_e}")

    # 1) DashScope
    try:
        if ds_key:
            try:
                import dashscope  # type: ignore
                from dashscope import Generation  # type: ignore
                dashscope.api_key = ds_key
                resp = Generation.call(
                    model=ds_model,
                    prompt=prompt,
                    result_format='message',
                    max_tokens=800,
                    temperature=0.2,
                    top_p=0.8,
                )
                if getattr(resp, 'status_code', None) == 200:
                    txt = resp.output.choices[0].message.content.strip()
                else:
                    txt = str(getattr(resp, 'message', ''))
                try:
                    return _json.loads(txt)
                except Exception:
                    import re as _re
                    m = _re.search(r"\{[\s\S]*\}", txt or "")
                    if m:
                        return _json.loads(m.group(0))
            except Exception as _e:
                print(f"⚠ Problem-LLM(DashScope) 调用失败: {_e}")
    except Exception as _e:
        print(f"⚠ Problem-LLM 读取 DashScope 配置失败: {_e}")

    # 2) OpenAI
    try:
        if oa_key:
            import openai  # type: ignore
            client = openai.OpenAI(api_key=oa_key)
            sys = "你是机器人执行问题的裁决器。只输出严格的JSON，不要输出其它文字。"
            msg = [
                {"role": "system", "content": sys},
                {"role": "user", "content": prompt},
            ]
            resp = client.chat.completions.create(model=oa_model, messages=msg, temperature=0.2)
            txt = resp.choices[0].message.content.strip()
            try:
                return _json.loads(txt)
            except Exception:
                import re as _re
                m = _re.search(r"\{[\s\S]*\}", txt or "")
                if m:
                    return _json.loads(m.group(0))
                return None
    except Exception as _e:
        print(f"⚠ Problem-LLM(OpenAI) 调用失败: {_e}")
    print("ℹ️ Problem-LLM 未配置或未返回有效JSON，回退为本地兜底策略（跳过该步，继续后续）")
    return None

# === 用户自然语言命令：输入窗口 + LLM 解析 -> 动作计划 ===
user_command_mode = False
_user_cmd_thread = None
latest_event = None  # 每帧更新，供LLM解析使用


def _action_priority(a: dict) -> int:
    """粗略的动作优先级：安全相关最高；收纳/清洁其次；位移最低。"""
    name = (a or {}).get('action','') or ''
    n = name.lower()
    if n in ('toggleobjectoff',):
        return 100
    if n in ('closeobject',):
        return 80
    if n in ('cleanobject','dirtyobject'):
        return 70
    if n in ('pickupobject',):
        return 60
    if n in ('putobject',):
        return 55
    if n in ('openobject',):
        return 45
    if n in ('toggleobjecton',):
        return 40
    if n in ('goto',):
        return 20
    return 30


def _plan_summary_for_llm(max_items: int = 20) -> list[dict]:
    """将当前计划压缩为简要列表，并附带 priority 与来源 source，按优先级排序展示。"""
    items = []
    try:
        with _plan_lock:
            for i, a in enumerate(planned_actions[:max_items]):
                item = {
                    'idx': i,
                    'action': a.get('action'),
                    'params': {k: v for k, v in (a.get('params') or {}).items() if k in ('objectId','receptacleObjectId','dst','objectType')},
                    'source': a.get('source'),
                }
                item['priority'] = _action_priority(a)
                items.append(item)
    except Exception:
        pass
    # 优先级高在上
    items.sort(key=lambda x: (-x.get('priority',0), x.get('idx',0)))
    return items


def _visible_objects_for_llm(event) -> list[dict]:
    """只返回已探索区域中的物体信息，不使用仿真平台的全知信息"""
    objs = []
    try:
        # 使用探索系统记录的物体信息，而不是仿真平台的全知信息
        explored_objects = _get_explored_objects_only(event)
        ax, az = _agent_pos(event)

        for o in explored_objects[:160]:  # 限制数量避免过载
            oid = o.get('objectId')
            if not oid:
                continue
            pos = o.get('position') or o.get('pos', {})
            x, z = float(pos.get('x', 0.0)), float(pos.get('z', 0.0))
            d2 = (x - ax) ** 2 + (z - az) ** 2
            objs.append({
                'objectId': oid,
                'objectType': o.get('objectType'),
                'visible': o.get('visible', True),  # 探索过的物体默认认为是"已知"的
                'pickupable': o.get('pickupable'),
                'openable': o.get('openable'),
                'receptacle': o.get('receptacle'),
                'isOnFloor': o.get('isOnFloor'),
                'pos': {'x': x, 'z': z},
                'dist2': d2,
                'explored': True,  # 标记为已探索
            })
    except Exception as e:
        print(f"⚠ 获取已探索物体信息失败: {e}")
    return objs


def _get_explored_objects_only(event) -> list[dict]:
    """获取仅限于已探索区域的物体信息"""
    explored_objects = []

    try:
        # 方法1: 从探索进度文件读取
        exploration_file = EXPLORATION_JSON_PATH
        if os.path.exists(exploration_file):
            with open(exploration_file, 'r', encoding='utf-8') as f:
                exploration_data = json.load(f)

            # 获取已探索的物体
            explored_objects.extend(exploration_data.get('objects', []))
            print(f"📚 从探索文件获取 {len(explored_objects)} 个已知物体")

        # 方法2: 从结构化实时对象文件读取（如果存在且更新）
        try:
            structured_file = getattr(SEXP, 'OUTPUT_PATH', 'semantic_maps/realtime_objects_structured.json')
            if os.path.exists(structured_file):
                with open(structured_file, 'r', encoding='utf-8') as f:
                    structured_data = json.load(f)

                # 合并结构化数据中的物体
                structured_objects = structured_data.get('objects', [])
                if structured_objects:
                    explored_objects.extend(structured_objects)
                    print(f"📊 从结构化文件额外获取 {len(structured_objects)} 个物体")
        except Exception:
            pass

        # 方法3: 如果没有探索数据，只使用当前可见的物体（严格模式）
        if not explored_objects and event:
            current_visible = []
            for o in event.metadata.get('objects', []):
                if o.get('visible', False):  # 只有当前真正可见的物体
                    current_visible.append(o)
            explored_objects = current_visible
            print(f"👁️ 严格模式：只使用当前可见的 {len(current_visible)} 个物体")

    except Exception as e:
        print(f"⚠ 读取探索数据失败: {e}")
        # 最后的回退：只使用当前可见物体
        if event:
            explored_objects = [o for o in event.metadata.get('objects', []) if o.get('visible', False)]

    return explored_objects


def _llm_plan_from_user_text(text: str, event) -> dict | None:
    """将用户自然语言解析为“可直接执行的步骤”，尽量在规划时就确定每个物体的 objectId。"""

    # 检查LLM配置 - 从embodied B1/config.py读取
    import os

    has_dashscope = False
    has_openai = bool(os.environ.get('OPENAI_API_KEY'))

    # 尝试从embodied B1/config.py读取DashScope配置
    try:
        config_path = os.path.join(os.getcwd(), 'embodied B1', 'config.py')
        if os.path.exists(config_path):
            # 动态导入config模块
            import importlib.util
            spec = importlib.util.spec_from_file_location("config", config_path)
            config_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(config_module)

            # 获取API密钥
            api_key = getattr(config_module, 'DASHSCOPE_API_KEY', None)
            if api_key and api_key.startswith('sk-'):
                has_dashscope = True
                # 设置环境变量供后续使用
                os.environ['DASHSCOPE_API_KEY'] = api_key
                print(f"✅ 从embodied B1/config.py加载DashScope配置成功")
            else:
                print(f"⚠ embodied B1/config.py中的API密钥无效")
        else:
            print(f"⚠ 未找到embodied B1/config.py配置文件")
    except Exception as e:
        print(f"⚠ 读取embodied B1/config.py失败: {e}")

    # 如果还没有找到DashScope配置，检查环境变量
    if not has_dashscope:
        has_dashscope = bool(os.environ.get('DASHSCOPE_API_KEY'))
        if has_dashscope:
            print(f"✅ 从环境变量获取DashScope配置")

    print(f"🔍 LLM配置检查: DashScope={has_dashscope}, OpenAI={has_openai}")

    if not (has_dashscope or has_openai):
        print("⚠ 未配置LLM API密钥，使用本地简单解析")
        return _simple_local_parse(text, event)

    # 获取已探索的物体信息（不使用仿真平台的全知信息）
    explored_objects = _visible_objects_for_llm(event) if event else []

    ctx = {
        'user_text': text,
        'visible_objects': explored_objects,
        'current_plan': _plan_summary_for_llm(),
        'agent_inventory': [o.get('objectId') for o in (event.metadata.get('inventoryObjects', []) if event else [])],
    }


    prompt = (
        "你是家务机器人指挥官（具备优先级评估能力）。请完整理解用户句子：动词/目标对象/处理方式/目的地。\n"
        "例如：‘把电脑放床上’ -> 若手里已拿着 Laptop，则直接 GoTo(Bed) + PutObject(Laptop -> Bed)；若手里拿着别的物体，应先安全 Put 到最近台面。\n"
        "输出严格 JSON，不要附加解释：\n"
        "{\n  \"rationale\": \"简短中文说明你的理解\",\n  \"steps\": [{\"action\": ..., \"params\": {...}} , ...],\n  \"schedule\": \"now|after_current|after_index\",\n  \"after_index\": 可选整数\n}\n"
        "要求：\n"
        "- steps 仅使用 AI2-THOR 允许的动作：GoTo/PickupObject/PutObject/OpenObject/CloseObject/ToggleObjectOn/ToggleObjectOff/CleanObject/DirtyObject/RotateRight/RotateLeft/MoveAhead/MoveBack/LookUp/LookDown 等。\n"
        "- 规划时就尽量填充具体 ID：\n  · 对每个对象优先给出 objectId / receptacleObjectId；\n  · 若只有语义类型，临时给出 objectType，后处理会据可见对象最近匹配补全 ID。\n"
        "- 模糊消歧：\n  · 目标物体优先选 isOnFloor=true 且最近；\n  · 放置目的地优先选 Table/CounterTop/Shelf/Bed 等类型中最近可达的容器/表面；\n  · 需要时自行补充 OpenObject/CloseObject。\n"
        "- 手里已有待放物体时跳过 Pickup；若手里拿着其它物体且影响动作，请先 Put 到最近安全表面；\n"
        "- 根据 current_plan 的重要性决定 schedule（高优先级用 now）。\n"
        f"上下文: {_json.dumps(ctx, ensure_ascii=False)}"
    )
    return _try_call_llm_for_decision(prompt)


def _simple_local_parse(text: str, event) -> dict:
    """简单的本地指令解析，用于LLM不可用时的备用方案"""
    text = text.lower().strip()

    # 获取可见物体
    visible_objects = []
    if event:
        for obj in event.metadata.get('objects', []):
            if obj.get('visible', False):
                visible_objects.append(obj)

    # 简单的模式匹配
    steps = []

    # 模式1: "拿/捡 X"
    if any(word in text for word in ['拿', '捡', '取', 'pickup', 'take']):
        # 寻找目标物体
        for obj in visible_objects:
            obj_type = obj.get('objectType', '').lower()
            if any(keyword in text for keyword in [obj_type.lower(), obj.get('objectId', '').lower()]):
                steps = [
                    {"action": "GoTo", "params": {"objectId": obj.get('objectId')}},
                    {"action": "PickupObject", "params": {"objectId": obj.get('objectId')}}
                ]
                break

    # 模式2: "放 X 到 Y" 或 "把 X 放到 Y"
    elif any(word in text for word in ['放', 'put', '放到', '放在']):
        # 寻找目标容器
        for obj in visible_objects:
            obj_type = obj.get('objectType', '').lower()
            if obj.get('receptacle', False) and any(keyword in text for keyword in ['桌', 'table', '台', 'counter', '柜', 'cabinet']):
                # 假设要放置手中的物体
                agent_inventory = event.metadata.get('inventoryObjects', []) if event else []
                if agent_inventory:
                    held_obj = agent_inventory[0].get('objectId')
                    steps = [
                        {"action": "GoTo", "params": {"objectId": obj.get('objectId')}},
                        {"action": "PutObject", "params": {"objectId": held_obj, "receptacleObjectId": obj.get('objectId')}}
                    ]
                break

    # 如果没有匹配到模式，返回一个通用的探索动作
    if not steps:
        steps = [{"action": "Pass", "params": {}}]

    return {
        "rationale": f"本地解析: {text}",
        "steps": steps,
        "schedule": "now"
    }


def _resolve_llm_steps_object_ids(decision: dict, event) -> dict:
    """尽量在入列前为每个步骤补全 objectId/receptacleObjectId，避免执行期再追问。
    策略：
    - 若给了 objectType 而无 objectId，则在可见对象中按类型子串匹配并选最近者；
    - PutObject 缺少 objectId 时，优先使用手中物体（若手中仅有一个则直接使用，或按类型匹配）；
    - PutObject 缺少 receptacleObjectId 时，根据 receptacleObjectType/dst/中文同义词选择最近容器；
    - 支持常见中文 -> 英文类型同义映射（如 “电脑”->Laptop，“床”->Bed，“桌/桌子”->Table，“台面”->CounterTop）。
    """
    if not isinstance(decision, dict):
        return decision
    try:
        # 使用已探索的物体信息，而不是仿真平台的全知信息
        vis = _get_explored_objects_only(event)
        ax, az = _agent_pos(event) if event else (0.0, 0.0)
        inv = event.metadata.get('inventoryObjects', []) if event else []
        def _norm(s: str) -> str:
            return (s or '').lower()
        cn_map = {
            '电脑': 'laptop', '笔记本': 'laptop', '床': 'bed', '桌': 'table', '桌子': 'table',
            '台面': 'countertop', '餐桌': 'diningtable', '茶几': 'coffeetable', '水槽': 'sink',
        }
        def _match_type_sub(s: str, obj_type: str) -> bool:
            return _norm(s) in _norm(obj_type)
        def _find_nearest_by_type(type_hint: str):
            if not type_hint:
                return None
            th = _norm(cn_map.get(type_hint, type_hint))
            cand = []
            for o in vis:
                t = _norm(o.get('objectType') or o.get('name') or '')
                if th in t:
                    p = o.get('position') or {}
                    x, z = float(p.get('x', 0.0)), float(p.get('z', 0.0))
                    d2 = (x-ax)*(x-ax)+(z-az)*(z-az)
                    cand.append((d2, o.get('objectId')))
            if not cand:
                return None
            cand.sort(key=lambda t: t[0])
            return cand[0][1]
        def _find_inventory_by_type(type_hint: str):
            th = _norm(cn_map.get(type_hint, type_hint))
            for o in inv:
                t = _norm(o.get('objectType') or o.get('name') or '')
                if th in t:
                    return o.get('objectId')
            return inv[0].get('objectId') if inv else None

        steps = decision.get('steps') or []
        for a in steps:
            act = (a.get('action') or '').strip()
            params = a.setdefault('params', {}) or {}
            # 常规对象动作：补 objectId
            if act in ('PickupObject','OpenObject','CloseObject','ToggleObjectOn','ToggleObjectOff','GoTo'):
                if not params.get('objectId'):
                    typ = params.get('objectType') or params.get('type') or params.get('name')
                    oid = _find_nearest_by_type(str(typ)) if typ else None
                    if oid:
                        params['objectId'] = oid
            # PutObject：补 objectId 与 receptacleObjectId
            if act == 'PutObject':
                if not params.get('objectId'):
                    typ = params.get('objectType') or params.get('type') or 'laptop'
                    oid = _find_inventory_by_type(str(typ)) or _find_nearest_by_type(str(typ))
                    if oid:
                        params['objectId'] = oid
                if not params.get('receptacleObjectId'):
                    rtyp = params.get('receptacleObjectType') or params.get('dst') or 'table'
                    rid = _find_nearest_by_type(str(rtyp))
                    if rid:
                        params['receptacleObjectId'] = rid
        decision['steps'] = steps
        return decision
    except Exception:
        return decision


def _enqueue_llm_steps(decision: dict):
    # 先尽力补全 objectId
    ev = globals().get('latest_event')
    decision = _resolve_llm_steps_object_ids(decision, ev)
    steps = decision.get('steps') or []
    schedule = (decision.get('schedule') or 'after_current').lower()
    after_index = int(decision.get('after_index') or 0)
    steps = _sanitize_planned_actions(steps)
    if not steps:
        return False
    with _plan_lock:
        if schedule == 'now':
            globals()['planned_actions'] = steps + planned_actions
            globals()['executing_plan'] = True
        elif schedule == 'after_index' and 0 <= after_index <= len(planned_actions):
            globals()['planned_actions'] = planned_actions[:after_index+1] + steps + planned_actions[after_index+1:]
            globals()['executing_plan'] = True
        else:  # after_current
            globals()['planned_actions'] = planned_actions[:1] + steps + planned_actions[1:]
            globals()['executing_plan'] = True
    # 打印预览
    try:
        preview = _json.dumps(steps[:3], ensure_ascii=False)
        print(f"📝 计划预览(最多3步): {preview}")
    except Exception:
        pass
    return True


# ==================== 人工指令学习系统 ====================

def _load_preference_learning():
    """加载持久化的偏好学习数据"""
    global _preference_scores, _correction_history
    try:
        if os.path.exists(PREFERENCE_LEARNING_FILE):
            with open(PREFERENCE_LEARNING_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                _preference_scores = data.get('preference_scores', {})
                _correction_history = data.get('correction_history', [])
                print(f"📚 加载偏好学习数据: {len(_preference_scores)} 个物体类型, {len(_correction_history)} 条修正记录")
        else:
            _preference_scores = {}
            _correction_history = []
            print("📚 初始化新的偏好学习系统")
    except Exception as e:
        print(f"⚠ 加载偏好学习数据失败: {e}")
        _preference_scores = {}
        _correction_history = []

def _save_preference_learning():
    """保存偏好学习数据到文件"""
    try:
        os.makedirs(os.path.dirname(PREFERENCE_LEARNING_FILE), exist_ok=True)
        data = {
            'preference_scores': _preference_scores,
            'correction_history': _correction_history[-1000:],  # 只保留最近1000条记录
            'last_updated': time.time()
        }
        with open(PREFERENCE_LEARNING_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"💾 保存偏好学习数据: {len(_preference_scores)} 个物体类型")
    except Exception as e:
        print(f"⚠ 保存偏好学习数据失败: {e}")

def _apply_preference_decay():
    """应用遗忘衰减机制"""
    global _preference_scores
    for obj_type in _preference_scores:
        for target_type in _preference_scores[obj_type]:
            _preference_scores[obj_type][target_type] *= LEARNING_DECAY_RATE

def _record_human_correction(obj_type: str, obj_id: str, obj_state: dict,
                           original_target: str, human_target: str, context: dict):
    """记录人工修正事件"""
    global _correction_history, _preference_scores

    # 记录修正事件
    correction_event = {
        'timestamp': time.time(),
        'object_type': obj_type,
        'object_id': obj_id,
        'object_state': obj_state,
        'original_target': original_target,
        'human_target': human_target,
        'context': context
    }
    _correction_history.append(correction_event)

    # 更新偏好分数
    if obj_type not in _preference_scores:
        _preference_scores[obj_type] = {}

    # 正向激励：增加被选择目标的分数
    if human_target not in _preference_scores[obj_type]:
        _preference_scores[obj_type][human_target] = 5.0  # 初始分数
    _preference_scores[obj_type][human_target] += 2.0

    # 负向抑制：降低被放弃目标的分数
    if original_target not in _preference_scores[obj_type]:
        _preference_scores[obj_type][original_target] = 5.0
    _preference_scores[obj_type][original_target] -= 1.0

    # 确保分数不为负
    for target in _preference_scores[obj_type]:
        _preference_scores[obj_type][target] = max(0.1, _preference_scores[obj_type][target])

    print(f"📝 记录人工修正: {obj_type} {original_target}→{human_target}")
    print(f"   当前偏好分数: {_preference_scores[obj_type]}")

    # 保存到文件
    _save_preference_learning()

def _get_learned_preference(obj_type: str, available_targets: list) -> str:
    """根据学习到的偏好选择最佳目标"""
    if obj_type not in _preference_scores:
        return available_targets[0] if available_targets else None

    best_target = None
    best_score = -1

    for target in available_targets:
        score = _preference_scores[obj_type].get(target, 1.0)  # 默认分数1.0
        if score > best_score:
            best_score = score
            best_target = target

    return best_target or (available_targets[0] if available_targets else None)

def _prune_conflicting_actions(new_plan_actions: list):
    """清理与新人工指令冲突的旧计划动作"""
    global planned_actions

    if not new_plan_actions:
        return

    # 提取新计划中涉及的对象
    new_plan_objects = set()
    for action in new_plan_actions:
        params = action.get('params', {})
        if 'objectId' in params:
            new_plan_objects.add(params['objectId'])
        if 'receptacleObjectId' in params:
            new_plan_objects.add(params['receptacleObjectId'])

    # 从队列末尾开始检查，移除冲突的动作
    with _plan_lock:
        i = len(planned_actions) - 1
        removed_count = 0
        while i >= len(new_plan_actions):  # 不影响刚插入的新计划
            action = planned_actions[i]
            params = action.get('params', {})

            # 检查是否与新计划的对象冲突
            conflicts = False
            for obj_id in new_plan_objects:
                if (params.get('objectId') == obj_id or
                    params.get('receptacleObjectId') == obj_id):
                    conflicts = True
                    break

            if conflicts:
                removed_action = planned_actions.pop(i)
                removed_count += 1
                print(f"🗑️ 移除冲突动作: {removed_action.get('action')} {params}")

            i -= 1

        if removed_count > 0:
            print(f"🔄 清理了 {removed_count} 个与人工指令冲突的动作")

def _handle_user_command_text(text: str):
    global latest_event, _command_breakdown_callback, planned_actions, executing_plan
    if not text.strip():
        return

    print(f"🎯 收到人工指令: {text.strip()}")

    # 1. 暂停当前任务执行
    was_executing = executing_plan
    if executing_plan:
        executing_plan = False
        print("⏸️ 暂停当前自主任务，优先处理人工指令")

    try:
        # 2. 解析人工指令
        print("🤖 正在调用LLM解析人工指令...")
        decision = _llm_plan_from_user_text(text.strip(), latest_event)

        # 调试：显示LLM返回的原始结果
        print(f"🔍 LLM返回结果: {decision}")

        if not isinstance(decision, dict):
            print(f'⚠ LLM 返回类型错误，期望dict，实际: {type(decision)}')
            if hasattr(_command_breakdown_callback, '__call__'):
                _command_breakdown_callback(f"LLM返回类型错误: {type(decision)}")
            # 恢复执行
            if was_executing:
                executing_plan = True
            return

        if not decision.get('steps'):
            print(f'⚠ LLM 未返回有效步骤，steps字段: {decision.get("steps")}')
            if hasattr(_command_breakdown_callback, '__call__'):
                _command_breakdown_callback("LLM未返回有效步骤")
            # 恢复执行
            if was_executing:
                executing_plan = True
            return

        # 显示LLM理解
        if decision.get('rationale'):
            print(f"🗒️ LLM理解: {decision.get('rationale')}")

        # 3. 分析是否存在冲突并记录学习数据
        steps = decision.get('steps', [])
        _analyze_and_record_conflicts(steps, latest_event)

        # 4. 显示命令分解结果
        if steps and hasattr(_command_breakdown_callback, '__call__'):
            breakdown_text = f"🚨 人工指令优先! 分解为{len(steps)}个步骤:\n"
            for i, step in enumerate(steps[:5]):  # 只显示前5个步骤
                action = step.get('action', '未知动作')
                params = step.get('params', {})
                if 'objectId' in params:
                    breakdown_text += f"{i+1}. {action}({params['objectId']})\n"
                elif 'receptacleObjectId' in params:
                    breakdown_text += f"{i+1}. {action}(→{params['receptacleObjectId']})\n"
                else:
                    breakdown_text += f"{i+1}. {action}\n"
            if len(steps) > 5:
                breakdown_text += f"... 还有{len(steps)-5}个步骤"
            _command_breakdown_callback(breakdown_text.strip())

        # 5. 插入人工指令到队列最前端
        with _plan_lock:
            # 标记为人工指令，设置高优先级
            for step in steps:
                step['source'] = 'human_command'
                step['priority'] = 100  # 最高优先级

            # 插入到队列最前端
            planned_actions = steps + planned_actions

            # 6. 清理冲突的旧计划
            _prune_conflicting_actions(steps)

            # 恢复执行状态
            executing_plan = True

        print(f'✅ 人工指令已插入队列前端，共{len(steps)}个步骤')

    except Exception as e:
        print(f"⚠ 解析或处理人工指令失败: {e}")
        if hasattr(_command_breakdown_callback, '__call__'):
            _command_breakdown_callback(f"处理失败: {str(e)}")
        # 恢复执行状态
        if was_executing:
            executing_plan = True

def _analyze_and_record_conflicts(human_steps: list, event):
    """分析人工指令与当前计划的冲突，记录学习数据"""
    if not human_steps or not event:
        return

    # 查找人工指令中的PutObject动作
    for step in human_steps:
        if step.get('action') == 'PutObject':
            obj_id = step.get('params', {}).get('objectId')
            human_target = step.get('params', {}).get('receptacleObjectId')

            if not obj_id or not human_target:
                continue

            # 查找当前计划中对同一对象的PutObject动作
            original_target = None
            with _plan_lock:
                for planned_action in planned_actions:
                    if (planned_action.get('action') == 'PutObject' and
                        planned_action.get('params', {}).get('objectId') == obj_id):
                        original_target = planned_action.get('params', {}).get('receptacleObjectId')
                        break

            if original_target and original_target != human_target:
                # 发现冲突！记录学习数据
                obj_info = None
                for obj in event.metadata.get('objects', []):
                    if obj.get('objectId') == obj_id:
                        obj_info = obj
                        break

                if obj_info:
                    obj_type = obj_info.get('objectType', 'Unknown')
                    obj_state = {
                        'isDirty': obj_info.get('isDirty', False),
                        'isFilled': obj_info.get('isFilled', False),
                        'isOpen': obj_info.get('isOpen', False),
                        'temperature': obj_info.get('temperature', 'RoomTemp')
                    }

                    agent_pos = event.metadata.get('agent', {}).get('position', {})
                    context = {
                        'agent_position': agent_pos,
                        'scene_name': getattr(event, 'scene_name', 'Unknown'),
                        'timestamp': time.time()
                    }

                    print(f"🔍 检测到冲突: {obj_type} 系统计划→{original_target}, 人工指令→{human_target}")
                    _record_human_correction(obj_type, obj_id, obj_state, original_target, human_target, context)


def _start_user_command_window_async():
    """启动一个简单Tk窗口：左侧输入命令，右侧实时显示任务列表。"""
    import threading as _threading
    def _run():
        import tkinter as tk
        from tkinter import ttk
        global user_command_mode, _command_breakdown_callback
        user_command_mode = True
        try:
            root = tk.Tk()
            root.title('指令中心（Esc 关闭）')
            root.geometry('1200x700')  # 增大窗口尺寸

            # 主题与样式（含中文字体回退，字体调大）
            style = ttk.Style()
            try:
                style.theme_use('clam')
            except Exception:
                pass
            root.configure(bg='#0D1117')
            style.configure('TFrame', background='#0D1117')
            # 字体回退：尝试多种中文字体，确保兼容性
            font_candidates = [
                'Microsoft YaHei',  # Windows 微软雅黑
                'SimHei',           # Windows 黑体
                'PingFang SC',      # macOS 苹方
                'Noto Sans CJK SC', # Linux Noto
                'WenQuanYi Micro Hei', # Linux 文泉驿
                'DejaVu Sans',      # 通用字体
                'Arial Unicode MS', # 支持Unicode的Arial
                'Arial'             # 最后回退
            ]

            font_found = None
            for font_name in font_candidates:
                try:
                    # 测试字体是否可用
                    import tkinter.font as tkFont
                    test_font = tkFont.Font(family=font_name, size=12)
                    if test_font.actual('family') == font_name or font_name in ['Arial', 'DejaVu Sans']:
                        font_found = font_name
                        break
                except Exception:
                    continue

            if not font_found:
                font_found = 'Arial'

            print(f"🔤 使用字体: {font_found}")

            try:
                style.configure('TLabel', background='#0D1117', foreground='#E6EDF3', font=(font_found, 12))
                style.configure('Header.TLabel', background='#0D1117', foreground='#E6EDF3', font=(font_found, 14, 'bold'))
                style.configure('TButton', font=(font_found, 12))
                style.configure('Status.TLabel', background='#0D1117', foreground='#58A6FF', font=(font_found, 11))
                style.configure('Task.TLabel', background='#0D1117', foreground='#7DD3FC', font=(font_found, 11))
            except Exception as e:
                print(f"⚠ 字体配置失败: {e}")
                # 最终回退到系统默认字体
                style.configure('TLabel', background='#0D1117', foreground='#E6EDF3')
                style.configure('Header.TLabel', background='#0D1117', foreground='#E6EDF3')
                style.configure('TButton', background='#0D1117', foreground='#E6EDF3')
                style.configure('Status.TLabel', background='#0D1117', foreground='#58A6FF')
                style.configure('Task.TLabel', background='#0D1117', foreground='#7DD3FC')


            # 关闭逻辑
            def _on_close():
                try:
                    globals()['user_command_mode'] = False
                    globals()['_command_breakdown_callback'] = None  # 清理回调函数
                finally:
                    try:
                        root.destroy()
                    except Exception:
                        pass
            root.protocol('WM_DELETE_WINDOW', lambda: _on_close())
            root.bind('<Escape>', lambda e: _on_close())

            # 主布局：左右分栏 + 底部状态栏
            main = ttk.Frame(root)
            main.pack(fill='both', expand=True)
            main.columnconfigure(0, weight=3)
            main.columnconfigure(1, weight=4)
            main.rowconfigure(0, weight=1)
            main.rowconfigure(1, weight=0)  # 底部状态区域

            # 左侧：命令输入
            left = ttk.Frame(main)
            left.grid(row=0, column=0, sticky='nsew', padx=10, pady=10)
            left.rowconfigure(1, weight=1)  # 让文本框可扩展
            left.rowconfigure(3, weight=1)  # 让状态区域可扩展

            ttk.Label(left, text='📝 命令输入', style='Header.TLabel').grid(row=0, column=0, sticky='w', pady=(0,6))
            txt = tk.Text(left, height=6, wrap='word', bg='#101418', fg='#E6EDF3', insertbackground='#E6EDF3', relief='flat', font=(font_found, 12))
            txt.grid(row=1, column=0, sticky='nsew', pady=(0,8))

            btns = ttk.Frame(left)
            btns.grid(row=2, column=0, sticky='ew', pady=(0,8))
            ttk.Button(btns, text='提交 (Enter)', command=lambda: (_handle_user_command_text(txt.get('1.0','end')), txt.delete('1.0','end'))).pack(side='left')
            ttk.Button(btns, text='清空', command=lambda: txt.delete('1.0','end')).pack(side='left', padx=8)

            ttk.Label(left, text='示例：把闹钟捡起来放在桌上 / 打开台灯 / 把地上的碗放到台面', foreground='#8B949E').grid(row=3, column=0, sticky='w')

            # 左下方：当前任务和目标状态
            status_frame = ttk.Frame(left)
            status_frame.grid(row=4, column=0, sticky='nsew', pady=(20,0))
            status_frame.rowconfigure(1, weight=1)

            ttk.Label(status_frame, text='🎯 当前状态', style='Header.TLabel').grid(row=0, column=0, sticky='w', pady=(0,6))

            # 当前任务显示
            current_task_var = tk.StringVar(value="等待任务...")
            current_goal_var = tk.StringVar(value="无目标")
            command_breakdown_var = tk.StringVar(value="")

            # 设置命令分解回调函数
            def update_command_breakdown(text):
                command_breakdown_var.set(text)
            _command_breakdown_callback = update_command_breakdown

            ttk.Label(status_frame, text='正在执行:', style='Status.TLabel').grid(row=1, column=0, sticky='w')
            current_task_label = ttk.Label(status_frame, textvariable=current_task_var, style='Task.TLabel', wraplength=300)
            current_task_label.grid(row=2, column=0, sticky='w', padx=(20,0))

            ttk.Label(status_frame, text='当前目标:', style='Status.TLabel').grid(row=3, column=0, sticky='w', pady=(10,0))
            current_goal_label = ttk.Label(status_frame, textvariable=current_goal_var, style='Task.TLabel', wraplength=300)
            current_goal_label.grid(row=4, column=0, sticky='w', padx=(20,0))

            ttk.Label(status_frame, text='命令分解:', style='Status.TLabel').grid(row=5, column=0, sticky='w', pady=(10,0))
            command_breakdown_label = ttk.Label(status_frame, textvariable=command_breakdown_var, style='Task.TLabel', wraplength=300)
            command_breakdown_label.grid(row=6, column=0, sticky='w', padx=(20,0))

            # 学习状态显示
            learning_status_var = tk.StringVar(value="")
            ttk.Label(status_frame, text='学习状态:', style='Status.TLabel').grid(row=7, column=0, sticky='w', pady=(10,0))
            learning_status_label = ttk.Label(status_frame, textvariable=learning_status_var, style='Task.TLabel', wraplength=300)
            learning_status_label.grid(row=8, column=0, sticky='w', padx=(20,0))

            # 绑定回车提交并阻止换行
            txt.bind('<Return>', lambda e: (_handle_user_command_text(txt.get('1.0','end')), txt.delete('1.0','end'), 'break'))

            # 右侧：任务列表（按优先级从高到低显示）
            right = ttk.Frame(main)
            right.grid(row=0, column=1, sticky='nsew', padx=10, pady=10)
            right.rowconfigure(1, weight=2)
            right.rowconfigure(3, weight=3)
            ttk.Label(right, text='📋 任务列表（优先级高的在上面）', style='Header.TLabel').grid(row=0, column=0, sticky='w')
            list_frame = ttk.Frame(right)
            list_frame.grid(row=1, column=0, sticky='nsew', pady=(6,0))
            sb = ttk.Scrollbar(list_frame, orient='vertical')
            lb = tk.Listbox(list_frame, yscrollcommand=sb.set, bg='#0D1117', fg='#E6EDF3', highlightthickness=0, selectbackground='#1F6FEB', relief='flat', width=68, font=(font_found, 11))
            sb.config(command=lb.yview)
            lb.pack(side='left', fill='both', expand=True)
            sb.pack(side='right', fill='y')

            ttk.Label(right, text='🧠 3-LLM 上层任务与验收', style='Header.TLabel').grid(row=2, column=0, sticky='w', pady=(10,0))
            board_frame = ttk.Frame(right)
            board_frame.grid(row=3, column=0, sticky='nsew', pady=(6,0))
            board_frame.rowconfigure(0, weight=1)
            board_frame.rowconfigure(1, weight=1)
            board_frame.columnconfigure(0, weight=1)

            board_macro_lb = tk.Listbox(board_frame, bg='#0D1117', fg='#E6EDF3', highlightthickness=0,
                                        selectbackground='#1F6FEB', relief='flat', width=68, font=(font_found, 10))
            board_verify_lb = tk.Listbox(board_frame, bg='#0D1117', fg='#E6EDF3', highlightthickness=0,
                                         selectbackground='#1F6FEB', relief='flat', width=68, font=(font_found, 10))
            board_macro_lb.grid(row=0, column=0, sticky='nsew', pady=(0,6))
            board_verify_lb.grid(row=1, column=0, sticky='nsew')

            def _shorten(d: dict, maxlen: int = 88) -> str:
                try:
                    s = str(d)
                    return s if len(s) <= maxlen else s[:maxlen-1] + '…'
                except Exception:
                    return ''

            def refresh_list():
                lb.delete(0, 'end')
                items = _plan_summary_for_llm(64)
                board_macro_lb.delete(0, 'end')
                board_verify_lb.delete(0, 'end')

                # 更新当前任务状态
                if items:
                    current_item = items[0]  # 第一个是当前正在执行的
                    current_task_var.set(f"{current_item['action']} {_shorten(current_item.get('params',{}), 40)}")

                    # 尝试提取目标信息
                    params = current_item.get('params', {})
                    if 'objectId' in params:
                        current_goal_var.set(f"目标对象: {params['objectId']}")
                    elif 'receptacleObjectId' in params:
                        current_goal_var.set(f"目标容器: {params['receptacleObjectId']}")
                    else:
                        current_goal_var.set("执行中...")
                else:
                    current_task_var.set("等待任务...")
                    current_goal_var.set("无目标")

                # 更新学习状态
                if _preference_scores:
                    learned_count = len(_preference_scores)
                    total_corrections = len(_correction_history)
                    top_learned = []
                    for obj_type, scores in list(_preference_scores.items())[:3]:
                        best_target = max(scores.keys(), key=lambda k: scores[k])
                        top_learned.append(f"{obj_type}→{best_target}")
                    learning_text = f"已学习{learned_count}类物体，{total_corrections}次修正\n" + "；".join(top_learned)
                    learning_status_var.set(learning_text)
                else:
                    learning_status_var.set("暂无学习数据")

                # 填充任务列表
                for a in items:
                    p = int(a.get('priority', 0))
                    icon = '🔴' if p >= 80 else ('🟡' if p >= 60 else '🟢')
                    idx = lb.size()
                    is_current = idx == 0  # 第一个是当前任务
                    prefix = "▶ " if is_current else "  "
                    lb.insert('end', f"{prefix}{icon} [{a['idx']:02d}] P{p:03d} {a['action']} {_shorten(a.get('params',{}))}")
                    # Problem-agent  steps in red
                    # 当前任务高亮显示
                    if is_current:
                        try:
                            lb.itemconfig(idx, foreground='#7DD3FC', selectforeground='#7DD3FC')
                        except Exception:
                            pass
                    elif str(a.get('source') or '') == 'problem_agent':
                        try:
                            lb.itemconfig(idx, foreground='#FF5A5A')
                        except Exception:
                            pass

                # 3-LLM上层任务看板
                board = _llm_task_board_for_ui(24)
                b_state = board.get('state', 'idle')
                generated = board.get('generated_tasks') or []
                verification = board.get('verification') or []

                board_macro_lb.insert('end', f"[状态] {b_state}")
                if not generated:
                    board_macro_lb.insert('end', "[上层任务] 暂无")
                else:
                    board_macro_lb.insert('end', "[上层任务]")
                    for t in generated:
                        status = t.get('status', '待执行')
                        icon = '✅' if status == '已完成' else ('❌' if status == '未完成' else '⏳')
                        issue = str(t.get('issue_description') or '')
                        action = str(t.get('implied_action') or '')
                        board_macro_lb.insert('end', f"{icon} {status} | {action} | {issue[:58]}")

                if not verification:
                    board_verify_lb.insert('end', "[验收结果] 待任务执行完成后生成")
                else:
                    board_verify_lb.insert('end', "[验收结果]")
                    for v in verification:
                        status = v.get('status', '待人工确认')
                        icon = '✅' if status == '已完成' else ('❌' if status == '未完成' else '🟡')
                        reason = str(v.get('reason') or '')
                        issue = str(v.get('issue_description') or '')
                        board_verify_lb.insert('end', f"{icon} {status} | {issue[:36]} | {reason[:40]}")

                root.after(800, refresh_list)
            refresh_list()

            # 底部状态栏
            status = ttk.Label(root, text='Esc 关闭 • Enter 提交 • m 可再次触发场景理解', anchor='w', font=(font_found, 11))
            status.pack(fill='x', padx=10, pady=(0,6))

            # 键盘快捷
            def _on_key(event):
                if event.keysym == 'Return':
                    _handle_user_command_text(txt.get('1.0','end'))
                    txt.delete('1.0','end')
                elif event.keysym == 'Escape':
                    _on_close()
            root.bind('<Key>', _on_key)

            root.mainloop()
        except Exception as _e:
            print(f"⚠ 命令窗口创建失败: {_e}")
        finally:
            user_command_mode = False
            _command_breakdown_callback = None  # 清理回调函数

    t = _threading.Thread(target=_run, daemon=True)
    t.start()




def _apply_llm_decision(decision: dict, default_reason: str, current_step_only: bool = True) -> bool:
    """依据LLM返回的决策修改计划队列。
    返回True表示已处理（不再执行默认跳过逻辑）；False表示未处理。
    支持：
      - {decision: "skip"}
      - {decision: "replan"}
      - {decision: "replace_steps", steps: [...]}  # 用新步骤替换当前一步
      - {decision: "insert_steps", steps: [...]}   # 在当前一步之前插入
    说明：当 current_step_only=True（默认，用于问题求解LLM），不允许触发全局“三LLM”重规划，
    遇到 replan 将就地处理为“跳过当前步”或“仅在当前步前插入替代步骤”。
    """
    global planned_actions, executing_plan
    if not isinstance(decision, dict):
        return False
    d = (decision.get('decision') or '').lower()
    steps = decision.get('steps') or []

    # 1) 明确跳过
    if d in ('skip', 'skip_step'):
        with _plan_lock:
            if planned_actions:
                planned_actions.pop(0)
                if not planned_actions:
                    executing_plan = False
        return True

    # 2) 重规划：根据 current_step_only 选择就地处理还是触发全局重规划
    if d in ('replan','re_plan','rethink'):
        if current_step_only:
            # 不触发三LLM，仅就地处理当前步：若给了 steps，当做 insert；否则跳过
            if isinstance(steps, list) and steps:
                src = decision.get('source') or decision.get('decision_source')
                if src:
                    for s in steps:
                        if isinstance(s, dict) and 'source' not in s:
                            s['source'] = src
                with _plan_lock:
                    safe = _sanitize_planned_actions(steps)
                    planned_actions = safe + planned_actions
                    executing_plan = True
                return True
            else:
                with _plan_lock:
                    if planned_actions:
                        planned_actions.pop(0)
                        if not planned_actions:
                            executing_plan = False
                return True
        else:
            # 允许触发全局重规划（仅在明确需要时才会使用）
            try:
                _maybe_trigger_replan(None, reason=f"LLM裁决：{default_reason}")
            except Exception:
                pass
            with _plan_lock:
                if planned_actions:
                    planned_actions.pop(0)
                    if not planned_actions:
                        executing_plan = False
            return True

    # 3) 用新步骤替换当前一步
    if d in ('replace_steps','replace') and isinstance(steps, list) and steps:
        src = decision.get('source') or decision.get('decision_source')
        if src:
            for s in steps:
                if isinstance(s, dict) and 'source' not in s:
                    s['source'] = src
        with _plan_lock:
            if planned_actions:
                planned_actions.pop(0)
                safe = _sanitize_planned_actions(steps)
                planned_actions = safe + planned_actions
                executing_plan = True
        return True

    # 4) 在当前一步之前插入
    if d in ('insert_steps','insert') and isinstance(steps, list) and steps:
        src = decision.get('source') or decision.get('decision_source')
        if src:
            for s in steps:
                if isinstance(s, dict) and 'source' not in s:
                    s['source'] = src
        with _plan_lock:
            safe = _sanitize_planned_actions(steps)
            planned_actions = safe + planned_actions
            executing_plan = True
        return True

    return False


def _on_action_failure_llm(name: str, params: dict, event, err_msg: str) -> bool:
    """在动作失败时将上下文发给LLM裁决。返回True表示已处理队列。"""
    should_trigger, fail_cnt, sig = _should_trigger_problem_llm(name, params, err_msg)
    if not should_trigger:
        print(f"⏳ 同类错误累计 {fail_cnt}/{PROBLEM_LLM_TRIGGER_COUNT}，暂不触发Problem-LLM: {sig[0]} | {sig[1]}")
        return False

    ctx = _build_failure_context(event, name, params, err_msg)

    # 添加失败历史上下文
    if name == "GoTo" and "objectId" in params:
        fail_count = _goto_failure_counts.get(params["objectId"], 0)
        ctx['failure_history'] = f"GoTo this object has already failed {fail_count} times."
    elif name in ["OpenObject", "CloseObject", "PickupObject", "PutObject"] and "objectId" in params:
        # 对于其他动作，检查其目标对象的GoTo失败历史
        target_id = params.get("objectId") or params.get("receptacleObjectId")
        if target_id:
            fail_count = _goto_failure_counts.get(target_id, 0)
            if fail_count > 0:
                ctx['failure_history'] = f"Navigation to this target has failed {fail_count} times."

    prompt = (
        "以下是机器人在执行一步动作时遇到的错误。你是专门的'问题求解智能体'，职责是：\n"
        "1) 先判断失败原因（如：手上拿着不相关物体/目标容器未打开/路径被柜门阻挡/目标不可达等）；\n"
        "2) 给出解决方案，并细化为 AI2-THOR 可执行的动作序列（如先 Put 到最近台面、OpenObject 柜门、重新 GoTo 等）。\n"
        "重要：如果 failure_history 显示导航到目标已多次失败，说明目标可能不可达，应考虑 skip 而非继续重试。\n"
        "只输出严格 JSON，不要解释其它文字。允许的决策：\n"
        "- {\"decision\": \"skip\"}\n"
        "- {\"decision\": \"replan\"}\n"
        "- {\"decision\": \"replace_steps\", \"steps\": [{\"action\": ..., \"params\": {...}}, ...]}\n"
        "附加要求：\n- 尽量在 steps 中给出具体 objectId/receptacleObjectId；若只知道类型也可先给 objectType，执行端会就近补全。\n"
        "- 遇到 PutObject 的容器问题（如无可放置点）时，优先返回可替换的可用容器，并用 replace_steps 替换当前任务。\n"
        "- 遇到 PutObject 失败且手中有物体时，先 Put 到最近的安全表面（Table/CounterTop/Shelf/Bed）。\n- 遇到 GoTo 失败且有柜门阻挡时，先 OpenObject 再 GoTo。\n"
        f"上下文: {_json.dumps(ctx, ensure_ascii=False)}"
    )
    print(f"🧠 Problem-LLM 提交: action={name}, error={err_msg}")
    # 使用独立的问题求解LLM通道（与场景理解/命令解析分离）
    decision = _call_problem_llm(prompt)
    if decision is None:
        local_decision = _build_local_problem_recovery_decision(name, params, event, err_msg)
        if isinstance(local_decision, dict):
            print("🧠 Problem-LLM 无响应，启用本地兜底修复策略")
            try:
                local_decision['source'] = 'problem_agent_local'
            except Exception:
                pass
            return _apply_llm_decision(local_decision, default_reason=f"{name} 本地兜底")

        print("🧠 Problem-LLM 未配置或无有效响应 -> 默认跳过当前步骤，继续后续任务")
        with _plan_lock:
            if planned_actions:
                planned_actions.pop(0)
                if not planned_actions:
                    globals()['executing_plan'] = False
        return True
    else:
        try:
            _preview = _json.dumps({k: decision.get(k) for k in ('decision','steps')}, ensure_ascii=False)[:180]
            print(f"🧠 Problem-LLM 决策: {_preview} ...")
        except Exception:
            pass
        # 标注来源，便于 UI 用红色显示
        if isinstance(decision, dict):
            decision['source'] = 'problem_agent'
    return _apply_llm_decision(decision, default_reason=f"{name} 失败：{err_msg}")

def execute_next_planned_action(controller, event):
    """从计划队列执行一步（含导航与交互）。返回新的 event。"""
    global planned_actions, executing_plan, _nav_state, semantic_map, _action_retry_count, _hand_clear_put_fail_counts
    # 先取当前要执行的动作（peek），避免在导航未完成前弹出
    with _plan_lock:
        if not executing_plan or not planned_actions:
            executing_plan = False
            return event
        step = planned_actions[0]
    _record_llm_executed_step_started(step)
    name = step.get('action')
    params = step.get('params') or {}

    # 生成动作的唯一标识符
    action_key = f"{name}_{params.get('objectId', '')}"

    # 检查重试次数
    retry_count = _action_retry_count.get(action_key, 0)
    if retry_count > 3:  # 降低重试次数从8到3，避免无限循环
        print(f"⚠ 动作 {action_key} 重试次数过多，跳过")
        with _plan_lock:
            planned_actions.pop(0)
            if not planned_actions:
                executing_plan = False
        _action_retry_count.pop(action_key, None)
        return event

    _action_retry_count[action_key] = retry_count + 1

    # 验证对象是否存在于当前场景中
    if 'objectId' in params:
        oid = params.get('objectId')
        resolved = _resolve_object_id_from_hint(event, oid)
        if resolved is None:
            print(f"⚠ 对象 {oid} 在当前场景中不存在，跳过该动作")
            print(f"   可能原因：1) 对象未被探索到 2) 对象ID不正确 3) 对象在其他房间")
            with _plan_lock:
                planned_actions.pop(0)
                if not planned_actions:
                    executing_plan = False
            _action_retry_count.pop(action_key, None)
            return event

    try:
        # 1) 高层导航：GoTo(objectId=...) - 使用AI2-THOR内置NavMesh
        if name == "GoTo":
            # 检查是否已经在导航中（包括 action_sequence 模式和 TSET_STYLE 模式）
            if _nav_state.get("active") and (_nav_state.get("action_sequence") or _nav_state.get("method") == "TSET_STYLE"):
                # 检查导航超时
                start_time = _nav_state.get("start_time", 0)
                timeout = _nav_state.get("timeout", 30.0)
                if time.time() - start_time > timeout:
                    print(f"⚠ 导航超时({timeout}秒)，跳过该动作")
                    _nav_state = {"active": False, "target": None, "action_sequence": None, "idx": 0, "start_time": None, "timeout": 30.0}
                    with _plan_lock:
                        planned_actions.pop(0)
                        if not planned_actions:
                            executing_plan = False
                    return event

                # 继续执行当前导航
                event, reached = _execute_navmesh_navigation(controller, event)
                if reached:
                    # 读取结果并记录（成功/失败）
                    tgt_id = _nav_state.get("target")
                    success_nav = not _nav_state.get("last_failed", False)
                    # 完成导航：清空状态并出队
                    _nav_state = {"active": False, "target": None, "action_sequence": None, "idx": 0, "start_time": None, "timeout": 30.0}
                    with _plan_lock:
                        planned_actions.pop(0)
                        if not planned_actions:
                            executing_plan = False
                    # 清理重试计数
                    _action_retry_count.pop(action_key, None)
                    # 记录GoTo结果
                    _mark_goto_result(tgt_id, success_nav)
                    print(f"✅ 导航完成")
                return event

            # 开始新的导航 - 使用AI2-THOR内置NavMesh
            oid = params.get('objectId')
            resolved = _resolve_object_id_from_hint(event, oid)
            if resolved is None:
                print(f"⚠ 未能解析对象ID: {oid}，稍后重试")
                return event

            print(f"✅ 已解析对象: {oid} -> {resolved}，位置: {_obj_pos(event, resolved)}")


            # 预处理：关闭可能阻塞通路的打开柜门/抽屉
            try:
                event, closed_ids = _preclose_blocking_openables(controller, event, resolved)
                if closed_ids:
                    print(f"🧹 为通路清障，已预关闭 {len(closed_ids)} 个打开物体: {closed_ids}")
            except Exception:
                pass

            # 尝试使用AI2-THOR内置的NavMesh导航
            if _should_skip_goto(resolved):
                print("⏭️ 连续失败多次，跳过本次 GoTo")
                with _plan_lock:
                    planned_actions.pop(0)
                    if not planned_actions:
                        executing_plan = False
                _action_retry_count.pop(action_key, None)
                return event
            success = _start_navmesh_navigation(controller, event, resolved)
            if not success:
                print(f"⚠ 无法规划到目标的路径")
                _mark_goto_result(resolved, False)
                # 交给 LLM 决策：改策略/跳过/重规划
                try:
                    handled = _on_action_failure_llm("GoTo", {"objectId": resolved}, event, "NavMesh无法规划到目标")
                    if handled:
                        return event
                except Exception as _e:
                    print(f"⚠ LLM 决策失败，采用默认跳过: {_e}")
                with _plan_lock:
                    planned_actions.pop(0)
                    if not planned_actions:
                        executing_plan = False
                return event

            # 开始导航的第一步
            event, reached = _execute_navmesh_navigation(controller, event)
            if reached:
                # 读取结果并记录（成功/失败）
                tgt_id = _nav_state.get("target")
                success_nav = not _nav_state.get("last_failed", False)
                # 完成导航：清空状态并出队
                _nav_state = {"active": False, "target": None, "action_sequence": None, "idx": 0, "start_time": None, "timeout": 30.0}
                with _plan_lock:
                    planned_actions.pop(0)
                    if not planned_actions:
                        executing_plan = False
                # 清理重试计数
                _action_retry_count.pop(action_key, None)
                # 记录GoTo结果
                _mark_goto_result(tgt_id, success_nav)
                print(f"✅ 导航完成")
            return event

        # 2) 交互动作（一次性执行并出队）
        elif name == "OpenObject":
            # 打开对象（柜子、抽屉等）
            oid = params.get('objectId')
            resolved = _resolve_object_id_from_hint(event, oid)
            if resolved is None:
                print(f"⚠ 未能解析对象ID: {oid}")
                with _plan_lock:
                    planned_actions.pop(0)
                    if not planned_actions:
                        executing_plan = False
                return event
            elif _should_skip_target_completely(resolved):
                print(f"⏭️ 目标不可达，跳过 OpenObject: {resolved}")
                with _plan_lock:
                    planned_actions.pop(0)
                    if not planned_actions:
                        executing_plan = False
                _action_retry_count.pop(action_key, None)
                return event

            # 检查距离
            agent_pos = _agent_pos(event)
            obj_pos = _obj_pos(event, resolved)
            if obj_pos:
                distance = _distance(agent_pos[0], agent_pos[1], obj_pos[0], obj_pos[1])
                if distance > 1.5:
                    print(f"⚠ Agent距离对象太远，无法打开: {resolved}")
                    print(f"   Agent位置: {agent_pos}")
                    print(f"   对象位置: {obj_pos}")
                    print(f"   距离: {distance:.2f}m > 打开范围: 1.5m")
                    print("   自动插入GoTo动作导航到对象位置")

                    # 自动插入GoTo动作到队列前面
                    with _plan_lock:
                        if _should_skip_goto(resolved):
                            print(f"   ⏭️ 跳过插入导航: GoTo({resolved})")
                        else:
                            goto_action = {'action': 'GoTo', 'params': {'objectId': resolved}}
                            planned_actions.insert(0, goto_action)  # 插入到当前动作前面
                            print(f"   已插入导航动作: GoTo({resolved})")

                    # 如果因为连续失败而跳过了GoTo，则同时处理当前OpenObject，避免原地死循环
                    if _should_skip_goto(resolved):
                        try:
                            handled = _on_action_failure_llm("OpenObject", {"objectId": resolved}, event, "导航到对象多次失败，放弃本次打开")
                            if handled:
                                _action_retry_count.pop(action_key, None)
                                return event
                        except Exception as _e:
                            print(f"⚠ LLM 决策失败（OpenObject-跳过分支），采用默认跳过: {_e}")
                        with _plan_lock:
                            if planned_actions:
                                planned_actions.pop(0)
                                if not planned_actions:
                                    executing_plan = False
                        _action_retry_count.pop(action_key, None)
                        return event

                    # 清理重试计数，避免无限重试
                    _action_retry_count.pop(action_key, None)
                    return event

            print(f"🚪 尝试打开对象: {resolved}")
            open_event = controller.step(action="OpenObject", objectId=resolved, forceAction=True)
            if open_event.metadata.get("lastActionSuccess", False):
                print(f"✅ 成功打开: {resolved}")
            else:
                error_msg = open_event.metadata.get("errorMessage", "Unknown error")
                print(f"❌ 打开失败: {error_msg}")
                # 失败也交给LLM裁决
                try:
                    handled = _on_action_failure_llm("OpenObject", {"objectId": resolved}, open_event, str(error_msg))
                    if handled:
                        return open_event
                except Exception as _e:
                    print(f"⚠ LLM 决策失败，采用默认跳过: {_e}")

            with _plan_lock:
                planned_actions.pop(0)
                if not planned_actions:
                    executing_plan = False
            return open_event

        elif name == "CloseObject":
            # 关闭对象（柜子、抽屉等）
            oid = params.get('objectId')
            resolved = _resolve_object_id_from_hint(event, oid)
            if resolved is None:
                print(f"⚠ 未能解析对象ID: {oid}")
                with _plan_lock:
                    planned_actions.pop(0)
                    if not planned_actions:
                        executing_plan = False
                return event
            elif _should_skip_target_completely(resolved):
                print(f"⏭️ 目标不可达，跳过 CloseObject: {resolved}")
                with _plan_lock:
                    planned_actions.pop(0)
                    if not planned_actions:
                        executing_plan = False
                _action_retry_count.pop(action_key, None)
                return event

            # 检查距离
            agent_pos = _agent_pos(event)
            obj_pos = _obj_pos(event, resolved)
            if obj_pos:
                distance = _distance(agent_pos[0], agent_pos[1], obj_pos[0], obj_pos[1])
                if distance > 1.5:
                    print(f"⚠ Agent距离对象太远，无法关闭: {resolved}")
                    print(f"   Agent位置: ({agent_pos[0]:.2f}, {agent_pos[1]:.2f})")
                    print(f"   对象位置: ({obj_pos[0]:.2f}, {obj_pos[1]:.2f})")
                    print(f"   距离: {distance:.2f}m > 关闭范围: 1.5m")
                    print("   自动插入GoTo动作导航到对象位置")

                    # 自动插入GoTo动作到队列前面
                    with _plan_lock:
                        if _should_skip_goto(resolved):
                            print(f"   ⏭️ 跳过插入导航: GoTo({resolved})")
                        else:
                            goto_action = {'action': 'GoTo', 'params': {'objectId': resolved}}
                            planned_actions.insert(0, goto_action)  # 插入到当前动作前面
                            print(f"   已插入导航动作: GoTo({resolved})")

                    # 如果因为连续失败而跳过了GoTo，则同时处理当前CloseObject，避免原地死循环
                    if _should_skip_goto(resolved):
                        try:
                            handled = _on_action_failure_llm("CloseObject", {"objectId": resolved}, event, "导航到对象多次失败，放弃本次关闭")
                            if handled:
                                _action_retry_count.pop(action_key, None)
                                return event
                        except Exception as _e:
                            print(f"⚠ LLM 决策失败（CloseObject-跳过分支），采用默认跳过: {_e}")
                        with _plan_lock:
                            if planned_actions:
                                planned_actions.pop(0)
                                if not planned_actions:
                                    executing_plan = False
                        _action_retry_count.pop(action_key, None)
                        return event

                    # 清理重试计数，避免无限重试
                    _action_retry_count.pop(action_key, None)
                    return event

            print(f"🚪 尝试关闭对象: {resolved}")
            close_event = controller.step(action="CloseObject", objectId=resolved, forceAction=True)
            if close_event.metadata.get("lastActionSuccess", False):
                print(f"✅ 成功关闭: {resolved}")
            else:
                error_msg = close_event.metadata.get("errorMessage", "Unknown error")
                print(f"❌ 关闭失败: {error_msg}")
                try:
                    handled = _on_action_failure_llm("CloseObject", {"objectId": resolved}, close_event, str(error_msg))
                    if handled:
                        return close_event
                except Exception as _e:
                    print(f"⚠ LLM 决策失败，采用默认跳过: {_e}")

            with _plan_lock:
                planned_actions.pop(0)
                if not planned_actions:
                    executing_plan = False
            return close_event

        elif name == "ToggleObjectOn":
            # 打开设备（灯、电视等）
            oid = params.get('objectId')
            resolved = _resolve_object_id_from_hint(event, oid)
            if resolved is None:
                print(f"⚠ 未能解析对象ID: {oid}")
                with _plan_lock:
                    planned_actions.pop(0)
                    if not planned_actions:
                        executing_plan = False
                return event

            print(f"💡 尝试打开设备: {resolved}")
            toggle_event = controller.step(action="ToggleObjectOn", objectId=resolved, forceAction=True)
            if toggle_event.metadata.get("lastActionSuccess", False):
                print(f"✅ 成功打开设备: {resolved}")
            else:
                error_msg = toggle_event.metadata.get("errorMessage", "Unknown error")
                print(f"❌ 打开设备失败: {error_msg}")
                try:
                    handled = _on_action_failure_llm("ToggleObjectOn", {"objectId": resolved}, toggle_event, str(error_msg))
                    if handled:
                        return toggle_event
                except Exception as _e:
                    print(f"⚠ LLM 决策失败，采用默认跳过: {_e}")

            with _plan_lock:
                planned_actions.pop(0)
                if not planned_actions:
                    executing_plan = False
            return toggle_event

        elif name == "ToggleObjectOff":
            # 关闭设备（灯、电视等）
            oid = params.get('objectId')
            resolved = _resolve_object_id_from_hint(event, oid)
            if resolved is None:
                print(f"⚠ 未能解析对象ID: {oid}")
                with _plan_lock:
                    planned_actions.pop(0)
                    if not planned_actions:
                        executing_plan = False
                return event

            print(f"💡 尝试关闭设备: {resolved}")
            toggle_event = controller.step(action="ToggleObjectOff", objectId=resolved, forceAction=True)
            if toggle_event.metadata.get("lastActionSuccess", False):
                print(f"✅ 成功关闭设备: {resolved}")
            else:
                error_msg = toggle_event.metadata.get("errorMessage", "Unknown error")
                print(f"❌ 关闭设备失败: {error_msg}")
                try:
                    handled = _on_action_failure_llm("ToggleObjectOff", {"objectId": resolved}, toggle_event, str(error_msg))
                    if handled:
                        return toggle_event
                except Exception as _e:
                    print(f"⚠ LLM 决策失败，采用默认跳过: {_e}")

            with _plan_lock:
                planned_actions.pop(0)
                if not planned_actions:
                    executing_plan = False
            return toggle_event

        elif name in ("CleanObject", "DirtyObject"):
            # 清洁/弄脏对象：若不在范围内自动插入 GoTo，随后执行 CleanObject/DirtyObject
            oid = params.get('objectId')
            resolved = _resolve_object_id_from_hint(event, oid)
            if resolved is None:
                print(f"⚠ 未能解析对象ID: {oid}")
                with _plan_lock:
                    planned_actions.pop(0)
                    if not planned_actions:
                        executing_plan = False
                return event

            elif _should_skip_target_completely(resolved):
                print(f"⏭️ 目标不可达，跳过 ToggleObjectOn: {resolved}")
                with _plan_lock:
                    planned_actions.pop(0)
                    if not planned_actions:
                        executing_plan = False
                _action_retry_count.pop(action_key, None)
                return event

            # 距离检查（与交互一致约1.5m）
            agent_pos = _agent_pos(event)
            obj_pos = _obj_pos(event, resolved)
            if obj_pos is None:
                print(f"⚠ 无法获取对象位置: {resolved}")
                with _plan_lock:
                    planned_actions.pop(0)
                    if not planned_actions:
                        executing_plan = False
                return event
            distance = _distance(agent_pos[0], agent_pos[1], obj_pos[0], obj_pos[1])
            if distance > 1.5:
                print(f"⚠ 距离对象过远，自动插入GoTo: {resolved}")
                with _plan_lock:
                    if _should_skip_goto(resolved):
                        print(f"   ⏭️ 跳过插入导航: GoTo({resolved})")
                    else:
                        planned_actions.insert(0, {'action': 'GoTo', 'params': {'objectId': resolved}})

                # 如果因为连续失败而跳过了GoTo，则同时处理当前ToggleObjectOn，避免原地死循环
                if _should_skip_goto(resolved):
                    try:
                        handled = _on_action_failure_llm("ToggleObjectOn", {"objectId": resolved}, event, "导航到对象多次失败，放弃本次开启")
                        if handled:
                            _action_retry_count.pop(action_key, None)
                            return event
                    except Exception as _e:
                        print(f"⚠ LLM 决策失败（ToggleObjectOn-跳过分支），采用默认跳过: {_e}")
                    with _plan_lock:
                        if planned_actions:
                            planned_actions.pop(0)
                            if not planned_actions:
                                executing_plan = False
                    _action_retry_count.pop(action_key, None)
                    return event

                _action_retry_count.pop(action_key, None)
                return event

            # 执行 CleanObject / DirtyObject
            # 地图标注本次清洁/弄脏操作
            try:
                pos_mark = _obj_pos(event, resolved)
                if pos_mark:
                    semantic_map.setdefault("task_points", []).append({"x": pos_mark[0], "z": pos_mark[1], "label": f"{name}->{resolved}"})
            except Exception:
                pass

            print(f"🧽 执行 {name} -> {resolved}")
            ev2 = controller.step(action=name, objectId=resolved, forceAction=True)
            if ev2.metadata.get('lastActionSuccess', False):
                print(f"✅ {name} 成功: {resolved}")
                event = ev2
            else:
                err = ev2.metadata.get('errorMessage', 'Unknown error')
                print(f"⚠ {name} 失败: {err}")
                # 交给LLM裁决下一步（跳过/替换/重规划）
                try:
                    handled = _on_action_failure_llm(name, {"objectId": resolved}, ev2, str(err))
                    if handled:
                        return ev2
                except Exception as _e:
                    print(f"⚠ LLM 决策失败，采用默认跳过: {_e}")
                event = ev2
            with _plan_lock:
                planned_actions.pop(0)
                if not planned_actions:
                    executing_plan = False
            return event
        elif name == "PickupObject":
            oid = params.get('objectId')
            resolved = _resolve_object_id_from_hint(event, oid)
            if resolved is None:
                print(f"⚠ 未能解析可拾取的 objectId: {oid}，跳过")
            elif _should_skip_target_completely(resolved):
                print(f"⏭️ 目标不可达，跳过 PickupObject: {resolved}")
                with _plan_lock:
                    planned_actions.pop(0)
                    if not planned_actions:
                        executing_plan = False
                _action_retry_count.pop(action_key, None)
                return event
            else:
                # 首先检查Agent手中是否已有物体
                agent_inventory = event.metadata.get("inventoryObjects", [])
                if agent_inventory:
                    current_object = agent_inventory[0].get("objectId")
                    print(f"⚠ Agent手中已有物体: {current_object} -> 先安全放置再拾取")

                    fail_n = _hand_clear_put_fail_counts.get(current_object, 0)
                    if fail_n >= 2:
                        print(f"🛟 清手放置已连续失败 {fail_n} 次，改用 DropHandObject 兜底")
                        drop_ev = controller.step(action='DropHandObject', forceAction=True)
                        if drop_ev.metadata.get('lastActionSuccess', False):
                            print("✅ 已执行 DropHandObject，下一步将重试当前拾取")
                            _hand_clear_put_fail_counts.pop(current_object, None)
                            _action_retry_count.pop(action_key, None)
                            return drop_ev
                        else:
                            print(f"⚠ DropHandObject 失败: {drop_ev.metadata.get('errorMessage', 'Unknown error')}，跳过当前拾取避免卡住")
                            with _plan_lock:
                                planned_actions.pop(0)
                                if not planned_actions:
                                    executing_plan = False
                            _action_retry_count.pop(action_key, None)
                            return drop_ev

                    # 选择最近的安全容器（桌面/台面等），优先可视
                    safe_dst = _find_nearest_safe_receptacle(event)
                    if safe_dst:
                        with _plan_lock:
                            # 插入一个将当前物体放置到安全容器的动作；PutObject 会在必要时自动插入 GoTo
                            planned_actions.insert(0, {"action": "PutObject", "params": {"objectId": current_object, "receptacleObjectId": safe_dst}})
                        _action_retry_count.pop(action_key, None)
                        return event
                    else:
                        print("❌ 未找到可用的安全容器，无法清空手部，取消整个计划")
                        with _plan_lock:
                            planned_actions.clear()  # 清空整个计划，而不是只跳过当前动作
                            executing_plan = False
                        # 尝试让LLM重新规划
                        try:
                            print("🤖 请求LLM重新规划...")
                            _on_action_failure_llm("PickupObject", {"objectId": resolved}, event, "无法清空手部，所有安全容器都不可达")
                        except Exception as e:
                            print(f"⚠ LLM重规划失败: {e}")
                        return event

                # 检查Agent是否在物体附近
                agent_pos = _agent_pos(event)
                obj_pos = _obj_pos(event, resolved)

                if obj_pos is None:
                    print(f"⚠ 无法获取物体位置: {resolved}")
                else:
                    distance = _distance(agent_pos[0], agent_pos[1], obj_pos[0], obj_pos[1])
                    pickup_range = 1.5  # AI2-THOR的拾取范围大约是1.5米

                    if distance > pickup_range:
                        print(f"⚠ Agent距离物体太远，无法拾取: {resolved}")
                        print(f"   Agent位置: ({agent_pos[0]:.2f}, {agent_pos[1]:.2f})")
                        print(f"   物体位置: ({obj_pos[0]:.2f}, {obj_pos[1]:.2f})")
                        print(f"   距离: {distance:.2f}m > 拾取范围: {pickup_range}m")
                        print("   自动插入GoTo动作导航到物体位置")

                        # 自动插入GoTo动作到队列前面
                        with _plan_lock:
                            if _should_skip_goto(resolved):
                                print(f"   ⏭️ 跳过插入导航: GoTo({resolved})")
                            else:
                                goto_action = {'action': 'GoTo', 'params': {'objectId': resolved}}
                                planned_actions.insert(0, goto_action)  # 插入到当前动作前面
                                print(f"   已插入导航动作: GoTo({resolved})")

                        # 如果因为连续失败而跳过了GoTo，则同时处理当前PickupObject，避免原地死循环
                        if _should_skip_goto(resolved):
                            try:
                                handled = _on_action_failure_llm("PickupObject", {"objectId": resolved}, event, "导航到物体多次失败，放弃本次拾取")
                                if handled:
                                    _action_retry_count.pop(action_key, None)
                                    return event
                            except Exception as _e:
                                print(f"⚠ LLM 决策失败（PickupObject-跳过分支），采用默认跳过: {_e}")
                            with _plan_lock:
                                if planned_actions:
                                    planned_actions.pop(0)
                                    if not planned_actions:
                                        executing_plan = False
                            _action_retry_count.pop(action_key, None)
                            return event

                        # 清理重试计数，避免无限重试
                        _action_retry_count.pop(action_key, None)
                        return event
                    else:
                        print(f"✅ Agent在拾取范围内: {distance:.2f}m <= {pickup_range}m")
                        # 地图标注
                        try:
                            semantic_map.setdefault("task_points", []).append({"x": obj_pos[0], "z": obj_pos[1], "label": f"Pickup {resolved}"})
                        except Exception:
                            pass
                        event = controller.step(action="PickupObject", objectId=resolved, forceAction=True)
                        if event.metadata.get("lastActionSuccess", False):
                            print(f"✅ 成功拾取 {resolved}")
                        else:
                            err = event.metadata.get('errorMessage', 'Unknown error')
                            print(f"⚠ 拾取失败: {err}")
                            try:
                                handled = _on_action_failure_llm("PickupObject", {"objectId": resolved}, event, str(err))
                                if handled:
                                    return event
                            except Exception as _e:
                                print(f"⚠ LLM 决策失败，采用默认跳过: {_e}")

            with _plan_lock:
                planned_actions.pop(0)
                if not planned_actions:
                    executing_plan = False
            return event
        elif name == "DropHandObject":
            event = controller.step(action="DropHandObject", forceAction=True)
            with _plan_lock:
                planned_actions.pop(0)
                if not planned_actions:
                    executing_plan = False
            return event
        elif name in ("OpenObject", "CloseObject", "ToggleObjectOn", "ToggleObjectOff"):
            oid = params.get('objectId')
            resolved = _resolve_object_id_from_hint(event, oid)
            if resolved is None:
                print(f"⚠ 未能解析对象ID: {oid}，跳过 {name}")
            else:
                try:
                    pos = _obj_pos(event, resolved)
                    if pos:
                        semantic_map.setdefault("task_points", []).append({"x": pos[0], "z": pos[1], "label": f"{name.replace('Object','')} {resolved}"})
                except Exception:
                    pass
                event = controller.step(action=name, objectId=resolved)
            with _plan_lock:
                planned_actions.pop(0)
                if not planned_actions:
                    executing_plan = False
            return event
        elif name == "PutObject":
            # PutObject需要Agent已经拿着要放置的对象，并且在目标容器附近
            # 参数说明:
            # - objectId: 要放置的对象ID (Agent必须已经拿着)
            # - receptacleObjectId: 目标容器ID (AI2-THOR API中应该作为objectId参数传递)

            src_hint = params.get('objectId')  # 要放置的对象
            dst_hint = params.get('receptacleObjectId')  # 目标容器

            if not src_hint or not dst_hint:
                print(f"⚠ PutObject 缺少必要参数: objectId={src_hint}, receptacleObjectId={dst_hint}")
                with _plan_lock:
                    planned_actions.pop(0)
                    if not planned_actions:
                        executing_plan = False
                return event

            # 解析对象ID
            src = _resolve_object_id_from_hint(event, src_hint) if src_hint else None
            dst = _resolve_object_id_from_hint(event, dst_hint) if dst_hint else None

            # 检查目标容器是否可达
            if dst and _should_skip_target_completely(dst):
                print(f"⏭️ 目标容器不可达，跳过 PutObject: {src_hint} -> {dst}")
                with _plan_lock:
                    planned_actions.pop(0)
                    if not planned_actions:
                        executing_plan = False
                _action_retry_count.pop(action_key, None)
                return event

            if not src or not dst:
                print(f"⚠ PutObject 参数解析失败: src={src_hint}->{src}, dst={dst_hint}->{dst}")

                # 如果目标容器解析失败，尝试依据“柜体标签”进行语义匹配选择替代容器（非预设）
                if src and not dst:
                    print(f"🔍 基于柜体标签进行容器语义匹配...")
                    src_type = _get_obj_type_by_id(event, src) or ""
                    lab_dst = _choose_receptacle_by_labels(event, src_type)
                    if lab_dst:
                        print(f"✅ 标签匹配选择: {src_type} -> {lab_dst}")
                        dst = lab_dst
                    # 若标签也未命中，使用贝叶斯评分选择最合适的容器
                    if not dst:
                        bayes_dst = _choose_container_bayes(event, src)
                        if bayes_dst:
                            print(f"✅ 贝叶斯评分选择: {src_type} -> {bayes_dst}")
                            dst = bayes_dst

                    # 若语义偏好未命中，回退到通用可见容器
                    if not dst:
                        print(f"🔍 偏好未命中，回退到通用容器搜索")
                        all_objects = event.metadata.get("objects", [])
                        container_candidates = []
                        container_types = ["Bowl", "Plate", "Cup", "Mug", "Pot", "Pan", "Sink", "CounterTop", "Table", "Shelf", "Dresser", "Desk", "SideTable", "CoffeeTable", "DiningTable"]
                        for obj in all_objects:
                            obj_type = obj.get("objectType", "")
                            obj_id = obj.get("objectId", "")
                            if any(container_type.lower() in obj_type.lower() for container_type in container_types):
                                if obj.get("visible", False) and obj.get("receptacle", False):
                                    container_candidates.append((obj_type, obj_id))
                        priority_order = ["Bowl", "Plate", "CounterTop", "Table", "Shelf", "Desk"]
                        for priority_type in priority_order:
                            for obj_type, obj_id in container_candidates:
                                if priority_type.lower() in obj_type.lower():
                                    print(f"✅ 找到替代容器: {obj_type} -> {obj_id}")
                                    dst = obj_id
                                    break
                            if dst:
                                break
                        if not dst and container_candidates:
                            obj_type, obj_id = container_candidates[0]
                            print(f"✅ 使用第一个可用容器: {obj_type} -> {obj_id}")
                            dst = obj_id

                if not src or not dst:
                    with _plan_lock:
                        planned_actions.pop(0)
                        if not planned_actions:
                            executing_plan = False
                    return event

            # 检查Agent是否拿着要放置的对象
            agent_inventory = event.metadata.get("inventoryObjects", [])
            holding_target = any(obj.get("objectId") == src for obj in agent_inventory)

            print(f"🔍 PutObject检查: 要放置={src}, 目标容器={dst}")
            print(f"🔍 Agent当前拿着: {[obj.get('objectId') for obj in agent_inventory]}")

            if not holding_target:
                print(f"⚠ Agent没有拿着要放置的对象: {src}")
                print("⚠ 需要先GoTo源对象位置，然后PickupObject，再GoTo目标位置")
                with _plan_lock:
                    planned_actions.pop(0)
                    if not planned_actions:
                        executing_plan = False
                return event

            # 检查Agent是否在目标容器附近
            agent_pos = _agent_pos(event)
            dst_pos = _obj_pos(event, dst)

            if dst_pos is None:
                print(f"⚠ 无法获取目标容器位置: {dst}")
                with _plan_lock:
                    planned_actions.pop(0)
                    if not planned_actions:
                        executing_plan = False
                # 如果目标容器是可开并且当前为打开状态，则按流程规范在放置后关闭柜门/抽屉
                try:
                    target = None
                    for o in event.metadata.get('objects', []) or []:
                        if o.get('objectId') == dst:
                            target = o
                            break
                    if target and target.get('openable') and target.get('isOpen'):
                        ce = controller.step(action='CloseObject', objectId=dst, forceAction=True)
                        if ce.metadata.get('lastActionSuccess', False):
                            print(f"  : {dst}")
                            event = ce
                        else:
                            print(f"  : {ce.metadata.get('errorMessage','')}")
                except Exception:
                    pass

                return event

            distance = _distance(agent_pos[0], agent_pos[1], dst_pos[0], dst_pos[1])
            put_range = 1.0  # 更严格的放置距离，提升精确度

            if distance > put_range:
                print(f"⚠ Agent距离目标容器太远，无法放置: {dst}")
                print(f"   Agent位置: ({agent_pos[0]:.2f}, {agent_pos[1]:.2f})")
                # 在“距离过远无法放置”的情况下，也记录一次预期容器学习，便于后续决策
                try:
                    if src and dst:
                        # 根据源物体类型，更新目标容器的 ObservedContents
                        s_obj = next((o for o in event.metadata.get('objects', []) or [] if o.get('objectId') == src), None)
                        s_type = (s_obj.get('objectType') if s_obj else (_get_obj_type_by_id(event, src) or ''))
                        grp = SS.obj_group(str(s_type))
                        if grp:
                            lst = semantic_map.setdefault('container_contents', {}).setdefault(dst, [])
                            if grp not in lst:
                                lst.append(grp)
                except Exception:
                    pass

                print(f"   容器位置: ({dst_pos[0]:.2f}, {dst_pos[1]:.2f})")
                print(f"   距离: {distance:.2f}m > 放置范围: {put_range}m")
                print("   自动插入GoTo动作导航到目标容器")

                # 自动插入GoTo动作到队列前面
                with _plan_lock:
                    if _should_skip_goto(dst):
                        print(f"   ⏭️ 跳过插入导航: GoTo({dst})")
                    else:
                        goto_action = {'action': 'GoTo', 'params': {'objectId': dst}}
                        planned_actions.insert(0, goto_action)  # 插入到当前动作前面
                        print(f"   已插入导航动作: GoTo({dst})")

                # 如果因为连续失败而跳过了GoTo，则同时处理当前PutObject，避免原地死循环
                if _should_skip_goto(dst):
                    try:
                        handled = _on_action_failure_llm("PutObject", {"objectId": src, "receptacleObjectId": dst}, event, "导航到容器多次失败，放弃本次放置")
                        if handled:
                            _action_retry_count.pop(action_key, None)
                            return event
                    except Exception as _e:
                        print(f"⚠ LLM 决策失败（PutObject-跳过分支），采用默认跳过: {_e}")
                    with _plan_lock:
                        if planned_actions:
                            planned_actions.pop(0)
                            if not planned_actions:
                                executing_plan = False
                    _action_retry_count.pop(action_key, None)
                    return event

                # 清理重试计数，避免无限重试
                _action_retry_count.pop(action_key, None)
                return event

            print(f"✅ Agent已经拿着 {src}，在放置范围内: {distance:.2f}m <= {put_range}m")

            # 地图标注目的地 receptacle
            try:
                semantic_map.setdefault("task_points", []).append({"x": dst_pos[0], "z": dst_pos[1], "label": f"Put->{dst}"})
            except Exception:
                pass

            # 放置到目标容器 (AI2-THOR的PutObject只需要objectId参数，指向目标容器)
            put_event = controller.step(action="PutObject", objectId=dst, forceAction=True, placeStationary=True)
            if put_event.metadata.get("lastActionSuccess", False):
                print(f"✅ 成功放置到 {dst}")
                event = put_event
                if src:
                    _hand_clear_put_fail_counts.pop(src, None)
                # 更新容器观察到的内容（学习）
                try:
                    if src and dst:
                        s_obj = next((o for o in put_event.metadata.get('objects', []) or [] if o.get('objectId') == src), None)
                        s_type = (s_obj.get('objectType') if s_obj else (_get_obj_type_by_id(event, src) or ''))
                        grp = SS.obj_group(str(s_type))
                        if grp:
                            lst = semantic_map.setdefault('container_contents', {}).setdefault(dst, [])
                            if grp not in lst:
                                lst.append(grp)
                except Exception:
                    pass

            else:
                err_msg = put_event.metadata.get('errorMessage', 'Unknown error')
                print(f"⚠ 放置到 {dst} 失败: {err_msg}")
                if src and ('No valid positions to place object found' in str(err_msg)):
                    _hand_clear_put_fail_counts[src] = _hand_clear_put_fail_counts.get(src, 0) + 1
                    print(f"📉 清手放置失败计数 {src}: {_hand_clear_put_fail_counts[src]}")
                    
                    # 【新增】如果清手失败已经 >= 2 次，改用 DropHandObject 强制丢弃
                    if _hand_clear_put_fail_counts[src] >= 2:
                        print(f"🛟 清手放置已连续失败 {_hand_clear_put_fail_counts[src]} 次，改用 DropHandObject 兜底")
                        drop_ev = controller.step(action='DropHandObject', forceAction=True)
                        if drop_ev.metadata.get('lastActionSuccess', False):
                            print("✅ 已执行 DropHandObject，跳过当前PutObject")
                            _hand_clear_put_fail_counts.pop(src, None)
                            with _plan_lock:
                                planned_actions.pop(0)
                                if not planned_actions:
                                    executing_plan = False
                            _action_retry_count.pop(action_key, None)
                            return drop_ev
                        else:
                            print(f"⚠ DropHandObject 失败: {drop_ev.metadata.get('errorMessage', 'Unknown error')}，跳过当前PutObject避免卡住")
                            _hand_clear_put_fail_counts.pop(src, None)
                            with _plan_lock:
                                planned_actions.pop(0)
                                if not planned_actions:
                                    executing_plan = False
                            _action_retry_count.pop(action_key, None)
                            return drop_ev
                
                # 回退策略：若目标可开且当前为关闭，尝试先开门/抽屉后再放置；否则回退到安全容器
                try:
                    target = None
                    for o in event.metadata.get('objects', []) or []:
                        if o.get('objectId') == dst:
                            target = o
                            break
                    if target and target.get('openable') and not target.get('isOpen'):
                        oe = controller.step(action="OpenObject", objectId=dst)
                        if oe.metadata.get('lastActionSuccess', False):
                            print(f"↪️ 已打开容器后重试放置: {dst}")
                            pe2 = controller.step(action="PutObject", objectId=dst, forceAction=True, placeStationary=True)
                            if pe2.metadata.get('lastActionSuccess', False):
                                print("✅ 开启后放置成功")
                                event = pe2
                            else:
                                print(f"⚠ 开启后放置仍失败: {pe2.metadata.get('errorMessage','Unknown error')}")
                                safe_dst = _find_nearest_safe_receptacle(event)
                                if safe_dst and safe_dst != dst:
                                    print(f"↪️ 回退放置到安全容器: {safe_dst}")
                                    pe3 = controller.step(action="PutObject", objectId=safe_dst, forceAction=True, placeStationary=True)
                                    event = pe3
                        else:
                            # 目标不可开/已开，直接回退到安全容器
                            safe_dst = _find_nearest_safe_receptacle(event)
                            if safe_dst and safe_dst != dst:
                                print(f"↪️ 回退放置到安全容器: {safe_dst}")
                                pe3 = controller.step(action="PutObject", objectId=safe_dst, forceAction=True, placeStationary=True)
                                event = pe3
                except Exception:
                    # 忽略回退异常，保持 put_event 结果
                    event = put_event

                # 若回退后已成功放置，则直接完成本步，避免重复失败链路
                try:
                    if event.metadata.get("lastActionSuccess", False):
                        print("✅ 回退放置成功，继续后续任务")
                        if src:
                            _hand_clear_put_fail_counts.pop(src, None)
                        with _plan_lock:
                            planned_actions.pop(0)
                            if not planned_actions:
                                executing_plan = False
                        _action_retry_count.pop(action_key, None)
                        return event
                except Exception:
                    pass

                # 让LLM裁决改策略/跳过/重规划（仅在失败时调用）
                try:
                    handled = _on_action_failure_llm("PutObject", {"objectId": src, "receptacleObjectId": dst}, event, str(err_msg))
                    if handled:
                        return event
                except Exception as _e:
                    print(f"⚠ LLM 决策失败，采用默认跳过: {_e}")
            with _plan_lock:
                planned_actions.pop(0)
                if not planned_actions:
                    executing_plan = False
            return event

        # 3) 兼容已有原子动作（若仍出现）
        elif name in ("MoveAhead", "MoveBack", "MoveLeft", "MoveRight", "Pass"):
            event = controller.step(action=name)
            if not event.metadata.get("lastActionSuccess", True):
                err = event.metadata.get("errorMessage", "Unknown error")
                try:
                    handled = _on_action_failure_llm(name, params, event, err)
                    if handled:
                        return event
                except Exception as _e:
                    print(f"⚠ LLM 决策失败，采用默认跳过: {_e}")
        elif name in ("RotateLeft", "RotateRight"):
            if 'degrees' in params:
                event = controller.step(action=name, degrees=params['degrees'])
            else:
                event = controller.step(action=name)
        elif name in ("LookDown", "LookUp"):
            if 'degrees' in params:
                event = controller.step(action=name, degrees=params['degrees'])
            else:
                event = controller.step(action=name)
        elif name == "ExploreToFind":
            # 将元动作 ExploreToFind 在执行端展开为平台许可的原子动作序列（GoTo/OpenObject），避免下发不支持的动作
            search_type = str(params.get('objectType') or params.get('type') or '').lower()
            try:
                objs = event.metadata.get('objects', []) or []
                ax, az = _agent_pos(event)
                def _dist2(o):
                    p = o.get('position') or {}
                    x, z = float(p.get('x', 0.0)), float(p.get('z', 0.0))
                    return (x - ax) ** 2 + (z - az) ** 2
                def _is_candidate(o):
                    # 仅在可开启的柜体/抽屉中探索；不对电器执行开启
                    if not o.get('openable', False):
                        return False
                    t = str(o.get('objectType') or '')
                    return any(k in t for k in ['Cabinet', 'Drawer', 'Cupboard', 'UpperCabinet', 'LowerCabinet'])
                candidates = sorted([o for o in objs if _is_candidate(o)], key=_dist2)[:6]
                if not candidates:
                    print(f"ℹ️ ExploreToFind({search_type}) 无可用容器，跳过")
                else:
                    seq = []
                    for c in candidates:
                        cid = c.get('objectId')
                        if not cid:
                            continue
                        seq.append({'action': 'GoTo', 'params': {'objectId': cid}, 'source': 'explore_to_find'})
                        seq.append({'action': 'OpenObject', 'params': {'objectId': cid}, 'source': 'explore_to_find'})
                        seq.append({'action': 'PickupIfFound', 'params': {'objectType': search_type}, 'source': 'explore_to_find'})
                    with _plan_lock:
                        # 将展开后的原子动作插入到当前队列前端，并移除当前元动作
                        planned_actions.pop(0)
                        planned_actions = seq + planned_actions
                    # 清理重试计数并提前返回，让主循环按新队列执行
                    _action_retry_count.pop(action_key, None)
                    print(f"🔎 ExploreToFind({search_type}) 已展开：每开一个容器后都会立即检查并尝试拾取")
                    return event
            except Exception as _e:
                print(f"⚠ ExploreToFind 展开失败，跳过: {_e}")
            # 若展开失败，按不支持处理
            print(f"⚠ 未支持的动作: {name}，跳过")

        elif name == "PickupIfFound":
            search_type = str(params.get('objectType') or params.get('type') or '').strip()
            target_obj = _find_visible_pickup_target_by_type(event, search_type)
            if target_obj is None:
                print(f"🔍 未发现目标 {search_type}，继续探索下一个容器")
                with _plan_lock:
                    if planned_actions:
                        planned_actions.pop(0)
                        if not planned_actions:
                            executing_plan = False
                _action_retry_count.pop(action_key, None)
                return event

            target_id = target_obj.get('objectId')
            target_type = target_obj.get('objectType')
            print(f"✅ 已发现目标: {target_type} ({target_id})，立即插入拾取动作")

            with _plan_lock:
                if planned_actions:
                    planned_actions.pop(0)  # 移除当前 PickupIfFound

                # 找到目标后，移除本轮 ExploreToFind 后续探索动作
                while planned_actions and planned_actions[0].get('source') == 'explore_to_find':
                    planned_actions.pop(0)

                planned_actions.insert(0, {
                    'action': 'PickupObject',
                    'params': {'objectId': target_id},
                    'source': 'explore_to_find'
                })
                executing_plan = True

            _action_retry_count.pop(action_key, None)
            return event
        else:
            print(f"⚠ 未支持的动作: {name}，跳过")
        # 非导航动作：执行后直接出队
        with _plan_lock:
            planned_actions.pop(0)
            if not planned_actions:
                executing_plan = False
        # 清理重试计数
        _action_retry_count.pop(action_key, None)
        return event
    except Exception as e:
        print(f"❌ 执行动作失败 {name}: {e}")
        return event


# —— 导航状态与工具函数 ——
_nav_state = {"active": False, "target": None, "path": None, "idx": 0, "start_time": None, "timeout": 30.0}
_action_retry_count = {}  # 记录每个动作的重试次数
_hand_clear_put_fail_counts = {}  # 记录“清手放置”失败次数，避免 Pickup/Put 死循环


# —— GoTo 失败计数，用于避免在同一目标上反复插入导航 ——
MAX_GOTO_FAILURES = 3
_goto_failure_counts = {}
_failure_reset_counter = 0  # 用于定期重置失败计数器


def _normalize_search_token(text: str) -> str:
    return str(text or '').strip().lower().replace('_', '').replace('-', '').replace(' ', '')


def _match_object_type_for_search(obj_type: str, search_type: str) -> bool:
    """匹配 ExploreToFind 的目标类型，支持中英文和常见别名。"""
    ot = _normalize_search_token(obj_type)
    st = _normalize_search_token(search_type)
    if not st:
        return False
    if st == ot or st in ot or ot in st:
        return True

    alias_groups = {
        'rag': ['rag', 'cloth', 'dishtowel', 'towel', 'dishsponge', 'sponge', '抹布', '洗碗布', '海绵'],
        'cloth': ['rag', 'cloth', 'dishtowel', 'towel', 'dishsponge', 'sponge', '抹布', '洗碗布', '海绵'],
        '抹布': ['rag', 'cloth', 'dishtowel', 'towel', 'dishsponge', 'sponge', '抹布', '洗碗布', '海绵'],
        'sponge': ['dishsponge', 'sponge', 'rag', 'cloth', '抹布', '海绵'],
    }
    expanded = alias_groups.get(st, [st])
    expanded_norm = [_normalize_search_token(x) for x in expanded]
    return any((x == ot) or (x in ot) or (ot in x) for x in expanded_norm)


def _find_visible_pickup_target_by_type(event, search_type: str):
    """在当前可见且可拾取对象中，选一个匹配类型且最近的目标。"""
    objs = event.metadata.get('objects', []) or []
    ax, az = _agent_pos(event)
    candidates = []
    for o in objs:
        if not o.get('visible') or not o.get('pickupable'):
            continue
        otype = str(o.get('objectType') or '')
        if not _match_object_type_for_search(otype, search_type):
            continue
        pos = o.get('position') or {}
        dx = float(pos.get('x', 0.0)) - ax
        dz = float(pos.get('z', 0.0)) - az
        candidates.append((dx * dx + dz * dz, o))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]

def _should_skip_goto(oid: str) -> bool:
    """若某目标的 GoTo 已连续失败超过阈值，则应跳过再次尝试。"""
    if not oid:
        return False
    cnt = _goto_failure_counts.get(oid, 0)
    if cnt >= MAX_GOTO_FAILURES:
        print(f"⏭️ 该目标 GoTo 已连续失败 {cnt} 次，选择跳过: {oid}")
        return True
    return False

def _should_skip_target_completely(oid: str) -> bool:
    """若某目标的 GoTo 已连续失败超过阈值，则该目标相关的所有动作都应跳过。"""
    return _should_skip_goto(oid)


def _mark_goto_result(oid: str, success: bool):
    """记录某目标 GoTo 的成功/失败次数。成功则清零，失败则自增。"""
    global _failure_reset_counter
    if not oid:
        return
    if success:
        if oid in _goto_failure_counts:
            print(f"🔁 清除 {oid} 的GoTo失败计数（导航成功）")
        _goto_failure_counts.pop(oid, None)
    else:
        _goto_failure_counts[oid] = _goto_failure_counts.get(oid, 0) + 1
        print(f"❌ 记录 GoTo 失败（{_goto_failure_counts[oid]}/{MAX_GOTO_FAILURES}）：{oid}")

    # 定期重置机制：每50次失败记录后，清理一些旧的失败记录
    _failure_reset_counter += 1
    if _failure_reset_counter >= 50:
        _failure_reset_counter = 0
        if len(_goto_failure_counts) > 5:  # 如果失败记录太多
            # 清理一半的失败记录，给系统重新尝试的机会
            items = list(_goto_failure_counts.items())
            items.sort(key=lambda x: x[1])  # 按失败次数排序
            to_clear = items[:len(items)//2]  # 清理失败次数较少的一半
            for oid_to_clear, _ in to_clear:
                _goto_failure_counts.pop(oid_to_clear, None)
            print(f"🔄 定期清理失败记录，已清理 {len(to_clear)} 个目标的失败计数")


def _agent_pos(event):
    a = event.metadata.get("agent", {})
    p = a.get("position", {"x": 0.0, "z": 0.0})
    return float(p.get("x", 0.0)), float(p.get("z", 0.0))


def _obj_pos(event, object_id: str):
    for o in event.metadata.get("objects", []) or []:
        if o.get("objectId") == object_id:
            pos = o.get("position") or {"x": 0.0, "z": 0.0}
            return float(pos.get("x", 0.0)), float(pos.get("z", 0.0))
    return None


def _distance(ax, az, bx, bz):
    dx, dz = ax - bx, az - bz
    return (dx * dx + dz * dz) ** 0.5



def _find_nearest_safe_receptacle(event) -> str | None:
    """返回最近的安全放置容器ID（优先可见的CounterTop/Table类表面），排除不可达目标。"""
    global _goto_failure_counts  # 引用失败计数器
    try:
        all_objects = event.metadata.get("objects", []) or []
        ax, az = _agent_pos(event)
        priority = [
            "CounterTop", "Table", "DiningTable", "CoffeeTable",
            "Desk", "SideTable", "Shelf"
        ]
        def match_priority(obj_type: str) -> bool:
            t = (obj_type or "").lower()
            return any(p.lower() in t for p in priority)

        candidates = []  # 存储所有候选容器

        # 先找可见，再找不可见
        for visibility in (True, False):
            for o in all_objects:
                if not o.get("receptacle", False):
                    continue
                if o.get("visible", False) != visibility:
                    continue
                if not match_priority(o.get("objectType", "")):
                    continue

                oid = o.get("objectId")
                # 关键检查：如果该对象是已知不可达的，则跳过
                if _goto_failure_counts.get(oid, 0) >= MAX_GOTO_FAILURES:
                    print(f"🚫 跳过不可达容器: {oid} (失败{_goto_failure_counts[oid]}次)")
                    continue

                pos = o.get("position") or {"x": 0.0, "z": 0.0}
                x, z = float(pos.get("x", 0.0)), float(pos.get("z", 0.0))
                d2 = (x - ax) ** 2 + (z - az) ** 2
                candidates.append((oid, d2, visibility))

        # 按距离排序，优先选择可见的
        candidates.sort(key=lambda item: (not item[2], item[1]))  # 可见优先，然后按距离

        if candidates:
            best_id = candidates[0][0]
            print(f"📍 选择安全容器: {best_id} (距离: {candidates[0][1]**0.5:.2f}m)")
            return best_id

        # 紧急情况：所有安全容器都被标记为不可达
        # 检查是否有被标记为失败的容器，如果有，重置它们的失败计数
        failed_containers = []
        for o in all_objects:
            if not o.get("receptacle", False):
                continue
            if not match_priority(o.get("objectType", "")):
                continue
            oid = o.get("objectId")
            if _goto_failure_counts.get(oid, 0) >= MAX_GOTO_FAILURES:
                failed_containers.append(oid)

        if failed_containers:
            print(f"🔄 紧急重置失败计数器，重新尝试 {len(failed_containers)} 个容器")
            for oid in failed_containers:
                _goto_failure_counts.pop(oid, None)

            # 递归调用自己，重新寻找
            return _find_nearest_safe_receptacle(event)

        print("⚠ 未找到任何可达的安全容器")
        return None

    except Exception as e:
        print(f"⚠ 查找安全容器失败: {e}")
        return None


def _get_obj_type_by_id(event, oid: str) -> str | None:
    for o in event.metadata.get("objects", []) or []:
        if o.get("objectId") == oid:
            return o.get("objectType") or o.get("name")
    return None



# load_container_labels 已迁移到 exploration_io.py


def _choose_receptacle_by_labels(event, obj_type: str) -> str | None:
    """基于 semantic_map['container_labels'] 与 obj_type 的简单匹配，返回最近的合适容器ID。"""
    try:
        labels_map: dict = semantic_map.get("container_labels", {}) or {}
        if not labels_map or not obj_type:
            return None
        obj_l = str(obj_type).lower()
        ax, az = _agent_pos(event)
        candidates = []
        for cid, info in labels_map.items():
            labels = [str(s).lower() for s in (info.get('labels') or [])]
            if any(obj_l in s for s in labels):
                # 距离
                o = next((o for o in event.metadata.get('objects', []) or [] if o.get('objectId') == cid), None)
                if not o:
                    continue
                p = o.get('position') or {}
                x, z = float(p.get('x', 0.0)), float(p.get('z', 0.0))
                d2 = (x - ax) ** 2 + (z - az) ** 2
                candidates.append((d2, cid))
        if not candidates:
            return None
        candidates.sort(key=lambda t: t[0])
        return candidates[0][1]
    except Exception:
        return None


def _area_name_at(x: float, z: float) -> str | None:
    try:
        areas = semantic_map.get("areas", {}) or {}
        for a in areas.values():
            b = a.get("boundary", {})
            if b and (b.get("min_x") <= x <= b.get("max_x")) and (b.get("min_z") <= z <= b.get("max_z")):
                return str(a.get("name"))
    except Exception:
        pass
    return None


def _choose_container_bayes(event, obj_id: str) -> str | None:
    try:
        # object features
        obj = next((o for o in event.metadata.get('objects', []) or [] if o.get('objectId') == obj_id), None)
        if not obj:
            return None
        obj_type = str(obj.get('objectType', ''))
        og = SS.obj_group(obj_type)

        # preferred zone from priors and norms (both accepted, norms override if present)
        def _norm_zone_to_canonical(z):
            s = str(z or '').lower()
            if not s:
                return None
            if ('\u51b7\u85cf' in z) or ('fridge' in s) or ('refrig' in s):
                return 'Storage'
            if ('\u6d17' in z) or ('\u6e05\u6d01' in z) or ('clean' in s):
                return 'Cleaning'
            if ('\u5207\u914d' in z) or ('\u5907\u9910' in z) or ('\u5904\u7406' in z) or ('prep' in s):
                return 'Prep'
            if ('\u70f9\u996a' in z) or ('cook' in s):
                return 'Cooking'
            if ('\u7528\u9910' in z) or ('dining' in s):
                return 'Dining'
            if ('\u50a8\u5b58' in z) or ('\u6536\u7eb3' in z) or ('storage' in s):
                return 'Storage'
            return None

        pref_zone = None
        try:
            priors = getattr(SP, 'REGION_PRIORS', {})
            pmap = priors.get(obj_type) or {}
            if pmap:
                top_region = max(pmap.items(), key=lambda kv: kv[1])[0]
                pref_zone = SS.region_to_zone(top_region)
        except Exception:
            pass
        # norms override
        try:
            n = ON.NORMS.get(obj_type, {})
            z = n.get('preferred_zone')
            nz = _norm_zone_to_canonical(z)
            if nz:
                pref_zone = nz
        except Exception:
            pass

        # candidate containers
        ax, az = _agent_pos(event)
        candidates = []
        for c in event.metadata.get('objects', []) or []:
            if not c.get('receptacle'):
                continue
            cid = c.get('objectId')
            ctype = c.get('objectType') or ''
            cls = SS.container_class(ctype)
            p = c.get('position') or {}
            x, z = float(p.get('x', 0.0)), float(p.get('z', 0.0))
            aname = _area_name_at(x, z)
            lz = SS.region_to_zone(aname) if aname else None
            observed = list(semantic_map.get('container_contents', {}).get(cid, []) or [])
            sc = SS.score_container(og, pref_zone, cls, lz, observed)
            d2 = (x - ax) ** 2 + (z - az) ** 2
            candidates.append((sc, -d2, cid))  # prefer nearer when scores tie
        if not candidates:
            return None
        candidates.sort(reverse=True)
        best = candidates[0]
        # 可设置阈值，当前只要有候选就返回最佳
        return best[2]
    except Exception:
        return None




# —— 实时进展与二次规划触发（框架化） ——
_last_replan_trigger_ts = 0.0


# load_replan_triggers 已迁移到 exploration_io.py


def _progress(msg: str):
    # 统一的进展输出，可后续换成事件总线/Webhook
    try:
        # 过滤空消息/控制字符，避免异常字符串导致的刷屏
        if not msg or msg.strip() == "" or "\x1b" in msg:
            return
        print(f"[Progress] {msg}")
    except Exception:
        pass


def _maybe_trigger_replan(event, reason: str):
    """在节流的前提下触发一次异步二次规划（不阻塞主循环）。"""
    import time
    global _last_replan_trigger_ts
    now = time.time()
    min_interval = float(semantic_map.get("replan_triggers", {}).get("min_interval_sec", 6))
    if now - _last_replan_trigger_ts < min_interval:
        return
    _last_replan_trigger_ts = now
    _progress(f"触发二次规划：{reason}")
    try:
        trigger_llm_scene_understanding_async(event=event)
    except Exception as e:
        print(f"⚠ 触发二次规划失败: {e}")


def _on_object_discovered(event, obj_type: str, oid: str):
    """当发现新对象时，依据配置判断是否触发二次规划。"""
    try:
        cfg = semantic_map.get("replan_triggers", {}) or {}
        watch_types = {str(t).lower() for t in cfg.get("when_discovered_objectTypes", [])}
        if watch_types and obj_type and obj_type.lower() in watch_types:
            _maybe_trigger_replan(event, reason=f"发现关键对象类型 {obj_type} ({oid})")
    except Exception:
        pass

def _update_semantic_roles(event):
    """依据启发式规则将柜体/台面加入预定义语义角色，用于智能放置。
    规则：
    - 距离 Sink 最近的柜体 -> cup_cabinet
    - 距离 Stove/Range 最近的柜体 -> utensil_cabinet / pan_pot_cabinet
    - UpperCabinet 优先归为 cup/bowl/plate；LowerCabinet 优先归为 utensil/pan_pot
    - 所有 CounterTop/Table 都加入 countertop 角色
    """
    try:
        roles = semantic_map.setdefault("roles", {})
        for k in ("cup_cabinet", "utensil_cabinet", "bowl_cabinet", "plate_cabinet", "pan_pot_cabinet", "countertop"):
            roles.setdefault(k, [])
        objs = event.metadata.get("objects", []) or []
        # 参考点
        sink_pos = []
        stove_pos = []
        for o in objs:
            t = (o.get("objectType") or "").lower()
            p = o.get("position") or {}
            if any(s in t for s in ["sink", "sinkbasin"]):
                sink_pos.append((float(p.get("x", 0.0)), float(p.get("z", 0.0))))
            if any(s in t for s in ["stove", "stoveburner", "range"]):
                stove_pos.append((float(p.get("x", 0.0)), float(p.get("z", 0.0))))
        def nearest_dist2(px, pz, lst):
            if not lst: return 1e9
            return min((px - x) ** 2 + (pz - z) ** 2 for x, z in lst)
        # 填充角色
        for o in objs:
            oid = o.get("objectId")
            t = (o.get("objectType") or "").lower()
            p = o.get("position") or {}
            x, z = float(p.get("x", 0.0)), float(p.get("z", 0.0))
            # countertop
            if any(s in t for s in ["counter", "table", "desk", "shelf"]):
                if oid not in roles["countertop"]:
                    roles["countertop"].append(oid)
            # 柜体
            is_cabinet = any(s in t for s in ["cabinet", "drawer"])
            if not is_cabinet:
                continue
            d_sink = nearest_dist2(x, z, sink_pos)
            d_stove = nearest_dist2(x, z, stove_pos)
            is_upper = "upper" in t
            # 归类：
            if is_upper or d_sink < d_stove:
                # 更靠近水槽或上柜 -> 杯/碗/盘
                for role in ("cup_cabinet", "bowl_cabinet", "plate_cabinet"):
                    if oid not in roles[role]:
                        roles[role].append(oid)
            else:
                # 更靠近炉灶或下柜 -> 工具/锅具
                for role in ("utensil_cabinet", "pan_pot_cabinet"):
                    if oid not in roles[role]:
                        roles[role].append(oid)
    except Exception:
        pass


def _choose_preferred_receptacle_for_type(event, obj_type: str) -> str | None:
    """依据学习到的偏好和 semantic_map.preferences 选择合适的容器角色，并返回最近的该角色实体objectId。"""
    prefs = semantic_map.get("preferences", {})
    roles = semantic_map.get("roles", {})
    if not obj_type:
        return None

    # 1. 首先尝试使用学习到的偏好
    learned_roles = []
    if obj_type in _preference_scores:
        # 按学习到的分数排序角色
        role_scores = _preference_scores[obj_type]
        learned_roles = sorted(role_scores.keys(), key=lambda r: role_scores[r], reverse=True)
        print(f"🧠 使用学习偏好 {obj_type}: {[(r, f'{role_scores[r]:.1f}') for r in learned_roles[:3]]}")

    # 2. 回退到默认偏好
    default_roles = prefs.get(obj_type, [])

    # 3. 合并偏好列表（学习偏好优先，但不重复）
    desired_roles = learned_roles + [r for r in default_roles if r not in learned_roles]

    # 位置计算
    ax, az = _agent_pos(event)
    def nearest(cands):
        best = None
        best_d2 = None
        for oid in cands:
            pos = _obj_pos(event, oid)
            if not pos:
                continue
            d2 = (pos[0] - ax) ** 2 + (pos[1] - az) ** 2
            if best is None or d2 < best_d2:
                best, best_d2 = oid, d2
        return best

    # 按优先级尝试角色（学习偏好优先）
    for role in desired_roles:
        cands = roles.get(role) or []
        oid = nearest(cands)
        if oid:
            # 应用衰减机制
            _apply_preference_decay()
            print(f"📍 选择容器: {obj_type} -> {role} ({oid})")
            return oid

    # 退化到 countertop
    cands = roles.get("countertop") or []
    result = nearest(cands)
    if result:
        print(f"📍 回退到台面: {obj_type} -> countertop ({result})")
    return result

def _bearing_deg(ax, az, bx, bz):
    # 返回 agent->target 的平面角度（相对世界坐标系的朝向），用于近似旋转
    import math
    return math.degrees(math.atan2(bz - az, bx - ax))


def _get_agent_yaw(event):
    rot = (event.metadata.get("agent") or {}).get("rotation") or {}
    return float(rot.get("y", 0.0))


def _normalize_deg(d):
    while d > 180: d -= 360
    while d < -180: d += 360
    return d


def _step_towards(controller, event, target_pos, reach_threshold=0.4):
    """执行一步朝向目标的动作（旋转小角度或前进一步）。返回(event, reached: bool)。"""
    ax, az = _agent_pos(event)
    tx, tz = target_pos
    if _distance(ax, az, tx, tz) <= reach_threshold:
        return event, True
    # 旋转对准


def detect_and_enqueue_tidy_tasks(event) -> bool:
    """基于简单启发式检测“可整理”的轻度混乱，并将一小段整理动作入列。
    返回 True 表示已入列任务；否则 False。
    规则示例：
    - 锅/平底锅 在 炉灶/地面 上 -> 放到柜子(锅具)或台面
    - 杯/碗/盘 在 地面 -> 放到杯柜/碗柜/盘柜或台面
    - 刀 在 地面 -> 放到餐具柜或台面
    - 兜底：任何不应在地面的可拾取物体（FLOOR_OK_TYPES 之外）在地面 -> 收纳
    """
    try:
        objs = event.metadata.get('objects', []) or []
        # 工具函数
        def is_on_floor(o: dict) -> bool:
            if o.get('isOnFloor') is True:
                return True
            parents = o.get('parentReceptacles') or []
            return any('Floor' in (p or '') for p in parents)
        def parent_in(o: dict, keyword: str) -> bool:
            parents = o.get('parentReceptacles') or []
            return any(keyword.lower() in (p or '').lower() for p in parents)
        # 候选排序：先地面，再炉灶
        messy_candidates = []
        for o in objs:
            if not o.get('visible', False):
                continue
            t = o.get('objectType') or ''
            oid = o.get('objectId')
            if not oid:
                continue
            # 锅具
            if t in ('Pot', 'Pan'):
                if is_on_floor(o) or parent_in(o, 'Stove') or parent_in(o, 'StoveBurner'):
                    messy_candidates.append(o)
            # 餐具器皿
            if t in ('Cup', 'Mug', 'Bowl', 'Plate') and is_on_floor(o):
                messy_candidates.append(o)
            # 刀
            if t == 'Knife' and is_on_floor(o):
                messy_candidates.append(o)
        # 兜底：任意不应在地面的可拾取物体
        if not messy_candidates:
            for o in objs:
                if o.get('pickupable', False) and is_on_floor(o):
                    t = o.get('objectType') or ''
                    if t not in getattr(ON, 'FLOOR_OK_TYPES', set()):
                        messy_candidates.append(o)
        if not messy_candidates:
            return False
        # 选择最近的一个进行整理
        ax, az = _agent_pos(event)
        def d2(o):
            p = o.get('position') or {}
            return (float(p.get('x', 0.0)) - ax) ** 2 + (float(p.get('z', 0.0)) - az) ** 2
        target = sorted(messy_candidates, key=d2)[0]
        tgt_id = target.get('objectId')
        tgt_type = target.get('objectType') or ''
        # 若目标为脏态，则不做轻度整理（避免“脏物直接收纳”），交由LLM规划清洁
        try:
            _st = semantic_map.get("objects", {}).get(tgt_id, {}).get("state", {})
        except Exception:
            _st = {}
        if target.get('isDirty') is True or _st.get('isDirty') is True:
            print('ℹ️ 目标为脏态，跳过收纳，等待LLM规划清洁')
            return False

        # 选择目的地（基于柜体标签的语义匹配；找不到则回退到安全台面/桌面）
        dst = _choose_receptacle_by_labels(event, tgt_type)
        if not dst:
            dst = _find_nearest_safe_receptacle(event)
        if not dst:
            return False
        # 入列：GoTo(tgt) -> Pickup(tgt) -> GoTo(dst) -> Put(dst)
        with _plan_lock:
            planned_actions.extend([
                {"action": "GoTo", "params": {"objectId": tgt_id}},
                {"action": "PickupObject", "params": {"objectId": tgt_id}},
                {"action": "GoTo", "params": {"objectId": dst}},
                {"action": "PutObject", "params": {"objectId": tgt_id, "receptacleObjectId": dst}},
            ])
            globals()['executing_plan'] = True
        print(f"🧹 已入列整理：{tgt_type} -> {dst}")
        return True
    except Exception as e:
        print(f"⚠ 检测整理任务失败: {e}")
        return False

# ---------------- 新增：全局任务检测与优先级排序 ----------------
from typing import TypedDict, Literal, Optional as _Optional

class SceneTask(TypedDict, total=False):
    id: str
    category: Literal['Safety','Cleaning','Tidy','Maintenance']
    issue: str
    implied_action: str
    object_id: str
    priority: int
    reason: str


def _is_on_like(o: dict) -> bool:
    return bool(o.get('isOn') or o.get('isToggled') or o.get('isRunning'))


def _type_contains(o: dict, kws: list[str]) -> bool:
    t = (o.get('objectType') or '').lower()
    name = (o.get('name') or '').lower()
    s = t + ' ' + name
    return any(k.lower() in s for k in kws)


def detect_scene_tasks(event) -> list[SceneTask]:
    """
    扫描全局元数据，产出结构化任务清单：优先识别安全/紧急任务，然后清洁/整理。
    仅做检测与排序，不执行动作。
    关键变更（泛化）：
    - 将“可接受放置”与“必须纠正”区分：例如 Tablet 在 Table/CounterTop 上视为可接受，不触发收纳；
    - 对“不应放在地上”的可拾取物，统一触发收纳（除非显式允许）。
    """
    tasks: list[SceneTask] = []
    objs = event.metadata.get('objects', []) or []

    # 局部工具：无需依赖热力图，也能判定“在地上/在炉灶上”等情况
    def _is_on_floor_loc(o: dict) -> bool:
        if o.get('isOnFloor') is True:
            return True
        parents = o.get('parentReceptacles') or []
        return any('Floor' in (p or '') for p in parents)

    def _parent_in(o: dict, keyword: str) -> bool:
        parents = o.get('parentReceptacles') or []
        return any(keyword.lower() in (p or '').lower() for p in parents)

    def _support_surface_type(o: dict) -> str | None:
        parents = o.get('parentReceptacles') or []
        SURF_KWS = [
            'Floor','CounterTop','DiningTable','CoffeeTable','Table','Desk','SideTable','Shelf','ShelvingUnit','Sofa','Stool'
        ]
        for kw in SURF_KWS:
            for p in parents:
                if kw.lower() in (p or '').lower():
                    return kw
        return None

    def _is_acceptable_here(otype: str, o: dict) -> bool:
        surf = _support_surface_type(o)
        if surf == 'Floor':
            return otype in getattr(ON, 'FLOOR_OK_TYPES', set())
        wl = set(getattr(ON, 'ACCEPTABLE_SURFACES', {}).get(otype, []))
        if wl:
            return surf in wl
        return surf in {'CounterTop','DiningTable','CoffeeTable','Table','Desk','SideTable','Shelf','Sofa'}

    DISH_LIKE = ('Cup', 'Mug', 'Bowl', 'Plate')
    COOKWARE = ('Pot', 'Pan')
    FOOD_LIKE = ('Apple', 'Potato', 'Tomato', 'Bread', 'Cereal', 'Egg', 'Lettuce')

    for o in objs:
        oid = o.get('objectId') or ''
        if not oid:
            continue
        otype = o.get('objectType') or ''

        # 1) Safety: 明火/加热设备开启
        if _type_contains(o, ['burner','stove','range']) and _is_on_like(o):
            tasks.append({
                'id': f'safety_open_flame::{oid}',
                'category': 'Safety',
                'issue': 'OpenFlame',
                'implied_action': 'ToggleObjectOff',
                'object_id': oid,
                'priority': 100,
                'reason': f'{otype} isOn=True（明火/加热）',
            })
            continue
        if _type_contains(o, ['microwave','toaster','coffee','kettle','heater']) and _is_on_like(o):
            tasks.append({
                'id': f'safety_hot_appliance::{oid}',
                'category': 'Safety',
                'issue': 'HotApplianceOn',
                'implied_action': 'ToggleObjectOff',
                'object_id': oid,
                'priority': 92,
                'reason': f'{otype} isOn=True（电器未关闭）',
            })
            continue
        # 水龙头未关
        if _type_contains(o, ['faucet','tap']) and _is_on_like(o):
            tasks.append({
                'id': f'safety_faucet_on::{oid}',
                'category': 'Safety',
                'issue': 'WaterRunning',
                'implied_action': 'ToggleObjectOff',
                'object_id': oid,
                'priority': 95,
                'reason': f'{otype} isOn=True（水龙头未关闭）',
            })
            continue

        # 2) Maintenance/Tidy: 重要门体长开（冰箱/烤箱/微波炉），非紧急但应关闭
        if o.get('openable') and o.get('isOpen') and _type_contains(o, ['fridge','refrigerator','oven','microwave','dishwasher','cabinet','drawer']):
            pri = 80 if _type_contains(o, ['fridge','refrigerator','oven','microwave','dishwasher']) else 50
            tasks.append({
                'id': f'close_openable::{oid}',
                'category': 'Tidy',
                'issue': 'DoorOpen',
                'implied_action': 'CloseObject',
                'object_id': oid,
                'priority': pri,
                'reason': f'{otype} isOpen=True（门未关闭）',
            })

        # 3) Cleaning: 脏/破损
        if o.get('dirtyable') and o.get('isDirty'):
            tasks.append({
                'id': f'clean_dirty::{oid}',
                'category': 'Cleaning',
                'issue': 'ObjectDirty',
                'implied_action': 'CleanObject',
                'object_id': oid,
                'priority': 60,
                'reason': f'{otype} isDirty=True（需要清洁）',
            })
        otype_low = (otype or '').lower()
        if (o.get('isBroken') is True) or ('cracked' in otype_low) or ('broken' in otype_low):
            tasks.append({
                'id': f'broken::{oid}',
                'category': 'Cleaning',
                'issue': 'BrokenMess',
                'implied_action': 'CleanUpBroken',
                'object_id': oid,
                'priority': 65,
                'reason': f'{otype} 破损/洒落（需要清理碎片/污渍）',
            })

        # 4) Misplaced: 区域不符，但若当前表面可接受则不收纳
        try:
            rid = (semantic_map.get("objects", {}).get(oid, {}) or {}).get("regionId")
            area_name = None
            if rid:
                area_name = (semantic_map.get("areas", {}).get(rid, {}) or {}).get("name")
            norms = ON.NORMS.get(otype)
            if norms:
                pref_zone = norms.get("preferred_zone")
                if pref_zone and area_name and pref_zone != area_name and o.get('pickupable', False):
                    if not _is_acceptable_here(otype, o):
                        tasks.append({
                            'id': f'misplaced::{oid}',
                            'category': 'Tidy',
                            'issue': 'Misplaced',
                            'implied_action': 'PutAwayObject',
                            'object_id': oid,
                            'priority': 55,
                            'reason': f'{otype} 当前区域={area_name}，期望区域={pref_zone}',
                        })
        except Exception:
            pass

        # 4b) 兜底：在地上/炉灶上 -> 收纳
        if o.get('pickupable', False):
            if otype in DISH_LIKE and _is_on_floor_loc(o):
                tasks.append({
                    'id': f'onfloor_dish::{oid}',
                    'category': 'Tidy',
                    'issue': 'OnFloor',
                    'implied_action': 'PutAwayObject',
                    'object_id': oid,
                    'priority': 57,
                    'reason': f'{otype} 在地上（应收纳到柜子/台面）',
                })
            elif otype in FOOD_LIKE and _is_on_floor_loc(o):
                tasks.append({
                    'id': f'onfloor_food::{oid}',
                    'category': 'Tidy',
                    'issue': 'OnFloor',
                    'implied_action': 'PutAwayObject',
                    'object_id': oid,
                    'priority': 58,
                    'reason': f'{otype} 在地上（应放入橱柜/台面）',
                })
            elif otype in COOKWARE and (_is_on_floor_loc(o) or _parent_in(o, 'Stove') or _parent_in(o, 'StoveBurner')):
                tasks.append({
                    'id': f'misplaced_cookware::{oid}',
                    'category': 'Tidy',
                    'issue': 'Misplaced',
                    'implied_action': 'PutAwayObject',
                    'object_id': oid,
                    'priority': 56,
                    'reason': f'{otype} 在地面/炉灶上（应收纳到炊具柜/台面）',
                })
            elif _is_on_floor_loc(o) and (otype not in getattr(ON, 'FLOOR_OK_TYPES', set())):
                tasks.append({
                    'id': f'onfloor::{oid}',
                    'category': 'Tidy',
                    'issue': 'OnFloor',
                    'implied_action': 'PutAwayObject',
                    'object_id': oid,
                    'priority': 54,
                    'reason': f'{otype} 在地上（应放至合适表面/柜体）',
                })

    # 排序：优先级降序
    tasks.sort(key=lambda t: t.get('priority', 0), reverse=True)
    return tasks



def print_A_candidates(event, limit: int = 12):
    try:
        print("步骤2: A 提取高层候选任务（多任务，来自本地扫描）")
        tasks = detect_scene_tasks(event)
        print_task_overview(tasks, limit=limit)
    except Exception as _e:
        print(f"⚠ 本地A候选打印失败: {_e}")


    # 排序：优先级降序
    tasks.sort(key=lambda t: t.get('priority', 0), reverse=True)
    return tasks


def print_task_overview(tasks: list[SceneTask], limit: int = 10):
    if not tasks:
        print('🧭 场景任务扫描：未发现需要立即处理的任务')
        return
    print('🧭 场景任务扫描（按优先级）：')
    for i, t in enumerate(tasks[:limit], start=1):
        print(f"  {i}. [{t['category']}] {t['issue']} -> {t['implied_action']}({t['object_id']})  优先级={t['priority']}  原因: {t['reason']}")


def enqueue_task_plan(event, task: SceneTask, auto_start: bool = True) -> bool:
    """将单个任务转换为最小可执行动作序列并入列。只做必要动作：
    - Safety: ToggleObjectOff / CloseObject
    - Cleaning: 调用洗碗/清洁流程（尽量不去随意开门，除非需要放置）
    - Tidy: 仅在门是开的情况下 CloseObject，不主动去开柜
    """
    try:
        act = task.get('implied_action')
        oid = task.get('object_id')
        if not act or not oid:
            return False
        with _plan_lock:
            if act in ('ToggleObjectOff', 'ToggleObjectOn', 'CloseObject', 'OpenObject'):
                planned_actions.extend([
                    {'action': 'GoTo', 'params': {'objectId': oid}},
                    {'action': act, 'params': {'objectId': oid, 'forceAction': True}},
                ])
            elif act == 'CleanObject':
                # 使用标准清洁流程（到水槽、开关水、清洁、再拿起与收纳）
                pass
            elif act == 'PutAwayObject':
                # 使用标准收纳流程
                pass
        if act == 'CleanObject':
            return enqueue_wash_workflow(event, oid)
        if act == 'PutAwayObject':
            return enqueue_putaway_workflow(event, oid)
        if auto_start:
            with _plan_lock:
                globals()['executing_plan'] = True
        print(f"✅ 已入列任务: [{task['category']}] {task['implied_action']} -> {oid}")
        return True
    except Exception as e:
        print(f"⚠ 入列任务失败: {e}")
        return False


    yaw = _get_agent_yaw(event)
    desired = _bearing_deg(ax, az, tx, tz)
    delta = _normalize_deg(desired - yaw)
    try:
        if abs(delta) > 12:
            step = 15 if delta > 0 else -15
            event = controller.step(action=("RotateRight" if step > 0 else "RotateLeft"), degrees=abs(step))
            return event, False
        # 朝前走一步
        event = controller.step(action="MoveAhead")
        # 若失败则尝试小角度调整
        if not event.metadata.get("lastActionSuccess", True):
            event = controller.step(action="RotateRight", degrees=15)
        return event, False
    except Exception:
        return event, False


# ======== Grid-based shortest path navigation (ported from tset.py style) ========
GRID_SIZE = 0.15  # 匹配AI2-THOR的gridSize设置

from collections import deque

def _round2(x: float) -> float:
    return round(float(x), 2)


def _nearest_reachable_point(pts: list, tx: float, tz: float) -> tuple[float, float] | None:
    best, best_d2 = None, float("inf")

    for p in pts:
        x, z = float(p.get("x", 0.0)), float(p.get("z", 0.0))
        dx, dz = x - tx, z - tz
        d2 = dx * dx + dz * dz
        if d2 < best_d2:
            best_d2 = d2
            best = (x, z)
    return best

# ======== 标准化流程：洗碗与收纳 ========

def _find_nearest_of_types(event, type_keywords: list[str]) -> str | None:
    objs = event.metadata.get('objects', []) or []
    ax, az = _agent_pos(event)
    cand = []
    for o in objs:
        t = (o.get('objectType') or '').lower()
        if any(k.lower() in t for k in type_keywords):
            p = o.get('position') or {}
            x, z = float(p.get('x', 0.0)), float(p.get('z', 0.0))
            d2 = (x - ax) ** 2 + (z - az) ** 2
            cand.append((d2, o.get('objectId')))
    if not cand:
        return None
    cand.sort(key=lambda x: x[0])
    return cand[0][1]


def enqueue_wash_workflow(event, obj_id: str) -> bool:
    """洗碗标准流程：Pickup -> 到水槽 -> 放入水槽 -> 开水 -> CleanObject -> 关水 -> 拿起 -> 入柜

    返回 True 表示序列已入列
    """
    try:
        # 选择水槽/水龙头实体
        sink = _find_nearest_of_types(event, ["sinkbasin", "sink"]) or _find_nearest_of_types(event, ["faucet"])
        faucet = _find_nearest_of_types(event, ["faucet"]) or sink
        if not sink:
            print("ℹ️ 未找到水槽，无法执行洗碗流程")
            return False
        # 选择收纳柜（依据标签或偏好）
        obj = next((o for o in event.metadata.get('objects', []) or [] if o.get('objectId') == obj_id), None)
        obj_type = obj.get('objectType') if obj else ''
        dst_cab = _choose_preferred_receptacle_for_type(event, obj_type) or _choose_receptacle_by_labels(event, obj_type)
        with _plan_lock:
            planned_actions.extend([
                {"action": "GoTo", "params": {"objectId": obj_id}},
                {"action": "PickupObject", "params": {"objectId": obj_id}},
                {"action": "GoTo", "params": {"objectId": sink}},
                {"action": "PutObject", "params": {"dst": sink}},  # 将手中物体放入水槽
            ])
            if faucet:
                planned_actions.append({"action": "ToggleObjectOn", "params": {"objectId": faucet}})
            planned_actions.append({"action": "CleanObject", "params": {"objectId": obj_id}})  # 清洗目标物体
            if faucet:
                planned_actions.append({"action": "ToggleObjectOff", "params": {"objectId": faucet}})
            planned_actions.append({"action": "PickupObject", "params": {"objectId": obj_id}})
            if dst_cab:
                planned_actions.extend([
                    {"action": "GoTo", "params": {"objectId": dst_cab}},
                    {"action": "OpenObject", "params": {"objectId": dst_cab}},
                    {"action": "PutObject", "params": {"dst": dst_cab}},
                    {"action": "CloseObject", "params": {"objectId": dst_cab}},
                ])
        globals()['executing_plan'] = True
        print(f"🧽 已入列洗碗流程: {obj_id} -> sink={sink} -> cabinet={dst_cab}")
        return True
    except Exception as e:
        print(f"⚠ 加入洗碗流程失败: {e}")
        return False


def enqueue_putaway_workflow(event, obj_id: str) -> bool:
    """收纳标准流程：Pickup -> 到柜子 -> 开柜 -> 放 -> 关柜"""
    try:
        obj = next((o for o in event.metadata.get('objects', []) or [] if o.get('objectId') == obj_id), None)
        obj_type = obj.get('objectType') if obj else ''
        dst_cab = _choose_preferred_receptacle_for_type(event, obj_type) or _choose_receptacle_by_labels(event, obj_type)
        # 若未命中标签/偏好，使用贝叶斯评分模型
        if not dst_cab and obj_id:
            bayes_dst = _choose_container_bayes(event, obj_id)
            if bayes_dst:
                dst_cab = bayes_dst
        # 仍未找到则回退到安全台面/桌面
        if not dst_cab:
            dst_cab = _find_nearest_safe_receptacle(event)
        if not dst_cab:
            print("ℹ️ 未找到合适的收纳容器")
            return False
        with _plan_lock:
            planned_actions.extend([
                {"action": "GoTo", "params": {"objectId": obj_id}},
                {"action": "PickupObject", "params": {"objectId": obj_id}},
                {"action": "GoTo", "params": {"objectId": dst_cab}},
                {"action": "OpenObject", "params": {"objectId": dst_cab}},
                {"action": "PutObject", "params": {"dst": dst_cab}},
                {"action": "CloseObject", "params": {"objectId": dst_cab}},
            ])
        globals()['executing_plan'] = True
        print(f"🗃️ 已入列收纳流程: {obj_id} -> cabinet={dst_cab}")
        return True
    except Exception as e:
        print(f"⚠ 加入收纳流程失败: {e}")
        return False

def _get_reachable_points(controller) -> list[dict]:
    ev = controller.step(action="GetReachablePositions")
    return ev.metadata.get("actionReturn", []) or []


def _bfs_shortest_path(start_pos: tuple[float, float], target_pos: tuple[float, float], reachable_pts: list[dict], grid: float = 0.25):
    # Build a set of rounded grid coords for fast lookup
    reach_set = {(_round2(p.get("x", 0.0)), _round2(p.get("z", 0.0))) for p in reachable_pts}
    sx, sz = _round2(start_pos[0]), _round2(start_pos[1])
    tx, tz = _round2(target_pos[0]), _round2(target_pos[1])

    # Snap goal to nearest reachable grid if needed
    if (tx, tz) not in reach_set:
        near = _nearest_reachable_point(reachable_pts, tx, tz)
        if not near:
            return None
        tx, tz = _round2(near[0]), _round2(near[1])
    start = (sx, sz)
    goal = (tx, tz)
    if start == goal:
        return [start]

    q = deque([ [start] ])
    visited = {start}
    steps = [ (0, grid), (0, -grid), (grid, 0), (-grid, 0) ]
    while q:
        path = q.popleft()
        cx, cz = path[-1]
        if (cx, cz) == goal:
            return path
        for dx, dz in steps:
            nx = _round2(cx + dx)
            nz = _round2(cz + dz)
            nxt = (nx, nz)
            if nxt in reach_set and nxt not in visited:
                visited.add(nxt)
                q.append(path + [nxt])
    return None



def _nearest_reachable_to(target_pos: tuple[float, float], reachable_pts: list[dict], grid: float = 0.25) -> tuple[float, float] | None:
    try:
        tx, tz = _round2(target_pos[0]), _round2(target_pos[1])
        candidates = [(_round2(p.get('x', 0.0)), _round2(p.get('z', 0.0))) for p in reachable_pts]
        if not candidates:
            return None
        return min(candidates, key=lambda p: (p[0]-tx)**2 + (p[1]-tz)**2)
    except Exception:
        return None


def estimate_travel_time_to_object(controller, event, object_id: str, grid: float = GRID_SIZE) -> dict:
    """估算从当前Agent到指定对象的行走成本（不修改导航状态）。
    返回 {steps, meters, est_seconds}；若无法估算，返回空字典。
    """
    try:
        obj_pos = _obj_pos(event, object_id)
        if not obj_pos:
            return {}
        ax, az = _agent_pos(event)
        reachable = _get_reachable_points(controller)
        if not reachable:
            return {}
        goal = _nearest_reachable_to((obj_pos[0], obj_pos[1]), reachable, grid)
        if not goal:
            return {}
        path = _bfs_shortest_path((ax, az), goal, reachable, grid=grid)
        if not path:
            return {}
        steps = max(0, len(path) - 1)
        meters = steps * grid
        # 估算每步时间：移动一步 + 可能的轻微转向；使用全局 ACTION_DELAY_SEC 作为近似
        step_time = max(0.05, float(ACTION_DELAY_SEC))
        est_seconds = steps * step_time
        return {"steps": steps, "meters": round(meters, 2), "est_seconds": round(est_seconds, 2)}
    except Exception:
        return {}


# ---- Open-door aware helpers -------------------------------------------------

def _distance_point_to_segment(px: float, pz: float, ax: float, az: float, bx: float, bz: float) -> float:
    """Distance from point P to segment AB in XZ-plane."""
    try:
        apx, apz = px - ax, pz - az
        abx, abz = bx - ax, bz - az
        ab2 = abx * abx + abz * abz
        if ab2 <= 1e-6:
            # A and B are the same point
            dx, dz = px - ax, pz - az
            return math.hypot(dx, dz)
        # Project AP onto AB, clamp to [0,1]
        t = (apx * abx + apz * abz) / ab2
        if t < 0.0:
            qx, qz = ax, az
        elif t > 1.0:
            qx, qz = bx, bz
        else:
            qx, qz = ax + t * abx, az + t * abz
        return math.hypot(px - qx, pz - qz)
    except Exception:
        return 1e9


def _preclose_blocking_openables(controller, event, target_object_id: str,
                                 agent_radius: float = 1.4,
                                 corridor_half_width: float = 0.55) -> tuple:
    """
    Before navigation, proactively close nearby opened doors/drawers that likely block the path.
    Heuristic rules:
    - Close any openable object that is currently open AND
      a) within agent_radius of the agent, OR
      b) within a corridor around the straight line segment from agent to target
         with half-width corridor_half_width.
    - Skip if the openable objectId equals the navigation target.
    Returns (new_event, closed_ids)
    """
    try:
        objs = event.metadata.get('objects', []) or []
        ax, az = _agent_pos(event)
        tgt_pos = _obj_pos(event, target_object_id)
        if not tgt_pos:
            return event, []
        tx, tz = tgt_pos[0], tgt_pos[1]

        closed_ids = []
        for o in objs:
            try:
                if not (o.get('openable') and o.get('isOpen')):
                    continue
                oid = o.get('objectId')
                if not oid or oid == target_object_id:
                    continue
                p = o.get('position') or {}
                x, z = float(p.get('x', 0.0)), float(p.get('z', 0.0))
                # Near agent?
                near_agent = (x - ax) ** 2 + (z - az) ** 2 <= agent_radius ** 2
                # Near corridor?
                dist_seg = _distance_point_to_segment(x, z, ax, az, tx, tz)
                near_corridor = dist_seg <= corridor_half_width
                if near_agent or near_corridor:
                    # 检查对象是否可见，只关闭可见的对象
                    if not o.get('visible', False):
                        print(f"↪️ 跳过不可见的打开物体: {oid}")
                        continue

                    print(f"↪️ 预关闭可能阻塞通路的打开物体: {oid} (near_agent={near_agent}, near_path={near_corridor})")
                    ce = controller.step(action="CloseObject", objectId=oid, forceAction=True)
                    if ce.metadata.get('lastActionSuccess', False):
                        print(f"   ✅ 已关闭: {oid}")
                        event = ce
                        closed_ids.append(oid)
                    else:
                        err_msg = ce.metadata.get('errorMessage', '')
                        # 如果是可见性问题，静默处理
                        if "not found within the specified visibility" in str(err_msg):
                            print(f"   ℹ️ 对象不在视野内，跳过关闭: {oid}")
                        else:
                            print(f"   ⚠ 关闭失败: {oid} -> {err_msg}")
                        # Keep going; do not raise
            except Exception:
                continue
        return event, closed_ids
    except Exception:
        return event, []



def _ensure_nav_path(controller, event, target_pos: tuple[float, float]) -> bool:
    """Plan path once and cache into _nav_state; also update semantic_map['planned_path']."""
    global _nav_state, semantic_map
    try:
        ax, az = _agent_pos(event)
        reachable = _get_reachable_points(controller)
        if not reachable:
            print("⚠ 可达点为空，无法规划路径")
            return False
        path = _bfs_shortest_path((ax, az), target_pos, reachable, grid=GRID_SIZE)
        if not path or len(path) < 1:
            print("⚠ 无法规划到目标的路径")
            return False
        _nav_state["path"] = path
        _nav_state["idx"] = 0
        _nav_state["active"] = True
        _nav_state["target"] = {"type": "position", "pos": target_pos}
        _nav_state["start_time"] = time.time()  # 记录导航开始时间
        # Visualize planned path on semantic map
        try:
            semantic_map.setdefault("planned_path", [])
            semantic_map["planned_path"] = path[:]
            semantic_map.setdefault("task_points", []).append({"x": target_pos[0], "z": target_pos[1], "label": "target"})
            if len(semantic_map["task_points"]) > 30:
                semantic_map["task_points"] = semantic_map["task_points"][-30:]
        except Exception:
            pass
        print(f"🧭 规划路径长度: {len(path)}")
        return True
    except Exception as e:
        print(f"⚠ 规划路径失败: {e}")
        return False


def _step_follow_nav(controller, event) -> tuple:
    """Execute one step along cached path with advanced obstacle avoidance. Returns (event, reached: bool)."""
    global _nav_state
    path = _nav_state.get("path") or []
    idx = int(_nav_state.get("idx", 0))
    if not path or idx >= len(path) - 1:
        return event, True

    # 初始化避障状态
    if "obstacle_state" not in _nav_state:
        _nav_state["obstacle_state"] = {
            "blocked_count": 0,
            "last_position": None,
            "stuck_count": 0,
            "avoidance_mode": False,
            "avoidance_attempts": 0
        }

    obstacle_state = _nav_state["obstacle_state"]
    current_pos = _agent_pos(event)

    # 检查是否卡住（位置没有变化）
    if obstacle_state["last_position"]:
        if _distance(current_pos[0], current_pos[1],
                    obstacle_state["last_position"][0], obstacle_state["last_position"][1]) < 0.05:
            obstacle_state["stuck_count"] += 1
        else:
            obstacle_state["stuck_count"] = 0
    obstacle_state["last_position"] = current_pos

    cur = path[idx]
    nxt = path[idx + 1]
    dx = _round2(nxt[0] - cur[0])
    dz = _round2(nxt[1] - cur[1])

    # Determine target yaw (world-aligned)
    target_yaw = 0
    if dz < 0: target_yaw = 180
    elif dx > 0: target_yaw = 90
    elif dx < 0: target_yaw = 270

    yaw = _get_agent_yaw(event) % 360
    rot_diff = target_yaw - yaw
    if rot_diff > 180: rot_diff -= 360
    if rot_diff < -180: rot_diff += 360

    try:
        # 遇到多次卡住：先执行一次90度转弯避障并尝试前进，失败则进入重规划/高级避障
        if obstacle_state["stuck_count"] >= 3 and not obstacle_state.get("did_90_bypass"):
            direction = "RotateLeft" if (obstacle_state["stuck_count"] % 2 == 1) else "RotateRight"
            print(f"⤴️ 多次卡住，先{direction} 90° 进行避障")
            event = controller.step(action=direction, degrees=90)
            obstacle_state["did_90_bypass"] = True
            # 尝试前进一步
            ev_try = controller.step(action="MoveAhead")
            if ev_try.metadata.get("lastActionSuccess", True):
                print("✅ 90°转弯后前进成功")
                obstacle_state["stuck_count"] = 0
                return ev_try, False
            else:
                print("⚠ 90°转弯后前进失败，触发路径重规划")
                return _trigger_replan(controller, event)

        # 如果卡住太久，启动高级避障模式
        if obstacle_state["stuck_count"] > 5:
            return _advanced_obstacle_avoidance(controller, event, path, idx)

        # 正常导航逻辑
        if abs(rot_diff) > 1:
            step_deg = 15 if rot_diff > 0 else -15
            event = controller.step(action=("RotateRight" if step_deg > 0 else "RotateLeft"), degrees=abs(step_deg))
            return event, False

        # Move ahead one grid
        ev2 = controller.step(action="MoveAhead", forceAction=True)
        if ev2.metadata.get("lastActionSuccess", True):
            _nav_state["idx"] = idx + 1
            obstacle_state["blocked_count"] = 0  # 重置阻挡计数
            obstacle_state["stuck_count"] = 0    # 重置卡住计数
            obstacle_state.pop("did_90_bypass", None)
            return ev2, (_nav_state["idx"] >= len(path) - 1)
        else:
            # 移动失败，增加阻挡计数
            obstacle_state["blocked_count"] += 1
            return _handle_movement_blocked(controller, event, obstacle_state)

    except Exception as e:
        print(f"⚠ 导航异常: {e}")
        return event, False

def _handle_movement_blocked(controller, event, obstacle_state):
    """处理移动被阻挡的情况"""
    blocked_count = obstacle_state["blocked_count"]

    try:
        if blocked_count <= 3:
            # 尝试小角度左右转动
            angle = 10 + (blocked_count * 5)  # 10, 15, 20度
            direction = "RotateRight" if blocked_count % 2 == 1 else "RotateLeft"
            print(f"🔄 尝试{direction} {angle}度避障 (尝试{blocked_count}/3)")
            event = controller.step(action=direction, degrees=angle)
            return event, False

        elif blocked_count <= 6:
            # 尝试后退
            print(f"⬅ 尝试后退避障 (尝试{blocked_count-3}/3)")
            event = controller.step(action="MoveBack")
            if not event.metadata.get("lastActionSuccess", True):
                # 后退失败，尝试转向
                event = controller.step(action="RotateRight", degrees=30)
            return event, False

        elif blocked_count <= 10:
            # 尝试大角度转向
            angle = 45 + ((blocked_count - 6) * 15)  # 45, 60, 75, 90度
            direction = "RotateRight" if blocked_count % 2 == 1 else "RotateLeft"
            print(f"🔄 尝试大角度{direction} {angle}度 (尝试{blocked_count-6}/4)")
            event = controller.step(action=direction, degrees=angle)
            return event, False

        else:
            # 阻挡次数过多，触发重新规划
            print("⚠ 阻挡次数过多，需要重新规划路径")
            return _trigger_replan(controller, event)

    except Exception as e:
        print(f"⚠ 避障处理异常: {e}")
        return event, False


def _advanced_obstacle_avoidance(controller, event, path, idx):
    """高级避障模式：当Agent长时间卡住时使用"""
    global _nav_state
    obstacle_state = _nav_state["obstacle_state"]
    current_pos = _agent_pos(event)

    # 初始化避障历史记录
    if "avoidance_history" not in obstacle_state:
        obstacle_state["avoidance_history"] = []
        obstacle_state["escape_attempts"] = 0

    print(f"🚨 启动高级避障模式 (卡住{obstacle_state['stuck_count']}次)")

    try:
        # 检查是否在重复相同位置（防止转圈）
        for hist_pos in obstacle_state["avoidance_history"]:
            if _distance(current_pos[0], current_pos[1], hist_pos[0], hist_pos[1]) < 0.1:
                obstacle_state["escape_attempts"] += 1
                print(f"⚠ 检测到重复位置，脱困尝试 {obstacle_state['escape_attempts']}")
                break

        # 记录当前位置
        obstacle_state["avoidance_history"].append(current_pos)
        if len(obstacle_state["avoidance_history"]) > 10:
            obstacle_state["avoidance_history"].pop(0)  # 只保留最近10个位置

        # 如果脱困尝试过多，使用激进策略
        if obstacle_state["escape_attempts"] > 3:
            print("🚨 使用激进脱困策略")
            return _aggressive_escape(controller, event)

        # 策略1: 尝试多方向探索 + 移动组合
        if obstacle_state["avoidance_attempts"] < 12:  # 增加尝试次数
            # 更复杂的探索模式：转向 + 移动 + 后退
            attempt = obstacle_state["avoidance_attempts"]

            if attempt < 4:
                # 前4次：基本方向探索
                angles = [45, -45, 90, -90]
                angle = angles[attempt]
                print(f"🔍 基本探索: 转向{angle}度")
                event = controller.step(action=("RotateRight" if angle > 0 else "RotateLeft"), degrees=abs(angle))

            elif attempt < 8:
                # 5-8次：大角度 + 后退组合
                angles = [135, -135, 180, -180]
                angle = angles[attempt - 4]
                print(f"🔄 大角度探索: 转向{angle}度 + 后退")
                event = controller.step(action=("RotateRight" if angle > 0 else "RotateLeft"), degrees=abs(angle))
                if event.metadata.get("lastActionSuccess", True):
                    event = controller.step(action="MoveBack")

            else:
                # 9-12次：随机角度 + 多步移动
                import random
                angle = random.choice([30, -30, 60, -60, 120, -120, 150, -150])
                print(f"🎲 随机探索: 转向{angle}度 + 多步移动")
                event = controller.step(action=("RotateRight" if angle > 0 else "RotateLeft"), degrees=abs(angle))
                if event.metadata.get("lastActionSuccess", True):
                    # 尝试连续移动
                    for _ in range(2):
                        move_event = controller.step(action="MoveAhead")
                        if move_event.metadata.get("lastActionSuccess", True):
                            event = move_event
                            break

            # 检查是否成功脱困
            new_pos = _agent_pos(event)
            if _distance(current_pos[0], current_pos[1], new_pos[0], new_pos[1]) > 0.1:
                print("✅ 成功脱困，重置避障状态")
                obstacle_state["stuck_count"] = 0
                obstacle_state["avoidance_attempts"] = 0
                obstacle_state["escape_attempts"] = 0
                obstacle_state["avoidance_history"] = []
                return event, False

            obstacle_state["avoidance_attempts"] += 1
            return event, False

        else:
            # 策略2: 重新规划路径
            print("🔄 所有探索失败，触发路径重规划")
            return _trigger_replan(controller, event)

    except Exception as e:
        print(f"⚠ 高级避障异常: {e}")
        return event, False


def _aggressive_escape(controller, event):
    """激进脱困策略：当常规避障失败时使用"""
    global _nav_state

    print("🚨 执行激进脱困策略")

    try:
        # 策略1: 连续后退 + 大角度转向
        print("   1. 连续后退脱离当前区域")
        success_moves = 0
        for i in range(3):  # 尝试后退3步
            back_event = controller.step(action="MoveBack")
            if back_event.metadata.get("lastActionSuccess", True):
                success_moves += 1
                event = back_event
                print(f"     后退步骤 {i+1}/3 成功")
            else:
                print(f"     后退步骤 {i+1}/3 失败")
                break

        if success_moves > 0:
            print(f"   ✅ 成功后退 {success_moves} 步")

            # 策略2: 180度转向 + 前进
            print("   2. 180度转向寻找新方向")
            turn_event = controller.step(action="RotateRight", degrees=180)
            if turn_event.metadata.get("lastActionSuccess", True):
                event = turn_event

                # 尝试前进
                forward_event = controller.step(action="MoveAhead")
                if forward_event.metadata.get("lastActionSuccess", True):
                    print("   ✅ 激进脱困成功")
                    # 重置所有避障状态
                    _nav_state["obstacle_state"] = {
                        "blocked_count": 0,
                        "last_position": None,
                        "stuck_count": 0,
                        "avoidance_mode": False,
                        "avoidance_attempts": 0,
                        "avoidance_history": [],
                        "escape_attempts": 0
                    }
                    return forward_event, False

        # 策略3: 如果后退失败，尝试原地多次转向
        print("   3. 原地多次转向寻找出路")
        for angle in [60, -60, 120, -120]:
            turn_event = controller.step(action=("RotateRight" if angle > 0 else "RotateLeft"), degrees=abs(angle))
            if turn_event.metadata.get("lastActionSuccess", True):
                event = turn_event
                move_event = controller.step(action="MoveAhead")
                if move_event.metadata.get("lastActionSuccess", True):
                    print(f"   ✅ 转向{angle}度后成功前进")
                    # 重置避障状态
                    _nav_state["obstacle_state"]["stuck_count"] = 0
                    _nav_state["obstacle_state"]["escape_attempts"] = 0
                    return move_event, False

        # 所有激进策略都失败，触发重规划
        print("   ❌ 激进脱困失败，触发重规划")
        return _trigger_replan(controller, event)

    except Exception as e:
        print(f"⚠ 激进脱困异常: {e}")
        return _trigger_replan(controller, event)


def _trigger_replan(controller, event):
    """触发路径重新规划"""
    global _nav_state

    try:
        print("🔄 重新规划路径...")
        target = _nav_state.get("target")
        if not target:
            print("⚠ 无法重新规划：目标信息丢失")
            return event, True  # 结束导航

        # 清除当前路径状态
        _nav_state["path"] = None
        _nav_state["idx"] = 0
        _nav_state["obstacle_state"] = {
            "blocked_count": 0,
            "last_position": None,
            "stuck_count": 0,
            "avoidance_mode": False,
            "avoidance_attempts": 0
        }

        # 重新规划路径
        target_pos = target.get("pos")
        if target_pos and _ensure_nav_path(controller, event, target_pos):
            print("✅ 路径重规划成功")
            return event, False
        else:
            print("❌ 路径重规划失败，结束导航")
            return event, True

    except Exception as e:
        print(f"⚠ 重规划异常: {e}")
        return event, True


# ======== NavMesh-based Navigation System ========

# ======================== 请用这个版本替换原函数 ========================
def _start_navmesh_navigation(controller, event, target_object_id):
    """使用标准AI2-THOR导航方法（参考tset.py实现）"""
    global _nav_state

    try:
        print(f"🧭 开始导航到: {target_object_id}")

        target_obj = next((obj for obj in event.metadata.get("objects", []) if obj.get("objectId") == target_object_id), None)

        if not target_obj:
            print(f"❌ 目标对象不存在: {target_object_id}")
            return False

        target_position = target_obj.get("position")
        if not target_position:
            print(f"❌ 无法获取目标对象位置: {target_object_id}")
            return False

        print(f"🎯 目标对象位置: {target_position}")

        print(f"🔍 获取可达位置...")
        reachable_event = controller.step(action="GetReachablePositions")
        if not reachable_event.metadata.get("lastActionSuccess", False):
            print("❌ 获取可达位置失败")
            return False

        reachable_positions_list = reachable_event.metadata.get("actionReturn", [])
        if not reachable_positions_list:
            print("❌ 没有可达位置")
            return False

        # 先按打开的柜门/抽屉过滤，再按静态台面安全缓冲过滤
        filtered_positions = _filter_positions_near_open_objects(reachable_positions_list, event)
        filtered_positions = _filter_positions_near_static_receptacles(filtered_positions, event, margin=0.2)
        reachable_positions_set = {(round(p['x'], 2), round(p['z'], 2)) for p in filtered_positions}
        print(f"✅ 获取到{len(filtered_positions)}个可达位置（已过滤{len(reachable_positions_list)-len(filtered_positions)}个被阻挡位置）")

        # 若可达集异常偏小（可能卡在角落/缝隙），尝试脱困一次并重取
        if len(filtered_positions) < 10:
            print("⚠ 可达点极少，可能处于角落/缝隙，尝试脱困...")
            _, escaped = _escape_from_corner(controller, event)
            chk = controller.step(action="GetReachablePositions")
            r2 = chk.metadata.get("actionReturn", []) or []
            if escaped and len(r2) > len(filtered_positions):
                filtered_positions = _filter_positions_near_open_objects(r2, event)
                filtered_positions = _filter_positions_near_static_receptacles(filtered_positions, event, margin=0.2)
                reachable_positions_set = {(round(p['x'], 2), round(p['z'], 2)) for p in filtered_positions}
                print(f"✅ 脱困后可达点恢复到 {len(filtered_positions)} 个")
            else:
                print("⚠ 脱困未能显著恢复可达集，继续尝试规划")

        agent_meta = event.metadata["agent"]
        start_position_agent = (round(agent_meta["position"]['x'], 2), round(agent_meta["position"]['z'], 2))
        print(f"📍 当前位置: {start_position_agent}")

        path = _find_shortest_path_tset_style(start_position_agent, target_obj, reachable_positions_set, 0.15)

        if path and len(path) > 1:
            print(f"✅ 路径规划成功，共{len(path)}步")
            print(f"   路径预览: {path[:3]}{'...' if len(path) > 3 else ''}")

            _nav_state = {
                "active": True,
                "target": target_object_id,
                "path": path,
                "idx": 0,
                "start_time": time.time(),
                "timeout": 30.0,
                "method": "TSET_STYLE",
                "recovery_attempts": 0,
                "last_failed": False,
                "aligned": False,  # --- 重置对齐状态 ---
                "alignment_failures": 0,  # --- 新增：对齐失败计数器 ---
                "soft_start": False  # --- 新增：软启动标记 ---
            }
            return True
        else:
            print("❌ 路径规划失败")
            return False

    except Exception as e:
        print(f"❌ 导航启动失败: {e}")
        return False


def _distance(x1, z1, x2, z2):
    """计算两点间的距离"""
    return ((x2 - x1) ** 2 + (z2 - z1) ** 2) ** 0.5


def _get_agent_yaw(event):
    """获取智能体当前朝向角度"""
    return event.metadata["agent"]["rotation"]["y"]


def _bearing_deg(x1, z1, x2, z2):
    """计算从点1到点2的方位角（度）"""
    import math
    dx = x2 - x1
    dz = z2 - z1
    angle_rad = math.atan2(dx, dz)
    angle_deg = math.degrees(angle_rad)
    return angle_deg % 360


def _normalize_deg(angle):
    """标准化角度到[-180, 180]范围"""
    while angle > 180:
        angle -= 360
    while angle < -180:
        angle += 360
    return angle


def _perform_physical_alignment(controller, current_pos, target_pos):
    """执行物理对齐：让机器人从当前位置移动到目标位置"""
    try:
        # 计算需要移动的方向和距离
        dx = target_pos[0] - current_pos[0]
        dz = target_pos[1] - current_pos[1]
        distance = (dx ** 2 + dz ** 2) ** 0.5

        if distance < 0.05:  # 已经很接近了
            return True

        print(f"🎯 开始物理对齐：移动 {distance:.2f}m")

        # 如果距离太远，直接放弃
        if distance > 2.0:
            print(f"❌ 对齐距离过远 ({distance:.2f}m > 2.0m)，放弃对齐")
            return False

        # 计算需要的旋转角度
        import math
        target_angle = math.atan2(dx, dz) * 180 / math.pi

        # 获取当前朝向
        event = controller.step(action="Pass")
        current_rotation = event.metadata['agent']['rotation']['y']

        # 计算需要旋转的角度
        angle_diff = target_angle - current_rotation
        while angle_diff > 180:
            angle_diff -= 360
        while angle_diff < -180:
            angle_diff += 360

        # 执行旋转
        rotation_steps = int(abs(angle_diff) / 90)
        if rotation_steps > 0:
            action = "RotateRight" if angle_diff > 0 else "RotateLeft"
            for _ in range(rotation_steps):
                result = controller.step(action=action)
                if not result.metadata.get("lastActionSuccess", False):
                    print(f"❌ 对齐旋转失败，放弃对齐")
                    return False

        # 执行移动，限制最大移动步数
        move_steps = min(max(1, int(distance / 0.15)), 10)  # 最多10步
        successful_moves = 0
        for i in range(move_steps):
            result = controller.step(action="MoveAhead", moveMagnitude=0.15, forceAction=True)
            if result.metadata.get("lastActionSuccess", False):
                successful_moves += 1
            else:
                print(f"⚠ 对齐移动失败 (步骤 {i+1}/{move_steps})")
                # 如果连续3步都失败，就放弃
                if i - successful_moves >= 3:
                    print(f"❌ 连续移动失败过多，放弃对齐")
                    return False

        # 检查最终位置
        final_event = controller.step(action="Pass")
        final_pos = _agent_pos(final_event)
        if final_pos:
            final_distance = ((final_pos[0] - target_pos[0]) ** 2 + (final_pos[1] - target_pos[1]) ** 2) ** 0.5
            if final_distance < 0.3:  # 放宽到30cm内认为成功
                print(f"✅ 物理对齐成功，最终距离: {final_distance:.2f}m")
                return True
            else:
                print(f"❌ 物理对齐失败，距离: {final_distance:.2f}m")
                return False

        return False

    except Exception as e:
        print(f"❌ 物理对齐异常: {e}")
        return False


def _filter_positions_near_open_objects(reachable_positions, event):
    """过滤掉被打开的柜子/抽屉阻挡的可达位置"""
    if not reachable_positions or not event:
        return reachable_positions

    try:
        # 获取所有打开的可开启物体
        open_objects = []
        for obj in event.metadata.get('objects', []):
            if obj.get('openable') and obj.get('isOpen'):
                pos = obj.get('position', {})
                if pos:
                    open_objects.append({
                        'id': obj.get('objectId', ''),
                        'type': obj.get('objectType', ''),
                        'x': pos.get('x', 0),
                        'z': pos.get('z', 0),
                        'rotation': obj.get('rotation', {}).get('y', 0)
                    })

        if not open_objects:
            return reachable_positions

        # 过滤可达位置
        filtered_positions = []
        blocked_count = 0

        for pos in reachable_positions:
            px, pz = pos.get('x', 0), pos.get('z', 0)
            is_blocked = False

            # 检查是否被打开的物体阻挡
            for open_obj in open_objects:
                ox, oz = open_obj['x'], open_obj['z']

                # 计算距离
                distance = ((px - ox) ** 2 + (pz - oz) ** 2) ** 0.5

                # 如果位置太靠近打开的物体，认为可能被阻挡
                # 柜子门通常向外开启约0.5-0.8米
                if distance < 0.8:  # 0.8米安全距离
                    is_blocked = True
                    blocked_count += 1
                    break

            if not is_blocked:
                filtered_positions.append(pos)

        if blocked_count > 0:
            print(f"🚧 过滤掉{blocked_count}个可能被打开物体阻挡的位置")

        return filtered_positions

    except Exception as e:
        print(f"⚠ 位置过滤失败: {e}")
        return reachable_positions

# === 额外导航辅助：静态结构安全缓冲 + 角落脱困 ===
def _filter_positions_near_static_receptacles(reachable_positions, event, margin: float = 0.2):
    """避开靠近大型静态可放置台面的可达点，减小贴边擦碰（ManipulaTHOR 底盘更宽）。
    通过对象的 receptacle 属性（且不可拾取/不可移动）作为近似过滤标准。
    """
    try:
        static_surfaces = []
        for o in (event.metadata.get("objects", []) or []):
            if o.get("receptacle") and not o.get("pickupable") and not o.get("moveable"):
                p = o.get("position") or {}
                static_surfaces.append((float(p.get("x", 0.0)), float(p.get("z", 0.0))))
        if not static_surfaces:
            return reachable_positions
        out = []
        for pos in reachable_positions:
            px, pz = float(pos.get('x', 0.0)), float(pos.get('z', 0.0))
            too_close = False
            for sx, sz in static_surfaces:
                if ((px - sx) ** 2 + (pz - sz) ** 2) ** 0.5 < margin:
                    too_close = True
                    break
            if not too_close:
                out.append(pos)
        removed = len(reachable_positions) - len(out)
        if removed > 0:
            print(f"🛡️ 靠近静态台面移除了 {removed} 个可达点（安全缓冲 {margin}m）")
        return out
    except Exception as e:
        print(f"⚠ 静态结构缓冲过滤失败: {e}")
        return reachable_positions


def _escape_from_corner(controller, event, max_trials: int = 4):
    """尝试从角落/缝隙中脱困：后退、小步前进、侧移与小角度旋转混合。返回 (event, escaped: bool)。"""
    try:
        for t in range(max_trials):
            print(f"   -> 脱困尝试 {t+1}/{max_trials}")
            e = controller.step(action="MoveBack")
            if e.metadata.get("lastActionSuccess", False):
                ev = controller.step(action="RotateRight", degrees=30)
                ev = controller.step(action="MoveAhead", moveMagnitude=0.12, forceAction=True)
                if ev.metadata.get("lastActionSuccess", False):
                    # 检查可达集是否恢复
                    chk = controller.step(action="GetReachablePositions")
                    rp = chk.metadata.get("actionReturn", []) or []
                    if len(rp) >= 20:
                        print("   ✅ 脱困成功：可达点恢复")
                        return ev, True
            # 尝试侧移
            side = "MoveRight" if (t % 2 == 0) else "MoveLeft"
            ev = controller.step(action=side)
            # 即使侧移失败也继续下一轮
        return event, False
    except Exception as e:
        print(f"   ❌ 脱困异常: {e}")
        return event, False



def _get_wall_positions(event):
    """获取场景中所有墙壁的位置信息"""
    wall_positions = []
    try:
        for obj in event.metadata["objects"]:
            if obj["objectType"] in ["Wall", "WallPanel", "wall_panel"] or "wall" in obj["objectType"].lower():
                pos = obj["position"]
                bbox = obj.get("axisAlignedBoundingBox", {})

                # 获取墙壁的边界框信息
                if bbox:
                    center = bbox.get("center", pos)
                    size = bbox.get("size", {"x": 0.1, "y": 2.0, "z": 0.1})

                    wall_info = {
                        "objectId": obj["objectId"],
                        "center": (center["x"], center["z"]),
                        "size": (size["x"], size["z"]),
                        "bounds": {
                            "min_x": center["x"] - size["x"]/2,
                            "max_x": center["x"] + size["x"]/2,
                            "min_z": center["z"] - size["z"]/2,
                            "max_z": center["z"] + size["z"]/2
                        }
                    }
                    wall_positions.append(wall_info)

    except Exception as e:
        print(f"⚠️ 获取墙壁位置时出错: {e}")

    return wall_positions


def _is_path_blocked_by_wall(pos1, pos2, wall_positions, safety_margin=0.1):
    """检查两点之间的路径是否被墙壁阻挡"""
    x1, z1 = pos1
    x2, z2 = pos2

    # 检查路径是否与任何墙壁相交
    for wall in wall_positions:
        bounds = wall["bounds"]

        # 扩展墙壁边界以增加安全边距
        min_x = bounds["min_x"] - safety_margin
        max_x = bounds["max_x"] + safety_margin
        min_z = bounds["min_z"] - safety_margin
        max_z = bounds["max_z"] + safety_margin

        # 检查线段是否与扩展的墙壁矩形相交
        if _line_intersects_rect(x1, z1, x2, z2, min_x, min_z, max_x, max_z):
            return True

    return False


def _line_intersects_rect(x1, z1, x2, z2, rect_min_x, rect_min_z, rect_max_x, rect_max_z):
    """检查线段是否与矩形相交"""
    # 使用线段与矩形相交的算法
    # 首先检查线段端点是否在矩形内
    if (rect_min_x <= x1 <= rect_max_x and rect_min_z <= z1 <= rect_max_z) or \
       (rect_min_x <= x2 <= rect_max_x and rect_min_z <= z2 <= rect_max_z):
        return True

    # 检查线段是否与矩形的任何边相交
    # 矩形的四条边
    rect_edges = [
        (rect_min_x, rect_min_z, rect_max_x, rect_min_z),  # 下边
        (rect_max_x, rect_min_z, rect_max_x, rect_max_z),  # 右边
        (rect_max_x, rect_max_z, rect_min_x, rect_max_z),  # 上边
        (rect_min_x, rect_max_z, rect_min_x, rect_min_z)   # 左边
    ]

    for edge in rect_edges:
        if _lines_intersect(x1, z1, x2, z2, edge[0], edge[1], edge[2], edge[3]):
            return True

    return False


def _lines_intersect(x1, y1, x2, y2, x3, y3, x4, y4):
    """检查两条线段是否相交"""
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-10:
        return False  # 平行线

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom

    return 0 <= t <= 1 and 0 <= u <= 1


def _find_shortest_path_with_walls(start_pos, end_pos_obj, reachable_pos_set, wall_positions, grid_size):
    """增强的路径规划函数，考虑墙壁阻挡"""
    import queue
    import math

    # 找到离目标对象最近的可达位置
    min_dist = float('inf')
    end_pos_agent = None
    for pos in reachable_pos_set:
        dist = math.sqrt((pos[0] - end_pos_obj['x'])**2 + (pos[1] - end_pos_obj['z'])**2)
        if dist < min_dist:
            min_dist = dist
            end_pos_agent = pos

    if not end_pos_agent:
        return None

    print(f"🎯 导航终点（离物体最近的可达点）: {end_pos_agent}")

    # BFS路径搜索，考虑墙壁阻挡
    q = queue.Queue()
    q.put([start_pos])
    visited = set()
    visited.add(start_pos)

    while not q.empty():
        path = q.get()
        current_pos = path[-1]

        if current_pos == end_pos_agent:
            return path

        # 生成相邻位置
        for dx in [-grid_size, 0, grid_size]:
            for dz in [-grid_size, 0, grid_size]:
                if dx == 0 and dz == 0:
                    continue

                next_pos = (round(current_pos[0] + dx, 2), round(current_pos[1] + dz, 2))

                if next_pos in visited or next_pos not in reachable_pos_set:
                    continue

                # 检查路径是否被墙壁阻挡
                if _is_path_blocked_by_wall(current_pos, next_pos, wall_positions):
                    continue

                visited.add(next_pos)
                new_path = path + [next_pos]
                q.put(new_path)

    return None


def _find_shortest_path_tset_style(start_pos, target_obj, reachable_pos_set, grid_size):
    """
    路径规划函数（tset.py风格）
    - 首先从 reachable_pos_set 中选取到“目标AABB（若无则用position）”的最近可达点作为导航终点
    - 增加靠墙安全距离，持物体时尽量避开高成本区域
    """
    import queue
    import math

    # 解析目标的AABB/位置
    bbox = None
    pos = None
    try:
        bbox = (target_obj or {}).get('axisAlignedBoundingBox')
    except Exception:
        bbox = None
    try:
        pos = (target_obj or {}).get('position')
    except Exception:
        pos = None

    def _dist_to_target(px: float, pz: float) -> float:
        # 点到AABB的平面距离（XZ），在AABB内距离为0
        if bbox and isinstance(bbox, dict) and ('center' in bbox) and ('size' in bbox):
            cx = bbox['center'].get('x', 0.0)
            cz = bbox['center'].get('z', 0.0)
            sx = bbox['size'].get('x', 0.0)
            sz = bbox['size'].get('z', 0.0)
            dx = max(abs(px - cx) - sx / 2.0, 0.0)
            dz = max(abs(pz - cz) - sz / 2.0, 0.0)
            return math.hypot(dx, dz)
        if pos and isinstance(pos, dict):
            return math.hypot(px - pos.get('x', 0.0), pz - pos.get('z', 0.0))
        return float('inf')

    # 选择离AABB最近的可达点作为终点
    min_dist = float('inf')
    end_pos_agent = None
    for rp in reachable_pos_set:
        d = _dist_to_target(rp[0], rp[1])
        if d < min_dist:
            min_dist = d
            end_pos_agent = rp

    if not end_pos_agent:
        return None

    print(f"🎯 导航终点（离目标AABB最近的可达点）: {end_pos_agent}")

    # 简化：仅在持物体时进行基本的边缘避让
    try:
        last_evt = controller.last_event  # 使用全局controller的最新事件
    except NameError:
        last_evt = None

    # 检测是否持有物体
    is_holding_object = False
    if last_evt is not None:
        try:
            is_holding_object = len(last_evt.metadata.get("inventoryObjects", [])) > 0
        except Exception:
            is_holding_object = False

    # 只有在持物体时才进行边缘检测，且更保守
    costly_nodes = set()
    if is_holding_object:
        reachable_set = set(reachable_pos_set)
        edge_count = 0

        for rp in reachable_pos_set:
            x, z = rp
            # 只检查直接相邻的4个方向，如果3个或以上方向不可达，才标记为边缘
            blocked_directions = 0
            for dx, dz in [(0, grid_size), (0, -grid_size), (grid_size, 0), (-grid_size, 0)]:
                neighbor = (round(x + dx, 2), round(z + dz, 2))
                if neighbor not in reachable_set:
                    blocked_directions += 1

            # 只有当3个或以上方向被阻挡时，才认为是危险的边缘位置
            if blocked_directions >= 3:
                costly_nodes.add(rp)
                edge_count += 1
                # 限制边缘节点数量，避免过度限制
                if edge_count > len(reachable_pos_set) // 4:  # 最多标记1/4的节点为边缘
                    break

    print(f"🚧 检测到 {len(costly_nodes)} 个边缘节点，持物体: {is_holding_object}")

    # 检查起始位置和目标位置是否在可达集合中
    reachable_set = set(reachable_pos_set)

    # 先尝试将起始位置对齐到网格
    aligned_start = (round(start_pos[0] / grid_size) * grid_size, round(start_pos[1] / grid_size) * grid_size)
    aligned_start = (round(aligned_start[0], 2), round(aligned_start[1], 2))

    print(f"🔍 起始位置对齐: {start_pos} -> {aligned_start}")

    # 起点容差：在连续坐标与离散可达点之间允许小偏差，不必强制物理对齐
    start_tolerance = max(0.10, grid_size * 0.9)

    if aligned_start in reachable_set:
        start_pos = aligned_start
        print(f"✅ 使用对齐后的起始位置: {start_pos}")
    elif start_pos not in reachable_set and aligned_start not in reachable_set:
        print(f"⚠ 起始位置不在可达集合中: {start_pos}")
        # 寻找最近的可达位置作为起始点
        min_dist = float('inf')
        nearest_start = None
        for rp in reachable_pos_set:
            dist = ((rp[0] - start_pos[0]) ** 2 + (rp[1] - start_pos[1]) ** 2) ** 0.5
            if dist < min_dist:
                min_dist = dist
                nearest_start = rp
        if nearest_start and min_dist <= 1.2:
            print(f"🔄 使用最近的可达位置作为起始点: {nearest_start} (距离: {min_dist:.2f}m)")
            # 关键策略：先进行“软投影”确保图搜索可启动，再尽力物理对齐
            original_pos = (start_pos[0], start_pos[1])
            start_pos = nearest_start
            if min_dist > start_tolerance:
                print(f"🎯 需要物理对齐：从 {original_pos} 移动到 {nearest_start} (距离: {min_dist:.2f}m)")
                alignment_success = _perform_physical_alignment(controller, original_pos, nearest_start)
                if not alignment_success:
                    print(f"⚠ 物理对齐失败，改为使用投影起点继续规划")
                else:
                    print(f"✅ 物理对齐完成")
        else:
            print(f"❌ 无法找到合适的起始位置，最近距离: {min_dist:.2f}m")
            # 显示前5个最近的可达位置用于调试
            distances = [(((rp[0] - start_pos[0]) ** 2 + (rp[1] - start_pos[1]) ** 2) ** 0.5, rp) for rp in reachable_pos_set]
            distances.sort()
            print(f"🔍 最近的5个可达位置:")
            for i, (dist, pos) in enumerate(distances[:5]):
                print(f"  {i+1}. {pos} (距离: {dist:.2f}m)")
            return None

    if end_pos_agent not in reachable_set:
        print(f"⚠ 目标位置不在可达集合中: {end_pos_agent}")
        # 显示目标位置附近的可达位置
        distances = [(((rp[0] - end_pos_agent[0]) ** 2 + (rp[1] - end_pos_agent[1]) ** 2) ** 0.5, rp) for rp in reachable_pos_set]
        distances.sort()
        print(f"🔍 目标位置附近最近的5个可达位置:")
        for i, (dist, pos) in enumerate(distances[:5]):
            print(f"  {i+1}. {pos} (距离: {dist:.2f}m)")
        return None

    # 容差兜底：若仍不在集合中，强制投影到最近可达点
    if start_pos not in reachable_set:
        nearest_start = None
        min_dist = float('inf')
        for rp in reachable_pos_set:
            dist = ((rp[0] - start_pos[0]) ** 2 + (rp[1] - start_pos[1]) ** 2) ** 0.5
            if dist < min_dist:
                min_dist = dist
                nearest_start = rp
        if nearest_start is not None:
            print(f"🔁 起点最终兜底投影: {start_pos} -> {nearest_start} (距离: {min_dist:.2f}m)")
            start_pos = nearest_start

    # 最终检查：确保起始位置在可达集合中
    if start_pos not in reachable_set:
        print(f"❌ 最终检查失败：起始位置仍不在可达集合中: {start_pos}")
        return None

    print(f"🎯 BFS搜索: {start_pos} -> {end_pos_agent}")

    # 简化的BFS路径搜索
    q = queue.Queue()
    q.put([start_pos])
    visited = {start_pos}
    explored_count = 0
    max_explorations = 10000  # 防止无限搜索

    while not q.empty() and explored_count < max_explorations:
        path = q.get()
        current_pos = path[-1]
        explored_count += 1

        if current_pos == end_pos_agent:
            print(f"🛤️ 找到路径，步数: {len(path)-1}，探索了{explored_count}个节点")
            return path

        # 四个方向移动
        neighbors_added = 0
        for dx, dz in [(0, grid_size), (0, -grid_size), (grid_size, 0), (-grid_size, 0)]:
            next_pos = (round(current_pos[0] + dx, 2), round(current_pos[1] + dz, 2))

            # 调试第一步的邻居检查
            if explored_count == 1:
                in_reachable = next_pos in reachable_pos_set
                in_visited = next_pos in visited
                in_costly = next_pos in costly_nodes
                print(f"  邻居 {next_pos}: 可达={in_reachable}, 已访问={in_visited}, 边缘={in_costly}")

            if next_pos not in reachable_pos_set or next_pos in visited:
                continue

            # 只有在持物体且下一个位置是极端边缘时才避开
            if is_holding_object and next_pos in costly_nodes:
                continue

            new_path = list(path)
            new_path.append(next_pos)
            q.put(new_path)
            visited.add(next_pos)
            neighbors_added += 1

        # 调试第一步的结果
        if explored_count == 1:
            print(f"  第一步添加了 {neighbors_added} 个邻居到队列")

        # 每1000次探索输出一次进度
        if explored_count % 1000 == 0:
            print(f"🔍 BFS进度: 探索了{explored_count}个节点，队列大小: {q.qsize()}")

    if explored_count >= max_explorations:
        print(f"⚠ BFS搜索超时，探索了{explored_count}个节点")
    else:
        print(f"❌ BFS搜索完成但未找到路径，探索了{explored_count}个节点，访问了{len(visited)}个位置")

    return None


# ======================== 请用这个版本替换原函数 ========================
def _execute_navmesh_navigation(controller, event):
    """执行导航的一步（包含对齐逻辑）"""
    global _nav_state

    if not _nav_state.get("active", False):
        return event, True

    try:
        # --- 检查超时 ---
        if time.time() - _nav_state.get("start_time", 0) > _nav_state.get("timeout", 30.0):
            print("⏰ 导航超时")
            _nav_state["last_failed"] = True
            return event, True

        method = _nav_state.get("method", "TSET_STYLE")

        if method == "TSET_STYLE":
            path = _nav_state.get("path", [])
            if not path:
                return event, True

            # --- 核心改动：对齐逻辑（带失败降级） ---
            if not _nav_state.get("aligned", False):
                planned_start_pos = path[0]
                current_pos = _agent_pos(event)

                # 如果与规划起点的距离大于半个格子，则需要对齐
                if _distance(current_pos[0], current_pos[1], planned_start_pos[0], planned_start_pos[1]) > 0.075:
                    # 检查对齐失败次数
                    align_failures = _nav_state.get("alignment_failures", 0)
                    
                    # 失败次数超过阈值：强制跳过对齐，直接软启动
                    if align_failures >= 3:
                        print(f"⚠ 对齐已失败 {align_failures} 次，跳过对齐，从最近可达点软启动路径")
                        _nav_state["aligned"] = True
                        _nav_state["soft_start"] = True  # 标记为软启动模式
                        # 不再尝试对齐，直接进入路径执行逻辑
                    else:
                        print(f"👣 需要对齐: 从 {current_pos} 移动到路径起点 {planned_start_pos} (尝试 {align_failures + 1}/3)")

                        agent_yaw = _get_agent_yaw(event)
                        target_yaw = _bearing_deg(current_pos[0], current_pos[1], planned_start_pos[0], planned_start_pos[1])
                        rot_diff = _normalize_deg(target_yaw - agent_yaw)

                        if abs(rot_diff) > 5:  # 旋转容差5度
                            action = "RotateRight" if rot_diff > 0 else "RotateLeft"
                            degrees = min(abs(rot_diff), 30) # 每次最多转30度
                            print(f"   🔄 正在旋转 {degrees} 度以对齐...")
                            align_result = controller.step(action=action, degrees=degrees)
                            if not align_result.metadata.get("lastActionSuccess", False):
                                _nav_state["alignment_failures"] = align_failures + 1
                            return align_result, False
                        else:
                            print(f"   ➡️ 正在前进以对齐...")
                            align_result = controller.step(action="MoveAhead")
                            # 检查对齐移动是否成功
                            if not align_result.metadata.get("lastActionSuccess", False):
                                _nav_state["alignment_failures"] = align_failures + 1
                                print(f"   ⚠ 对齐移动失败 (次数: {_nav_state['alignment_failures']}/3)")
                            else:
                                # 检查是否真的接近了起点
                                new_pos = _agent_pos(align_result)
                                dist_after = _distance(new_pos[0], new_pos[1], planned_start_pos[0], planned_start_pos[1])
                                if dist_after > 0.075:
                                    _nav_state["alignment_failures"] = align_failures + 1
                                    print(f"   ⚠ 移动后仍未对齐 (距离: {dist_after:.2f}m, 次数: {_nav_state['alignment_failures']}/3)")
                            return align_result, False
                else:
                    print("✅ 对齐完成，开始执行路径。")
                    _nav_state["aligned"] = True
                    _nav_state["alignment_failures"] = 0
                    # 对齐完成后，直接进入下面的正式路径执行逻辑

            # --- 正式路径执行 ---
            idx = _nav_state.get("idx", 0)
            if idx >= len(path) - 1:
                print("🎯 导航完成")
                return event, True

            # 软启动模式：首次执行时跳到最近路径点
            if _nav_state.get("soft_start", False):
                current_pos = _agent_pos(event)
                # 找到距离当前位置最近的路径点
                min_dist = float('inf')
                closest_idx = idx
                for i in range(idx, len(path)):
                    dist = _distance(current_pos[0], current_pos[1], path[i][0], path[i][1])
                    if dist < min_dist:
                        min_dist = dist
                        closest_idx = i
                
                if closest_idx > idx:
                    print(f"🔧 软启动: 从当前位置 {current_pos} 跳到最近路径点 {path[closest_idx]} (路径索引 {closest_idx})")
                    _nav_state["idx"] = closest_idx
                    idx = closest_idx
                
                _nav_state["soft_start"] = False  # 只执行一次

            current_pos_on_path = path[idx]
            next_pos_on_path = path[idx + 1]

            print(f"🚶 导航步骤 {idx+1}/{len(path)-1}: 从{current_pos_on_path}到{next_pos_on_path}")

            dx = round(next_pos_on_path[0] - current_pos_on_path[0], 2)
            dz = round(next_pos_on_path[1] - current_pos_on_path[1], 2)

            target_yaw = 0
            if dz > 0: target_yaw = 0
            elif dz < 0: target_yaw = 180
            elif dx > 0: target_yaw = 90
            elif dx < 0: target_yaw = 270

            current_yaw = round(event.metadata["agent"]["rotation"]["y"]) % 360
            rotation_diff = _normalize_deg(target_yaw - current_yaw)

            if abs(rotation_diff) > 1:
                # ManipulaTHOR 底盘更宽，分段旋转更稳（最多每次90度）
                rotate_deg = min(abs(rotation_diff), 90)
                print(f"   🔄 从{current_yaw}度旋转到{target_yaw}度（本次旋转 {rotate_deg} 度）")
                nav_event = controller.step(action=("RotateRight" if rotation_diff > 0 else "RotateLeft"), degrees=rotate_deg)
                if not nav_event.metadata.get("lastActionSuccess", False):
                    # 旋转失败也走统一恢复逻辑
                    return _handle_navmesh_failure(controller, nav_event)
                return nav_event, False
            else:
                print(f"   ➡️ 向前移动到{next_pos_on_path}")
                # 动态调整步长：靠近障碍或操作台时，用更小步长以减少碰撞
                try:
                    ax, az = _agent_pos(event)
                    dist_to_next = max(0.0, ((next_pos_on_path[0] - ax) ** 2 + (next_pos_on_path[1] - az) ** 2) ** 0.5)
                except Exception:
                    dist_to_next = 0.15
                step_mag = min(0.15, max(0.07, dist_to_next))
                nav_event = controller.step(action="MoveAhead", moveMagnitude=step_mag, forceAction=True)
                if nav_event.metadata.get("lastActionSuccess", False):
                    # 只在确实接近目标网格点时推进索引，避免小步长造成的过早推进
                    try:
                        nx, nz = _agent_pos(nav_event)
                        rem = ((next_pos_on_path[0] - nx) ** 2 + (next_pos_on_path[1] - nz) ** 2) ** 0.5
                    except Exception:
                        rem = 0.0
                    if rem <= 0.08:
                        _nav_state["idx"] += 1
                        if _nav_state["idx"] >= len(path) - 1:
                            return nav_event, True
                    return nav_event, False
                else:
                    error_msg = nav_event.metadata.get("errorMessage", "Unknown error")
                    print(f"   ❌ 移动失败: {error_msg}")
                    # 若可达集已明显塌缩，优先尝试脱困
                    try:
                        chk = controller.step(action="GetReachablePositions")
                        rp = chk.metadata.get("actionReturn", []) or []
                        if len(rp) <= 3:
                            print("   ⚠ 可达集塌缩，仅有少量可达点，尝试脱困...")
                            ev2, esc = _escape_from_corner(controller, nav_event)
                            if esc:
                                return ev2, False
                    except Exception:
                        pass
                    return _handle_navmesh_failure(controller, nav_event)

        elif method in ("ShortestPath", "Precise") or _nav_state.get("action_sequence"):
            # 执行动作序列（内置最短路径或精确模式生成的动作）
            actions = _nav_state.get("action_sequence") or []
            idx = _nav_state.get("idx", 0)

            if not actions or idx >= len(actions):
                print("🎯 导航完成")
                return event, True

            a = actions[idx]
            try:
                if isinstance(a, dict):
                    nav_event = controller.step(a)
                elif isinstance(a, str):
                    nav_event = controller.step(action=a)
                else:
                    print(f"   ⚠ 跳过无法识别的动作项: {a}")
                    _nav_state["idx"] = idx + 1
                    return event, False
            except Exception as e:
                print(f"   ❌ 执行动作异常: {e}")
                return _handle_navmesh_failure(controller, event)

            if nav_event.metadata.get("lastActionSuccess", False):
                _nav_state["idx"] = idx + 1
                # 是否完成序列
                if _nav_state["idx"] >= len(actions):
                    print("🎯 导航完成")
                    try:
                        _nav_state["recovery_attempts"] = 0
                    except Exception:
                        pass
                    _nav_state["last_failed"] = False
                    return nav_event, True
                else:
                    return nav_event, False
            else:
                msg = nav_event.metadata.get("errorMessage", "Unknown error")
                print(f"   ❌ 序列动作失败: {msg}")
                return _handle_navmesh_failure(controller, nav_event)

        else:
            print(f"⚠ 未知的导航方法: {method}")
            return event, True

    except Exception as e:
        print(f"⚠ 导航执行异常: {e}")
        return _handle_navmesh_failure(controller, event)


def _move_to_position(controller, event, target_pos):
    """移动到指定位置（使用基本移动动作）"""
    try:
        current_pos = _agent_pos(event)
        dx = target_pos[0] - current_pos[0]
        dz = target_pos[1] - current_pos[1]

        # 计算需要的移动方向
        distance = (dx*dx + dz*dz)**0.5

        """

    # :  3  LLM 
    try:
        if attempt >= 3 and (attempt == 3 or attempt % 3 == 0):
            _maybe_trigger_replan(event, reason=f" {attempt} ")
    except Exception:
        pass
        """


        if distance < 0.1:  # 已经很接近了
            return event

        # 计算目标角度
        import math
        target_angle = math.atan2(dx, -dz) * 180 / math.pi  # AI2-THOR坐标系
        current_rotation = event.metadata["agent"]["rotation"]["y"]
        angle_diff = target_angle - current_rotation

        # 标准化角度到[-180, 180]
        while angle_diff > 180:
            angle_diff -= 360
        while angle_diff < -180:
            angle_diff += 360

        # 先旋转到正确方向
        if abs(angle_diff) > 45:  # 需要旋转
            step = min(90, abs(angle_diff))
            if angle_diff > 0:
                return controller.step(action="RotateRight", degrees=step)
            else:
                return controller.step(action="RotateLeft", degrees=step)
        else:
            # 向前移动，使用forceAction尝试绕过一些碰撞检测问题
            return controller.step(action="MoveAhead", forceAction=True)

    except Exception as e:
        print(f"⚠ 移动到位置失败: {e}")
        return event


def _handle_navmesh_failure(controller, event):
    """
    (增强版) 处理NavMesh导航失败的智能恢复逻辑
    """
    global _nav_state

    # 引入一个状态变量来跟踪恢复尝试的次数
    if "recovery_attempts" not in _nav_state:
        _nav_state["recovery_attempts"] = 0

    _nav_state["recovery_attempts"] += 1
    attempt = _nav_state["recovery_attempts"]

    print(f"🔧 NavMesh导航步骤失败, 尝试恢复... (第 {attempt} 次)")

    # 优先针对静态结构碰撞做本地避障（ManipulaTHOR底盘更宽，需更保守）
    try:
        err = (event.metadata or {}).get("errorMessage", "NavMesh step failed") if hasattr(event, 'metadata') else "NavMesh step failed"
        if isinstance(err, str) and "Collided with static structure" in err:
            print("   -> 检测到与静态结构碰撞，先尝试本地避障绕行")
            # 尝试右转90并前进一步
            e1 = controller.step(action="RotateRight", degrees=90)
            e2 = controller.step(action="MoveAhead", moveMagnitude=0.1, forceAction=True)
            if e2.metadata.get("lastActionSuccess", False):
                print("   ✅ 本地绕行成功（右转+前进）")
                return e2, False
            # 尝试左转180（相当于向另一侧避让）并前进一步
            e3 = controller.step(action="RotateLeft", degrees=180)
            e4 = controller.step(action="MoveAhead", moveMagnitude=0.1, forceAction=True)
            if e4.metadata.get("lastActionSuccess", False):
                print("   ✅ 本地绕行成功（左转+前进）")
                return e4, False
            # 尝试侧移（左右交替）
            side = "MoveRight" if attempt % 2 == 1 else "MoveLeft"
            es = controller.step(action=side)
            if es.metadata.get("lastActionSuccess", False):
                print("   ✅ 本地侧移成功")
                return es, False
    except Exception as _e:
        print(f"   ⚠ 本地避障失败: {_e}")

    # 再将该失败上报给 LLM 进行一次裁决；若LLM建议跳过/替换/重规划，则结束当前导航
    try:
        handled = _on_action_failure_llm("GoTo", {"nav_target": _nav_state.get("target")}, event, f"NavMesh导航步骤失败(第{attempt}次): {err}")
        if handled:
            _nav_state["last_failed"] = True
            return event, True
    except Exception as _e:
        print(f"⚠ LLM 决策失败（导航步骤失败上报）: {_e}")

    # 若恢复已尝试多次，向 LLM 报告以便重规划（节流由 _maybe_trigger_replan 内部处理）
    try:
        if attempt >= 3 and (attempt == 3 or attempt % 3 == 0):
            _maybe_trigger_replan(event, reason=f"导航恢复失败累计 {attempt} 次")
    except Exception:
        pass


    recovery_event = event
    # 优先尝试关闭近距离且已打开的可开物体（如柜门），为移动清障
    try:
        objs = event.metadata.get('objects', []) or []
        ax, az = _agent_pos(event)
        for o in objs:
            if o.get('openable') and o.get('isOpen') and o.get('objectId'):
                p = o.get('position') or {}
                x, z = float(p.get('x', 0.0)), float(p.get('z', 0.0))
                d2 = (x - ax) ** 2 + (z - az) ** 2
                if d2 <= 1.4 ** 2:  # 约1.4m范围内
                    # 检查对象是否可见
                    if not o.get('visible', False):
                        continue
                    print(f"   -> 尝试关闭附近打开的物体: {o.get('objectId')}")
                    ce = controller.step(action="CloseObject", objectId=o.get('objectId'), forceAction=True)
                    if ce.metadata.get('lastActionSuccess', False):
                        print("   ✅ 已关闭附近打开的柜门/抽屉，继续导航")
                        return ce, False
                    else:
                        err_msg = ce.metadata.get('errorMessage', '')
                        if "not found within the specified visibility" not in str(err_msg):
                            print(f"   ⚠ 关闭失败: {err_msg}")
    except Exception:
        pass

    # 若报错包含阻挡物体，尝试清障（拾取可拾取的小物体）
    try:
        err = (event.metadata or {}).get("errorMessage", "") or ""
        block_id = None
        if err:
            for o in (event.metadata.get("objects", []) or []):
                oid = o.get("objectId")
                if oid and oid in err:
                    block_id = oid
                    break
        if block_id:
            for o in (event.metadata.get("objects", []) or []):
                if o.get("objectId") == block_id and o.get("pickupable"):
                    p = o.get("position") or {}
                    x, z = float(p.get("x", 0.0)), float(p.get("z", 0.0))
                    ax, az = _agent_pos(event)
                    d2 = (x - ax) ** 2 + (z - az) ** 2
                    if d2 <= 1.2 ** 2:
                        print(f"   -> 尝试拾取阻挡物体以清障: {block_id}")
                        pe = controller.step(action="PickupObject", objectId=block_id, forceAction=True)
                        if pe.metadata.get("lastActionSuccess", False):
                            print("   ✅ 已拾取阻挡物体，继续导航")
                            return pe, False
    except Exception:
        pass

    # 尝试横向侧移避让（左右交替），为 MoveAhead 创造空间
    try:
        side = "MoveRight" if attempt % 2 == 1 else "MoveLeft"
        print(f"   -> 尝试侧移避让: {side}")
        se = controller.step(action=side)
        if se.metadata.get("lastActionSuccess", False):
            return se, False
    except Exception:
        pass


    try:
        # 策略1：先尝试后退，创造空间 (这是最关键的一步)
        if 1 <= attempt <= 2:
            print("   -> 恢复策略1: 尝试后退一步")
            recovery_event = controller.step(action="MoveBack")
            if recovery_event.metadata.get("lastActionSuccess", False):
                print("   ✅ 恢复动作 'MoveBack' 成功")
                return recovery_event, False  # 继续导航
            else:
                print("   ⚠ 'MoveBack' 失败，尝试小角度旋转作为兜底")
                # 在同一轮次立即做一个小角度旋转，避免直接判定失败而结束导航
                fallback_angle = 20 if attempt == 1 else 30
                direction = "RotateRight" if attempt == 1 else "RotateLeft"
                recovery_event = controller.step(action=direction, degrees=fallback_angle)
                return recovery_event, False

        # 策略2：小角度交替旋转
        elif 3 <= attempt <= 6:
            angle = 15 + (attempt - 3) * 10 # 15, 25, 35, 45度
            direction = "RotateRight" if attempt % 2 != 0 else "RotateLeft"
            print(f"   -> 恢复策略2: 尝试 {direction} {angle} 度")
            recovery_event = controller.step(action=direction, degrees=angle)
            if recovery_event.metadata.get("lastActionSuccess", False):
                print(f"   ✅ 恢复动作 '{direction}' 成功")
                return recovery_event, False
            else:
                 print(f"   ⚠ '{direction}' 失败")

        # 策略3：如果所有简单尝试都失败了，彻底重新规划路径
        else:
            print("   -> 恢复策略3: 所有简单恢复均失败，触发从当前位置的路径重规划！")
            target_id = _nav_state.get("target")

            # 清空当前失败的路径和恢复计数
            _nav_state = {"active": False, "target": None, "action_sequence": None, "idx": 0, "start_time": None, "timeout": 30.0}

            # 重新启动一次完整的导航规划
            if target_id and _start_navmesh_navigation(controller, event, target_id):
                 print("   ✅ 路径重规划成功，将执行新路径")
                 # 让主循环在下一帧开始执行新路径
                 return controller.step(action="Pass"), False
            else:
                print("   ❌ 路径重规划也失败了，放弃当前导航")
                # 将该失败上报给LLM进行裁决（是否跳过此GoTo、替换为其他步骤、或触发更高层重规划）
                try:
                    params = {"nav_target": target_id}
                    err = (event.metadata or {}).get("errorMessage", "Nav step failure") if hasattr(event, 'metadata') else "Nav step failure"
                    handled = _on_action_failure_llm("GoTo", params, event, str(err))
                    if handled:
                        _nav_state["last_failed"] = True
                        return event, True
                except Exception as _e:
                    print(f"⚠ LLM 决策失败（导航失败上报）: {_e}")
                _nav_state["last_failed"] = True
                return event, True # 返回 True 表示导航彻底结束

    except Exception as e:
        print(f"   ❌ 恢复过程中发生异常: {e}")
        _nav_state["last_failed"] = True
        return event, True # 发生未知异常，结束导航

    # 如果上面的所有尝试都没有成功返回，则默认导航结束
    print("   ❌ 未知恢复错误，结束导航")
    _nav_state["last_failed"] = True
    return recovery_event, True


def _start_simple_navigation(controller, event, target_object_id):
    """改进的简单导航模式（精确导航）"""
    global _nav_state

    try:
        obj_pos = _obj_pos(event, target_object_id)
        if not obj_pos:
            return False

        # 使用更精确的导航策略
        agent_pos = _agent_pos(event)
        dx = obj_pos[0] - agent_pos[0]
        dz = obj_pos[1] - agent_pos[1]
        distance = (dx*dx + dz*dz)**0.5

        print(f"🎯 精确导航: 从{agent_pos}到{obj_pos}, 距离{distance:.2f}m")

        # 生成精确的移动序列
        actions = []

        # 计算需要的角度
        import math
        target_angle = math.atan2(dx, -dz) * 180 / math.pi  # AI2-THOR坐标系
        current_rotation = event.metadata["agent"]["rotation"]["y"]
        angle_diff = target_angle - current_rotation

        # 标准化角度到[-180, 180]
        while angle_diff > 180:
            angle_diff -= 360
        while angle_diff < -180:
            angle_diff += 360

        # 转向目标
        if abs(angle_diff) > 5:  # 5度容差
            if angle_diff > 0:
                turn_steps = max(1, int(abs(angle_diff) / 90))
                actions.extend(["RotateRight"] * turn_steps)
            else:
                turn_steps = max(1, int(abs(angle_diff) / 90))
                actions.extend(["RotateLeft"] * turn_steps)

        # 精确移动 - 根据距离调整步数（使用正确的网格尺寸）
        move_steps = max(1, int(distance / 0.15))
        # 限制移动步数，避免过度移动
        if distance < 1.0:
            # 短距离：少量移动
            actions.extend(["MoveAhead"] * min(move_steps, 3))
        elif distance < 3.0:
            # 中距离：适中移动
            actions.extend(["MoveAhead"] * min(move_steps, 8))
        else:
            # 长距离：更多移动但有上限
            actions.extend(["MoveAhead"] * min(move_steps, 12))

        # 只在长距离时添加微调动作
        if distance > 2.0:
            actions.extend(["RotateRight", "RotateLeft"])  # 微调朝向
            actions.extend(["MoveAhead"] * 1)  # 一步额外接近

        print(f"🚶 精确导航模式，共{len(actions)}步")

        _nav_state = {
            "active": True,
            "target": target_object_id,
            "action_sequence": actions,
            "idx": 0,
            "start_time": time.time(),
            "timeout": 30.0,
            "method": "Precise"
        }
        return True

    except Exception as e:
        print(f"⚠ 精确导航启动失败: {e}")
        return False


# ======== End of NavMesh navigation ========

def update_point_cloud_visualizer(point_cloud):
    """更新点云可视化器中的点云数据"""
    global vis, pcd_vis, vis_running

    if not vis_running or vis is None or pcd_vis is None:
        return False

    try:
        # 检查点云数据
        if len(point_cloud.points) == 0:
            print("⚠ 点云数据为空，跳过更新")
            return False

        # 更新点云数据
        pcd_vis.points = point_cloud.points
        pcd_vis.colors = point_cloud.colors

        # 更新几何体
        vis.update_geometry(pcd_vis)
        vis.poll_events()
        vis.update_renderer()

        # 每100帧打印一次调试信息
        if hasattr(update_point_cloud_visualizer, 'frame_count'):
            update_point_cloud_visualizer.frame_count += 1
        else:
            update_point_cloud_visualizer.frame_count = 1

        if update_point_cloud_visualizer.frame_count % 100 == 0:
            # 显示当前帧点云信息
            total_points = len(point_cloud.points)
            print(f"�️ 全局地图更新第 {update_point_cloud_visualizer.frame_count} 帧，总点数: {total_points}")

        return True

    except Exception as e:
        print(f"❌ 更新点云可视化失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def close_point_cloud_visualizer():
    """关闭点云可视化器"""
    global vis, vis_running

    if vis is not None:
        try:
            vis.destroy_window()
            vis_running = False
            print("✓ 点云可视化器已关闭")
        except Exception as e:
            print(f"❌ 关闭点云可视化器失败: {e}")

# YOLO相关实现从 main_with_depth 中抽离至独立模块，便于复用与后续拆分
from yolo_utils import initialize_yolo, detect_objects

# 基于仿真器分割的检测（替代YOLO，准确且稳定）
def detect_objects_from_segmentation(event, image_rgb):
    """使用AI2-THOR的实例分割与元数据，生成检测结果并在图像上叠加分割着色与边框。
    返回：(annotated_bgr, detections)
    detections: [{'bbox':(x1,y1,x2,y2), 'confidence':1.0, 'class':str, 'class_id':-1, 'objectId':str}]
    """
    base = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    detections = []
    try:
        objs = event.metadata.get('objects', []) or []
        masks = getattr(event, 'instance_masks', None)
        if masks is None or len(masks) == 0:
            # 兜底：仅用可见物体的AABB在图像上画框（无像素级分割）
            for o in objs:
                if not o.get('visible', False):
                    continue
                # 无精确像素mask时，跳过绘制掩膜，仅记录类别
                detections.append({'bbox': (0,0,0,0), 'confidence': 1.0,
                                   'class': o.get('objectType') or o.get('name') or 'Object',
                                   'class_id': -1, 'objectId': o.get('objectId')})
            return base, detections

        overlay = base.copy()
        for o in objs:
            if not o.get('visible', False):
                continue
            oid = o.get('objectId')
            if oid not in masks:
                continue
            mask = masks[oid]
            if mask is None:
                continue
            # 计算bbox
            ys, xs = np.where(mask)
            if xs.size == 0 or ys.size == 0:
                continue
            x1, x2 = int(xs.min()), int(xs.max())
            y1, y2 = int(ys.min()), int(ys.max())

            cls_name = o.get('objectType') or o.get('name') or 'Object'
            color = get_type_color(cls_name)  # BGR

            # 叠加半透明掩膜
            overlay[mask] = (0.4 * np.array(color) + 0.6 * overlay[mask]).astype(np.uint8)
            # 轮廓线
            try:
                cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay, cnts, -1, color, 1)
            except Exception:
                pass
            # 边框与标签
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 1)
            cv2.putText(overlay, cls_name, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

            detections.append({'bbox': (x1, y1, x2, y2), 'confidence': 1.0,
                               'class': cls_name, 'class_id': -1, 'objectId': oid})
        base = overlay
    except Exception as e:
        print(f"⚠ 分割检测出错: {e}")
    return base, detections


# 创建保存目录（仅在需要时）
os.makedirs("semantic_maps", exist_ok=True)
if SAVE_CAPTURES:
    for d in CAPTURE_DIRS:
        os.makedirs(d, exist_ok=True)
    print("📁 保存目录已创建: captured_* 与 semantic_maps/")
else:
    print("📁 仅创建 semantic_maps/；已禁用自动生成 captured_* 目录")


# 初始化偏好学习系统
print("📚 初始化偏好学习系统...")
_load_preference_learning()

# 为兼容当前 ai2-thor 版本，清理可能遗留的无效提交号环境变量
# 某些环境下会设置 AI2THOR_COMMIT_ID=旧的提交号，导致 find_build 抛出
# “Invalid commit_id ... no build exists for arch=Windows” 错误。
os.environ.pop("AI2THOR_COMMIT_ID", None)

# 初始化 AI2-THOR 控制器，启用深度图像和导航网格
controller = Controller(
    width=1280,  # 宽度
    height=720,  # 高度
    fieldOfView=110,  # 视野角度
    scene=SCENE_NAME,  # 改为客厅场景，通常可移动物体更多
    gridSize=0.15,  # 移动步长（从0.25减小到0.15，提高导航精度）
    rotateStepDegrees=90,  # 旋转步长
    renderDepthImage=True,  # 启用深度图像
    renderInstanceSegmentation=True,  # 启用实例分割（可选）

    # 启用导航网格支持
    snapToGrid=False,  # 关闭网格对齐以支持更精确的导航
    visibilityDistance=1.5  # 设置可见距离
)
# 初始化二维语义地图（基于仿真器）
try:
    init_semantic_map(controller)
except Exception as e:
    print(f"⚠ 语义地图初始化失败: {e}")

# 初始化渐进式探索JSON文件
try:
    init_exploration_json()
except Exception as e:
    print(f"⚠ 探索JSON初始化失败: {e}")

# 载入（可选）柜体/容器标签，用于“非预设”的语义放置
try:
    EIO.load_container_labels(semantic_map)
except Exception as e:
    print(f"⚠ 容器标签加载失败: {e}")
# 载入（可选）“二次规划触发器”配置，用于在运行中根据新发现动态触发重规划
try:
    EIO.load_replan_triggers(semantic_map)
except Exception as e:
    print(f"⚠ 二次规划触发器加载失败: {e}")


# 实时探索JSON文件将在首次更新时自动创建


print("🎮 按键说明:")
print("  WASD控制移动，QE旋转，F抓取，G放下")
print("  n: 放入最近容器  o: 放到最近物体中心（失败自动回退）")
print("  v: 启用/禁用自主探索  V: 显示探索状态")
print("  w: 启用/禁用多视角导航  W: 显示多视角导航状态")
print("  0: 启用/禁用全图Frontier导航  9: 显示全图Frontier状态")
print("  M: 理解场景（调用LLM），N: 测试导航，C: 制造混乱")
print("  H: 切换热力图模式（ON/OFF）")
print("  X: 取消计划执行，P: 截图保存，ESC: 退出")
print(f"🖼️  窗口说明：左上=RGB图像，右上=深度图像，左下=实例分割，右下=Detection ({DETECTION_MODE})")

# —— 启动时将场景中的碗设置为“有污渍” ——
try:
    set_all_bowls_dirty(controller, dirty=True)
    print("🧽 已将场景中的碗设置为：有污渍")
except Exception as e:
    print(f"⚠ 设置碗为有污渍失败: {e}")

# —— 启动时自动制造混乱（按 INITIAL_CHAOS_LEVEL 控制） ——
try:
    apply_initial_chaos(controller, level=INITIAL_CHAOS_LEVEL)
except Exception as e:
    print(f"⚠ 初始化混乱执行失败: {e}")

# 初始化俯视图相机管理器
if TOPDOWN_CAMERA_ENABLED:
    try:
        topdown_manager = TopDownViewManager()
        topdown_manager.setup_topdown_camera(controller)
        print("✓ 俯视图相机已初始化")
    except Exception as e:
        print(f"⚠ 俯视图相机初始化失败: {e}")
        TOPDOWN_CAMERA_ENABLED = False

# 初始化YOLO模型
yolo_enabled = initialize_yolo()
print("🔍 [DEBUG] YOLO初始化完成，准备启动轻量化监控...")

# 初始化场景状态管理器
print("🔄 初始化场景状态管理器...")
SSM.scene_state_manager.initialize_on_startup()

# 🆕 初始化自主探索器
print("🤖 初始化自主探索器...")
explorer = AN.AutonomousExplorer(grid_size=0.15)
print("✅ 自主探索器已初始化（按v启用/禁用，按V显示状态）")

# 🆕 初始化多视角导航器（支持视角变化的探索）
print("🎯 初始化多视角导航器...")
viewpoint_navigator = VN.ViewpointNavigator(grid_size=0.15, cluster_radius=0.5)
print("✅ 多视角导航器已初始化（按w启用/禁用，按W显示状态）")
print("   特性: 考虑视角变化，优先探索 Frontier 和多角度观察")

# 🆕 初始化 PointNav 导航器（推荐：使用 AI2-THOR 原生 PointNav 功能）
pointnav_nav = PNN.PointNavNavigator(grid_size=0.15)
print("✅ PointNav 导航器已初始化 (推荐使用) (按0启用/禁用，按9显示状态)")
print("   特性: 使用 AI2-THOR PointNavExpertAction 直接导航，无需手动路径规划")

# 保留已知地图导航器（备选方案）
known_map_nav = KMN.KnownMapNavigator(grid_size=0.15)
print("✅ 已知地图导航器已初始化 (备选) (按8启用/禁用，按7显示状态)")

# 保留旧的 Frontier 导航器（备用，默认禁用）
frontier_nav = FFN.FrontierFullMapNavigator(grid_size=0.15)

# —— 启动轻量化LLM监控系统 ——
print("🔍 [DEBUG] 准备启动轻量化LLM监控系统...")
try:
    # 获取API密钥
    api_key = ""
    try:
        import os
        import sys
        api_key = os.environ.get('DASHSCOPE_API_KEY', '')
        print(f"🔍 [DEBUG] 环境变量API密钥: {'存在' if api_key else '不存在'}")

        if not api_key:
            # 尝试从配置文件获取 - 使用绝对路径
            script_dir = os.path.dirname(os.path.abspath(__file__))
            cfg_py = os.path.join(script_dir, 'embodied B1', 'config.py')
            print(f"🔍 [DEBUG] 脚本目录: {script_dir}")
            print(f"🔍 [DEBUG] 检查配置文件: {cfg_py}")
            if os.path.exists(cfg_py):
                print("🔍 [DEBUG] 配置文件存在，尝试直接读取...")
                # 直接读取配置文件内容
                try:
                    with open(cfg_py, 'r', encoding='utf-8') as f:
                        config_content = f.read()
                    print(f"🔍 [DEBUG] 配置文件内容长度: {len(config_content)}")

                    # 执行配置文件内容获取API密钥
                    config_globals = {}
                    exec(config_content, config_globals)
                    if 'DASHSCOPE_API_KEY' in config_globals:
                        api_key = config_globals['DASHSCOPE_API_KEY']
                        print(f"🔍 [DEBUG] 成功从配置文件获取API密钥，长度: {len(api_key)}")
                    else:
                        print("🔍 [DEBUG] 配置文件中未找到DASHSCOPE_API_KEY")
                except Exception as e:
                    print(f"🔍 [DEBUG] 读取配置文件失败: {e}")
            else:
                print("🔍 [DEBUG] 配置文件不存在")
    except Exception as e:
        print(f"🔍 [DEBUG] 获取API密钥异常: {e}")

    if api_key:
        # 创建回调函数来触发完整理解，避免循环导入
        def trigger_full_understanding_callback():
            try:
                trigger_llm_scene_understanding_async()
            except Exception as e:
                print(f"⚠️ 触发完整理解回调失败: {e}")

        LLM_MONITOR.start_lightweight_monitor(api_key, check_interval=15, trigger_callback=trigger_full_understanding_callback)
        print("🤖 轻量化LLM监控已启动，每15秒检查场景变化")
    else:
        print("⚠️ 未找到API密钥，跳过轻量化监控")
except Exception as e:
    print(f"⚠️ 启动轻量化监控失败: {e}")

# —— 启用第三视角相机（初始放置在Agent后上方） ——
try:
    _ev0 = controller.step(action="Pass")
    _apos = _ev0.metadata.get('agent', {}).get('position', {})
    _arot = _ev0.metadata.get('agent', {}).get('rotation', {})
    _yaw = float(_arot.get('y', 0.0))
    import math
    _dist, _height = 1.5, 1.6
    _rad = math.radians(_yaw)
    _tx = _apos.get('x', 0.0) - math.sin(_rad) * _dist
    _ty = _apos.get('y', 0.0) + _height
    _tz = _apos.get('z', 0.0) - math.cos(_rad) * _dist
    if THIRD_PERSON_ENABLED:
        controller.step(action="AddThirdPartyCamera",
                        position={"x": _tx, "y": _ty, "z": _tz},
                        rotation={"x": 35.0, "y": _yaw, "z": 0.0},
                        fieldOfView=90)
        print("🎥 已开启第三视角：窗口名 'Third-Person View'")
    
    # 添加俯视图相机（垂直向下）
    if TOPDOWN_CAMERA_ENABLED:
        # 俯视图相机放在房间上方，垂直向下
        topdown_pos = {
            "x": _apos.get('x', 0.0),
            "y": _apos.get('y', 0.0) + 4.0,  # 放在较高位置（4米高）
            "z": _apos.get('z', 0.0)
        }
        try:
            controller.step(action="AddThirdPartyCamera",
                            position=topdown_pos,
                            rotation={"x": 90.0, "y": 0.0, "z": 0.0},  # 完全向下看
                            fieldOfView=90)
            print("📷 已开启俯视图相机：显示房间真实俯视图")
        except Exception as e:
            print(f"⚠ 俯视图相机初始化失败: {e}")
except Exception as e:
    print(f"⚠ 第三视角相机初始化失败: {e}")

if yolo_enabled:
    print("🎯 YOLO物体检测已启用")
else:
    print("⚠ YOLO物体检测未启用")

# 点云可视化器：默认关闭（避免占用过多内存/窗口）并确保已关闭
pointcloud_vis_enabled = False
try:
    PCU.close_point_cloud_visualizer()
except Exception:
    pass

# 创建保存图像的文件夹
if SAVE_CAPTURES:
    for folder in ["captured_images", "captured_depth", "captured_segmentation"]:
        if not os.path.exists(folder):
            os.makedirs(folder)

# 定义全局变量用于存储按键状态
key_pressed = None

image_counter = 0  # 用于图像文件命名



def display_and_save_images(event, save_image=False):
    """显示RGB、深度和分割图像，可选择保存"""
    global image_counter, frame_count, SHOW_HEATMAPS

    # 获取RGB图像 (RGB格式)
    if INPUT_MODE == 'external':
        ext = read_external_frame(EXTERNAL_CFG)
        rgb_frame = getattr(ext, 'rgb', None)
    else:
        rgb_frame = event.frame
    if rgb_frame is None:
        # 外接或仿真均可能出现空帧，直接跳过本帧
        return

    # --- 核心转换区 ---
    # 1. 将原始RGB图转为BGR，作为后续处理的基础
    bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

    # 获取深度图像
    if INPUT_MODE == 'external':
        depth_frame = getattr(ext, 'depth', None)
    else:
        depth_frame = event.depth_frame
    if depth_frame is None:
        depth_colored = np.zeros_like(bgr_frame)
    else:
        # 将深度图像标准化到0-255范围用于显示
        depth_normalized = cv2.normalize(depth_frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)

    # 获取实例分割图像
    instance_frame = event.instance_segmentation_frame
    if instance_frame is None:
        print("⚠ 警告: 分割图像为空")
        instance_colored = np.zeros_like(bgr_frame)
    else:
        instance_colored = cv2.applyColorMap(instance_frame, cv2.COLORMAP_HSV)

    # 獲取俯視圖相機的RGB圖像（使用官方TopDownViewManager）
    topdown_frame = None
    topdown_rgb = None
    if TOPDOWN_CAMERA_ENABLED and topdown_manager is not None:
        try:
            topdown_rgb = topdown_manager.get_topdown_image(event)
            if topdown_rgb is not None:
                topdown_frame = cv2.cvtColor(topdown_rgb, cv2.COLOR_RGB2BGR)
        except Exception as e:
            # 靜默失敗，繼續顯示其他內容
            pass

    # 2. 物体检测：根据 DETECTION_MODE 选择 GT 分割或 YOLO
    if DETECTION_MODE == 'gt':
        det_frame_bgr, detections = detect_objects_from_segmentation(event, rgb_frame)
    else:
        det_frame_bgr, detections = detect_objects(rgb_frame)
    # ------------------

    # 调整图像大小以便并排显示
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
    cv2.putText(combined, f"Detection ({DETECTION_MODE})", (display_width + 10, display_height + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

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

    # 🆕 显示真实房间俯视图（来自官方TopDownViewManager）
    if TOPDOWN_CAMERA_ENABLED and topdown_rgb is not None and topdown_manager is not None:
        try:
            # 从事件中获取Agent信息
            agent = event.metadata.get("agent", {})
            agent_pos = agent.get("position", {"x": 0, "y": 0, "z": 0})
            agent_rotation = agent.get("rotation", {}).get("y", 0)
            
            # 使用TopDownViewManager的标注功能
            annotated_topdown = topdown_manager.annotate_topdown_image(
                topdown_rgb,
                agent_position={"x": agent_pos.get("x", 0), "z": agent_pos.get("z", 0)},
                agent_rotation=agent_rotation
            )
            
            # 显示标注后的俯视图
            topdown_manager.display_topdown(
                annotated_topdown,
                window_name="TopDown View - Bird's Eye",
                annotate=False  # 已经在上面标注过了
            )
        except Exception as e:
            # 降级处理：显示原始俯视图
            try:
                topdown_manager.display_topdown(
                    topdown_rgb,
                    window_name="TopDown View - Bird's Eye"
                )
            except Exception:
                pass
    else:
        # 如果俯视图相机未启用或未获得数据，可以显示其他信息
        pass

    # 🆕 禁用第三视角窗口显示
    # try:
    #     tp_frames = getattr(event, 'third_party_camera_frames', None)
    #     ...
    # except Exception:
    #     pass

    # 更新语义地图
    frame_count += 1
    try:
        update_semantic_map(event, frame_count)
    except Exception as e:
        print(f"⚠ 语义地图更新失败: {e}")

    # 🆕 禁用点云可视化
    # if depth_frame is not None and vis_running:
    #     try:
    #         pcd = PCU.generate_point_cloud(rgb_frame, depth_frame)
    #         PCU.update_point_cloud_visualizer(pcd)
    #     except Exception:
    #         pass

    # 保存图像（如果需要）
    if SAVE_CAPTURES and save_image:
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
        det_filename = f"captured_images/{DETECTION_MODE}_detect_{image_counter:04d}_{timestamp}.jpg"
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
                f.write(f"Detection Results ({DETECTION_MODE}) - {timestamp}\n")
                f.write(f"Total objects detected: {len(detections)}\n\n")
                for i, det in enumerate(detections):
                    f.write(f"Object {i+1}:\n")
                    f.write(f"  Class: {det['class']}\n")
                    f.write(f"  Confidence: {det['confidence']:.3f}\n")
                    f.write(f"  Bounding Box: {det['bbox']}\n\n")

        # 生成和保存点云（静默）
        if depth_frame is not None:
            try:
                pcd = PCU.generate_point_cloud(rgb_frame, depth_frame)
                pointcloud_filename = f"captured_pointclouds/pointcloud_{image_counter:04d}_{timestamp}.ply"
                PCU.save_point_cloud(pcd, pointcloud_filename)
                # 同时保存为PCD
                pcd_filename = f"captured_pointclouds/pointcloud_{image_counter:04d}_{timestamp}.pcd"
                PCU.save_point_cloud(pcd, pcd_filename)
            except Exception:
                pass

        print(f"图像已保存: RGB={rgb_filename}, Depth={depth_vis_filename}, Segmentation={seg_filename}, Detection={det_filename}")
        if detections:
            print(f"检测结果已保存: {detection_txt}")
        # 保存语义地图（PNG + JSON）
        try:
            EIO.export_semantic_map(semantic_map_img, semantic_map, f"semmap_{image_counter:04d}_{timestamp}")
        except Exception as e:
            print(f"⚠ 语义地图保存失败: {e}")
        image_counter += 1

    # 处理OpenCV窗口事件（当 pynput 不可用时，降级使用窗口按键）
    _poll_key_from_cv2_if_needed()


def _poll_key_from_cv2_if_needed():
    """当 pynput 不可用时，从 OpenCV 窗口轮询按键并映射到既有按键处理逻辑。"""
    global key_pressed, user_command_mode
    if PYNPUT_AVAILABLE:
        cv2.waitKey(1)
        return

    code = cv2.waitKey(1)
    if code < 0:
        return

    key_code = code & 0xFF
    if key_code == 255:
        return
    if key_code == 27:
        key_pressed = keyboard.Key.esc
        return

    if 0 <= key_code < 128:
        ch = chr(key_code).lower()
        if ch == 'm':
            if not user_command_mode:
                try:
                    _start_user_command_window_async()
                except Exception as _e:
                    print(f"⚠ 无法启动命令窗口: {_e}")
            key_pressed = 'm'
            return
        if user_command_mode:
            return
        key_pressed = ch

# 键盘按下事件处理函数
def on_press(key):
    global key_pressed, user_command_mode
    try:
        ch = key.char.lower()
        # 按下 m：同时触发场景理解，并打开命令窗口
        if ch == 'm':
            if not user_command_mode:
                try:
                    _start_user_command_window_async()
                except Exception as _e:
                    print(f"⚠ 无法启动命令窗口: {_e}")
            # 仍然将 'm' 透传给主循环，保持原有“场景理解”功能
            key_pressed = 'm'
        if user_command_mode:
            return  # 在命令窗口期间忽略所有键盘指令，避免误操作
        key_pressed = ch
    except AttributeError:
        if user_command_mode:
            return
        key_pressed = key

# 键盘释放事件处理函数
def on_release(key):
    if key == keyboard.Key.esc:  # 如果按下 ESC 键，退出监听
        return False

# 启动键盘监听器
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

# 主循环
frame_idx = 0  # 添加帧计数器
try:
    while True:
        frame_idx += 1  # 每帧递增
        # 保持场景更新
        event = controller.step(action="Pass")
        # 记录最新事件供命令LLM解析使用
        globals()['latest_event'] = event

        # 快照并清空按键，避免线程间竞争导致丢键
        current_key = key_pressed
        key_pressed = None

        # 三LLM流程结束且任务执行完后，自动恢复触发前的导航状态
        _auto_restore_navigation_after_llm_tasks(controller)

        # 🆕 执行自主探索（如果启用）
        if explorer.is_enabled and explorer.is_exploring:
            event = explorer.execute_exploration_step(controller, event)

        # 🆕 执行多视角导航（如果启用）
        if viewpoint_navigator.is_enabled and viewpoint_navigator.is_exploring:
            event = viewpoint_navigator.execute_exploration_step(controller, event)

        # 🆕 执行导航（PointNav 优先，然后是已知地图导航）
        if pointnav_nav.is_enabled:
            event = pointnav_nav.step(controller, event)
        elif known_map_nav.is_enabled:
            event = known_map_nav.step(controller, event)
        elif frontier_nav.is_enabled:
            event = frontier_nav.step(controller, event)

        # 处理 m/M：触发场景理解，并确保命令窗口已打开
        if current_key in ('m', 'M'):
            try:
                if not user_command_mode:
                    _start_user_command_window_async()
                print('🤖 启动LLM场景理解...')
                try:
                    # 重复触发保护
                    if _llm_understanding_in_progress:
                        print("⚠ 3-LLM场景理解已在进行中，拒绝重复触发。请稍候...")
                    else:
                        # 在触发完整理解前，若存在候选的实时探索文件则推广为正式文件
                        try:
                            EIO.promote_candidate_to_realtime()
                        except Exception:
                            pass
                        start_llm_scene_understanding(event=event)
                except Exception as _e:
                    print(f"⚠ 场景理解启动失败: {_e}")
            except Exception as _e:
                print(f"⚠ 处理 m 键失败: {_e}")

        # 自动执行计划优先；执行期间仅支持按 x 取消，其他按键忽略
        if executing_plan:
            if current_key == 'x':
                with _plan_lock:
                    planned_actions.clear()
                    globals()['executing_plan'] = False
                print('⏹ 已取消当前计划执行')
            else:
                event = execute_next_planned_action(controller, event)
                _sleep_if_slow()
                continue
        else:
            # 根据按键执行相应动作

            # 

            # 支持键盘滚动图例（PageUp/PageDown/上/下）以便在不支持鼠标滚轮的环境下使用
            if current_key in (keyboard.Key.page_up, keyboard.Key.up):
                legend_scroll_offset = max(0, legend_scroll_offset - LEGEND_ROW_H * 5)
                continue
            if current_key in (keyboard.Key.page_down, keyboard.Key.down):
                legend_scroll_offset = legend_scroll_offset + LEGEND_ROW_H * 5
                continue

            if current_key == keyboard.Key.esc:  # 按下 ESC 键退出
                break
            elif current_key == 'w':  # 前进
                event = controller.step(action="MoveAhead")
            elif current_key == 's':  # 后退
                event = controller.step(action="MoveBack")
            elif current_key == 'a':  # 左移
                event = controller.step(action="MoveLeft")
            elif current_key == 'd':  # 右移
                event = controller.step(action="MoveRight")
            elif current_key == 'q':  # 左转
                event = controller.step(action="RotateLeft")
            elif current_key == 'e':  # 右转
                event = controller.step(action="RotateRight")
            elif current_key == 'c':  # 制造混乱：苹果落地 + 椅子放倒
                try:
                    event = chaos_drop_apple_on_floor(controller, event)
                except Exception as e:
                    print(f"⚠ drop apple 失败: {e}")
                try:
                    event = chaos_tip_chair(controller, event)
                except Exception as e:
                    print(f"⚠ tip chair 失败: {e}")

            elif current_key == 'g':  # 放下手中物体
                inv = event.metadata.get('inventoryObjects', [])
                if inv:
                    drop_event = controller.step(action='DropHandObject', forceAction=True)
                    if drop_event.metadata.get('lastActionSuccess', False):
                        print('✓ 已放下手中物体')
                        event = drop_event
                    else:
                        print(f"⚠ 放下失败: {drop_event.metadata.get('errorMessage', 'Unknown error')}")
                else:
                    print('❌ 手中没有物体可放下')

            elif current_key == 'n':  # 快捷：将手中物体放入最近容器（优先冰箱/橱柜）
                inv = event.metadata.get('inventoryObjects', []) or []
                if not inv:
                    print('❌ 手中没有物体，无法放入容器')
                else:
                    held = inv[0].get('objectId')
                    objs = [o for o in (event.metadata.get('objects', []) or []) if o.get('visible') and o.get('receptacle')]

                    def is_fridge_or_cabinet(o: dict) -> bool:
                        text = ' '.join([
                            str(o.get('objectType') or ''),
                            str(o.get('objectId') or ''),
                            str(o.get('name') or ''),
                        ]).lower()
                        return any(k in text for k in ('fridge', 'refrigerator', 'cabinet', 'kitchencabinet', 'cupboard', 'drawer'))

                    dst = None
                    preferred = [o for o in objs if is_fridge_or_cabinet(o)]
                    if preferred:
                        ax, az = _agent_pos(event)
                        target = min(preferred, key=lambda o: (o.get('position', {}).get('x', 0) - ax) ** 2 + (o.get('position', {}).get('z', 0) - az) ** 2)
                        dst = target.get('objectId')

                    if not dst:
                        dst = _find_nearest_safe_receptacle(event)

                    if not dst:
                        # 回退：找最近可见容器
                        if objs:
                            ax, az = _agent_pos(event)
                            target = min(objs, key=lambda o: (o.get('position', {}).get('x', 0) - ax) ** 2 + (o.get('position', {}).get('z', 0) - az) ** 2)
                            dst = target.get('objectId')

                    if not dst:
                        print('❌ 附近没有可用容器')
                    else:
                        put_event = controller.step(action='PutObject', objectId=dst, forceAction=True, placeStationary=True)
                        if put_event.metadata.get('lastActionSuccess', False):
                            print(f"✅ 已放入最近容器: {held} -> {dst}")
                            event = put_event
                        else:
                            print(f"⚠ 放入容器失败: {put_event.metadata.get('errorMessage', 'Unknown error')}")

            elif current_key == 'o':  # 快捷：将手中物体放到最近物体中心（失败回退到容器）
                inv = event.metadata.get('inventoryObjects', []) or []
                if not inv:
                    print('❌ 手中没有物体，无法放置')
                else:
                    held = inv[0].get('objectId')
                    objs = [
                        o for o in (event.metadata.get('objects', []) or [])
                        if o.get('visible') and o.get('objectId') != held
                    ]

                    if not objs:
                        print('❌ 视野内没有可用目标物体')
                    else:
                        ax, az = _agent_pos(event)
                        target = min(objs, key=lambda o: (o.get('position', {}).get('x', 0) - ax) ** 2 + (o.get('position', {}).get('z', 0) - az) ** 2)
                        target_id = target.get('objectId')
                        center = ((target.get('axisAlignedBoundingBox') or {}).get('center') or {}).copy()
                        if not center:
                            p = target.get('position') or {}
                            center = {
                                'x': float(p.get('x', 0.0)),
                                'y': float(p.get('y', 0.0)) + 0.2,
                                'z': float(p.get('z', 0.0)),
                            }

                        placed = controller.step(action='PlaceObjectAtPoint', objectId=held, position=center, forceAction=True)
                        if placed.metadata.get('lastActionSuccess', False):
                            print(f"✅ 已放到最近物体中心: {held} -> {target_id}")
                            event = placed
                        else:
                            # 回退：若目标可作为容器，尝试 PutObject
                            if target.get('receptacle'):
                                put_event = controller.step(action='PutObject', objectId=target_id, forceAction=True, placeStationary=True)
                                if put_event.metadata.get('lastActionSuccess', False):
                                    print(f"✅ 中心放置失败，已回退放入目标容器: {held} -> {target_id}")
                                    event = put_event
                                else:
                                    print(f"⚠ 放置失败: {put_event.metadata.get('errorMessage', 'Unknown error')}")
                            else:
                                print(f"⚠ 中心放置失败且目标非容器: {placed.metadata.get('errorMessage', 'Unknown error')}")

            elif current_key == 'k':  # 打开最近的可开对象
                objs = [o for o in event.metadata.get('objects', []) or [] if o.get('visible') and o.get('openable')]
                if objs:
                    ax, az = _agent_pos(event)
                    target = min(objs, key=lambda o: (o.get('position',{}).get('x',0)-ax)**2 + (o.get('position',{}).get('z',0)-az)**2)
                    ev2 = controller.step(action='OpenObject', objectId=target.get('objectId'), forceAction=True)
                    print('✅ 打开' if ev2.metadata.get('lastActionSuccess') else f"⚠ 打开失败: {ev2.metadata.get('errorMessage','')}" )
                    event = ev2
                else:
                    print('❌ 附近没有可打开的对象')
            elif current_key == 'l':  # 关闭最近的可开对象
                objs = [o for o in event.metadata.get('objects', []) or [] if o.get('visible') and o.get('openable')]
                if objs:
                    ax, az = _agent_pos(event)
                    target = min(objs, key=lambda o: (o.get('position',{}).get('x',0)-ax)**2 + (o.get('position',{}).get('z',0)-az)**2)
                    ev2 = controller.step(action='CloseObject', objectId=target.get('objectId'), forceAction=True)
                    print('✅ 关闭' if ev2.metadata.get('lastActionSuccess') else f"⚠ 关闭失败: {ev2.metadata.get('errorMessage','')}" )
                    event = ev2
                else:
                    print('❌ 附近没有可关闭的对象')
            elif current_key == 't':  # 打开最近的可切换对象（如水龙头、灯）
                objs = [o for o in event.metadata.get('objects', []) or [] if o.get('visible') and (o.get('toggleable') or o.get('canToggle'))]
                if objs:
                    ax, az = _agent_pos(event)
                    target = min(objs, key=lambda o: (o.get('position',{}).get('x',0)-ax)**2 + (o.get('position',{}).get('z',0)-az)**2)
                    ev2 = controller.step(action='ToggleObjectOn', objectId=target.get('objectId'), forceAction=True)
                    print('✅ Toggle On' if ev2.metadata.get('lastActionSuccess') else f"⚠ 操作失败: {ev2.metadata.get('errorMessage','')}" )
                    event = ev2
                else:
                    print('❌ 附近没有可切换的对象')
            elif current_key == 'y':  # 关闭煤气/炉灶优先；找不到则关闭最近可切换对象
                objs = [o for o in event.metadata.get('objects', []) or [] if o.get('visible') and (o.get('toggleable') or o.get('canToggle'))]
                if objs:
                    ax, az = _agent_pos(event)
                    def _is_gas_related(o: dict) -> bool:
                        text = ' '.join([
                            str(o.get('objectType') or ''),
                            str(o.get('objectId') or ''),
                            str(o.get('name') or ''),
                        ]).lower()
                        return any(k in text for k in ('stoveknob', 'stoveburner', 'burner', 'gas', 'range'))

                    gas_objs = [o for o in objs if _is_gas_related(o)]
                    target_pool = gas_objs if gas_objs else objs
                    target = min(target_pool, key=lambda o: (o.get('position',{}).get('x',0)-ax)**2 + (o.get('position',{}).get('z',0)-az)**2)
                    ev2 = controller.step(action='ToggleObjectOff', objectId=target.get('objectId'), forceAction=True)
                    print('✅ Toggle Off' if ev2.metadata.get('lastActionSuccess') else f"⚠ 操作失败: {ev2.metadata.get('errorMessage','')}" )
                    event = ev2
                else:
                    print('❌ 附近没有可切换的对象')
            elif current_key == 'j':  # 为手中容器注水
                inv = event.metadata.get('inventoryObjects', []) or []
                if inv:
                    held = inv[0].get('objectId')
                    ev2 = controller.step(action='FillObjectWithLiquid', objectId=held, fillLiquid='water', forceAction=True)
                    print('✅ 注水成功' if ev2.metadata.get('lastActionSuccess') else f"⚠ 注水失败: {ev2.metadata.get('errorMessage','')}" )
                    event = ev2
                else:
                    print('❌ 手中没有容器')
            elif current_key == 'u':  # 倒空手中液体
                inv = event.metadata.get('inventoryObjects', []) or []
                if inv:
                    held = inv[0].get('objectId')
                    ev2 = controller.step(action='EmptyLiquidFromObject', objectId=held, forceAction=True)
                    print('✅ 倒空成功' if ev2.metadata.get('lastActionSuccess') else f"⚠ 倒空失败: {ev2.metadata.get('errorMessage','')}" )
                    event = ev2
                else:
                    print('❌ 手中没有容器')
            elif current_key == 'r':  # 清洁最近的可清洁对象
                objs = [o for o in event.metadata.get('objects', []) or [] if o.get('visible') and o.get('dirtyable') and o.get('isDirty')]
                if objs:
                    ax, az = _agent_pos(event)
                    target = min(objs, key=lambda o: (o.get('position',{}).get('x',0)-ax)**2 + (o.get('position',{}).get('z',0)-az)**2)
                    ev2 = controller.step(action='CleanObject', objectId=target.get('objectId'), forceAction=True)
                    print('✅ 清洁完成' if ev2.metadata.get('lastActionSuccess') else f"⚠ 清洁失败: {ev2.metadata.get('errorMessage','')}" )
                    event = ev2
                else:
                    print('ℹ️ 附近没有需要清洁的对象')
            elif current_key == 'b':  # 弄脏最近的可清洁对象（演示）
                objs = [o for o in event.metadata.get('objects', []) or [] if o.get('visible') and o.get('dirtyable') and not o.get('isDirty')]
                if objs:
                    ax, az = _agent_pos(event)
                    target = min(objs, key=lambda o: (o.get('position',{}).get('x',0)-ax)**2 + (o.get('position',{}).get('z',0)-az)**2)
                    ev2 = controller.step(action='DirtyObject', objectId=target.get('objectId'), forceAction=True)
                    print('✅ 已弄脏' if ev2.metadata.get('lastActionSuccess') else f"⚠ 操作失败: {ev2.metadata.get('errorMessage','')}" )
                    event = ev2
                else:
                    print('ℹ️ 附近没有可弄脏的对象')
            elif current_key == 'i':  # 切片最近的可切片对象（如 Apple/Bread）
                objs = [o for o in event.metadata.get('objects', []) or [] if o.get('visible') and (o.get('sliceable') or o.get('canBeSliced'))]
                if objs:
                    ax, az = _agent_pos(event)
                    target = min(objs, key=lambda o: (o.get('position',{}).get('x',0)-ax)**2 + (o.get('position',{}).get('z',0)-az)**2)
                    ev2 = controller.step(action='SliceObject', objectId=target.get('objectId'), forceAction=True)
                    print('✅ 已切片' if ev2.metadata.get('lastActionSuccess') else f"⚠ 切片失败: {ev2.metadata.get('errorMessage','')}" )
                    event = ev2
                else:
                    print('ℹ️ 附近没有可切片的对象')
            elif current_key == 'z':  # 进行一次“轻度整理”检测并入列
                ok = detect_and_enqueue_tidy_tasks(event)
                if not ok:
                    print('ℹ️ 暂无需要整理的轻度混乱')
            elif current_key == 'h':  # 切换热力图显示模式
                SHOW_HEATMAPS = not SHOW_HEATMAPS
                mode_text = "已启用热力图" if SHOW_HEATMAPS else "已禁用热力图"
                print(f"热力图模式: {mode_text}")
            elif current_key == 'T':  # 扫描并打印“全局任务清单”（含紧急/混乱）
                tasks = detect_scene_tasks(event)
                print_task_overview(tasks, limit=12)
            elif current_key == 'H':  # 自动处理最高优先级的安全任务（如关闭明火/水龙头）
                tasks = detect_scene_tasks(event)
                top_safety = next((t for t in tasks if t.get('category') == 'Safety'), None)
                if top_safety:
                    ok = enqueue_task_plan(event, top_safety)
                    if not ok:
                        print('⚠ 入列安全任务失败')
                else:
                    print('ℹ️ 未发现可自动修复的安全任务')
            elif current_key == 'v':  # 🆕 启用/禁用自主探索
                if explorer.is_enabled:
                    explorer.disable()
                else:
                    explorer.enable()
            elif current_key == 'V':  # 🆕 显示自主探索状态
                status = explorer.get_status()
                print(f"🤖 自主探索状态:")
                for k, v in status.items():
                    print(f"   {k}: {v}")
            elif current_key == '0':  # 启用/禁用 PointNav 导航（推荐）
                if pointnav_nav.is_enabled:
                    pointnav_nav.disable()
                else:
                    # 首次启用时初始化地图
                    if len(pointnav_nav.known_points) == 0:
                        print("⏳ 首次启用 PointNav，正在加载地图...")
                        if pointnav_nav.initialize_map(controller):
                            pointnav_nav.enable()
                        else:
                            print("❌ 地图加载失败")
                    else:
                        pointnav_nav.enable()
            elif current_key == '9':  # 显示 PointNav 导航状态（推荐）
                status = pointnav_nav.get_status()
                print("🗺️  PointNav 导航状态:")
                for k, v in status.items():
                    print(f"   {k}: {v}")
            elif current_key == '8':  # 启用/禁用已知地图导航（备选）
                if known_map_nav.is_enabled:
                    known_map_nav.disable()
                else:
                    # 首次启用时初始化地图
                    if len(known_map_nav.known_points) == 0:
                        print("⏳ 首次启用已知地图导航，正在加载地图...")
                        if known_map_nav.initialize_map(controller):
                            known_map_nav.enable()
                        else:
                            print("❌ 地图加载失败")
                    else:
                        known_map_nav.enable()
            elif current_key == '7':  # 显示已知地图导航状态（备选）
                status = known_map_nav.get_status()
                print("🗺️  已知地图导航状态:")
                for k, v in status.items():
                    print(f"   {k}: {v}")
            elif current_key == '?':  # 打印按键帮助
                print('— 按键帮助 —')
                print('  w/s/a/d/q/e: 移动/旋转')
                print('  f: 抓取最近可拾取  g: 放下')
                print('  o: 打开最近的柜子  k: 打开最近可开  l: 关闭最近可开')
                print('  t/y: 打开可切换对象 / 关闭时优先煤气(炉灶旋钮/燃烧器)')
                print('  j/u: 为手中容器注水/倒空液体')
                print('  r/b: 清洁/弄脏 最近的可清洁对象')
                print('  i: 切片最近可切片对象')
                print('  h: 切换热力图显示模式')
                print('  z: 检测并入列一次整理任务')
                print('  v: 启用/禁用自主探索  V: 显示自主探索状态')
                print('  w: 启用/禁用多视角导航  W: 显示多视角导航状态')
                print('  0: 启用/禁用全图Frontier导航  9: 显示全图Frontier状态')
                print('  m: 启动LLM场景理解  n: 放入最近容器(优先冰箱/橱柜)  p: 截图  x: 取消当前计划  ESC: 退出')

            elif current_key in ('m', 'M'):  # 使用实时探索数据触发LLM场景理解
                try:
                    print(f"\n🤖 启动LLM场景理解（使用实时探索数据）...")
                    # 重复触发保护
                    if _llm_understanding_in_progress:
                        print("⚠ 3-LLM场景理解已在进行中，拒绝重复触发。请稍候...")
                    else:
                        # 若存在候选文件，优先推广为正式 realtime 文件
                        try:
                            promoted = EIO.promote_candidate_to_realtime()
                            if promoted:
                                print("🔁 已将候选实时探索文件推广为正式文件")
                        except Exception:
                            pass
                        print(f"📄 使用文件: {REALTIME_EXPLORATION_JSON}")
                        if os.path.exists(REALTIME_EXPLORATION_JSON):
                            t = threading.Thread(target=start_llm_scene_understanding,
                                               args=(REALTIME_EXPLORATION_JSON, event), daemon=True)
                            t.start()
                            print("⏳ 已启动场景理解（后台运行），请稍候...")
                        else:
                            print("❌ 实时探索数据文件不存在，请先移动探索场景")
                except Exception as e:
                    print(f"⚠ 触发LLM场景理解失败: {e}")

            elif current_key in ('L',):  # 热重载柜体标签与重规划触发器
                try:
                    EIO.load_container_labels(semantic_map)
                    EIO.load_replan_triggers(semantic_map)
                    print('🔄 已重载: container_labels.json 与 replan_triggers.json')
                    _maybe_trigger_replan(event, reason='配置重载')
                except Exception as e:
                    print(f"⚠ 配置重载失败: {e}")

            elif current_key == 'f':  # 抓取最近的可见且可拾取物体
                all_objects = event.metadata.get('objects', [])
                visible = [o for o in all_objects if o.get('visible')]
                pickupable = [o for o in visible if o.get('pickupable')]
                target = None
                if pickupable:
                    ax, az = _agent_pos(event)
                    def _dist2(o):
                        p = o.get('position', {})
                        return (p.get('x', 0) - ax) ** 2 + (p.get('z', 0) - az) ** 2
                    target = min(pickupable, key=_dist2)
                elif visible:
                    target = visible[0]
                if target:
                    pick_event = controller.step(action='PickupObject', objectId=target['objectId'], forceAction=True)
                    if pick_event.metadata.get('lastActionSuccess', False):
                        print(f"✓ 已抓取: {target['objectId']}")
                        event = pick_event
                    else:
                        print(f"⚠ 抓取失败: {pick_event.metadata.get('errorMessage', 'Unknown error')}")
                else:
                    print('❌ 没有可抓取的物体')
            elif current_key == 'O':  # 打开最近的可见柜子（Cabinet/KitchenCabinet）
                try:
                    objs = event.metadata.get('objects', [])
                    agent_pos = event.metadata.get('agent', {}).get('position', {})
                    ax, az = agent_pos.get('x', 0.0), agent_pos.get('z', 0.0)

                    def is_cabinet(o: dict) -> bool:
                        ot = (o.get('objectType') or '').lower()
                        oid = (o.get('objectId') or '').lower()
                        name = (o.get('name') or '').lower()
                        if not o.get('openable'):
                            return False
                        # 仅限柜类对象；尽量保守匹配，避免误开其他容器
                        return ('cabinet' in ot) or ('kitchencabinet' in ot) or ('cabinet' in oid) or ('cabinet' in name)

                    cand = [o for o in objs if o.get('visible') and is_cabinet(o)]
                    if not cand:
                        print('❌ 视野内没有可打开的“柜子”对象')
                    else:
                        def dist2(o: dict) -> float:
                            p = o.get('position', {})
                            return (p.get('x', 0.0) - ax) ** 2 + (p.get('z', 0.0) - az) ** 2
                        target = min(cand, key=dist2)
                        if target.get('isOpen'):
                            print(f"ℹ️ 已经打开: {target.get('objectId')}")
                        else:
                            open_ev = controller.step(action='OpenObject', objectId=target.get('objectId'), forceAction=True)
                            if open_ev.metadata.get('lastActionSuccess', False):
                                print(f"✅ 已打开柜子: {target.get('objectId')}")
                                event = open_ev
                            else:
                                print(f"⚠ 打开失败: {open_ev.metadata.get('errorMessage', 'Unknown error')}")
                except Exception as e:
                    print(f"⚠ 打开柜子操作异常: {e}")

            elif current_key == 'N':  # 测试导航到最近的可见物体
                try:
                    all_objects = event.metadata.get("objects", [])
                    visible_objects = [obj for obj in all_objects if obj.get("visible")]
                    pickupable_objects = [obj for obj in visible_objects if obj.get("pickupable")]

                    print(f"📊 场景统计: 总物体{len(all_objects)}个, 可见{len(visible_objects)}个, 可拾取{len(pickupable_objects)}个")

                    if visible_objects:
                        print("👁️ 可见物体:")
                        for i, obj in enumerate(visible_objects[:5]):  # 只显示前5个
                            print(f"  {i+1}. {obj['objectId']} ({obj['objectType']}) - 可拾取: {obj.get('pickupable', False)}")

                    # 优先选择可拾取的物体，如果没有就选择任何可见物体
                    target_objects = pickupable_objects if pickupable_objects else visible_objects

                    if target_objects:
                        target_obj = target_objects[0]
                        print(f"🧭 测试导航到: {target_obj['objectId']} ({target_obj['objectType']})")
                        success = _start_navmesh_navigation(controller, event, target_obj['objectId'])
                        if success:
                            print("✅ 导航启动成功")
                        else:
                            print("❌ 导航启动失败")
                    else:
                        print("❌ 没有找到可导航的物体")
                except Exception as e:
                    print(f"⚠ 测试导航失败: {e}")

            elif current_key in ('v', 'V'):  # v: 启用/禁用自主探索  V: 显示状态
                if current_key == 'v':
                    if explorer.is_enabled:
                        explorer.disable()
                    else:
                        explorer.enable()
                else:  # V
                    status = explorer.get_status()
                    print(f"🤖 自主探索状态:")
                    print(f"   启用: {status['enabled']}")
                    print(f"   探索中: {status['exploring']}")
                    print(f"   访问位置: {status['visited_positions']}")
                    print(f"   总距离: {status['total_distance']}")
                    print(f"   已用时间: {status['elapsed_time']}")

            elif current_key in ('w', 'W'):  # w: 启用/禁用多视角导航  W: 显示状态
                if current_key == 'w':
                    if viewpoint_navigator.is_enabled:
                        viewpoint_navigator.disable()
                    else:
                        viewpoint_navigator.enable()
                else:  # W
                    status = viewpoint_navigator.get_status()
                    print(f"🎯 多视角导航状态:")
                    print(f"   启用: {status['enabled']}")
                    print(f"   探索中: {status['exploring']}")
                    print(f"   访问位置: {status['visited_positions']}")
                    print(f"   访问视角数: {status['visited_views']}")
                    print(f"   总距离: {status['total_distance']}")
                    print(f"   已用时间: {status['elapsed_time']}")

        # Failsafe: ensure planned actions execute even if earlier block is malformed
        if executing_plan:
            event = execute_next_planned_action(controller, event)
            _sleep_if_slow()
            continue

        # 第三视角相机跟随Agent（每帧更新）
        if THIRD_PERSON_ENABLED:
            try:
                apos = event.metadata.get('agent', {}).get('position', {})
                arot = event.metadata.get('agent', {}).get('rotation', {})
                yaw = float(arot.get('y', 0.0))
                import math
                dist, height = 1.6, 1.7
                rad = math.radians(yaw)
                cx = apos.get('x', 0.0) - math.sin(rad) * dist
                cy = apos.get('y', 0.0) + height
                cz = apos.get('z', 0.0) - math.cos(rad) * dist
                event = controller.step(action="UpdateThirdPartyCamera",
                                        thirdPartyCameraId=THIRD_PERSON_CAMERA_ID,
                                        position={"x": cx, "y": cy, "z": cz},
                                        rotation={"x": 35.0, "y": yaw, "z": 0.0})
            except Exception:
                pass
        
        # 俯视图相机更新（始终显示房间上方的俯视图）
        if TOPDOWN_CAMERA_ENABLED:
            try:
                apos = event.metadata.get('agent', {}).get('position', {})
                # 俯视图相机在房间上方，垂直向下看，位置跟随Agent的x/z，但y固定较高
                topdown_pos = {
                    "x": apos.get('x', 0.0),
                    "y": apos.get('y', 0.0) + 4.0,  # 固定在高度+4.0米
                    "z": apos.get('z', 0.0)
                }
                event = controller.step(action="UpdateThirdPartyCamera",
                                        thirdPartyCameraId=TOPDOWN_CAMERA_ID,
                                        position=topdown_pos,
                                        rotation={"x": 90.0, "y": 0.0, "z": 0.0})  # 垂直向下
            except Exception:
                pass

        # 如果用户按下了任意键或在执行计划，遵循慢速模式
        if SLOW_EXECUTION and (current_key is not None or executing_plan):
            _sleep_if_slow()



        # 如果有进行中的导航（tset.py风格），推进一步
        if _nav_state.get('active', False):
            event, reached = _execute_navmesh_navigation(controller, event)
            if reached:
                _nav_state['active'] = False

        # 实时显示图像（每帧都显示）
        save_image = (current_key == 'p')
        display_and_save_images(event, save_image)



        # 检查OpenCV窗口是否被关闭
        try:
            if cv2.getWindowProperty('AI2-THOR Multi-View', cv2.WND_PROP_VISIBLE) < 1:
                break
        except:
            # 如果窗口属性检查失败，继续运行
            pass

except KeyboardInterrupt:
    print("\n程序被用户中断")
finally:
    # 清理资源
    print("正在清理资源...")

    # 停止轻量化监控
    try:
        LLM_MONITOR.stop_lightweight_monitor()
        print("🤖 轻量化LLM监控已停止")
    except Exception as e:
        print(f"⚠ 停止轻量化监控失败: {e}")

    # 保存学习数据
    try:
        _save_preference_learning()
        print("💾 偏好学习数据已保存")
    except Exception as e:
        print(f"⚠ 保存偏好学习数据失败: {e}")

    cv2.destroyAllWindows()
    PCU.close_point_cloud_visualizer()
    controller.stop()
    listener.stop()
    print("程序已退出")


if __name__ == '__main__':
    # 主程序入口：当直接运行此脚本时启动
    print("🚀 启动 AI2-THOR 多模态探索系统...")
    print("📋 控制说明:")
    print("   WASD - 移动, QE - 转向, F - 抓取, O - 打开柜子")
    print("   M - LLM场景理解, P - 保存图像, ESC - 退出")
    print("   更多控制请查看代码注释")
    print("=" * 50)
