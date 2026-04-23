"""
点云/可视化 工具集：从 main_with_depth.py 抽离
- 生成点云 generate_point_cloud
- 保存点云 save_point_cloud
- 实时可视化器 init/update/close
"""
from __future__ import annotations
import numpy as np
import open3d as o3d
import cv2

# 可视化器模块级状态
vis = None
pcd_vis = None
vis_running = False


def generate_point_cloud(rgb_image, depth_image, camera_intrinsics=None):
    """从RGB与深度图生成点云 (Open3D)
    Args:
        rgb_image: (H, W, 3) RGB np.uint8
        depth_image: (H, W) float/np.uint16 深度
        camera_intrinsics: (fx, fy, cx, cy) 可选
    Returns:
        o3d.geometry.PointCloud
    """
    height, width = depth_image.shape
    if camera_intrinsics is None:
        fx = fy = width / (2.0 * np.tan(np.radians(45)))  # FOV=90°
        cx, cy = width / 2.0, height / 2.0
    else:
        fx, fy, cx, cy = camera_intrinsics

    u, v = np.meshgrid(np.arange(width), np.arange(height))
    valid_depth = depth_image > 0

    z = depth_image[valid_depth]
    x = (u[valid_depth] - cx) * z / fx
    y = (v[valid_depth] - cy) * z / fy

    colors = rgb_image[valid_depth] / 255.0
    points = np.column_stack((x, y, z))

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def save_point_cloud(pcd, filename: str) -> bool:
    """保存点云到文件。返回是否成功。"""
    try:
        o3d.io.write_point_cloud(filename, pcd)
        return True
    except Exception:
        return False


def init_point_cloud_visualizer() -> bool:
    """初始化Open3D实时点云可视化器。"""
    global vis, pcd_vis, vis_running
    try:
        print("🔄 创建Open3D可视化器...")
        vis = o3d.visualization.Visualizer()
        print("🔄 创建点云窗口...")
        vis.create_window("Real-time Point Cloud", width=800, height=600)
        print("✓ 点云窗口创建成功")

        # 初始点
        pcd_vis = o3d.geometry.PointCloud()
        test_points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        test_colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]])
        pcd_vis.points = o3d.utility.Vector3dVector(test_points)
        pcd_vis.colors = o3d.utility.Vector3dVector(test_colors)
        vis.add_geometry(pcd_vis)

        render_option = vis.get_render_option()
        render_option.point_size = 3.0
        render_option.background_color = np.array([0.1, 0.1, 0.1])

        vis.poll_events()
        vis.update_renderer()

        vis_running = True
        print("✓ 实时点云可视化器初始化成功")
        print("📋 点云窗口应该已经显示，标题为 'Real-time Point Cloud'")
        return True
    except Exception as e:
        print(f"❌ 点云可视化器初始化失败: {e}")
        return False


def update_point_cloud_visualizer(point_cloud) -> bool:
    """将新的点云数据推送到可视化窗口。"""
    global vis, pcd_vis, vis_running
    if not vis_running or vis is None or pcd_vis is None:
        return False
    try:
        if len(point_cloud.points) == 0:
            return False
        pcd_vis.points = point_cloud.points
        pcd_vis.colors = point_cloud.colors
        vis.update_geometry(pcd_vis)
        vis.poll_events()
        vis.update_renderer()
        # 可选：每100帧打印一次
        if hasattr(update_point_cloud_visualizer, 'frame_count'):
            update_point_cloud_visualizer.frame_count += 1
        else:
            update_point_cloud_visualizer.frame_count = 1
        return True
    except Exception as e:
        print(f"❌ 更新点云可视化失败: {e}")
        return False


def close_point_cloud_visualizer() -> None:
    """关闭点云可视化窗口。"""
    global vis, vis_running
    if vis is not None:
        try:
            vis.destroy_window()
            vis_running = False
            print("✓ 点云可视化器已关闭")
        except Exception as e:
            print(f"❌ 关闭点云可视化器失败: {e}")

