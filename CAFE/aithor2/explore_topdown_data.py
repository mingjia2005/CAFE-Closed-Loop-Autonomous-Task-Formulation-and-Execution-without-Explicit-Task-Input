#!/usr/bin/env python3
"""
深入探索GetMapViewCameraProperties返回的数据
"""

import sys
import json
from ai2thor.controller import Controller


def explore_mapview_data():
    """探索MapView数据"""
    print("🔬 深入探索GetMapViewCameraProperties数据")
    print("=" * 60)
    
    # 初始化控制器
    controller = Controller(scene='FloorPlan10', width=1280, height=720)
    event = controller.step(action="Pass")
    
    # 获取MapView相机信息
    print("\n📐 获取GetMapViewCameraProperties...")
    map_event = controller.step(action="GetMapViewCameraProperties")
    
    props = map_event.metadata.get("actionReturn", {})
    print("\n🎯 完整的actionReturn数据结构:")
    print(json.dumps(props, indent=2))
    
    # 检查event中的其他图像数据
    print("\n\n📸 检查event对象中的图像数据...")
    print(f"event.frame.shape: {event.frame.shape}")
    print(f"event.frame.dtype: {event.frame.dtype}")
    print(f"event.frame的值范围: min={event.frame.min()}, max={event.frame.max()}")
    
    # 检查是否有其他属性
    print("\n\n🔍 event中的关键属性:")
    attrs_to_check = [
        'third_party_camera_frames',  # 第三方相机
        'instance_segmentation_frame',
        'depth_frame',
        'semantic_segmentation_frame',
        'normal_frame',
        'object_segmentation_frame'
    ]
    
    for attr in attrs_to_check:
        if hasattr(event, attr):
            val = getattr(event, attr)
            if val is not None:
                if isinstance(val, dict):
                    print(f"✓ {attr}: dict with keys={list(val.keys())[:3]}...")
                elif hasattr(val, 'shape'):
                    print(f"✓ {attr}: array with shape={val.shape}")
                else:
                    print(f"✓ {attr}: {type(val)}")
            else:
                print(f"  {attr}: None")
        else:
            print(f"  {attr}: not found")
    
    # 列出event所有属性
    print("\n\n📋 Event对象的所有公开属性:")
    public_attrs = [attr for attr in dir(event) if not attr.startswith('_')]
    for i, attr in enumerate(public_attrs[:20]):  # 显示前20个
        try:
            val = getattr(event, attr)
            if not callable(val):
                if hasattr(val, 'shape'):
                    print(f"  - {attr}: array {val.shape}")
                elif isinstance(val, dict):
                    print(f"  - {attr}: dict")
                elif isinstance(val, (int, float, str, bool)):
                    print(f"  - {attr}: {type(val).__name__}")
                else:
                    print(f"  - {attr}: {type(val).__name__}")
        except Exception as e:
            print(f"  - {attr}: <Error: {e}>")
    
    # 尝试获取俯视图的其他方法
    print("\n\n🧪 尝试其他可能的俯视图方法...")
    
    # 1. 尝试参数化GetMapViewCameraProperties
    print("\n1️⃣  尝试GetMapViewCameraProperties with parameters...")
    try:
        # 尝试添加mode参数
        test_event = controller.step(
            action="GetMapViewCameraProperties",
            mode="top_down"
        )
        print("   ✓ mode参数接受但没有报错")
    except Exception as e:
        print(f"   ❌ mode参数: {e}")
    
    # 2. 尝试RenderObjectImage的变体
    print("\n2️⃣  尝试RenderObjectImage变体...")
    variants = [
        {"action": "RenderImage", "objectId": ""},
        {"action": "RenderTopDownImage"},
        {"action": "GetTopDownImage"},
        {"action": "RenderMapViewImage"},
    ]
    
    for variant in variants:
        try:
            test_event = controller.step(**variant)
            if hasattr(test_event, 'frame') and test_event.frame is not None:
                print(f"   ✅ {variant}: 成功! 返回frame={test_event.frame.shape}")
            else:
                print(f"   ⚠️  {variant}: 执行但无frame")
        except Exception as e:
            print(f"   ❌ {variant}: {str(e)[:50]}")
    
    # 3. 尝试agent.image with prefix
    print("\n3️⃣  检查Agent视角图像...")
    if hasattr(event, 'third_party_camera_frames'):
        print(f"   ✓ third_party_camera_frames 存在")
    else:
        print("   ✗ third_party_camera_frames 不存在")
    
    controller.stop()
    print("\n" + "=" * 60)
    print("✅ 探索完成")


if __name__ == "__main__":
    explore_mapview_data()
