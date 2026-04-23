"""
YOLO 辅助函数：加载模型与检测
从 main_with_depth.py 剥离，便于复用与后续拆分。
"""
from ultralytics import YOLO
import numpy as np
import cv2

# 模型句柄（模块级缓存）
yolo_model = None


def initialize_yolo():
    """初始化YOLO模型（COCO预训练 yolov8n.pt）。
    返回 True/False 表示是否成功。
    """
    global yolo_model

    # 检查网络连接
    import subprocess
    import os

    try:
        # 快速检查网络连接
        print("🔄 检查网络连接...")
        result = subprocess.run(['ping', '-c', '1', '-W', '2', 'github.com'],
                              capture_output=True, timeout=3)
        if result.returncode != 0:
            raise ConnectionError("无法连接到GitHub")

        print("🔄 加载YOLOv8n预训练模型...")
        yolo_model = YOLO('yolov8n.pt')  # 使用COCO预训练模型
        print("✓ YOLOv8n预训练模型加载成功")
        print(f"📦 模型支持的类别数量: {len(yolo_model.names)}")
        print(f"📋 前10个类别: {list(yolo_model.names.values())[:10]}")
        return True
    except Exception as e:
        print(f"❌ YOLO模型加载失败: {e}")
        print("💡 提示：网络连接问题或模型文件下载失败")
        print("💡 建议：使用仿真器分割模式 (DETECTION_MODE='gt')")
        print("💡 程序将继续使用仿真器分割功能")
        yolo_model = None
        return False


def detect_objects(image_rgb):
    """使用YOLO检测物体。输入为RGB格式图像，输出为(BGR标注图, detections)。
    detections: [ { 'bbox':(x1,y1,x2,y2), 'confidence':float, 'class':str, 'class_id':int } ]
    若模型未加载，返回(原图BGR, [])。
    """
    if yolo_model is None:
        return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR), []

    try:
        results = yolo_model(image_rgb, verbose=False, conf=0.25)
        r = results[0]
        detections = []

        if r.boxes is not None and len(r.boxes) > 0:
            boxes = r.boxes.xyxy.cpu().numpy()
            confidences = r.boxes.conf.cpu().numpy()
            class_ids = r.boxes.cls.cpu().numpy()

            print(f"🔍 YOLO检测到 {len(boxes)} 个物体")
            for box, conf, cls_id in zip(boxes, confidences, class_ids):
                detections.append({
                    'bbox': tuple(map(int, box)),
                    'confidence': float(conf),
                    'class': yolo_model.names[int(cls_id)],
                    'class_id': int(cls_id)
                })
                print(f"  ✓ {yolo_model.names[int(cls_id)]}: {conf:.3f} at {tuple(map(int, box))}")
        else:
            print("🔍 YOLO没有检测到任何物体")

        annotated_image_bgr = r.plot()  # 返回BGR格式的、已标注的图像
        return annotated_image_bgr, detections

    except Exception as e:
        print(f"⚠ YOLO检测出错: {e}")
        return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR), []

