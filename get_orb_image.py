import os
import sys
import time
import yaml
import cv2
import numpy as np
from pyorbbecsdk import Pipeline, Config, AlignFilter, OBStreamType, OBSensorType, OBFormat, OBFrameAggregateOutputMode, VideoFrame
from typing import Union, Any, Optional

# ==================== 常量 ====================
ESC_KEY = 27
MIN_DEPTH = 20    # 最小有效深度距离 mm
MAX_DEPTH = 10000 # 最大有效深度距离 mm


def frame_to_bgr_image(frame: VideoFrame) -> Union[Optional[np.array], Any]:
    """将Orbbec相机帧转换为BGR图像"""
    width = frame.get_width()
    height = frame.get_height()
    color_format = frame.get_format()
    data = np.asanyarray(frame.get_data())
    image = np.zeros((height, width, 3), dtype=np.uint8)
    if color_format == OBFormat.RGB:
        image = np.resize(data, (height, width, 3))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    elif color_format == OBFormat.BGR:
        image = np.resize(data, (height, width, 3))
    elif color_format == OBFormat.YUYV:
        image = np.resize(data, (height, width, 2))
        image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR_YUYV)
    elif color_format == OBFormat.MJPG:
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    elif color_format == OBFormat.UYVY:
        image = np.resize(data, (height, width, 2))
        image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR_UYVY)
    else:
        print(f"Unsupported color format: {color_format}")
        return None
    return image

def callback(frame):