# coding=utf-8
"""
标定验证脚本 - 点击图像获取3D位置
点击图像上的点，结合深度信息，计算该点在相机和机械臂基座坐标系下的位置

验证方法：
1. 点击法兰上的一个点
2. 获取该点的相机坐标系3D位置
3. 计算机械臂坐标系下的位置
4. 与机械臂返回的TCP位置对比，验证标定准确性
"""

import os
import sys
import time
import yaml
import cv2
import numpy as np
from pyorbbecsdk import Pipeline, Config, AlignFilter, OBStreamType, OBSensorType, OBFormat, OBFrameAggregateOutputMode, VideoFrame
from typing import Union, Any, Optional


np.set_printoptions(precision=4, suppress=True)

# ==================== 常量 ====================
ESC_KEY = 27
MIN_DEPTH = 20    # 最小有效深度距离 mm
MAX_DEPTH = 10000 # 最大有效深度距离 mm




# ==================== 全局变量 ====================
clicked_point = None  # 存储点击的像素坐标


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


def load_calibration_data(calib_file_path):
    """
    加载标定数据

    Returns:
        camera_matrix: 相机内参矩阵 (3x3)
        dist_coeffs: 畸变系数
        T_cam2base: 相机到基座的变换矩阵 (4x4)
    """
    if not os.path.exists(calib_file_path):
        raise FileNotFoundError(f"标定文件不存在: {calib_file_path}")

    with open(calib_file_path, 'r', encoding='utf-8') as f:
        calib_data = yaml.safe_load(f)

    # 解析相机内参
    if 'mtx_c2p' in calib_data:
        camera_matrix = np.array(calib_data['mtx_c2p'])
    elif 'camera_matrix' in calib_data:
        cam_mat_data = calib_data['camera_matrix']['data']
        camera_matrix = np.array(cam_mat_data).reshape(3, 3)
    else:
        raise KeyError("标定文件中未找到相机内参字段")

    # 解析畸变系数
    if 'dist' in calib_data:
        dist_coeffs = np.array(calib_data['dist'])
    elif 'distortion_coefficients' in calib_data:
        dist_coeffs = np.array(calib_data['distortion_coefficients']['data'])
    else:
        dist_coeffs = np.zeros((5, 1))

    # 解析变换矩阵 T_cam_base 是相机到基座的变换
    if 'T_cam_base' in calib_data:
        T_cam2base = np.array(calib_data['T_cam_base']).reshape(4, 4)
    elif 't_c2b' in calib_data and 'R_c2b' in calib_data:
        R = np.array(calib_data['R_c2b'])
        t = np.array(calib_data['t_c2b']).flatten()
        T_cam2base = np.eye(4)
        T_cam2base[:3, :3] = R
        T_cam2base[:3, 3] = t
    else:
        raise KeyError("标定文件中未找到变换矩阵")

    print(f"相机内参矩阵:\n{camera_matrix}")
    print(f"畸变系数: {dist_coeffs.flatten()}")
    print(f"相机→基座变换矩阵:\n{T_cam2base}")

    return camera_matrix, dist_coeffs, T_cam2base


def get_depth_at_point(depth_frame, u, v):
    """
    获取指定像素点的深度值

    Args:
        depth_frame: 深度帧
        u, v: 像素坐标

    Returns:
        depth: 深度值（单位：米）
    """
    if depth_frame is None:
        return None

    # width = depth_frame.get_width()
    # height = depth_frame.get_height()

    # if u < 0 or u >= width or v < 0 or v >= height:
    #     print(f"坐标超出范围: ({u}, {v}), 深度图尺寸: {width} x {height}")
    #     return None

    # 获取深度数据并reshape为(height, width)
    depth_value = depth_frame[v, u]
    depth_m = depth_value / 1000.0

    return depth_m


def pixel_to_camera_3d(u, v, depth, camera_matrix, dist_coeffs):
    """
    将像素坐标和深度转换为相机坐标系下的3D点

    Args:
        u, v: 像素坐标
        depth: 深度值（米）
        camera_matrix: 相机内参矩阵
        dist_coeffs: 畸变系数

    Returns:
        point_cam: 相机坐标系下的3D点 [x, y, z]（米）
    """
    # 去畸变
    point_undistorted = cv2.undistortPoints(
        np.array([[u, v]], dtype=np.float32).reshape(1, 1, 2),
        camera_matrix,
        dist_coeffs,
        None,
        camera_matrix
    )
    u_undist, v_undist = point_undistorted[0, 0]

    # 计算归一化相机坐标
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]

    x_norm = (u_undist - cx) / fx
    y_norm = (v_undist - cy) / fy

    # 转换为相机坐标系下的3D点
    x_cam = x_norm * depth
    y_cam = y_norm * depth
    z_cam = depth

    return np.array([x_cam, y_cam, z_cam])


def camera_3d_to_base_3d(point_cam, T_cam2base):
    """
    将相机坐标系下的3D点转换到机械臂基座坐标系

    Args:
        point_cam: 相机坐标系下的3D点 [x, y, z]（米）
        T_cam2base: 相机到基座的变换矩阵 (4x4)

    Returns:
        point_base: 基座坐标系下的3D点 [x, y, z]（米）
    """
    point_cam_homo = np.array([point_cam[0], point_cam[1], point_cam[2], 1.0])
    point_base_homo = T_cam2base @ point_cam_homo
    return point_base_homo[:3]


def mouse_callback(event, x, y, flags, param):
    """鼠标点击回调函数"""
    global clicked_point
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point = (x, y)
        print(f"\n点击位置: ({x}, {y})")


def main():
    """主程序"""

    # ==================== 配置 ====================
    script_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_path, "eye_hand_data")

    data_folders = [f for f in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, f))]
    if not data_folders:
        print("错误: 未找到标定数据文件夹")
        sys.exit(1)

    data_folders.sort(reverse=True)
    latest_folder = data_folders[0]
    CALIB_FILE_PATH = os.path.join(data_path, latest_folder, "camera_robot_pose.yaml")

    print(f"使用标定文件: {CALIB_FILE_PATH}")

    # ==================== 加载标定数据 ====================
    camera_matrix, dist_coeffs, T_cam2base = load_calibration_data(CALIB_FILE_PATH)

    # ==================== 初始化相机 ====================
    pipeline = Pipeline()
    config = Config()

    try:
        # 配置彩色流
        profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        color_profile = profile_list.get_video_stream_profile(0, 0, OBFormat.RGB, 0)
        config.enable_stream(color_profile)

        # 配置深度流
        profile_list = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        depth_profile = profile_list.get_default_video_stream_profile()
        config.enable_stream(depth_profile)

        # 确保获取完整帧集
        config.set_frame_aggregate_output_mode(OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)

    except Exception as e:
        print(f"流配置错误: {e}")
        return

    # 启用帧同步
    try:
        pipeline.enable_frame_sync()
    except Exception as e:
        print(f"帧同步错误: {e}")

    try:
        pipeline.start(config)
        print("相机启动成功")
    except Exception as e:
        print(f"Pipeline启动错误: {e}")
        return

    # 创建深度到彩色的对齐过滤器
    align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)


    # ==================== 创建窗口 ====================
    WINDOW_NAME = "Click to Verify Calibration"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

    print("\n=== 操作说明 ===")
    print("点击图像上的点，程序会显示:")
    print("  1. 该点在相机坐标系下的3D位置")
    print("  2. 该点在机械臂基座坐标系下的3D位置")
    print("按 'Q' 或 ESC 退出")
    print("按 'C' 清除点击标记")

    global clicked_point

    # ==================== 主循环 ====================
    try:
        while True:
            frames = pipeline.wait_for_frames(1000)
            if not frames:
                continue
                
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            
            if not color_frame or not depth_frame:
                continue

            # --- Spatial Alignment ---
            # Transforms one stream to the coordinate system/FOV of the other
            frames = align_filter.process(frames)
            if not frames:
                continue
            
            frames = frames.as_frame_set()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            
            if not color_frame or not depth_frame:
                continue

            # Convert raw color frame to BGR for OpenCV rendering
            color_image = frame_to_bgr_image(color_frame)
            if color_image is None:
                print("Failed to convert color frame")
                continue

            # --- Depth Image Processing --- only use color frame
            try:
                # Convert raw buffer to 2D numpy array
                depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape(
                    (depth_frame.get_height(), depth_frame.get_width()))
            except ValueError:
                print("Failed to reshape depth data")
                continue
            
            # Apply depth scale to get actual distance in mm and filter by range
            depth_data = depth_data.astype(np.float32) * depth_frame.get_depth_scale()
            depth_data = np.where((depth_data > MIN_DEPTH) & (depth_data < MAX_DEPTH), depth_data, 0)
            

            # 处理点击
            if clicked_point is not None:
                u, v = clicked_point

                depth = get_depth_at_point(depth_data, u, v)

                if depth is not None and depth > 0:
                    point_cam = pixel_to_camera_3d(u, v, depth, camera_matrix, dist_coeffs)
                    point_base = camera_3d_to_base_3d(point_cam, T_cam2base)

                    print(f"\n=== 计算结果 ===")
                    print(f"像素坐标: ({u}, {v})")
                    print(f"深度: {depth:.4f} m")
                    print(f"相机坐标系: [{point_cam[0]:.4f}, {point_cam[1]:.4f}, {point_cam[2]:.4f}] m")
                    print(f"基座坐标系: [{point_base[0]:.4f}, {point_base[1]:.4f}, {point_base[2]:.4f}] m")

                    cv2.circle(color_image, (u, v), 8, (0, 255, 0), 2)
                    cv2.circle(color_image, (u, v), 2, (0, 255, 0), -1)

                    info_text = [
                        f"Pixel: ({u}, {v})",
                        f"Depth: {depth:.3f} m",
                        f"Camera: ({point_cam[0]:.3f}, {point_cam[1]:.3f}, {point_cam[2]:.3f})",
                        f"Base: ({point_base[0]:.3f}, {point_base[1]:.3f}, {point_base[2]:.3f})"
                    ]

                    for i, text in enumerate(info_text):
                        cv2.putText(color_image, text, (u + 15, v + 15 + i * 25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                else:
                    print(f"无法获取有效深度值")
                    cv2.circle(color_image, (u, v), 8, (0, 0, 255), 2)
                    cv2.putText(color_image, "No Depth", (u + 15, v + 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            cv2.imshow(WINDOW_NAME, color_image)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q') or key == 27:
                break
            elif key == ord('c') or key == ord('C'):
                clicked_point = None
                print("点击标记已清除")

    except KeyboardInterrupt:
        print("\n程序被中断")

    finally:
        cv2.destroyAllWindows()
        pipeline.stop()
        print("程序结束")


if __name__ == '__main__':
    main()