# coding=utf-8
"""
眼在手外标定验证脚本
实时显示机器人末端在相机图像上的投影位置，验证标定结果是否准确
支持点击真实末端位置，输出点击点在两个坐标系下的坐标
"""

import os
import sys
import time
import json
import socket
import yaml
import cv2
import numpy as np
from pyorbbecsdk import Pipeline, Config, AlignFilter, OBStreamType, OBSensorType, OBFormat, OBFrameAggregateOutputMode, VideoFrame
from typing import Union, Any, Optional

import logging
from libs.log_setting import CommonLog
from libs.auxiliary import get_ip, popup_message, find_latest_data_folder

np.set_printoptions(precision=4, suppress=True)

logger_ = CommonLog(logging.getLogger(__name__))

ESC_KEY = 27

# ==================== 全局变量 ====================
clicked_point = None  # 用户点击的真实末端位置
blend_alpha = 0.5    # 彩色图权重（深度图权重为 1 - blend_alpha）
MIN_DEPTH = 20       # 最小有效深度 mm
MAX_DEPTH = 10000    # 最大有效深度 mm


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


def depth_to_color_image(depth_data, depth_scale, min_depth=MIN_DEPTH, max_depth=MAX_DEPTH):
    """
    将深度数据转换为彩色可视化图像

    Args:
        depth_data: 深度数据 (uint16, 单位: depth_scale)
        depth_scale: 深度缩放因子
        min_depth: 最小有效深度 (mm)
        max_depth: 最大有效深度 (mm)

    Returns:
        depth_color: 彩色深度图像 (BGR)
    """
    # 转换为毫米单位
    depth_mm = depth_data.astype(np.float32) * depth_scale

    # 过滤无效深度
    depth_mm = np.where((depth_mm > min_depth) & (depth_mm < max_depth), depth_mm, 0)

    # 归一化到 0-255
    depth_normalized = np.zeros_like(depth_mm, dtype=np.uint8)
    valid_mask = depth_mm > 0
    if valid_mask.any():
        depth_normalized[valid_mask] = np.clip(
            (depth_mm[valid_mask] - min_depth) / (max_depth - min_depth) * 255, 0, 255
        ).astype(np.uint8)

    # 应用 colormap (JET)
    depth_color = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)

    return depth_color


def load_calibration_yaml(calib_file_path):
    """
    读取眼在手外标定结果文件（camera_robot_pose.yaml）

    Returns:
        camera_matrix: 相机内参矩阵 (3x3)
        dist_coeffs: 相机畸变系数
        T_base_cam: 机器人基坐标系到相机坐标系的变换矩阵 (4x4)
        T_cam_base: 相机坐标系到机器人基坐标系的变换矩阵 (4x4)
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

    # 解析外参变换矩阵
    if 'robot_base_to_camera' in calib_data:
        T_base_cam = np.array(calib_data['robot_base_to_camera']).reshape(4, 4)
    elif 'T_R2C' in calib_data:
        T_base_cam = np.array(calib_data['T_R2C']).reshape(4, 4)
    elif 'T_cam_base' in calib_data:
        T_cam_base = np.array(calib_data['T_cam_base']).reshape(4, 4)
        T_base_cam = np.linalg.inv(T_cam_base)
    else:
        raise KeyError("标定文件中未找到外参变换矩阵")

    T_cam_base = np.linalg.inv(T_base_cam)

    print(f"相机内参矩阵:\n{camera_matrix}")
    print(f"畸变系数: {dist_coeffs.flatten()}")
    print(f"基座→相机变换矩阵:\n{T_base_cam}")

    return camera_matrix, dist_coeffs, T_base_cam, T_cam_base


def project_robot_end_to_image(end_pose, T_base_cam, camera_matrix, dist_coeffs):
    """将机器人末端4x4位姿矩阵投影到图像平面，返回像素坐标和相机坐标系下的坐标"""
    p_base = end_pose[:3, 3]
    p_base_homo = np.array([p_base[0], p_base[1], p_base[2], 1.0])

    # 转换到相机坐标系
    p_cam_homo = T_base_cam @ p_base_homo
    p_cam = p_cam_homo[:3]

    if p_cam[2] <= 0:
        print(f"警告: 末端点在相机后方 (z={p_cam[2]:.4f}), 投影可能不正确!")

    # 投影到图像平面
    p_cam_reshaped = p_cam.reshape(1, 1, 3)
    img_points, _ = cv2.projectPoints(
        p_cam_reshaped,
        np.zeros((3, 1)),
        np.zeros((3, 1)),
        camera_matrix,
        dist_coeffs
    )

    u = int(round(img_points[0][0][0]))
    v = int(round(img_points[0][0][1]))

    return (u, v), p_cam, p_base


def pixel_to_camera_3d(u, v, depth, camera_matrix, dist_coeffs):
    """
    将像素坐标和深度转换为相机坐标系下的3D点

    Args:
        u, v: 像素坐标
        depth: 深度值（米）
        camera_matrix: 相机内参矩阵
        dist_coeffs: 番变系数

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


def camera_3d_to_base_3d(point_cam, T_cam_base):
    """
    将相机坐标系下的3D点转换到机械臂基座坐标系

    Args:
        point_cam: 相机坐标系下的3D点 [x, y, z]（米）
        T_cam_base: 相机到基座的变换矩阵 (4x4)

    Returns:
        point_base: 基座坐标系下的3D点 [x, y, z]（米）
    """
    point_cam_homo = np.array([point_cam[0], point_cam[1], point_cam[2], 1.0])
    point_base_homo = T_cam_base @ point_cam_homo
    return point_base_homo[:3]


def send_cmd(client, cmd, get_pose=True):
    """发送命令到机械臂并获取姿态数据"""
    client.send(cmd.encode('utf-8'))

    if not get_pose:
        response = client.recv(1024).decode('utf-8')
        return True

    time.sleep(0.1)
    response = client.recv(4096).decode('utf-8')

    try:
        decoder = json.JSONDecoder()
        data_list = []
        index = 0

        while index < len(response):
            try:
                while index < len(response) and response[index].isspace():
                    index += 1
                if index >= len(response):
                    break
                obj, idx = decoder.raw_decode(response[index:])
                data_list.append(obj)
                index += idx
            except json.JSONDecodeError as e:
                print(f"JSON解析错误：{str(e)}")
                break

        target_data = None
        for data in reversed(data_list):
            if data.get("state") == "current_arm_state":
                target_data = data
                break

        if not target_data:
            return False, "未找到有效的机械臂状态响应"

        if target_data["arm_state"]["err"] != [0]:
            return False, f"机械臂报错: {target_data['arm_state']['err']}"

        pose_raw = target_data["arm_state"]["pose"]
        pose_converted = [
            pose_raw[0] / 1000000,
            pose_raw[1] / 1000000,
            pose_raw[2] / 1000000,
            pose_raw[3] / 1000,
            pose_raw[4] / 1000,
            pose_raw[5] / 1000
        ]

        return True, pose_converted

    except json.JSONDecodeError:
        return False, "JSON解析错误"
    except KeyError as e:
        return False, f"响应缺少关键字段: {str(e)}"
    except Exception as e:
        return False, f"处理响应时发生错误: {str(e)}"


def euler_angles_to_rotation_matrix(rx, ry, rz):
    """将欧拉角转换为旋转矩阵 (先绕x轴, 再绕y轴, 最后绕z轴)"""
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(rx), -np.sin(rx)],
                   [0, np.sin(rx), np.cos(rx)]])

    Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                   [0, 1, 0],
                   [-np.sin(ry), 0, np.cos(ry)]])

    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                   [np.sin(rz), np.cos(rz), 0],
                   [0, 0, 1]])

    R = Rz @ Ry @ Rx
    return R


def pose_6d_to_matrix(pose_6d):
    """将6D位姿转换为4x4变换矩阵"""
    x, y, z, rx, ry, rz = pose_6d
    R_mat = euler_angles_to_rotation_matrix(rx, ry, rz)
    T = np.eye(4)
    T[:3, :3] = R_mat
    T[:3, 3] = [x, y, z]
    return T


def get_current_end_pose(client):
    """获取当前机械臂末端位姿（4x4矩阵）"""
    socket_command = '{"command": "get_current_arm_state"}'
    state, pose = send_cmd(client, socket_command)

    if not state:
        print(f"获取位姿失败: {pose}")
        return None

    end_pose = pose_6d_to_matrix(pose)
    return end_pose


def mouse_callback(event, x, y, flags, param):
    """鼠标点击回调函数"""
    global clicked_point
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point = (x, y)


def main():
    """主验证程序"""

    # ==================== 配置参数 ====================
    current_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eye_hand_data")
    data_path = os.path.join("eye_hand_data",'data')
    CALIB_FILE_PATH = os.path.join(data_path, "camera_robot_pose.yaml")

    DISPLAY_WINDOW = "Calibration Verification"

    # ==================== 初始化机械臂 ====================
    print("=== 初始化机械臂 ===")
    robot_ip = get_ip()

    if not robot_ip:
        popup_message("提醒", "机械臂IP没有ping通")
        sys.exit(1)

    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        client.connect((robot_ip, 8080))
        socket_command = '{"command":"set_change_work_frame","frame_name":"Base"}'
        send_cmd(client, socket_command, get_pose=False)
        print("机械臂连接成功")
    except Exception as e:
        print(f"机械臂连接失败: {str(e)}")
        popup_message("提醒", "机械臂连接失败")
        sys.exit(1)

    # ==================== 加载标定文件 ====================
    print("=== 加载标定文件 ===")
    try:
        camera_matrix, dist_coeffs, T_base_cam, T_cam_base = load_calibration_yaml(CALIB_FILE_PATH)
        print(f"标定文件加载成功: {CALIB_FILE_PATH}")
    except Exception as e:
        print(f"标定文件加载失败: {str(e)}")
        client.close()
        sys.exit(1)

    # ==================== 初始化相机 ====================
    print("=== 初始化相机 ===")
    pipeline = Pipeline()
    config = Config()

    try:
        profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        color_profile = profile_list.get_video_stream_profile(0, 0, OBFormat.RGB, 0)
        config.enable_stream(color_profile)

        profile_list = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        depth_profile = profile_list.get_default_video_stream_profile()
        config.enable_stream(depth_profile)

        config.set_frame_aggregate_output_mode(OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)

    except Exception as e:
        print(f"流配置错误: {e}")
        client.close()
        sys.exit(1)

    try:
        pipeline.enable_frame_sync()
    except Exception as e:
        print(f"帧同步错误: {e}")

    try:
        pipeline.start(config)
        print("相机启动成功")
    except Exception as e:
        print(f"Pipeline启动错误: {e}")
        client.close()
        sys.exit(1)

    align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)

    # ==================== 创建显示窗口 ====================
    cv2.namedWindow(DISPLAY_WINDOW, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(DISPLAY_WINDOW, mouse_callback)

    print("=== 开始标定验证 ===")
    print("操作说明:")
    print("  - 红色圆点: 投影计算的末端位置")
    print("  - 点击图像: 标记真实末端位置，输出两个坐标系下的坐标")
    print("  - 按 'Q' 或 ESC: 退出程序")
    print("  - 按 'C': 清除点击标记")
    print("  - 按 'W': 增加彩色图权重 (blend_alpha += 0.1)")
    print("  - 按 'S': 减少彩色图权重 (blend_alpha -= 0.1)")
    print("  - 按 'R': 重置权重 (blend_alpha = 0.5)")

    global clicked_point, blend_alpha

    # ==================== 实时验证循环 ====================
    try:
        while True:
            frames = pipeline.wait_for_frames(1000)
            if not frames:
                continue

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            frames = align_filter.process(frames)
            if not frames:
                continue

            frames = frames.as_frame_set()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            color_image = frame_to_bgr_image(color_frame)
            if color_image is None:
                continue

            # 获取深度数据
            depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape(
                (depth_frame.get_height(), depth_frame.get_width()))
            depth_scale = depth_frame.get_depth_scale()

            # 将深度转换为彩色可视化图像
            depth_color_image = depth_to_color_image(depth_data, depth_scale)

            # 混合彩色图和深度图
            display_image = cv2.addWeighted(color_image, blend_alpha, depth_color_image, 1 - blend_alpha, 0)

            h, w = display_image.shape[:2]

            # 显示混合比例
            cv2.putText(display_image,
                        f"Blend: Color={blend_alpha:.1f}, Depth={1-blend_alpha:.1f}",
                        (w - 250, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # 获取机械臂末端位姿
            end_pose = get_current_end_pose(client)

            if end_pose is not None:
                try:
                    (u_proj, v_proj), p_cam_end, p_base_end = project_robot_end_to_image(
                        end_pose, T_base_cam, camera_matrix, dist_coeffs)

                    u_proj = np.clip(u_proj, 0, w - 1)
                    v_proj = np.clip(v_proj, 0, h - 1)

                    # 绘制投影位置（红色）
                    cv2.circle(display_image, (u_proj, v_proj), 10, (0, 0, 255), -1)

                    # 显示End在两个坐标系下的坐标
                    cv2.putText(display_image,
                                f"End Cam: ({p_cam_end[0]:.3f}, {p_cam_end[1]:.3f}, {p_cam_end[2]:.3f})",
                                (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    cv2.putText(display_image,
                                f"End Base: ({p_base_end[0]:.3f}, {p_base_end[1]:.3f}, {p_base_end[2]:.3f})",
                                (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                    # 如果用户点击了真实末端位置
                    if clicked_point is not None:
                        u_click, v_click = clicked_point

                        # 获取点击位置的深度（单位：米）
                        depth_value = depth_data[v_click, u_click]
                        depth_m = depth_value * depth_scale / 1000.0

                        if depth_m > 0:
                            # 计算点击点在相机坐标系下的坐标
                            p_cam_click = pixel_to_camera_3d(u_click, v_click, depth_m, camera_matrix, dist_coeffs)
                            print('p_cam_click :           ',p_cam_click)
                            # 计算点击点在机械臂坐标系下的坐标
                            p_base_click = camera_3d_to_base_3d(p_cam_click, T_cam_base)
                            print('p_base_click :           ',p_base_click)
                            # 绘制点击位置（绿色）
                            cv2.circle(display_image, (u_click, v_click), 10, (0, 255, 0), 2)

                            # 显示点击点在两个坐标系下的坐标 + 像素坐标
                            cv2.putText(display_image,
                                        f"Click Pixel: ({u_click}, {v_click})",
                                        (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            cv2.putText(display_image,
                                        f"Click Cam: ({p_cam_click[0]:.3f}, {p_cam_click[1]:.3f}, {p_cam_click[2]:.3f})",
                                        (20, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            cv2.putText(display_image,
                                        f"Click Base: ({p_base_click[0]:.3f}, {p_base_click[1]:.3f}, {p_base_click[2]:.3f})",
                                        (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                            # 输出到控制台
                            print(f"\n=== 坐标对比 ===")
                            print(f"End (机械臂返回):")
                            print(f"  相机坐标系: ({p_cam_end[0]:.4f}, {p_cam_end[1]:.4f}, {p_cam_end[2]:.4f}) m")
                            print(f"  机械臂坐标系: ({p_base_end[0]:.4f}, {p_base_end[1]:.4f}, {p_base_end[2]:.4f}) m")
                            print(f"Click (用户点击):")
                            print(f"  相机坐标系: ({p_cam_click[0]:.4f}, {p_cam_click[1]:.4f}, {p_cam_click[2]:.4f}) m")
                            print(f"  机械臂坐标系: ({p_base_click[0]:.4f}, {p_base_click[1]:.4f}, {p_base_click[2]:.4f}) m")
                        else:
                            cv2.circle(display_image, (u_click, v_click), 10, (0, 165, 255), 2)
                            cv2.putText(display_image, "No Depth", (u_click + 15, v_click + 15),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
                            print(f"点击位置 ({u_click}, {v_click}) 无有效深度数据")

                except Exception as e:
                    print(f"投影计算失败: {str(e)}")

            cv2.putText(display_image,
                        f"Calib: {os.path.basename(CALIB_FILE_PATH)}",
                        (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow(DISPLAY_WINDOW, display_image)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q') or key == ESC_KEY:
                print("用户退出程序")
                break
            elif key == ord('c') or key == ord('C'):
                clicked_point = None
                print("点击标记已清除")
            elif key == ord('w') or key == ord('W'):
                blend_alpha = min(1.0, blend_alpha + 0.1)
                print(f"彩色图权重: {blend_alpha:.1f}, 深度图权重: {1-blend_alpha:.1f}")
            elif key == ord('s') or key == ord('S'):
                blend_alpha = max(0.0, blend_alpha - 0.1)
                print(f"彩色图权重: {blend_alpha:.1f}, 深度图权重: {1-blend_alpha:.1f}")
            elif key == ord('r') or key == ord('R'):
                blend_alpha = 0.5
                print(f"权重重置: 彩色={blend_alpha:.1f}, 深度={1-blend_alpha:.1f}")

    except KeyboardInterrupt:
        print("程序被用户中断")
    except Exception as e:
        print(f"程序运行出错: {str(e)}")

    finally:
        print("=== 释放资源 ===")
        cv2.destroyAllWindows()
        pipeline.stop()
        client.close()
        print("所有资源已释放，程序退出")


if __name__ == '__main__':
    main()