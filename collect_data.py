# coding=utf-8
import json
import logging,os
import socket
import time
import sys
import numpy as np
import cv2
# import pyrealsense2 as rs
from pyorbbecsdk import Pipeline, Config, AlignFilter, OBStreamType, OBSensorType, OBFormat, OBFrameAggregateOutputMode, VideoFrame
from typing import Union, Any, Optional


from libs.log_setting import CommonLog
from libs.auxiliary import create_folder_with_date, get_ip, popup_message

cam0_origin_path = create_folder_with_date() # 提前建立好的存储照片文件的目录


logger_ = logging.getLogger(__name__)
logger_ = CommonLog(logger_)

# ==================== 常量 ====================
ESC_KEY = 27
MIN_DEPTH = 20    # 最小有效深度距离 mm
MAX_DEPTH = 10000 # 最大有效深度距离 mm



def frame_to_bgr_image(frame: VideoFrame) -> Union[Optional[np.array], Any]:
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
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif color_format == OBFormat.YUYV:
        image = np.resize(data, (height, width, 2))
        image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR_YUYV)
    elif color_format == OBFormat.MJPG:
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    elif color_format == OBFormat.I420:
        image = i420_to_bgr(data, width, height)
        return image
    elif color_format == OBFormat.NV12:
        image = nv12_to_bgr(data, width, height)
        return image
    elif color_format == OBFormat.NV21:
        image = nv21_to_bgr(data, width, height)
        return image
    elif color_format == OBFormat.UYVY:
        image = np.resize(data, (height, width, 2))
        image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR_UYVY)
    else:
        print("Unsupported color format: {}".format(color_format))
        return None
    return image


def callback(frame):

    scaling_factor = 0.25
    global count

    cv_img = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
    #cv_img = frame
    cv2.imshow("Capture_Video", cv_img)  # 窗口显示，显示名为 Capture_Video

    k = cv2.waitKey(30) & 0xFF  # 每帧数据延时 1ms，延时不能为 0，否则读取的结果会是静态帧

    if k == ord('s'):  # 若检测到按键 ‘s’，打印字符串

        socket_command = '{"command": "get_current_arm_state"}'
        state,pose = send_cmd(client,socket_command)
        logger_.info(f'获取状态：{"成功" if state else "失败"}，{f"当前位姿为{pose}" if state else None}')
        if state:

            filename = os.path.join(cam0_origin_path,"poses.txt")

            with open(filename, 'a+') as f:
                # 将列表中的元素用空格连接成一行
                pose_ = [str(i) for i in pose]
                new_line = f'{",".join(pose_)}\n'
                # 将新行附加到文件的末尾
                f.write(new_line)

            image_path = os.path.join(cam0_origin_path,f"{str(count)}.jpg")
            cv2.imwrite(image_path , frame)
            logger_.info(f"===采集第{count}次数据！")

        count += 1

    else:
        pass


def send_cmd(client, cmd, get_pose=True):
    """
    发送命令到机械臂并可选择性地获取姿态(pose)数据

    参数:
    client: socket客户端连接
    cmd: 要发送的命令字符串或JSON字符串
    get_pose: 是否需要获取pose数据

    返回:
    如果get_pose为True，返回tuple (状态, pose或错误信息)
    如果get_pose为False，返回布尔值表示命令是否成功发送
    """
    client.send(cmd.encode('utf-8'))

    if not get_pose:
        response = client.recv(1024).decode('utf-8')
        logger_.info(f"response:{response}")
        return True

    time.sleep(0.1)
    response = client.recv(4096).decode('utf-8')  # 增大接收缓冲区
    logger_.info(f'response:{response}')

    try:
        decoder = json.JSONDecoder()
        data_list = []
        index = 0
        # 分割并解析所有可能的JSON对象
        while index < len(response):
            try:
                # 跳过空白字符
                while index < len(response) and response[index].isspace():
                    index += 1
                if index >= len(response):
                    break
                obj, idx = decoder.raw_decode(response[index:])
                data_list.append(obj)
                index += idx
            except json.JSONDecodeError as e:
                logger_.error(f"JSON解析错误：{str(e)}")
                break

        # 寻找最后一个包含目标状态的响应
        target_data = None
        for data in reversed(data_list):
            if data.get("state") == "current_arm_state":
                target_data = data
                break

        if not target_data:
            return False, "未找到有效的机械臂状态响应"

        # 检查错误码
        if target_data["arm_state"]["err"] != [0]:
            return False, f"机械臂报错: {target_data['arm_state']['err']}"

        # 转换单位
        pose_raw = target_data["arm_state"]["pose"]
        pose_converted = [
            pose_raw[0] / 1000000,  # x: 0.001mm → m
            pose_raw[1] / 1000000,  # y: 0.001mm → m
            pose_raw[2] / 1000000,  # z: 0.001mm → m
            pose_raw[3] / 1000,    # rx: 0.001rad → rad
            pose_raw[4] / 1000,    # ry: 0.001rad → rad
            pose_raw[5] / 1000     # rz: 0.001rad → rad
        ]

        return True, pose_converted

    except json.JSONDecodeError:
        return False, "JSON解析错误"
    except KeyError as e:
        return False, f"响应缺少关键字段: {str(e)}"
    except Exception as e:
        return False, f"处理响应时发生错误: {str(e)}"
#
def displayD435():

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    try:
        pipeline.start(config)
    except Exception as e:
        logger_.error_(f"相机连接异常：{e}")
        popup_message("提醒", "相机连接异常")

        sys.exit(1)

    global count
    count = 1

    logger_.info(f"开始手眼标定程序，当前程序版号V1.0.0")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            callback(color_image)

    finally:

        pipeline.stop()
        cv2.destroyAllWindows()

def display_gemini_camera():
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

    # Set window size
    # window_width = 1280
    # window_height = 720
    # cv2.namedWindow("QuickStart Viewer", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("QuickStart Viewer", window_width, window_height)

    global count
    count = 1

    try:
        while True:

            # Retrieve a frameset with a 100ms timeout
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
            # try:
            #     # Convert raw buffer to 2D numpy array
            #     depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape(
            #         (depth_frame.get_height(), depth_frame.get_width()))
            # except ValueError:
            #     print("Failed to reshape depth data")
            #     continue
                

            # depth_scale = depth_frame.get_depth_scale()
            # depth_width = depth_frame.get_width()
            # depth_height = depth_frame.get_height()

            # # 检查分辨率并缩放内参
            # actual_width = color_frame.get_width()
            # actual_height = color_frame.get_height()

            # Apply depth scale to get actual distance in mm and filter by range
            # depth_data = depth_data.astype(np.float32) * depth_frame.get_depth_scale()
            # depth_data = np.where((depth_data > MIN_DEPTH) & (depth_data < MAX_DEPTH), depth_data, 0)
            
            # # Normalize and colormap for visualization
            # depth_data = depth_data.astype(np.uint16)
            # depth_image = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX)
            # depth_image = cv2.applyColorMap(depth_image.astype(np.uint8), cv2.COLORMAP_JET)
            

            # color_image = np.asanyarray(color_frame.get_data())
            # color_image_resized = cv2.resize(color_image, (window_width, window_height))
            # callback(color_image_resized)
            callback(color_image)


            # if cv2.waitKey(1) in [ord('q'), ESC_KEY]:
            #     break
    finally:
        cv2.destroyAllWindows()
        pipeline.stop()
        print("Pipeline stopped and all windows closed.")


import ctypes
import os
import sys
import time
from typing import Optional, Tuple

PERCIPIO_SDK_DIR = '/home/tengenx/wangyuhan/code/realman_calibration/hand_eye_calibration/pcammls'
if PERCIPIO_SDK_DIR not in sys.path:
    sys.path.insert(0, PERCIPIO_SDK_DIR)

libtycam_path = os.path.join(PERCIPIO_SDK_DIR, 'libtycam.so')
if os.path.exists(libtycam_path):
    ctypes.CDLL(libtycam_path)

import numpy as np
import pcammls
from pcammls import *
from pcammls import (
    PERCIPIO_STREAM_COLOR,
    PERCIPIO_STREAM_DEPTH,
    TY_EVENT_DEVICE_OFFLINE,
    PercipioSDK,
    image_data,
)


class PythonPercipioDeviceEvent(pcammls.DeviceEvent):
    Offline = False

    def __init__(self):
        pcammls.DeviceEvent.__init__(self)

    def run(self, handle, eventID):
        if eventID==TY_EVENT_DEVICE_OFFLINE:
          print('=== Event Callback: Device Offline!')
          self.Offline = True
        return 0

    def IsOffline(self):
        return self.Offline




def display_percipio():
    global count
    count = 1
    cl = PercipioSDK()

    # dev_list = cl.ListDevice()
    # for idx in range(len(dev_list)):
    #   dev = dev_list[idx]
    #   print ('{} -- {} \t {}'.format(idx,dev.id,dev.iface.id))
    # if  len(dev_list)==0:
    #   print ('no device')
    #   return
    # if len(dev_list) == 1:
    #     selected_idx = 0 
    # else:
    #     selected_idx  = int(input('select a device:'))
    # if selected_idx < 0 or selected_idx >= len(dev_list):
    #     return

    # sn = dev_list[selected_idx].id

    handle = cl.Open('207000154491')
    if not cl.isValidHandle(handle):
      err = cl.TYGetLastErrorCodedescription()
      print('no device found : ', end='')
      print(err)
      return
      
    event = PythonPercipioDeviceEvent()
    cl.DeviceRegiststerCallBackEvent(event)

    color_fmt_list = cl.DeviceStreamFormatDump(handle, PERCIPIO_STREAM_COLOR)
    if len(color_fmt_list) == 0:
      print ('device has no color stream.')
      return

    print ('color image format list:')
    for idx in range(len(color_fmt_list)):
        fmt = color_fmt_list[idx]
        print ('\t{} -size[{}x{}]\t-\t desc:{}'.format(idx, cl.Width(fmt), cl.Height(fmt), fmt.getDesc()))
    cl.DeviceStreamFormatConfig(handle, PERCIPIO_STREAM_COLOR, color_fmt_list[0])

    depth_fmt_list = cl.DeviceStreamFormatDump(handle, PERCIPIO_STREAM_DEPTH)
    if len(depth_fmt_list) == 0:
      print ('device has no depth stream.')
      return

    print ('depth image format list:')
    for idx in range(len(depth_fmt_list)):
        fmt = depth_fmt_list[idx]
        print ('\t{} -size[{}x{}]\t-\t desc:{}'.format(idx, cl.Width(fmt), cl.Height(fmt), fmt.getDesc()))
    cl.DeviceStreamFormatConfig(handle, PERCIPIO_STREAM_DEPTH, depth_fmt_list[0])

    err = cl.DeviceLoadDefaultParameters(handle)
    if err:
      print('Load default parameters fail: ', end='')
      print(cl.TYGetLastErrorCodedescription())
    else:
       print('Load default parameters successful')

    scale_unit = cl.DeviceReadCalibDepthScaleUnit(handle)
    print ('depth image scale unit :{}'.format(scale_unit))

    depth_calib = cl.DeviceReadCalibData(handle, PERCIPIO_STREAM_DEPTH)
    color_calib = cl.DeviceReadCalibData(handle, PERCIPIO_STREAM_COLOR)

    err = cl.DeviceStreamEnable(handle, PERCIPIO_STREAM_COLOR | PERCIPIO_STREAM_DEPTH)
    if err:
       print('device stream enable err:{}'.format(err))
       return
    
    print ('{} -- {} \t'.format(0,"Map depth to color coordinate(suggest)"))
    print ('{} -- {} \t'.format(1,"Map color to depth coordinate"))
    registration_mode = int(input('select registration mode(0 or 1):'))
    # selected_idx = 0
    # if selected_idx < 0 or selected_idx >= 2:
    #   registration_mode = 0

    cl.DeviceStreamOn(handle)
    img_registration_depth  = image_data()
    img_registration_render = image_data()
    img_parsed_color        = image_data()
    img_undistortion_color  = image_data()
    img_registration_color  = image_data()

    try:
        while True:
            if event.IsOffline():
                break
            image_list = cl.DeviceStreamRead(handle, 2000)
            if len(image_list) == 2:
                for i in range(len(image_list)):
                    frame = image_list[i]
                    if frame.streamID == PERCIPIO_STREAM_DEPTH:
                        img_depth = frame
                    if frame.streamID == PERCIPIO_STREAM_COLOR:
                        img_color = frame
                
                if 0 == registration_mode:
                    cl.DeviceStreamMapDepthImageToColorCoordinate(depth_calib, img_depth, scale_unit, color_calib, img_color.width, img_color.height, img_registration_depth)
                    
                    cl.DeviceStreamDepthRender(img_registration_depth, img_registration_render)
                    mat_depth_render = img_registration_render.as_nparray()
                    # cv2.imshow('registration', mat_depth_render)

                    cl.DeviceStreamImageDecode(img_color, img_parsed_color)
                    cl.DeviceStreamDoUndistortion(color_calib, img_parsed_color, img_undistortion_color)
                    mat_undistortion_color = img_undistortion_color.as_nparray()
                    callback(mat_undistortion_color)
                    # cv2.imshow('undistortion rgb', mat_undistortion_color)
                else:
                    cl.DeviceStreamImageDecode(img_color, img_parsed_color)
                    cl.DeviceStreamDoUndistortion(color_calib, img_parsed_color, img_undistortion_color)

                    cl.DeviceStreamMapRGBImageToDepthCoordinate(depth_calib, img_depth, scale_unit, color_calib, img_undistortion_color, img_registration_color)

                    cl.DeviceStreamDepthRender(img_depth, img_registration_render)
                    mat_depth_render = img_registration_render.as_nparray()
                    # cv2.imshow('depth', mat_depth_render)

                    mat_registration_color = img_registration_color.as_nparray()
                    # cv2.imshow('registration rgb', mat_registration_color)

            k = cv2.waitKey(10)
            if k==ord('q'): 
                break

        cl.DeviceStreamOff(handle)    
        cl.Close(handle)

    except KeyboardInterrupt:
        print('\nKeyboardInterrupt')




if __name__ == '__main__':

    robot_ip = get_ip()



    logger_.info(f'robot_ip:{robot_ip}')

    if robot_ip:

        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.connect((robot_ip, 8080))
        socket_command = '{"command":"set_change_work_frame","frame_name":"Base"}'
        send_cmd(client,socket_command,get_pose = False)

    else:

        popup_message("提醒", "机械臂ip没有ping通")
        sys.exit(1)

    display_percipio()
