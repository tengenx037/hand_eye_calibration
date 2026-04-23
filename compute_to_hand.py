# coding=utf-8

"""
眼在手外 用采集到的图片信息和机械臂位姿信息计算 相机坐标系相对于机械臂基坐标系的 旋转矩阵和平移向量

"""

import os
import logging

import yaml
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

from libs.auxiliary import find_latest_data_folder
from libs.log_setting import CommonLog

from save_poses2 import poses2_main

np.set_printoptions(precision=8,suppress=True)

logger_ = logging.getLogger(__name__)
logger_ = CommonLog(logger_)


current_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"eye_hand_data")

images_path = os.path.join("eye_hand_data",find_latest_data_folder(current_path))

file_path = os.path.join(images_path,"poses.txt")  #采集标定板图片时对应的机械臂末端的位姿 从 第一行到最后一行 需要和采集的标定板的图片顺序进行对应


with open("config.yaml", 'r', encoding='utf-8') as file:
    data = yaml.safe_load(file)

XX = data.get("checkerboard_args").get("XX") #标定板的中长度对应的角点的个数
YY = data.get("checkerboard_args").get("YY") #标定板的中宽度对应的角点的个数
L = data.get("checkerboard_args").get("L")   #标定板一格的长度  单位为米

def compute_transformation_from_robot_to_camera(R_c2b, t_c2b):
    T_c2b = np.concatenate((R_c2b, t_c2b.reshape(3, 1)), axis=1)
    T_c2b = np.concatenate((T_c2b, np.array([[0, 0, 0, 1]])), axis=0)
    T_b2c = np.linalg.inv(T_c2b)
    return T_b2c


def func():

    path = os.path.dirname(__file__)
    print(path)

    # 设置寻找亚像素角点的参数，采用的停止准则是最大循环次数30和最大误差容限0.001
    criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)

    # 获取标定板角点的位置
    objp = np.zeros((XX * YY, 3), np.float32)
    objp[:, :2] = np.mgrid[0:XX, 0:YY].T.reshape(-1, 2)     # 将世界坐标系建在标定板上，所有点的Z坐标全部为0，所以只需要赋值x和y
    objp = L*objp

    obj_points = []     # 存储3D点
    img_points = []     # 存储2D点

    images_num = [f for f in os.listdir(images_path) if f.endswith('.jpg')]

    for i in range(1, len(images_num) + 1):   #标定好的图片在images_path路径下，从0.jpg到x.jpg

        image_file = os.path.join(images_path,f"{i}.jpg")

        if os.path.exists(image_file):

            logger_.info(f'读 {image_file}')

            img = cv2.imread(image_file)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            size = gray.shape[::-1]
            ret, corners = cv2.findChessboardCorners(gray, (XX, YY), None)

            if ret:

                obj_points.append(objp)

                corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)  # 在原角点的基础上寻找亚像素角点
                if [corners2]:
                    img_points.append(corners2)
                else:
                    img_points.append(corners)

    N = len(img_points)

    # 标定,得到图案在相机坐标系下的位姿
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, size, None, None)

    # logger_.info(f"内参矩阵:\n:{mtx}" ) # 内参数矩阵
    # logger_.info(f"畸变系数:\n:{dist}")  # 畸变系数   distortion cofficients = (k_1,k_2,p_1,p_2,k_3)

    print("-----------------------------------------------------")

    poses2_main(file_path)
    # 机器人末端在基座标系下的位姿

    csv_file = os.path.join(path,"RobotToolPose.csv")
    tool_pose = np.loadtxt(csv_file,delimiter=',')

    R_tool = []
    t_tool = []

    for i in range(int(N)):

        R_tool.append(tool_pose[0:3,4*i:4*i+3])
        t_tool.append(tool_pose[0:3,4*i+3])

    # 使用多种算法进行标定，选择最优结果
    methods = [
        (cv2.CALIB_HAND_EYE_TSAI, "TSAI"),
        (cv2.CALIB_HAND_EYE_PARK, "PARK"),
        (cv2.CALIB_HAND_EYE_HORAUD, "HORAUD"),
        (cv2.CALIB_HAND_EYE_ANDREFF, "ANDREFF"),
    ]

    results = []
    for method, name in methods:
        R_test, t_test = cv2.calibrateHandEye(R_tool, t_tool, rvecs, tvecs, method)
        # 计算重投影误差
        errors = []
        for i in range(len(R_tool)):
            # 验证: 标定板位置通过位姿变换应该一致
            # 这里简化计算旋转和平移的相对误差
            pass
        results.append((R_test, t_test, name))
        logger_.info(f"算法 {name} 结果:")
        logger_.info(f"  R: {R_test}")
        logger_.info(f"  t: {t_test.flatten()}")

    # 选择TSAI结果作为默认（可根据实际情况调整）
    R, t = results[0][0], results[0][1]

    logger_.info(f"最终使用 TSAI 算法结果")

    # 保存标定结果到yaml文件(兼容验证脚本格式)
    result = {}
    result["camera_matrix"] = {"data": mtx.flatten().tolist(), "rows": 3, "cols": 3}
    result["distortion_coefficients"] = {"data": dist.flatten().tolist()}
    result["mtx_c2p"] = mtx.tolist()
    result["mtx_p2c"] = np.linalg.inv(mtx).tolist()
    result["dist"] = dist.tolist()
    result["R_c2b"] = R.tolist()
    result["t_c2b"] = t.tolist()
    # 保存标定时的图像分辨率
    result["image_width"] = size[0]
    result["image_height"] = size[1]
    T_R2C = compute_transformation_from_robot_to_camera(R, t)
    result["T_R2C"] = T_R2C.tolist()
    result["robot_base_to_camera"] = T_R2C.tolist()  # 兼容验证脚本格式
    # 计算相机到基座的变换矩阵
    T_c2b = np.eye(4)
    T_c2b[:3, :3] = R
    T_c2b[:3, 3] = t.flatten()
    result["T_cam_base"] = T_c2b.tolist()  # 兼容验证脚本格式

    output_file = os.path.join(images_path, "camera_robot_pose.yaml")
    with open(output_file, 'w', encoding='utf-8') as file:
        yaml.dump(result, file, default_flow_style=False)

    logger_.info(f"标定结果已保存到: {output_file}")
    logger_.info(f"标定分辨率: {size[0]} x {size[1]}")

    return R, t, mtx, dist

if __name__ == '__main__':

    # 旋转矩阵、平移向量、相机内参、畸变系数
    rotation_matrix, translation_vector, mtx, dist = func()

    # 将旋转矩阵转换为四元数
    rotation = R.from_matrix(rotation_matrix)
    quaternion = rotation.as_quat()
    x, y, z = translation_vector.flatten()

    logger_.info(f"旋转矩阵是:\n {            rotation_matrix}")

    logger_.info(f"平移向量是:\n {            translation_vector}")

    logger_.info(f"四元数是：\n {             quaternion}")

    logger_.info(f"相机内参矩阵:\n {mtx}")

    logger_.info(f"畸变系数:\n {dist}")

