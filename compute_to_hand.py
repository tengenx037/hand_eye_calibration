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

images_path = os.path.join("eye_hand_data",'data_0518_06')
# CALIB_FILE_PATH = os.path.join(data_path, "camera_robot_pose.yaml")

# DISPLAY_WINDOW = "Calibration Verification"

# images_path = os.path.join("eye_hand_data",find_latest_data_folder(current_path))

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


def rvec_to_rmat(rvec):
    """旋转向量 → 旋转矩阵"""
    rmat, _ = cv2.Rodrigues(rvec)
    return rmat


def rmat_to_rvec(rmat):
    """旋转矩阵 → 旋转向量"""
    rvec, _ = cv2.Rodrigues(rmat)
    return rvec


def compute_calibration_errors(
    obj_points, img_points, rvecs, tvecs, mtx, dist,
    R_tool, t_tool, R_handeye, t_handeye, logger,
    success_indices=None,
):
    """计算并输出手眼标定的各项误差指标。

    Args:
        success_indices: 成功检测的图片序号列表（1-based），用于输出文件名
    """
    N = len(obj_points)
    if success_indices is None:
        success_indices = list(range(1, N + 1))

    def _img_name(i):
        """返回第 i 个数据对应的图片文件名（i 为 0-based 数组索引）"""
        return f"{success_indices[i]}.jpg"

    # ========== 1. 重投影误差 (每张图 + 整体) ==========
    logger.info("=" * 60)
    logger.info("  误差指标 1: 重投影误差 (Reprojection Error)")
    logger.info("=" * 60)

    reproj_errors_px = []
    for i in range(N):
        img_pts_proj, _ = cv2.projectPoints(
            obj_points[i], rvecs[i], tvecs[i], mtx, dist,
        )
        error = cv2.norm(img_points[i], img_pts_proj, cv2.NORM_L2) / len(img_points[i])
        reproj_errors_px.append(error)

    reproj_errors_px = np.array(reproj_errors_px)
    logger.info(f"  图像数量:              {N}")
    logger.info(f"  平均重投影误差:         {reproj_errors_px.mean():.4f} px")
    logger.info(f"  最大重投影误差:         {reproj_errors_px.max():.4f} px  ({_img_name(np.argmax(reproj_errors_px))})")
    logger.info(f"  最小重投影误差:         {reproj_errors_px.min():.4f} px  ({_img_name(np.argmin(reproj_errors_px))})")
    logger.info(f"  重投影误差标准差:       {reproj_errors_px.std():.4f} px")

    # 逐张显示
    logger.info("")
    logger.info(f"  {'Image':>8s}  {'Error(px)':>10s}")
    logger.info(f"  {'-'*8}  {'-'*10}")
    for i in range(N):
        flag = " ***" if reproj_errors_px[i] == reproj_errors_px.max() else ""
        logger.info(f"  {_img_name(i):>8s}  {reproj_errors_px[i]:>10.4f}{flag}")

    # ========== 2. 手眼标定方程残差 AX = XB ==========
    # R_tool/t_tool 是 T_ee_base (基座在末端坐标系)，来自 RobotToolPose.csv
    # 方程: inv(T_ee_base[i]) * T_ee_base[0] * X = X * T_cam_board[i] * inv(T_cam_board[0])
    # 即: A_i * X = X * B_i,  X = T_base_cam
    logger.info("")
    logger.info("=" * 60)
    logger.info("  误差指标 2: 手眼方程残差 (AX - XB)")
    logger.info("=" * 60)

    t_handeye = t_handeye.flatten()
    rot_errors_deg = []
    trans_errors_mm = []

    # T_ee_base[0] (参考帧)
    T_ee_ref = np.eye(4)
    T_ee_ref[:3, :3] = R_tool[0]
    T_ee_ref[:3, 3] = t_tool[0]

    # T_cam_board[0] (参考帧)
    T_cb_ref = np.eye(4)
    T_cb_ref[:3, :3] = rvec_to_rmat(rvecs[0])
    T_cb_ref[:3, 3] = tvecs[0].flatten()

    for i in range(N):
        if i == 0:
            continue

        # T_ee_base[i]
        T_ee_i = np.eye(4)
        T_ee_i[:3, :3] = R_tool[i]
        T_ee_i[:3, 3] = t_tool[i]

        # A_i = inv(T_ee_base[i]) * T_ee_base[0]
        A = np.linalg.inv(T_ee_i) @ T_ee_ref

        # T_cam_board[i]
        T_cb_i = np.eye(4)
        T_cb_i[:3, :3] = rvec_to_rmat(rvecs[i])
        T_cb_i[:3, 3] = tvecs[i].flatten()

        # B_i = T_cam_board[i] * inv(T_cam_board[0])
        B = T_cb_i @ np.linalg.inv(T_cb_ref)

        # 手眼矩阵 X: T_base_cam
        X = np.eye(4)
        X[:3, :3] = R_handeye
        X[:3, 3] = t_handeye

        # 残差: A @ X - X @ B
        AX = A @ X
        XB = X @ B

        # 旋转误差
        R_diff = AX[:3, :3] @ XB[:3, :3].T
        angle = np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1, 1))
        rot_error_deg = np.degrees(angle)

        # 平移误差: m → mm
        trans_error = np.linalg.norm(AX[:3, 3] - XB[:3, 3]) * 1000

        rot_errors_deg.append(rot_error_deg)
        trans_errors_mm.append(trans_error)

    rot_errors_deg = np.array(rot_errors_deg)
    trans_errors_mm = np.array(trans_errors_mm)

    logger.info(f"  有效位姿对数:           {N - 1}")
    logger.info(f"  旋转残差 平均:          {rot_errors_deg.mean():.4f} deg")
    logger.info(f"  旋转残差 最大:          {rot_errors_deg.max():.4f} deg")
    logger.info(f"  旋转残差 标准差:         {rot_errors_deg.std():.4f} deg")
    logger.info(f"  平移残差 平均:          {trans_errors_mm.mean():.2f} mm")
    logger.info(f"  平移残差 最大:          {trans_errors_mm.max():.2f} mm")
    logger.info(f"  平移残差 标准差:         {trans_errors_mm.std():.2f} mm")

    # 逐张输出 (i=0 为参考帧，跳过)
    logger.info("")
    logger.info(f"  {'Pair':>10s}  {'Rot(deg)':>10s}  {'Trans(mm)':>10s}")
    logger.info(f"  {'-'*10}  {'-'*10}  {'-'*10}")
    for j in range(N - 1):
        img_ref = _img_name(0)
        img_i = _img_name(j + 1)
        r_flag = " ***" if rot_errors_deg[j] == rot_errors_deg.max() else ""
        t_flag = " ***" if trans_errors_mm[j] == trans_errors_mm.max() else ""
        logger.info(
            f"  {img_ref:>5s}→{img_i:<5s} {rot_errors_deg[j]:>10.4f}{r_flag}  {trans_errors_mm[j]:>10.2f}{t_flag}"
        )

    # ========== 3. 标定板位置一致性 (眼在手外) ==========
    logger.info("")
    logger.info("=" * 60)
    logger.info("  误差指标 3: 标定板在机械臂末端坐标系下的一致性")
    logger.info("  (Eye-to-Hand: 板固连在末端，T_ee_board 应恒定)")
    logger.info("  T_ee_board = T_ee_base * T_base_cam * T_cam_board")
    logger.info("=" * 60)

    X = np.eye(4)
    X[:3, :3] = R_handeye
    X[:3, 3] = t_handeye

    T_ee_board_list = []

    for i in range(N):
        # T_ee_base[i]: 基座在末端坐标系
        T_ee_base = np.eye(4)
        T_ee_base[:3, :3] = R_tool[i]
        T_ee_base[:3, 3] = t_tool[i]

        # T_cam_board[i]: 板在相机坐标系
        T_cam_board = np.eye(4)
        T_cam_board[:3, :3] = rvec_to_rmat(rvecs[i])
        T_cam_board[:3, 3] = tvecs[i].flatten()

        # T_ee_board = T_ee_base * X * T_cam_board
        T_ee_board = T_ee_base @ X @ T_cam_board
        T_ee_board_list.append(T_ee_board)

    # 计算均值 T_ee_board
    # 旋转用四元数平均
    quats = []
    translations = []
    for T in T_ee_board_list:
        r = R.from_matrix(T[:3, :3])
        quats.append(r.as_quat())
        translations.append(T[:3, 3])

    translations = np.array(translations)
    quats = np.array(quats)

    mean_translation = translations.mean(axis=0)
    # 四元数平均: SVD 方法
    Q = quats.T @ quats
    _, _, Vt = np.linalg.svd(Q)
    mean_quat = Vt[0]
    if mean_quat[3] < 0:  # w 分量应 > 0
        mean_quat = -mean_quat
    mean_rot = R.from_quat(mean_quat).as_matrix()

    # 每帧相对于均值的偏差
    pos_deviations_mm = []
    rot_deviations_deg = []
    for i in range(N):
        dp = np.linalg.norm(translations[i] - mean_translation) * 1000  # m → mm
        pos_deviations_mm.append(dp)

        dR = T_ee_board_list[i][:3, :3] @ mean_rot.T
        angle = np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))
        rot_deviations_deg.append(np.degrees(angle))

    pos_deviations_mm = np.array(pos_deviations_mm)
    rot_deviations_deg = np.array(rot_deviations_deg)

    logger.info(f"  位姿数量:               {N}")
    logger.info(f"  位置偏差 平均:           {pos_deviations_mm.mean():.2f} mm")
    logger.info(f"  位置偏差 最大:           {pos_deviations_mm.max():.2f} mm  ({_img_name(np.argmax(pos_deviations_mm))})")
    logger.info(f"  位置偏差 标准差:          {pos_deviations_mm.std():.2f} mm")
    logger.info(f"  旋转偏差 平均:           {rot_deviations_deg.mean():.4f} deg")
    logger.info(f"  旋转偏差 最大:           {rot_deviations_deg.max():.4f} deg  ({_img_name(np.argmax(rot_deviations_deg))})")
    logger.info(f"  旋转偏差 标准差:          {rot_deviations_deg.std():.4f} deg")

    # 逐张输出
    logger.info("")
    logger.info(f"  {'Image':>8s}  {'Pos(mm)':>8s}  {'Rot(deg)':>8s}")
    logger.info(f"  {'-'*8}  {'-'*8}  {'-'*8}")
    for i in range(N):
        p_flag = " ***" if pos_deviations_mm[i] == pos_deviations_mm.max() else ""
        r_flag = " ***" if rot_deviations_deg[i] == rot_deviations_deg.max() else ""
        logger.info(
            f"  {_img_name(i):>8s}  {pos_deviations_mm[i]:>8.2f}{p_flag}"
            f"  {rot_deviations_deg[i]:>8.4f}{r_flag}"
        )

    # ========== 汇总 ==========
    logger.info("")
    logger.info("=" * 60)
    logger.info("  误差汇总 (Summary)")
    logger.info("=" * 60)
    logger.info(f"  重投影误差 (RMSE):      {reproj_errors_px.mean():.3f} px")
    logger.info(f"  AX=XB 旋转残差:         {rot_errors_deg.mean():.3f} deg")
    logger.info(f"  AX=XB 平移残差:         {trans_errors_mm.mean():.2f} mm")
    logger.info(f"  板位姿一致性 位置偏差:   {pos_deviations_mm.mean():.2f} mm")
    logger.info(f"  板位姿一致性 旋转偏差:   {rot_deviations_deg.mean():.3f} deg")

    # 判断标定质量（T_ee_board 一致性是最可靠的实地指标）
    quality_msg = []
    if reproj_errors_px.mean() < 0.3:
        quality_msg.append("重投影: 优")
    elif reproj_errors_px.mean() < 1.0:
        quality_msg.append("重投影: 良")
    else:
        quality_msg.append("重投影: 差 *** 角点异常")

    if pos_deviations_mm.mean() < 2.0:
        quality_msg.append("板位姿一致性: 优")
    elif pos_deviations_mm.mean() < 5.0:
        quality_msg.append("板位姿一致性: 良")
    else:
        quality_msg.append("板位姿一致性: 差 *** 检查图-位姿配对")

    if rot_deviations_deg.mean() < 0.3:
        quality_msg.append("板旋转一致性: 优")
    elif rot_deviations_deg.mean() < 1.0:
        quality_msg.append("板旋转一致性: 良")
    else:
        quality_msg.append("板旋转一致性: 差 *** 检查图-位姿配对")

    logger.info(f"  质量评估: {' | '.join(quality_msg)}")
    logger.info("=" * 60)

    return {
        "reproj_mean_px": float(reproj_errors_px.mean()),
        "reproj_max_px": float(reproj_errors_px.max()),
        "reproj_std_px": float(reproj_errors_px.std()),
        "handeye_rot_mean_deg": float(rot_errors_deg.mean()),
        "handeye_rot_max_deg": float(rot_errors_deg.max()),
        "handeye_trans_mean_mm": float(trans_errors_mm.mean()),
        "handeye_trans_max_mm": float(trans_errors_mm.max()),
        "board_pos_mean_mm": float(pos_deviations_mm.mean()),
        "board_pos_max_mm": float(pos_deviations_mm.max()),
        "board_rot_mean_deg": float(rot_deviations_deg.mean()),
        "board_rot_max_deg": float(rot_deviations_deg.max()),
        "per_image_reproj_px": reproj_errors_px.tolist(),
        "per_image_board_pos_mm": pos_deviations_mm.tolist(),
        "per_image_board_rot_deg": rot_deviations_deg.tolist(),
    }


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
    success_indices = []  # 记录成功检测的图片序号（1-based，对应 poses.txt 行号）

    # 按数字序号排序文件名，支持非连续编号 (如 1.jpg, 3.jpg, 41.jpg)
    image_files = sorted(
        [f for f in os.listdir(images_path) if f.endswith('.jpg')],
        key=lambda x: int(os.path.splitext(x)[0])
    )

    for fname in image_files:
        image_file = os.path.join(images_path, fname)
        img_num = int(os.path.splitext(fname)[0])  # 从文件名提取序号

        logger_.info(f'读 {image_file}')

        img = cv2.imread(image_file)
        if img is None:
            logger_.warning(f'  无法读取 {fname}，已跳过')
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        size = gray.shape[::-1]
        ret, corners = cv2.findChessboardCorners(gray, (XX, YY), None)

        if ret:

            obj_points.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
            if [corners2]:
                img_points.append(corners2)
            else:
                img_points.append(corners)
            success_indices.append(img_num)
        else:
            logger_.warning(f'  {fname} 角点检测失败，已跳过')

    N = len(img_points)
    logger_.info(f'成功检测: {N}/{len(image_files)} 张图片')

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

    # 按成功检测的图片序号提取对应位姿，保证图-位姿一一对应
    for idx in success_indices:
        i = idx - 1  # 转为 0-based 索引
        R_tool.append(tool_pose[0:3, 4*i:4*i+3])
        t_tool.append(tool_pose[0:3, 4*i+3])

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

    # ========== 标定误差分析 ==========
    error_report = compute_calibration_errors(
        obj_points, img_points, rvecs, tvecs, mtx, dist,
        R_tool, t_tool, R, t, logger_,
        success_indices=success_indices,
    )

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

