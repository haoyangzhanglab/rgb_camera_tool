import cv2
import numpy as np
from cv2 import aruco
from pathlib import Path

# 设置相机参数
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 60)

# 需要根据相机、aruco标记进行修改
h, w = 480, 640  
markerlength = 0.035  # 单位是米

# 读取相机内参
current_file = Path(__file__)
current_dir = current_file.resolve().parent
save_dir = str(current_dir) + "/output" + "/intrinsics" + "/intrinsics_params.npy"
intrinsics = np.load(save_dir, allow_pickle=True).item()
mtx = intrinsics["mtx"]
dist = intrinsics["dist"]

# 获取ARUCO字典, 创建ARUCO检测器参数
aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_100)  
parameters = cv2.aruco.DetectorParameters_create()  

# 进入视频流检测
try:
    while True:
        ret, color_frame = cap.read()
        if not ret:
            continue

        gray_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejectImgPoints = cv2.aruco.detectMarkers(gray_frame, aruco_dict, parameters=parameters)

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(color_frame, corners, ids, borderColor=(255, 255, 0))
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners, markerlength, mtx, dist)
            for i in range(rvec.shape[0]):
                cv2.drawFrameAxes(color_frame, mtx, dist, rvec[i, :, :], tvec[i, :, :], 0.01)  # 在图像中绘制ARUCO标记的姿态轴线
                R, _ = cv2.Rodrigues(rvec[i])
                T = np.eye(4, 4, dtype=np.float64)
                T[0:3, 0:3], T[0:3, 3] = R, tvec[i]
                # 矩阵T表示aruco marker到相机坐标系的位姿

        cv2.imshow('video', color_frame)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
finally:
    cap.release()