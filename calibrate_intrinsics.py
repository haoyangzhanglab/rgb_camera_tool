import cv2
import glob
import numpy as np
from pathlib import Path

# 使用的棋盘格内角点数量，等于方格数减去1
# 需要根据标定使用的棋盘格进行修改，棋盘格格子尺寸不影响相机内参的计算
chess_w = 9
chess_h = 6

# 存储角点在二维平面的坐标
objpoints = []
imgpoints = []
objp = np.zeros((chess_w * chess_h, 3), np.float32)
objp[:, :2] = np.mgrid[0:chess_w, 0:chess_h].T.reshape(-1, 2)

# 设置终止条件
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 设置采集的棋盘格图片路径
current_file = Path(__file__)
current_dir = current_file.resolve().parent
save_dir = str(current_dir) + "/output" + "/intrinsics/"
img_save_path = save_dir + "*.png"
images = glob.glob(img_save_path)

# 开始检测角点
time_pause = 2000  # 每张图片显示的时间(ms)
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (chess_w, chess_h), None)

    if ret == True:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners)

        cv2.drawChessboardCorners(img, (chess_w, chess_h), corners2, ret)
        cv2.imshow("img", img)
        cv2.waitKey(time_pause)
    else:
        print("find no chessboard")

    cv2.destroyAllWindows()

# 计算相机内参
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# 以字典的形式保存相机内参
intrinsics = {"mtx": mtx, "dist": dist}
save_file_path = save_dir + "intrinsics_params.npy"
np.save(save_file_path, intrinsics)


# 计算重投影误差
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    mean_error += error
print("重投影平均误差:{}".format(mean_error/len(objpoints)))


print("去除畸变，显示校正后的图片")
h, w = gray.shape
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w, h))
for fname in images:
    img = cv2.imread(fname)
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    cv2.imshow('img', dst)
    cv2.waitKey(time_pause)
    cv2.destroyAllWindows()


