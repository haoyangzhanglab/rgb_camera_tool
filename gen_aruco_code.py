import cv2
import numpy as np
from cv2 import aruco
from pathlib import Path

# 定义保存的路径, 保存在当前文件夹下的output文件夹中
current_file = Path(__file__)
current_dir = current_file.resolve().parent
save_dir = str(current_dir) + "/output"

# 指定aruco码生成的字典、ID、像素大小
aruco_id = 25
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_100)
aruco_pixel_size = 400

# 生成的aruco码保存路径
raw_image = np.zeros((400, 400), dtype=np.int8)
marker_image = cv2.aruco.drawMarker(aruco_dict, id=aruco_id, sidePixels=aruco_pixel_size, img=raw_image, borderBits=1)

# 保存生成的aruco码
save_file_name = save_dir + "/aruco_marker.png"
cv2.imwrite(save_file_name, marker_image)

# 还可以直接从网站上下载生成的aruco码： https://chev.me/arucogen/， 可以指定大小(mm)
