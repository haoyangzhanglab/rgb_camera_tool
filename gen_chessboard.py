import cv2
import numpy as np
from pathlib import Path

# 定义保存的路径, 保存在当前文件夹下的output文件夹中
current_file = Path(__file__)
current_dir = current_file.resolve().parent
save_dir = str(current_dir) + "/output" + "/chessboard.png"


# 定义棋盘格的尺寸,单位是像素
# 棋盘格的实际大小不影响相机内参的计算
size = 100

# 定义标定板尺寸，在计算相机内参时，(w, h)表示的是四个方格的交点数量，即(boardx-1, boardy-1)
boardx = size * 10
boardy = size * 7

canvas = np.zeros((boardy, boardx, 1), np.uint8)  # 创建画布
for i in range(0, boardx):
    for j in range(0, boardy):
        if (int(i/size) + int(j/size)) % 2 != 0:  # 判定是否为奇数格
            canvas[j, i] = 255
cv2.imwrite(save_dir, canvas)


