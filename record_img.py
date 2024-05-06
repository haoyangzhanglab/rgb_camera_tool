import cv2
from pathlib import Path

# 在使用前记得把对应文件夹下原来的图片都删除掉

# 定义保存的路径, 保存在当前文件夹下的output文件夹中
current_file = Path(__file__)
current_dir = current_file.resolve().parent
save_dir = str(current_dir) + "/output" + "/intrinsics/"

# 打开视频流，设置图像采集的分辨率
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

# 图片计数器
image_counter = 0

while True:
    # 读取视频流中的帧
    ret, frame = cap.read()

    # 显示实时视频流
    cv2.imshow("Video Stream", frame)

    # 检测按键
    key = cv2.waitKey(1)

    # 按下 's' 保存图像
    if key == ord('s'):
        # 图片文件名按顺序递增
        image_counter += 1
        image_name = f"image_{image_counter}.png"
        image_file_name = save_dir + image_name

        # 保存图像
        cv2.imwrite(image_file_name, frame)
        print(f"保存图像: {image_file_name}")

    # 按下 'q' 退出程序
    if key == ord('q'):
        break

# 释放视频流和关闭窗口
cap.release()
cv2.destroyAllWindows()