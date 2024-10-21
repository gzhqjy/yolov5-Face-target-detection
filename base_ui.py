import sys
import numpy as np
import torch
import cv2
from PySide6.QtWidgets import QMainWindow, QApplication, QFileDialog, QLabel, QPushButton
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import QTimer
# 自定义的UI类，定义了主窗口的布局
from main_window import Ui_MainWindow

# 定义辅助函数，将Numpy数组转换成QImage，便于在Qt界面中显示
def convert2QImage(img):
    height, width, channel = img.shape
    return QImage(img, width, height, width * channel, QImage.Format_RGB888)

# 定义主窗口
class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)  # 初始化界面
        # 加载模型
        self.model = torch.hub.load("./", 'custom', "runs/train/exp10/weights/best.pt", source='local')
        self.capture = None  # 用于视频捕获
        self.timer = QTimer()  # 定时器，用于定时更新视频帧
        self.bind_slots()   # 绑定槽函数和信号（我认为是一种中断机制）

    # 图像检测
    def image_pred(self, file_path):
        # augment = False 不进行数据增强
        result = self.model(file_path, augment=False)

        # 确保结果图像是可写的
        for i in range(len(result.imgs)):
            result.imgs[i] = np.array(result.imgs[i]).copy()  # 创建可写副本

        image = result.render()[0]  # 获取渲染后的图像
        return convert2QImage(image)    # 将结果返回为QImage形式

    # 打开图片（相当于点击了button）
    def open_image(self):
        print("点击了检测图片")
        # 弹出文件选择对话框
        file_path = QFileDialog.getOpenFileName(self, dir="./data/images", filter="*.jpg;*.png;*.jpeg")
        # 检查用户是否选择了文件
        if file_path[0]:
            file_path = file_path[0]
            qimage = self.image_pred(file_path) # 图像检测
            self.input.setPixmap(QPixmap(file_path))    # 显示原始图像
            self.output.setPixmap(QPixmap.fromImage(qimage))    # 显示输出图像

    # 打开视频
    def open_video(self):
        print("点击了检测视频")
        self.capture = cv2.VideoCapture(0)  # 打开默认摄像头
        # 检查摄像头是否打开成功
        if not self.capture.isOpened():
            print("无法打开摄像头")
            return

        self.timer.timeout.connect(self.update_frame)  # 绑定定时器到更新帧的方法
        self.timer.start(30)  # 每30毫秒更新一次

    # 更新视频帧
    def update_frame(self):
        # ret表示是否成功读取
        ret, frame = self.capture.read()  # 从摄像头读取一帧
        if ret:
            # 将 BGR 转换为 RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 进行检测
            result = self.model(frame_rgb, augment=False)
            for i in range(len(result.imgs)):
                result.imgs[i] = np.array(result.imgs[i]).copy()  # 创建可写副本

            # 获取检测后的图像
            detected_image = result.render()[0]
            qimage = convert2QImage(detected_image)

            # 显示原始视频帧和检测结果
            self.input.setPixmap(QPixmap.fromImage(convert2QImage(frame_rgb)))
            self.output.setPixmap(QPixmap.fromImage(qimage))

    # 关闭摄像头
    def closeEvent(self, event):
        if self.capture is not None:
            self.capture.release()  # 释放摄像头
        event.accept()  # 接受关闭事件

    # 绑定槽
    def bind_slots(self):
        # 当点击“检测图片”按钮时，调用open_image方法
        self.dec_image.clicked.connect(self.open_image)
        # 当点击“检测视频”按钮时，调用open_video方法
        self.dec_video.clicked.connect(self.open_video)


if __name__ == '__main__':
    # 创建实例
    app = QApplication(sys.argv)
    # 主窗口
    window = MainWindow()
    window.show()
    # Qt循环
    app.exec()

