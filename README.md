# YOLO v5——目标检测

## 项目介绍和环境配置

![image-20240912151719437](img/image-20240912151719437.png)

![image-20240912153131274](img/image-20240912153131274.png)

可以搜索一些协议去链接摄像头，实现实时检测

![image-20240912153442715](img/image-20240912153442715.png)

将要检测的照片放入

![image-20240913083103816](img/image-20240913083103816.png)

## YoLo V5 模型检测

![image-20240913145337311](img/image-20240913145337311.png)

![image-20240913145934289](img/image-20240913145934289.png)

![image-20240913150154404](img/image-20240913150154404.png)

![image-20240913150354564](img/image-20240913150354564.png)

说明：torch.hub是一种加载数据集权重的方法，由于模型保存本地，所以采用了上面的方式设置权重

注意：我在跑代码的过程中发现由于yolov5版本落后，而我们所装的pytorch已经领先版本太多，所以此时只有两个处理方法：

1.修改源代码

2.增加图像数据处理操作

由于源代码过于繁琐，牵扯到的变量数量众多，所以采用了方式二

~~~python
import torch
import os
import numpy as np

# 确保模型路径存在
model_path = r'runs/train/exp10/weights/best.pt'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"模型文件未找到: {model_path}")

# 加载模型
model = torch.hub.load('.', 'custom', model_path, source='local')

# 指定图像路径
img_path = r'data/images/bus.jpg'
if not os.path.exists(img_path):
    raise FileNotFoundError(f"图像文件未找到: {img_path}")

# 进行推理
results = model(img_path, augment=False)

# 确保结果图像是可写的
for i in range(len(results.imgs)):
    results.imgs[i] = np.array(results.imgs[i]).copy()  # 创建可写副本

# 打印和显示结果
results.print()
results.show()
~~~



## YOLOv5 数据集构建

![image-20240913150726595](img/image-20240913150726595.png)

## YOLOv5模型训练

![image-20240913151959875](img/image-20240913151959875.png)

![image-20240913152007403](img/image-20240913152007403.png)

![image-20240913152351737](img/image-20240913152351737.png)

## 数据集说明

我使用的数据集是自制数据集，通过lux下载了一个哔哩哔哩视频

![image-20240914205224421](img/image-20240914205224421.png)

然后进行了抽帧的操作

~~~python
import cv2
import matplotlib.pyplot as plt

# 打开视频文件
video = cv2.VideoCapture("video01.mp4")
# 读取一帧
# ret,frame = video.read()

# plt.imshow(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))

# 抽取视频帧
num = 0
save_step = 30
while True:
    ret,frame = video.read()
    if not ret:
        break
    num += 1
    if num % save_step == 0:
        cv2.imwrite("images" + str(num) + ".jpg",frame)
~~~

之后形成的文件经过筛选形成初步图片数据集

然后对图片进行标签注释

通过labelimg（通过pip在虚拟环境中安装使用）进行标注，标注文件命名为labels

数据集制作好后按照指定的方式进行数据集处理（因为数据集量小，所以直接在文件夹上进行数据集的划分，如果数据集数量大，应当参考下面的处理方式）

~~~python
# 这个数据集采用了Kaggle上面经典的二分类数据集进行制作，原始数据集只有cats和dogs两个文件夹
import os
import shutil
import sys
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image
import torch.nn as nn
from model import *

# 判断可用设备
device = torch.device("cuda:0" if torch.cuda.is_availabsle() else "cpu")
print("using {} device.".format(device))

# 数据集路径
dataset_path = "PetImages"
output_path = "Cats_and_Dogs"

# 创建数据文件夹
os.makedirs(os.path.join(output_path, 'train', 'Cat'), exist_ok=True)
os.makedirs(os.path.join(output_path, 'train', 'Dog'), exist_ok=True)
os.makedirs(os.path.join(output_path, 'val', 'Cat'), exist_ok=True)
os.makedirs(os.path.join(output_path, 'val', 'Dog'), exist_ok=True)
os.makedirs(os.path.join(output_path, 'test', 'Cat'), exist_ok=True)
os.makedirs(os.path.join(output_path, 'test', 'Dog'), exist_ok=True)

# 复制图像的函数
def copy_images(imgs, src_folder, dest_folder):
    for img in imgs:
        src_path = os.path.join(src_folder, img)
        new_path = os.path.join(dest_folder, img)
        try:
            shutil.copy(src_path, new_path)
        except Exception as e:
            print(f"Error copying {src_path}: {e}")

# 复制训练、验证和测试集
cat_img_folder = os.path.join(dataset_path, 'Cat')
dog_img_folder = os.path.join(dataset_path, 'Dog')

cat_imgs = sorted(os.listdir(cat_img_folder))
dog_imgs = sorted(os.listdir(dog_img_folder))

cat_num = int(len(cat_imgs))
cat_train = cat_imgs[0:int(0.8 * cat_num)]
cat_val = cat_train[0:int(0.2 * len(cat_train))]
cat_test = cat_imgs[int(0.8 * cat_num):]

dog_num = int(len(dog_imgs))
dog_train = dog_imgs[0:int(0.8 * dog_num)]
dog_val = dog_train[0:int(0.2 * len(dog_train))]
dog_test = dog_imgs[int(0.8 * dog_num):]

copy_images(cat_test, cat_img_folder, os.path.join(output_path, 'test', 'Cat'))
copy_images(dog_test, dog_img_folder, os.path.join(output_path, 'test', 'Dog'))
copy_images(cat_train, cat_img_folder, os.path.join(output_path, 'train', 'Cat'))
copy_images(dog_train, dog_img_folder, os.path.join(output_path, 'train', 'Dog'))
copy_images(cat_val, cat_img_folder, os.path.join(output_path, 'val', 'Cat'))
copy_images(dog_val, dog_img_folder, os.path.join(output_path, 'val', 'Dog'))


# 自定义数据集类以处理图像加载异常
class CustomImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        try:
            sample = self.loader(path)
            # 转换为张量
            sample = self.transform(sample) if self.transform is not None else sample
        except (IOError, OSError, Image.UnidentifiedImageError) as e:
            print(f"Error loading image {path}: {e}")
            # 返回一个全零的张量和默认标签
            return torch.zeros(3, 224, 224), target  # 假设图像大小为224x224，通道数为3
        return sample, target

# 数据预处理
transforms_train = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transforms_test = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transforms_val = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 创建数据集和数据加载器
train_set = CustomImageFolder(root=os.path.join(output_path, "train"), transform=transforms_train)
train_dataloader = DataLoader(train_set, batch_size=64, shuffle=True)

test_set = CustomImageFolder(root=os.path.join(output_path, "test"), transform=transforms_test)
test_dataloader = DataLoader(test_set, batch_size=64, shuffle=True)

val_set = CustomImageFolder(root=os.path.join(output_path, "val"), transform=transforms_val)
val_dataloader = DataLoader(val_set, batch_size=64, shuffle=True)
~~~

数据集处理好后训练数据集共有465张，测试数据集有100张（放在了dataset文件中）

## 模型训练说明

模型训练主要在train.py中进行

![image-20240914211224233](img/image-20240914211224233.png)

首先改写了coco128.yaml文件（这是模型的预训练文件），形成bvn.yaml文件

~~~python
path: dataset
train: ./dataset/images/train/  # 128 images
val: ./dataset/images/val/  # 128 images
nc : 1

# class names
names: [ 'Person']
~~~

再在train.py文件中进行参数的设置

![image-20240914211511843](img/image-20240914211511843.png)

重点修改了训练的模型权重，yaml文件的存放位置，训练轮数

然后开始训练即可

在训练过程中由于numpy版本过高导致训练错误

以下有两种解决方式

1.降低numpy版本

2.修改源代码，将报错（主要是在np.int，修改为int或者是int32/64即可，使用pycharm的全局定位即可找到）修改为符合现代版本的程序

推荐使用第二种方法，因为将numpy版本降低后会引发一系列例如pandas和plt等包的版本不适配的问题，只能重新重装环境

![image-20240914212015366](img/image-20240914212015366.png)

模型训练后会得到weights文件，里面包含了最佳训练结果和最近一次的训练结果（权重）

## GUI界面制作

近几年，受益于人工智能的崛起，Python语言几乎以压倒性优势在众多编程语言中异军突起，成为AI时代的首选语言。在很多情况下，我们想要以图形化方式将我们的人工智能算法打包提供给用户使用，这时候选择以python为主的GUI框架就非常合适了。

QT是众多GUI框架里面非常著名的一款，它本身由C++开发，天然支持基于C++的GUI编程，编出来的图形化软件在当今众多GUI框架中运行效率几乎是天花板级别的，拥有完善的第三方库，极其适合数字图像处理、文档排版、多媒体、3D建模等专业软件开发。与此同时，QT还有一个强大的功能：支持跨平台，简单来理解，就是我们只需要编写一套代码就可以同时在windows、mac、linux上运行。

因为我选择的是pycharm，所以如果有使用jupyter编辑器的自行搜索解决安装QT的方法

具体pycharm环境下安装pyside6的方法在这里

[PyCharm下安装配置PySide6开发环境（Qt Designer(打开，编辑)、PyUIC和PyRCC）_pycharm配置pyside6-CSDN博客](https://blog.csdn.net/mengenqing/article/details/132489529)

安装后通过

Pyside6 QtDesinger进行图形化界面制作

![image-20240914212554855](img/image-20240914212554855.png)

然后将*.ui文件放入项目文件夹中，通过pyside6-uic处理成python文件（在设计图形化界面时可以通过属性管理器更改变量的名字，方便在项目后续中使用）

~~~python
# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'main_window.ui'
##
## Created by: Qt User Interface Compiler version 6.7.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QFrame, QLabel, QMainWindow,
    QPushButton, QSizePolicy, QStatusBar, QWidget)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(988, 428)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.input = QLabel(self.centralwidget)
        self.input.setObjectName(u"input")
        self.input.setGeometry(QRect(130, 50, 301, 231))
        self.input.setScaledContents(True)
        self.input.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.output = QLabel(self.centralwidget)
        self.output.setObjectName(u"output")
        self.output.setGeometry(QRect(560, 50, 321, 231))
        self.output.setScaledContents(True)
        self.output.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.line = QFrame(self.centralwidget)
        self.line.setObjectName(u"line")
        self.line.setGeometry(QRect(460, 30, 61, 291))
        self.line.setFrameShape(QFrame.Shape.VLine)
        self.line.setFrameShadow(QFrame.Shadow.Sunken)
        self.dec_image = QPushButton(self.centralwidget)
        self.dec_image.setObjectName(u"dec_image")
        self.dec_image.setGeometry(QRect(200, 350, 151, 41))
        self.dec_video = QPushButton(self.centralwidget)
        self.dec_video.setObjectName(u"dec_video")
        self.dec_video.setGeometry(QRect(680, 350, 151, 41))
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.input.setText(QCoreApplication.translate("MainWindow", u"\u663e\u793a\u539f\u59cb\u56fe\u7247", None))
        self.output.setText(QCoreApplication.translate("MainWindow", u"\u663e\u793a\u68c0\u6d4b\u7ed3\u679c", None))
        self.dec_image.setText(QCoreApplication.translate("MainWindow", u"\u56fe\u7247\u68c0\u6d4b", None))
        self.dec_video.setText(QCoreApplication.translate("MainWindow", u"\u89c6\u9891\u68c0\u6d4b", None))
    # retranslateUi
~~~

依据这段代码中的关键参数

![image-20240914212852784](img/image-20240914212852784.png)

我们进行可视化界面的制作

~~~python
import sys
import numpy as np
import torch
import cv2
from PySide6.QtWidgets import QMainWindow, QApplication, QFileDialog, QLabel, QPushButton
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import QTimer
from main_window import Ui_MainWindow

def convert2QImage(img):
    height, width, channel = img.shape
    return QImage(img, width, height, width * channel, QImage.Format_RGB888)

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        # 加载模型
        self.model = torch.hub.load("./", 'custom', "runs/train/exp10/weights/best.pt", source='local')
        self.capture = None  # 用于视频捕获
        self.timer = QTimer()  # 定时器
        self.bind_slots()

    def image_pred(self, file_path):
        result = self.model(file_path, augment=False)

        # 确保结果图像是可写的
        for i in range(len(result.imgs)):
            result.imgs[i] = np.array(result.imgs[i]).copy()  # 创建可写副本

        image = result.render()[0]  # 获取渲染后的图像
        return convert2QImage(image)

    # 打开图片
    def open_image(self):
        print("点击了检测图片")
        file_path = QFileDialog.getOpenFileName(self, dir="./data/images", filter="*.jpg;*.png;*.jpeg")
        if file_path[0]:
            file_path = file_path[0]
            qimage = self.image_pred(file_path)
            self.input.setPixmap(QPixmap(file_path))
            self.output.setPixmap(QPixmap.fromImage(qimage))

    # 打开视频
    def open_video(self):
        print("点击了检测视频")
        self.capture = cv2.VideoCapture(0)  # 打开默认摄像头
        if not self.capture.isOpened():
            print("无法打开摄像头")
            return

        self.timer.timeout.connect(self.update_frame)  # 绑定定时器到更新帧的方法
        self.timer.start(30)  # 每30毫秒更新一次

    def update_frame(self):
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

    def closeEvent(self, event):
        if self.capture is not None:
            self.capture.release()  # 释放摄像头
        event.accept()  # 接受关闭事件

    # 绑定槽
    def bind_slots(self):
        self.dec_image.clicked.connect(self.open_image)
        self.dec_video.clicked.connect(self.open_video)


if __name__ == '__main__':
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    app.exec()
~~~

![image-20240914213040845](img/image-20240914213040845.png)

点击即可运行

