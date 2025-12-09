import sys
import numpy as np
import cv2 as cv

from PyQt6 import QtCore
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QFileDialog,
    QLabel,
)

from PyQt6.QtGui import (
    QImage,
    QPixmap,
    QMouseEvent,
    QWheelEvent
)


class Image_Viewer_Base(QMainWindow):
    def __init__(self, H_show: int = 400, W_show: int = 600):
        super().__init__()

        # ---------------------------
        # 图像、矩阵初始化
        # ---------------------------
        self.image_cv = self.init_image_base()
        self.matrix_global = np.identity(3, dtype=np.float32)

        # 拖动状态
        self.dragging = False
        self.last_mouse_pos = None

        # ---------------------------
        # UI 组件
        # ---------------------------
        self.lb_img = QLabel()
        self.H_show = H_show
        self.W_show = W_show
        self.init_ui_base()
        self.refresh_image()

    # --------------------------------------------------------
    # UI 初始化
    # --------------------------------------------------------
    def init_ui_base(self):
        self.setWindowTitle('Image Viewer')
        self.lb_img.setFixedHeight(self.H_show)
        self.lb_img.setFixedWidth(self.W_show)
        self.setCentralWidget(self.lb_img)

    # --------------------------------------------------------
    # 选择图片
    # --------------------------------------------------------
    def init_image_base(self):
        return cv.imread(self.get_image())

    # --------------------------------------------------------
    # 刷新图像显示
    # --------------------------------------------------------
    def refresh_image(self):
        W = self.W_show
        H = self.H_show

        # 透视变换（核心）
        image_show = cv.warpPerspective(
            self.image_cv,
            self.matrix_global,
            (W, H),
            flags=cv.INTER_NEAREST
        )

        # 转成 QPixmap 显示
        qimg_show = QImage(
            np.ascontiguousarray(image_show).data,
            W,
            H,
            QImage.Format.Format_BGR888
        )
        qpixm_show = QPixmap(qimg_show)
        self.lb_img.setPixmap(qpixm_show)

    # --------------------------------------------------------
    # 鼠标按下：开始拖动 or RGB 采样准备
    # --------------------------------------------------------
    def mousePressEvent(self, e: QMouseEvent):
        self.dragging = True
        self.last_mouse_pos = np.array([e.pos().x(), e.pos().y()])
        return

    # --------------------------------------------------------
    # 鼠标松开：结束拖动
    # --------------------------------------------------------
    def mouseReleaseEvent(self, e: QMouseEvent):
        self.dragging = False
        return

    # --------------------------------------------------------
    # 鼠标移动：图片平移
    # --------------------------------------------------------
    def mouseMoveEvent(self, e: QMouseEvent):
        if not self.dragging:
            return

        now = np.array([e.pos().x(), e.pos().y()])
        delta = now - self.last_mouse_pos
        self.last_mouse_pos = now

        dx, dy = delta

        # 平移矩阵 T
        T = np.array([
            [1, 0, dx],
            [0, 1, dy],
            [0, 0, 1]
        ], dtype=np.float32)

        # 左乘
        self.matrix_global = T @ self.matrix_global
        self.refresh_image()

    # --------------------------------------------------------
    # 鼠标双击：输出 RGB 值
    # --------------------------------------------------------
    def mouseDoubleClickEvent(self, e: QMouseEvent):
        win_x, win_y = e.pos().x(), e.pos().y()
        p_win = np.array([win_x, win_y, 1], dtype=np.float32)

        # 反变换找原图坐标
        invM = np.linalg.inv(self.matrix_global)
        p_img = invM @ p_win
        x, y = int(p_img[0]), int(p_img[1])

        if 0 <= x < self.image_cv.shape[1] and 0 <= y < self.image_cv.shape[0]:
            b, g, r = self.image_cv[y, x]
            print(f"RGB = ({r}, {g}, {b})")
        else:
            print("点击位置不在图像内")

    # --------------------------------------------------------
    # 滚轮缩放：以鼠标为中心放缩
    # --------------------------------------------------------
    def wheelEvent(self, e: QWheelEvent):
        x, y = e.position().x(), e.position().y()

        # 缩放倍数
        zoom = 1.1 if e.angleDelta().y() > 0 else 0.9

        # 平移到原点
        T1 = np.array([
            [1, 0, -x],
            [0, 1, -y],
            [0, 0, 1]
        ], dtype=np.float32)

        # 缩放矩阵
        S = np.array([
            [zoom, 0, 0],
            [0, zoom, 0],
            [0, 0, 1]
        ], dtype=np.float32)

        # 平移回去
        T2 = np.array([
            [1, 0, x],
            [0, 1, y],
            [0, 0, 1]
        ], dtype=np.float32)

        # 合成
        self.matrix_global = T2 @ S @ T1 @ self.matrix_global
        self.refresh_image()

    # --------------------------------------------------------
    # 打开文件对话框
    # --------------------------------------------------------
    def get_image(self):
        img_name, ____ = QFileDialog.getOpenFileName(
            self,
            'Open Image File',
            '',
            '*.jpg;;*.png;;*.jpeg'
        )
        return img_name


if __name__ == '__main__':
    app = QApplication(sys.argv)
    im = Image_Viewer_Base()
    im.show()
    sys.exit(app.exec())
