import sys
import numpy as np
import cv2 as cv

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QLabel,
    QPushButton, QWidget, QVBoxLayout, QHBoxLayout
)
from PyQt6.QtGui import (
    QImage, QPixmap, QMouseEvent, QWheelEvent
)
from PyQt6.QtCore import Qt




class ImageViewer(QMainWindow):
    def __init__(self):
        super().__init__()

        # -----------------------------
        # 基础变量
        # -----------------------------
        self.image_cv = None
        self.matrix_global = np.identity(3, dtype=np.float32)

        self.dragging = False
        self.last_mouse_pos = None

        # -----------------------------
        # UI初始化
        # -----------------------------
        self.setWindowTitle("图片查看器 Image Viewer")

        self.lb_img = QLabel("请加载图片")
        self.lb_img.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.lb_img.setStyleSheet(
            """
            background-color: #ddd;
            border: 1px solid #888;
            font-size: 28px;          /* 字体大小 */
            color: #555;              /* 字体颜色更柔和 */
            font-weight: bold;        /* 粗体 */
            """
        )
        self.lb_img.setFixedSize(800, 600)

        # 选择图片按钮
        self.btn_load = QPushButton("选择图片")
        self.btn_load.setFixedHeight(40)
        self.btn_load.setStyleSheet(
            "font-size: 18px; padding: 6px; background-color: #5DA3FA;"
            "border-radius: 6px; color: white;"
        )
        self.btn_load.clicked.connect(self.load_image)

        # 布局
        layout_main = QVBoxLayout()
        layout_main.addWidget(self.lb_img)

        layout_bottom = QHBoxLayout()
        layout_bottom.addStretch()
        layout_bottom.addWidget(self.btn_load)
        layout_bottom.addStretch()

        layout_main.addLayout(layout_bottom)

        container = QWidget()
        container.setLayout(layout_main)
        self.setCentralWidget(container)

    # --------------------------------------------------
    # 选择图片
    # --------------------------------------------------
    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "", "Images (*.jpg *.png *.jpeg)"
        )
        if path:
            self.image_cv = cv.imread(path)
            self.matrix_global = np.identity(3, dtype=np.float32)
            self.refresh_image()

    # --------------------------------------------------
    # 刷新图像显示
    # --------------------------------------------------
    def refresh_image(self):
        if self.image_cv is None:
            return

        H = self.lb_img.height()
        W = self.lb_img.width()

        warped = cv.warpPerspective(
            self.image_cv,
            self.matrix_global,
            (W, H),
            flags=cv.INTER_NEAREST
        )

        qimg = QImage(
            warped.data,
            W,
            H,
            QImage.Format.Format_BGR888
        )
        self.lb_img.setPixmap(QPixmap(qimg))

    # --------------------------------------------------
    # 鼠标按下：拖动开始
    # --------------------------------------------------
    def mousePressEvent(self, e: QMouseEvent):
        if self.image_cv is None:
            return
        self.dragging = True
        self.last_mouse_pos = np.array([e.pos().x(), e.pos().y()])

    # --------------------------------------------------
    # 鼠标松开：拖动结束
    # --------------------------------------------------
    def mouseReleaseEvent(self, e: QMouseEvent):
        self.dragging = False

    # --------------------------------------------------
    # 鼠标移动：拖动平移
    # --------------------------------------------------
    def mouseMoveEvent(self, e: QMouseEvent):
        if self.image_cv is None or not self.dragging:
            return

        now = np.array([e.pos().x(), e.pos().y()])
        delta = now - self.last_mouse_pos
        self.last_mouse_pos = now

        dx, dy = delta
        T = np.array([
            [1, 0, dx],
            [0, 1, dy],
            [0, 0, 1]
        ], dtype=np.float32)

        self.matrix_global = T @ self.matrix_global
        self.refresh_image()

    # --------------------------------------------------
    # 双击输出 RGB
    # --------------------------------------------------
    def mouseDoubleClickEvent(self, e: QMouseEvent):
        if self.image_cv is None:
            return

        win_x, win_y = e.pos().x(), e.pos().y()
        p = np.array([win_x, win_y, 1])

        invM = np.linalg.inv(self.matrix_global)
        x_img, y_img, _ = invM @ p

        x_img = int(x_img)
        y_img = int(y_img)

        if 0 <= x_img < self.image_cv.shape[1] and 0 <= y_img < self.image_cv.shape[0]:
            b, g, r = self.image_cv[y_img, x_img]
            print(f"RGB = ({r}, {g}, {b})")
        else:
            print("图像范围外")

    # --------------------------------------------------
    # 滚轮缩放：以鼠标为中心
    # --------------------------------------------------
    def wheelEvent(self, e: QWheelEvent):
        if self.image_cv is None:
            return

        x = e.position().x()
        y = e.position().y()

        scale = 1.1 if e.angleDelta().y() > 0 else 0.9

        T1 = np.array([[1, 0, -x],
                       [0, 1, -y],
                       [0, 0, 1]], dtype=np.float32)

        S = np.array([[scale, 0, 0],
                      [0, scale, 0],
                      [0, 0, 1]], dtype=np.float32)

        T2 = np.array([[1, 0, x],
                       [0, 1, y],
                       [0, 0, 1]], dtype=np.float32)

        self.matrix_global = T2 @ S @ T1 @ self.matrix_global
        self.refresh_image()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = ImageViewer()
    viewer.show()
    sys.exit(app.exec())
