import sys, os, cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QComboBox, QLabel, QWidget, QSpinBox, QPushButton, QGridLayout, QVBoxLayout, QLineEdit
from PyQt5.QtGui import QFont, QImage, QPixmap, QPainter, QColor
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal
import numpy as np
import tkinter as tk
import numpy as np
# from IPython.lib.display import exists

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Image Processing")
        self.setGeometry(100, 100, 600, 480)

        # Создание виджета
        widget = QWidget(self)
        self.setCentralWidget(widget)

        # Создание компонентов интерфейса
        label_img_path = QLabel("Шаблоны:", self)
        self.combo_img_path = QComboBox(self)
        self.combo_img_path.addItems(["./27.01.000.98.02.001.jpg", "./27.01.000.98.02.002.jpg", "./27.01.000.98.02.003.jpg", "./27.01.000.98.02.004.jpg", "./cv2/27.07.004.01.01.005.jpg"])

        label_angle_int = QLabel("Угол поворота:", self)
        self.spin_angle_int = QSpinBox(self)
        self.spin_angle_int.setMinimum(-360)
        self.spin_angle_int.setMaximum(360)

        label_img_name = QLabel("Фото части:", self)
        self.combo_img_name = QComboBox(self)
        self.combo_img_name.addItems(['pic1.jpg'])

        label_x_set = QLabel("X смещение:", self)
        self.spin_x_set = QSpinBox(self)
        self.spin_x_set.setMinimum(-20)
        self.spin_x_set.setMaximum(20)

        label_y_set = QLabel("Y смещение:", self)
        self.spin_y_set = QSpinBox(self)
        self.spin_y_set.setMinimum(-20)
        self.spin_y_set.setMaximum(20)

        button_process = QPushButton("Process", self)
        button_process.clicked.connect(self.process_image)

        # Создание компоновщика
        layout = QVBoxLayout()
        layout.addWidget(label_img_path)
        layout.addWidget(self.combo_img_path)
        layout.addWidget(label_angle_int)
        layout.addWidget(self.spin_angle_int)
        layout.addWidget(label_img_name)
        layout.addWidget(self.combo_img_name)
        layout.addWidget(label_x_set)
        layout.addWidget(self.spin_x_set)
        layout.addWidget(label_y_set)
        layout.addWidget(self.spin_y_set)
        layout.addWidget(button_process)

                # Создание компоновщика
        layout = QGridLayout()
        layout.addWidget(label_img_path, 0, 0, 1, 2)
        
        # layout.addWidget(self.combo_img_path, 0, 3)
        # layout.addWidget(label_angle_int, 0, 2)
        # layout.addWidget(self.spin_angle_int, 0, 3)
        # layout.addWidget(label_img_name, 1, 0)
        # layout.addWidget(self.combo_img_name, 1, 1)
        # layout.addWidget(label_x_set, 1, 2)
        # layout.addWidget(self.spin_x_set, 1, 3)
        # layout.addWidget(label_y_set, 2, 0)
        # layout.addWidget(self.spin_y_set, 2, 1)
        # layout.addWidget(button_process, 2, 2, 1, 2)

        # Установка компоновщика для виджета
        widget.setLayout(layout)

    def process_image(self):
        img_path = self.combo_img_path.currentText()
        angle_int = self.spin_angle_int.value()
        img_name = self.combo_img_name.currentText()
        x_set = self.spin_x_set.value()
        y_set = self.spin_y_set.value()

        # Ваш код для обработки изображения

def get_self_elements(self):
    # Изменение оформления элементов окна
    elements = []
    for attr_name, attr_value in self.__dict__.items():
        elements.append((attr_name, type(attr_value).__name__))
        if isinstance(attr_value, (QLabel, QPushButton)):
            attr_value.setFont(QFont(attr_value.font().family(), int(attr_value.font().pointSize() * 1.5)))

def main():
  app = QApplication(sys.argv)
  window = MainWindow()
  window.setWindowTitle("Компьютерное зрение на службе качества ООО ФЛИМ")
  window.resize(400, 300)
  get_self_elements(window)
  window.show()
  sys.exit(app.exec_())

if __name__ == '__main__':
    main()