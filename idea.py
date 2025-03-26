import sys, os
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QComboBox, QWidget, QGridLayout
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor
from PyQt5.QtCore import QTimer

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Компьютерное зрение на службе ООО ФЛИМ")
        self.setGeometry(100, 100, 800, 600)

        # Создание виджета
        widget = QWidget(self)
        self.setCentralWidget(widget)
        self.combo_patt_select = QComboBox(self)
        self.combo_patt_select.setMinimumWidth(300)
        self.combo_patt_select.addItem("Добавьте чертежи в папку patterns")
        my_path = './patterns'

        if os.path.exists(my_path):
            my_files = os.listdir(my_path)

            if len(my_files) != 0:
                self.combo_patt_select.clear()
                for name in my_files:
                    my_file = os.path.join(my_path, name)
                    my_file = my_file.replace('\\','/')
                    self.combo_patt_select.addItem(my_file)
            
            self.combo_patt_select.activated.connect(self.combo_print)
    
    def combo_print(self):
        print(self.combo_patt_select.currentText())
    
    def exit_application(self):
        QApplication.quit()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())