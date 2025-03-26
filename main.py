import sys, os, datetime
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import cv2, json
import numpy as np

class View(QMainWindow):
    view_signal = pyqtSignal(str) # Сигнал для передачи текста кнопки

    def __init__(self):
        super().__init__()
        self.frame = None
        self.combo_msg_add_patt = 'Добавьте чертежи в папку patterns'
        self.combo_msg_calibrovka = 'Калибровка и измерения'
        self.envir = {'my_setting':'./setting.cfg',
                      'scale_percent':70, 
                      'mul':0.569, 
                      'path':'./patterns',
                      'high':600,
                      'width':600,
                      }
        if os.path.exists(self.envir['my_setting']):
            try:
                config = self.load_config()
                if config  is not None:
                    self.envir = config
            except:
                print('Файл с конфигурацией содержит ошибки')
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Компьютерное зрение на службе ООО ФЛИМ")
        self.set_geomenty_window()
        # self.setGeometry(50, 100, 800, 600)

        # Создание виджета
        widget = QTabWidget(self)
        self.setCentralWidget(widget)

        # Создание компонентов интерфейса cam_tab
        label_webcam = QLabel("Webcam Image:", self)
        self.label_webcam_image = QLabel()
        label_file = QLabel("File Image:", self)
        self.label_file_image = QLabel()
        self.current_dt = QLabel("дата и время:")
        self.combo_patt_select = QComboBox()
        button_exit = QPushButton("Exit", self)
        
        # Описане событий cam_tab
        self.combo_patt_select.addItem(self.combo_msg_add_patt)
        if os.path.exists(self.envir['path']):
            self.add_itm_combo()
        self.combo_patt_select.activated.connect(self.load_patt)
        button_exit.clicked.connect(self.exit_application)

        # Создание компоновщика cam_tab
        cam_tab = QWidget()
        root_layout = QVBoxLayout(cam_tab)
        first_layout = QHBoxLayout()
        second_layout = QHBoxLayout()
        thirth_layout = QHBoxLayout()
        fourth_layout = QHBoxLayout()
        root_layout.addLayout(first_layout)
        root_layout.addLayout(second_layout)
        root_layout.addLayout(thirth_layout)
        root_layout.addLayout(fourth_layout)
        first_layout.addWidget(label_webcam)
        first_layout.addWidget(label_file)
        second_layout.addWidget(self.label_webcam_image)
        second_layout.addWidget(self.label_file_image)
        thirth_layout.addWidget(self.combo_patt_select)
        thirth_layout.addWidget(self.current_dt)
        fourth_layout.addWidget(button_exit)

        # Создание компонентов интерфейса setting_tab
        self.settings = QTextEdit()
        formatted_json = json.dumps(self.envir, indent=4, ensure_ascii=False).encode('utf-8').decode('utf-8')
        self.settings.setPlainText(formatted_json)
        self.settings.textChanged.connect(lambda: self.view_signal.emit('Изменть конфиг'))
        self.btn_save = QPushButton("Нет изменений конфигурации")
        self.btn_save.pressed.connect(lambda: self.view_signal.emit('Сохранить конфиг')) 

        # Создание компановщика setting_tab
        setting_tab = QWidget(self)
        setting_layout = QVBoxLayout(setting_tab)
        setting_layout.addWidget(self.settings)
        setting_layout.addWidget(self.btn_save)

        # Создание компонентов интерфейса help_tab
        self.text_help = QTextEdit()
        self.text_help.setPlainText(self.create_text_help())

        # Создание компановщика
        help_tab = QWidget(self)
        help_lyaout = QVBoxLayout(help_tab)
        help_lyaout.addWidget(self.text_help)


        widget.addTab(cam_tab, "Измерение")
        widget.addTab(setting_tab, "Настройки")
        widget.addTab(help_tab, "Справка")
        
        

        # Запуск видеопотока с веб-камеры
        self.video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        print("Подключена веб-камера 0")
        if not self.video_capture.isOpened():
           self.video_capture = cv2.VideoCapture(1, cv2.CAP_DSHOW) 
           print("Подключена веб-камера 1")

        
        self.load_patt()
        self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 2592)#2048
        self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1944)#1536
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(50)
        
    def set_geomenty_window(self):
        self.setGeometry(50, 100, int(self.envir['width']), int(self.envir['high']))

    def load_config(self):
        my_setting = self.envir['my_setting']
        with open(my_setting, 'r') as file:
            params = json.load(file)
        return params
    
    def create_text_help(self):
        text = """
Программа предназначена для оптического распознования контура детали, сравнения контура с шаблоном, измерения параметров контура: длина, ширина, радиус.
Версия 1.0.0.0
Изменения 1.0.0.0
Добавлены закладки Измерение, Настройки, Справка

Выберите из списка шаблон для сравнения с контруром измеряемой детали.
Выберите из списка режим "Калибровка и измерения". Результаты измерения будут показаны в строке состояния.
Радиус величина расчетная и может быть расчитан только по трем точкам пересечения измеряемой дуги с осями измерения.
Для калибровки шаблона отключите микроскоп, выберите из списка шаболон, перейдите в режим "Калибровка и измерения".

Назначение параметров конфигурации
"scale_percent": 70, размер картинки в долях от 100%
"mul": 0.569, соотношение размера в сотых долях милиметра к размеру в пикселах 
"path": "./patterns", директория для хранения шаблонов
"high":600 минимальное значение высоты окна
"width":600 минимальное значение ширины окна
"""
        return text
    
    def add_itm_combo(self):
        my_path = self.envir['path']
        if os.path.exists(my_path):
            my_files = os.listdir(my_path)
            if len(my_files) != 0:
                self.combo_patt_select.clear()
                for name in my_files:
                    my_file = os.path.join(my_path, name)
                    my_file = my_file.replace('\\','/')
                    self.combo_patt_select.addItem(my_file)
        else:
            os.mkdir(my_path)
        if self.combo_msg_calibrovka not in [self.combo_patt_select.itemText(i) for i in range(self.combo_patt_select.count())]:
            self.combo_patt_select.addItem(self.combo_msg_calibrovka)
                    
    def convert_to_transparent(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        ret, mask = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        image_with_alpha = cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
        image_with_alpha[:, :, 3] = mask_inv

        yellow_color = np.array([0, 0, 0,   255], dtype=np.uint8)
        image_with_alpha[np.where((image_with_alpha == [0, 0, 0, 255]).all(axis=2))] = yellow_color

        return image_with_alpha
 
    def load_image(self):
        # Загрузка изображения из файла
        file_image = cv2.imread("./27.01.000.98.02.004.jp")
        if file_image is not None:
            file_image = cv2.cvtColor(file_image, cv2.COLOR_RGB2BGR)
            height, width, channel = file_image.shape
            bytes_per_line = 3 * width
            q_image = QImage(file_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
            self.label_file_image.setPixmap(QPixmap.fromImage(q_image))

    def create_img(self):
        # self.frame = frame[300:1100, 700:2000]
        if self.frame is None:
            col = 1300
            row = 800
        else:
            col = self.frame_shape[1]
            row = self.frame_shape[0]
        img = np.uint8(np.ones((row, col)) * 255)
        self.rule_x = col-col//3
        self.rule_y = row//2

        # Добавление вертикальной линии
        cv2.line(img, (self.rule_x, 0), (self.rule_x, row), (0, 0, 0), 1)
        
        # Добавление горизонтальной линии
        cv2.line(img, (0, self.rule_y), (col, self.rule_y), (0, 0, 0), 1)
        return img
    
    def remove_img_border(self, img):
        delta = 8
        if len(img.shape) == 3:
            y,x,_ = img.shape
        else:
            y,x = img.shape
        img[0:delta,:] = 255
        img[:,0:delta] = 255
        img[y-delta:y,:] = 255
        img[:,x-delta:x] = 255    
        return img
    
    def add_point(self,img, pints):
        for y, x in pints:
            cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        return img
    
    def poin2mm(self, ys, xs):
        dic_y = dict(zip(xs, ys))
        dic_x = dict(zip(ys, xs))
        point_x = [i for i in dic_y if dic_y[i]==self.rule_y]
        point_y = [i for i in dic_x if dic_x[i]==self.rule_x]
        try:
            measure_x = np.max(point_x)-np.min(point_x)
            measure_y = np.max(point_y)-np.min(point_y)
            h = np.max(point_x) - self.rule_x
            len_x = round(round(measure_x*self.envir['mul'], 1)/100, 3)
            len_y = round(round(measure_y*self.envir['mul'], 1)/100, 3)
            radius = round(((measure_y**2)/(h*4)+h)/2, 1) # в пикселах
            len_r = round(round(radius*self.envir['mul'], 1)/100, 3) # в сотках
        except ValueError:
            len_x = len_y = len_r = 0

        return len_x, len_y, len_r
    
    def measure(self, img):
        mask = self.create_img()
        img = self.remove_img_border(img)
        # Находим точки пересечения
        dimm_point = np.bitwise_or(img, mask)
        # eskiz = np.bitwise_and(img, mask)
        ys, xs = np.nonzero(np.invert(dimm_point))
        points = zip(ys, xs)
        len_x, len_y, len_r = self.poin2mm(ys, xs)
        msg = f'Ширина={len_y}мм, Длина={len_x}мм, Радиус={len_r}мм'
        self.statusBar().showMessage(msg)
        img = self.add_point(img, points)
        return img

    def update_frame(self):
        ret, frame = self.video_capture.read()
        current_datetime = datetime.datetime.now()
        self.current_dt.setText('Текущее дата и время: ' + current_datetime.strftime("%Y-%m-%d %H:%M:%S"))
        cv2.waitKey(1)
        if ret: 
            frame = frame[300:1100, 700:2000]
            frame_copy = frame.copy()
        else:
            frame = cv2.imread('./patt.jpg')
            frame_copy = frame.copy()
        is_view = False
        self.frame_shape = frame.shape

        gray_def = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        border_black = np.max(gray_def)/2 + 60
        if border_black > 90:
            frame = self.find_center(frame) #

            # self.frame_shape = frame.shape
            is_view = True
            if self.combo_patt_select.currentText() == self.combo_msg_calibrovka:
                frame = self.measure(frame)
            else:
                self.statusBar().showMessage(f'Ширина {frame.shape[1]} сравнить с 2048')
        
        if is_view:

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            # Делаем фон кадра прозрачным
            frame = self.convert_to_transparent(frame)
            frame = self.remove_img_border(frame)
            frame = self.overlay(self.patt, frame,  0, 65)#0, 65
        
        # Преобразуем картинку для вывода в QImage
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        frame_copy = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGBA)

        frame = self.downscale(frame)
        frame_copy = self.downscale(frame_copy)

        # Создание QImage из кадра
        image = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGBA8888)
        image_copy = QImage(frame_copy, frame_copy.shape[1], frame_copy.shape[0], QImage.Format_RGBA8888)

        # Установка QImage в QLabel
        self.label_webcam_image.setPixmap(QPixmap.fromImage(image)) # label_webcam_image
        self.label_file_image.setPixmap(QPixmap.fromImage(image_copy)) # label_webcam_image

    def downscale(self, img):
        width = int(img.shape[1] * self.envir['scale_percent'] / 100)
        height = int(img.shape[0] * self.envir['scale_percent'] / 100)
        dim = (width, height)
        # resize image
        resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        return resized

    def overlay(self, background, watermark, x, y):
        b = background.copy()
        if y + watermark.shape[0] > background.shape[0]:
            y = background.shape[0] - watermark.shape[0]
        if x + watermark.shape[1] > background.shape[1]:
            x = background.shape[1] - watermark.shape[1]
        place = b[y: y + watermark.shape[0], x: x + watermark.shape[1]]
        a = watermark[..., 3:].repeat(3, axis=2).astype('uint16')
        place[..., :3] = (place[..., :3].astype('uint16') * (255 - a) // 255) + watermark[..., :3].astype('uint16') * a // 255
        return b
    
    def show_image(self, image):
        cv2.imshow("Image", image)
        while True:
            key = cv2.waitKey(1)
            if key == ord("q"):
                break
        cv2.destroyAllWindows()
       
    def load_patt(self):
        img_path = self.combo_patt_select.currentText()
        is_save_img_patt = False
        match img_path:
            case self.combo_msg_add_patt:
                self.add_itm_combo()
                img = self.create_img() 
            case self.combo_msg_calibrovka:
                img = self.create_img()
            case _:
                # читаем файл шаблона img_path
                is_save_img_patt = True
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)#IMREAD_UNCHANGED
        
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # поворачиваем на 90 меняем фрмат шаблона с книжной на альбомную
        if img.shape[0] > img.shape[1]:
            cropped_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        else:
            cropped_img = img.copy()

        img_patt = self.remove_img_border(cropped_img)# удаляем рамку

        # устанавливаем длину шаблона на основе знания номера чертежа
        if '001' in img_path.split('.'):
            length = 415# 27.01.000.98.02.001 длина в сотках 0.01, 976px(979х588) длина в пикселах
            k_fact = 0.42520/1.0045 # Коэффициент введен по результатам замера шаблона 0.01мм/976рх
            k_chert = self.envir['mul']
            k_resize = k_chert/k_fact
        elif '002' in img_path.split('.'):
            length = 455# 27.01.000.98.02.002 длина  в сотках 0.01, 1070px(1073х589)  длина в пикселах
            k_fact = 0.42523/1.005 # Коэффициент введен по результатам замера шаблона
            k_chert = self.envir['mul']
            k_resize = k_chert/k_fact
        elif '003' in img_path.split('.'):
            length = 495# 27.01.000.98.02.003 длина в сотках 0.01, 1165px(1168х590) длина в пикселах
            k_fact = 0.42489/1.0045 # Коэффициент введен по результатам замера шаблона
            k_chert = self.envir['mul']
            k_resize = k_chert/k_fact
        elif '004' in img_path.split('.'):
            length = 535# 27.01.000.98.02.004 длина в сотках 0.01,  1259px(1262х589) длина в пикселах
            k_fact = 0.42494/1.004 # Коэффициент введен по результатам замера шаблона
            k_chert = self.envir['mul']
            k_resize = k_chert/k_fact
        elif '005' in img_path.split('.'):
            length = 2800# 27.07.004.01.01.005 длина в сотках 0.01, 495px длина в пикселах
            k_fact = 5.65657 # 2800/495
            k_chert = 3.34478
            k_resize = k_chert/k_fact
        else:
            length = 1300# размеры для белого поля, если папка patterns пустая
            k_fact = 1.0 # 2800/495
            k_chert = 1.0
            k_resize = k_chert/k_fact
        print('Коэффициент масштабирования', k_resize)
        # уменьшаем размер шаблона до размера и детали
        img_patt = cv2.resize(img_patt,(int(img_patt.shape[1]/k_resize), int(img_patt.shape[0]/k_resize)), cv2.INTER_NEAREST)

        # изменяем размеры шаблона с прозрачным фоном
        self.patt = img_patt#[100:,:]
        if is_save_img_patt:
            cv2.imwrite('./patt.jpg', img_patt)

    def find_center(self, img_in):
        #
        # Получаем cv2.imread, содаем контур, обрезаем, ищем центр масс
        #
        img_def = img_in.copy()

        # давим шум, риски на фото делаем потоньше
        img_def = cv2.medianBlur(img_def,5)

        # переводим рис в оттенки серого
        gray_def = cv2.cvtColor(img_def, cv2.COLOR_BGR2GRAY)

        # посчитаем величину границы черного
        border_black = np.max(gray_def)/2 + 60

        # перевоим в черно-белый по уровню 60. Чем темнее рис тем меньше уровень
        ret, thresh_def = cv2.threshold(gray_def, border_black, 255, 9)#cv2.THRESH_BINARY
        if np.average(thresh_def):
            ret, thresh_def = cv2.threshold(gray_def, border_black, 255, 8)#cv2.THRESH_BINARY

        # ищем массив замкнутых контуров
        contours_def, hierarchy = cv2.findContours(thresh_def, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)#cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE

        # Рисуем контур на белом фоне
        img_fon = np.uint8(np.ones((img_def.shape[0],img_def.shape[1]))*255) # белый фон
        cv2.drawContours(img_fon, contours_def, -1, (0,0,0), 1) # рисуем контур
        return img_fon 

    def exit_application(self):
        QApplication.quit()

class Presenter():
    def __init__(self, model, view):
        self.data = None
        self.model = model
        self.view = view
        view.view_signal.connect(self.press_btn) # Подключение сигнала к методу
        # self.is_changed = False

    def press_btn(self, btn_text): 
        match btn_text:
            case 'Изменть конфиг':
                print(f'Произошло событие {btn_text}')
                # if self.is_changed:
                print('Изменена конфигурация')
                txt = self.view.settings.toPlainText()
                self.is_none = self.model.text_to_dict(txt)
                if self.is_none is None:
                    self.view.btn_save.setText('Отменить изменения конфигурации, Ctrl+Z')
                else:
                    self.view.btn_save.setText('Сохранить изменения конфигурации')
                # self.is_changed = True
            case 'Сохранить конфиг':
                print(f'Произошло событие {btn_text}')
                # if self.is_changed:
                if self.is_none is None:
                    self.view.settings.undo()
                else:
                    # self.is_changed = False
                    self.view.btn_save.setText('Нет изменений конфигурации')
                    txt = self.view.settings.toPlainText()
                    dic = self.model.text_to_dict(txt)
                    self.model.save_config(self.view, dic)
                    self.view.envir = self.view.load_config()
                    # formatted_json = json.dumps(self.view.envir, indent=4, ensure_ascii=False)
                    # self.view.settings.setPlainText(formatted_json)
                self.view.set_geomenty_window()
            case _:
                print(f'Отсутствует обработчик для события {btn_text}')

class Model():
    def __init__(self):
        pass
    
    def text_to_dict(self, text):
        data = None
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                return data
            else:
                print("Ошибка: Текст не может быть преобразован в словарь.")
        except json.JSONDecodeError as e:
            print(f"Ошибка: Неверный формат JSON. {str(e)}")

    def save_config(self, view, params):
        my_setting = view.envir['my_setting']
        with open(my_setting, 'w') as file:
            json.dump(params, file)

if __name__ == "__main__":
    # pyinstaller --add-data="check.ico;." -i check.ico -n CheckFormPins -F main.py
    if getattr(sys, 'frozen', False):
        application_path = sys._MEIPASS
    elif __file__:
        application_path = os.path.dirname(__file__)

    iconFile = "./check.ico"
    path_ico = os.path.join(application_path, iconFile)
    # 
    app = QApplication(sys.argv)
    if os.path.exists(path_ico):
        app.setWindowIcon(QIcon(path_ico))
    view = View()
    model = Model()
    presenter = Presenter(model, view)
    view.show()
    sys.exit(app.exec_())