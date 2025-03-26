#@title <Измерение по шаблону>{ form-width: "20%" }
# Pin code 1 MP, 50mkm  x 1,68 = 84px масштаб:#1=0.96, #2=
# https://tproger.ru/translations/opencv-python-guide/
img_path = "./patterns/27.01.000.98.02.001.jpg" # @param ["./27.01.000.98.02.001.jpg", "./27.01.000.98.02.002.jpg", "./27.01.000.98.02.003.jpg", "./27.01.000.98.02.004.jpg", "./cv2/27.07.004.01.01.005.jpg"]
angle_int = -1 # @param {type:"integer"}
import cv2
import numpy as np
# from IPython.lib.display import exists
import os

def rotate_image(img_name):
  global angle_int
  # Загрузка изображения
  image = cv2.imread(img_name)

  # Получение размеров изображения
  height, width = image.shape[:2]
  # Вычисление центра изображения
  center = (width // 2, height // 2)
  # Задание угла поворота
  # Генерация матрицы преобразования
  matrix = cv2.getRotationMatrix2D(center, angle_int, 1.0)
  # Применение преобразования
  image = np.invert(image)
  rotated_image = cv2.warpAffine(image, matrix, (width, height))
  rotated_image[rotated_image < 2] = 130
  image = np.invert(rotated_image)
  cv2.imwrite('pic.jpg', image)

  return 'pic.jpg'

def overlay(back, front, x, y):
  #
  # Получаем фон cv2.imread jpg, получаем изображение cv2.imread png,
  # получаем координаты верхнего левого угла
  # возвращаем изображение на фоне
  #
  if back.shape > front.shape:
    background = back
    img = front
  else:
    background = front
    background = cv2.cvtColor(background, cv2.COLOR_RGB2RGBA)
    mask = np.bitwise_and.reduce(background[:,:,:-1], axis=2)
    background[:,:,3] = np.where(mask==255, 0, 255)
    img = back
    if img.shape[2] < 4:
      img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
  b = np.copy(background)
  place = b[y: y + img.shape[0], x: x + img.shape[1]]
  a = img[..., 3:].repeat(3, axis=2).astype('uint16')
  place[..., :3] = (place[..., :3].astype('uint16') * (255 - a) // 255) + img[..., :3].astype('uint16') * a // 255
  return b

def find_center(img_in):
  #
  # Получаем cv2.imread, содаем контур, обрезаем, ищем центр масс
  #
  img_def = img_in.copy()
  #   show_image(img_def)
  s0 = img_def.shape[0] # row
  s1 = img_def.shape[1] # column
  img_def = cv2.resize(img_def,(img_def.shape[1], int(img_def.shape[0]/1.014)), cv2.INTER_NEAREST) # удаляем искажение 1,014
  if s1 < 1100:
    return -1, -1, -1, -1, -1


  #   img_def = img_def[600:1600, 500:2000] # Немножко подрезаем под поле микроскопа 500:1800, 500:2000
  # img_def = cv2.GaussianBlur(img_def,(3,3), 0) # давим шум, риски на фото делаем потоньше
  img_def = cv2.medianBlur(img_def,5) # давим шум, риски на фото делаем потоньше
  # переводим рис в оттенки серого
  gray_def = cv2.cvtColor(img_def, cv2.COLOR_BGR2GRAY)
  #   cv2.imshow('img',gray_def)
  # посчитаем величину границы черного
  border_black = np.max(gray_def)/2 + 60
  # print('Уровень черного = ', border_black)
  # перевоим в черно-белый по уровню 60. Чем темнее рис тем меньше уровень
  ret, thresh_def = cv2.threshold(gray_def, border_black, 255, 9)#cv2.THRESH_BINARY
  if np.average(thresh_def):
    ret, thresh_def = cv2.threshold(gray_def, border_black, 255, 8)#cv2.THRESH_BINARY
  # cv2.imshow('img',thresh_def)
  # ищем массив замкнутых контуров
  contours_def, hierarchy = cv2.findContours(thresh_def, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)#cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
  if False:
    img_fon = np.uint8(np.ones((thresh_def.shape[0],thresh_def.shape[1]))*255) # белый фон
    cv2.drawContours(img_fon, contours_def, -1, (0,0,0), 1) # рисуем контур
    cv2.imshow('img',img_fon)
  # ищем контур с максимальной площадью
  max_area = 0
  max_contour = None
  for contour in contours_def:
    area = cv2.contourArea(contour)
    if area > 1400000.0:
      pass
    elif area > max_area:
      max_area = area
      max_contour = contour
#   print('max_area', max_area)
  # Находим координаты цетра масс контура
  M = cv2.moments(max_contour)
  cx = int(M['m10']/M['m00'])
  cy = int(M['m01']/M['m00'])

  # Рисуем контур на белом фоне
  # cv2.imshow('img',img_fon)
  img_fon = np.uint8(np.ones((img_def.shape[0],img_def.shape[1]))*255) # белый фон
  # show_image(img_def)
  cv2.drawContours(img_fon, contours_def, -1, (0,0,0), 1) # рисуем контур
  # cv2.imshow('img',img_fon)
  # Находим габариты контура
  invrt = np.invert(img_fon)
  ys, xs = np.nonzero(invrt)
  dimn = [np.min(ys)-40, np.max(ys)+40, np.min(xs)-40, np.max(xs)+40]

  # Рисуем контур и центр на оригинальном рисунке
  # cv2.drawContours(img_def, [max_contour], -1, (0,255,0), 1)
  # cv2.circle(img_def, (cx, cy), 5, (0, 0, 255), -1)
  # Изменяем габариты оригинального рисунка по габаритам контура
  #   img_def = img_def [dimn[0]:dimn[1], dimn[2]:dimn[3]]
  # Изменяем габариты контура по габаритам контура
  #   img_fon = img_fon [dimn[0]:dimn[1], dimn[2]:dimn[3]]

  return img_def, img_fon, cx, cy, contours_def

def show_image(image):
    cv2.imshow("Image", image)
    while True:
        key = cv2.waitKey(1)
        if key == ord("q"):
            break
    cv2.destroyAllWindows()

def drow_image():
    # читаем файл шаблона img_path
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)#IMREAD_UNCHANGED

    # поворачиваем на 90 меняем фрмат шаблона с книжной на альбомную
    if img.shape[0] > img.shape[1]:
        cropped_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    else:
        cropped_img = img.copy()
    y, x, w = cropped_img.shape
    corp_value = 8
    img_patt = cropped_img[corp_value:y-corp_value,corp_value:x-corp_value] # удаляем рамку


    # устанавливаем длину шаблона на основе знания номера чертежа
    if '001' in img_path.split('.'):
        length = 415# длина 27.01.000.98.02.001 в сотках 0.01 976px
        k_fact = 0.42520
        k_chert = 0.597
        k_resize = k_chert/k_fact
    elif '002' in img_path.split('.'):
        length = 455# длина 27.01.000.98.02.002 в сотках 0.01 1070px
        k_fact = 0.42523
        k_chert = 0.597
        k_resize = k_chert/k_fact
    elif '003' in img_path.split('.'):
        length = 495# длина 27.01.000.98.02.003 в сотках 0.01 1165px
        k_fact = 0.42489
        k_chert = 0.597
        k_resize = k_chert/k_fact
    elif '004' in img_path.split('.'):
        length = 535# длина 27.01.000.98.02.004 в сотках 0.01 1259px
        k_fact = 0.42494
        k_chert = 0.597
        k_resize = k_chert/k_fact
    elif '005' in img_path.split('.'):
        length = 2800# длина 27.07.004.01.01.005 в сотках 0.01 495px
        k_fact = 5.65657 # 2800/495
        k_chert = 3.34478
        k_resize = k_chert/k_fact

    # уменьшаем размер шаблона до размера и детали
    img_patt = cv2.resize(img_patt,(int(img_patt.shape[1]/k_resize), int(img_patt.shape[0]/k_resize)), cv2.INTER_NEAREST)
    # cv2.imshow('img',img_patt)

    # Измеряем координаты шаблона
    mask_patt = img_patt.copy()
    mask_patt[mask_patt == 255] = 0

    ys, xs = np.nonzero(mask_patt[:,:,0])
    print('граничные точки шаблона =', np.min(xs), np.min(ys), np.max(xs), np.max(ys))

    # изменяем размеры шаблона с прозрачным фоном
    img_patt=img_patt[np.min(ys)-40:np.max(ys)+200, np.min(xs)-40:np.max(xs)+200].copy()


    img_name = "./pic1.jpg" #@param ["None"] {allow-input: true}
    # flip_cnt = False #@param {type:"boolean"}
    x_set = 5 # @param {type:"slider", min:0, max:20, step:1}
    y_set = 2 #@param {type:"slider", min:0, max:20, step:1}

    if angle_int != 0:
        img_name = rotate_image(img_name)

    #Задаем констатны
    mul = 0.597 # коэффициент умножения 0.597
    x_set = int(x_set/0.597) # переводим в пикселы
    y_set = int(y_set/0.597) # переводим в пикселы
    # су и сх центр контура
    # orig оригинальное изображение
    # contr контур

    is_file = os.path.isfile(img_name)
    if is_file:
        img_def = cv2.imread(img_name)

        orig, img_pin, cx, cy, contours_def = find_center(img_def) # img_def img_name
        # cv2.imshow('img',orig)
        if type(orig) == np.ndarray:


            # Загрузка изображений
            # image1 = cv2.imread('pic1.jpg', cv2.IMREAD_UNCHANGED) # риунок
            # image2 = cv2.imread('pic2.jpg', cv2.IMREAD_UNCHANGED) # фон
            # временно уменьшаем размер
            img_pin = cv2.resize(img_pin,(int(img_pin.shape[1]/3), int(img_pin.shape[0]/3)), cv2.INTER_NEAREST)
            # Преобразование белого цвета в прозрачный
            img_pin = cv2.cvtColor(img_pin, cv2.COLOR_RGB2RGBA)
            mask = np.bitwise_and.reduce(img_pin[:,:,:-1], axis=2)
            img_pin[:,:,3] = np.where(mask==255, 0, 255)

            image3 = overlay (img_patt, img_pin, x_set, y_set)
            #
            # Сохранение объединенного изображения на диск
            # cv2.imwrite('img_pin.png', img_pin)
            # cv2.imwrite('patt.png', img_patt)
            # cv2.imwrite('pic3.png', image3)
            # cv2.imwrite('orig.png', orig)
            # cv2.imshow('Original', orig)
            show_image(image3)
        else:
            print('Ошибка. Требуется фото 5 мегапикселей')

def main():
   drow_image()

if __name__ == '__main__':
   main()