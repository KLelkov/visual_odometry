# -*- coding: utf-8 -*-
import cv2
import time
import numpy as np
from PCV1 import phi
# связываем видеопоток камеры с переменной capImg
capImg = cv2.VideoCapture('v1_496_rps.mp4')
# запускаем бесконечный цикл, чтобы следить
# в реальном времени
s = 0
Phi = [] # Список для записи углов поворота колеса
while(capImg.isOpened()):
    s += 1
    ret, frame = capImg.read()
    if frame is None:
        break
    cv2.imshow("Image", frame)

    # Записываем угол поворота колеса в каждом десятом кадре
    if s >= 10:
        image = frame[100:900, 300:1300]
        Phi.append(phi(image))
        s = 0

    key_press = cv2.waitKey(30)
    # если код нажатой клавиши совпадает с кодом
    # «q»(quit - выход),
    if key_press == ord('q'):
        # то прервать цикл while
        break
# освобождаем память от переменной capImg
capImg.release()
VPhi = [] # Список для записи скорости колеса в разные моменты времени
for i in range(len(Phi) - 1):
    VPhi.append((Phi[i + 1] - Phi[i]) * 3)
VPhi = np.array([VPhi])
print(VPhi)
# закрываем все окна opencv
cv2.destroyAllWindows()
