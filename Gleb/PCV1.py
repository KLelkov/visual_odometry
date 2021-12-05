# -*- coding: utf-8 -*-
import cv2
import numpy as np

def color_red_hsv(image):# Эта функция накладывает маску на изображение для обнаружения красной метки
	# Ranges for Blue, Green and Red values
	# Red range
	lower_red = np.array([0,140,180])
	upper_red = np.array([10,255,255])
	# Ultraviolet range
	lower_red1 = np.array([170,140,180])
	upper_red1 = np.array([180,255,255])
	# Apply color range to the image
	red_mask = cv2.inRange(image, lower_red, upper_red) + cv2.inRange(image, lower_red1, upper_red1)
	return red_mask

def red_mark(image):
	# Эта функция возвращает координаты красной метки

	img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	only_mark = color_red_hsv(img_hsv)
	moments = cv2.moments(only_mark, 1)
	x = int(moments['m10']/moments['m00'])
	y = int(moments['m01']/moments['m00'])
	return np.array([x, y])

def gray_mask(image):
	# Эта функция накладывает маску на изображение для обнаружения серого колеса
	lower_gray = np.array([190, int(0/100*255), int(40/100*255)])
	upper_gray = np.array([205, int(20/100*255), int(100/100*255)])

	lower_gray1 = np.array([30, int(0/100*255), int(40/100*255)])
	upper_gray1 = np.array([60, int(15/100*255), int(100/100*255)])

	gray_mask = cv2.inRange(image, lower_gray, upper_gray) + cv2.inRange(image, lower_gray1, upper_gray1)
	return gray_mask

def circl_centr(image):
	# эта функция ищет центр колеса
	image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	image = cv2.medianBlur(image,5)
	circles = cv2.HoughCircles( gray_mask(image),cv2.cv.CV_HOUGH_GRADIENT,1,20, param1=50,param2=40, minRadius=50,maxRadius=220) # Апрокимируем выделенное серое колесо до круга. Но функция находит много лишних кругов.
	circl = np.around(np.mean(circles[0,:], axis = 0)) # Находим среднее арифметическое из найденный кругов
	circl = np.uint16(np.around(circl))
	return np.array([circl[0], circl[1]])

def phi(image):
	# Эта функция возвращает угол наклона вектора, проведенного от центра колеса к карасной метке
	vec = red_mark(image) - circl_centr(image)
	phi = np.arctan2(vec[1], vec[0])
	return phi

if __name__ == '__main__':
	image = cv2.imread(r"c:\py\Computer_Vision\circl.png")
	img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	image2 =image[20:440, 80:520]

	print(phi(image2))

	xo, yo = circl_centr(image2)
	cv2.circle(image2,(xo,yo),2,(255,0,0),5)
	x, y = red_mark(image2)
	cv2.circle(image2,(x, y),3,(255,0,0),5)
	cv2.putText(image2, "%d, %d" % (x,y), (x - 120,y+50), cv2.FONT_HERSHEY_SIMPLEX, 1, (200,50,50), 2)
	cv2.imshow("Image", image2)
	cv2.waitKey(0)
