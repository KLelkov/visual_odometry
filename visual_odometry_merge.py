#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2 # used for image proccessing
import numpy as np # used for linear algebra math
import math # used for trigonometry functions
import time # used to track time, duh

from circles_on_steroids import find_circle_center_auto





def color_detection_bgr(image):
	# Ranges for Blue, Green and Red values
	lower_red = np.array([90,95,80])
	upper_red = np.array([140,150,255])
	# Apply color range to the image
	red_mask = cv2.inRange(image, lower_red, upper_red)
	return red_mask


def color_detection_hsv(image):
	# Ranges for Blue, Green and Red values
	# Red range
	lower_red = np.array([0,70,180])
	upper_red = np.array([10,255,255])
	# Ultraviolet range
	lower_red1 = np.array([170,70,180])
	upper_red1 = np.array([180,255,255])
	# Apply color range to the image
	red_mask = cv2.inRange(image, lower_red, upper_red) + cv2.inRange(image, lower_red1, upper_red1)
	return red_mask


def find_circle_center_manual(image):
	# The image should be grayscale with clearly visible wheel circle

	# param2 defines how many falsce circles are detected
	circles = cv2.HoughCircles(image,cv2.cv.CV_HOUGH_GRADIENT,1,20, param1=25, param2=30, minRadius=50) # Апрокимируем выделенное серое колесо до круга. Но функция находит много лишних кругов.

	# Display all the circles found
	#circles = np.uint16(np.around(circles))
	#for i in circles[0,:]:
	#	cv2.circle(image,(i[0],i[1]),i[2],(100),2)
	#	cv2.circle(image,(i[0],i[1]),5,(200),3)

	# Find the center
	center = np.around(np.mean(circles[0,:], axis = 0)) # Находим среднее арифметическое из найденный кругов
	# center array contains x, y, radius
	return (int(center[0]), int(center[1]), int(center[2]))


def find_contour_center(image):
	# This function will find contours rather then circles.
	# This gives advantage of possible ellipse approximation to account for image distortion.
	# The downside is however, that you need a solid circle contour. This will probably
	# require you to run the whole video analysis twice - one time to find center and the second
	# time - to find the actual rotation speed

	# The image should be grayscale with clearly visible wheel circle
	# Find contours
	contours, hierarchy = cv2.findContours(image.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	
	# Find the largest contour - it is the most likely wheel's circle
	maxlen = 0;
	truecont = contours[0];
	for icontour in contours:
		if len(icontour) > maxlen:
			maxlen = len(icontour)
			truecont = icontour

	# Find the ellipse approximation for the largest contour
	ellipse = cv2.fitEllipse(truecont)
	# Display it
	#cv2.ellipse(image, ellipse, (200), 5)
	
	return (int(ellipse[0][0]), int(ellipse[0][1]))


def visual_odometry():
	width = 1920
	height = 1080
	# Open video file
	cap = cv2.VideoCapture('test_4.488_rps.mp4')


	# Read one frame for demonstration
	ret, frame = cap.read()
	# If frame retrieved successefuly
	if ret:
		# Normalize image brightness
		cv2.normalize(frame, frame, 0, 255, cv2.NORM_MINMAX)
		# Blur the image to reduce noise
		blur = cv2.medianBlur(frame, 15)
		# Get color mask
		mask = color_detection_bgr(blur)
		# Merge the mask and the original image to highlight the color
		masked = cv2.bitwise_and(frame, frame, mask=mask)
		# Save image to file
		cv2.imwrite('detection_highlight.png', masked)

		# Correct image width and height for futher reference
		height, width, channels = frame.shape



	base_image = np.zeros((height,width), np.uint8)
	# Analyze first 50 frames to find the rotation center
	count = 0
	end_not_reached = True
	while end_not_reached:
	#for i in range(0, 450):
	#while(cap.isOpened()):
		# Attempt to get the frame from the video
		ret, frame = cap.read()
		end_not_reached = ret
		count += 1
		if count % 25 == 0:
			print(count)
		if ret:
			# Normalize image brightness
			cv2.normalize(frame, frame, 0, 255, cv2.NORM_MINMAX)
			# Blur the image to reduce noise
			blur = cv2.medianBlur(frame, 15)
			# Convert BGR to HSV
			# HSV images are easier to work with
			hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
			# Get the mask
			mask = color_detection_hsv(hsv)

			# add mask to the base_image
			base_image = base_image + mask
	print("total frames: {}".format(count))
	# TODO: Find actual rotation center on the base_image
	coords = find_circle_center_manual(base_image)
	#coords2 = find_contour_center(base_image)
	#coords3 = find_circle_center_auto(base_image, 1, coords[2] - 50, coords[2] + 50)
	print(coords)
	#print(coords2)
	#print(coords3)
	# Draw the center of the wheel          
	cv2.circle(base_image, (coords[0], coords[1]), 12, (100), -1)
	#cv2.circle(base_image, (coords2[0], coords2[1]), 8, (150), -1)
	#cv2.circle(base_image, (coords3[0], coords3[1]), 6, (200), -1)
	# Save image to file
	cv2.imwrite("rotation_center.png", base_image)
	# Show the image with marked rotation center
	width = int(base_image.shape[1] * 50 / 100)
	height = int(base_image.shape[0] * 50 / 100)
	cv2.imshow("center", cv2.resize(base_image, (width, height), interpolation = cv2.INTER_AREA))
	key_press = cv2.waitKey()


	# TODO: Analyse the video frame by frame and calculate displacement
	# of the marker relative to the rotatin center

	# TODO: Calculate the rotation speed of the wheel




	# Close the video
	cap.release()





if __name__ == '__main__':
	visual_odometry()

