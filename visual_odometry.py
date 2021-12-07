#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import cv2 # used for image proccessing
import numpy as np # used for linear algebra math
import math # used for trigonometry functions
import time # used to track time, duh

from circles_on_steroids import find_circle_center_auto




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

	args = [int(i) for i in sys.argv[1:]]
	methods = args
	print('')
	print("Script launched with {} arguments.".format(len(args)))
	if len(args) == 0:
		print("The default mode will be used with manual cv2.HoughCircles")
		print("approximation for rotation center.")
		print("You can add or remove additional approximations by adding")
		print("argumets to the script call. Example:")
		print("> python visual_odometry.py 1 0 1")
		print("This tells the script:")
		print("- to USE manual cv2.HoughCircles")
		print("- to NOT USE manual cv2.findContours")
		print("- to USE autmatic cv2.HoughCircles")
		print("Solution for each choosen approximation will be built separetly.")
		methods = [1, 0, 0]
	elif len(args) == 1:
		methods = [args[0], 0, 0]
	elif len(args) == 2:
		methods = [args[0], args[1], 0]
	elif len(args) > 3:
		print("Too many argumets. Extra arguments will be ignored.")
	print('')

	width = 1920
	height = 1080
	# Open video file
	cap = cv2.VideoCapture('crop_4.488rps.mp4')
	#cap = cv2.VideoCapture('crop_1.496rps.mp4')

	fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
	print("Video FPS: {}".format(fps))

	# Read one frame for demonstration
	ret, frame = cap.read()
	# If frame retrieved successefuly
	if ret:
		# Correct image width and height for futher reference
		height, width, channels = frame.shape
		print("Actual video dimensions are {}, {}".format(width, height))


	base_image = np.zeros((height,width), np.uint8)
	# Run the whole video once to find the rotation center
	count = 0
	marker_pos = []
	end_not_reached = True
	print('')
	print("Running the video once to find rotation center...")
	while end_not_reached:
	#for i in range(0, 100):
		# Attempt to get the frame from the video
		ret, frame = cap.read()
		end_not_reached = ret
		count += 1
		
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

			# --- marker position calculation
			moments = cv2.moments(mask, 1)
			x = -1
			y = -1
			if moments['m00'] != 0:
				x = int(moments['m10']/moments['m00'])
				y = int(moments['m01']/moments['m00'])
			marker_pos.append((x, y))
			#print(marker_pos[-1])
		if count % 50 == 0:
			print("Frames analysed: {} ...".format(count))
	print("Total frames in the video: {}".format(count))

	# Find actual rotation center on the base_image
	coords = find_circle_center_manual(base_image)
	coords2 = [0, 0]
	coords3 = [0, 0]
	if methods[0] == 1:
		print("Manual cv2.HoughCircles solution for rotation center: {}, {}".format(coords[0], coords[1]))
		cv2.circle(base_image, (coords[0], coords[1]), 12, (100), -1)
	if methods[1] == 1:
		coords2 = find_contour_center(base_image)
		print("Manual cv2.findContours solution for rotation center: {}, {}".format(coords2[0], coords2[1]))
		cv2.circle(base_image, (coords2[0], coords2[1]), 8, (150), -1)
	if methods[2] == 1:
		print("Running automatic cv2.HoughCircles solution. This will take a bunch of time!")
		coords3 = find_circle_center_auto(base_image, 1, coords[2] - 50, coords[2] + 50)
		print("Automatic cv2.HoughCircles solution for rotation center: {}, {}".format(coords3[0], coords3[1]))
		cv2.circle(base_image, (coords3[0], coords3[1]), 6, (200), -1)


	# Save image to file
	cv2.imwrite("rotation_center.png", base_image)
	# Show the image with marked rotation center
	width = int(base_image.shape[1] * 50 / 100)
	height = int(base_image.shape[0] * 50 / 100)
	cv2.namedWindow("Rotation center")
	cv2.moveWindow("Rotation center", 50, 50) 
	cv2.imshow("Rotation center", cv2.resize(base_image, (width, height), interpolation = cv2.INTER_AREA))
	cv2.waitKey(3000)

	cap.release()

	# Analyse the video frame by frame and calculate displacement
	# of the marker relative to the rotatin center
	print('')
	print("Calculating the rotation speed...")

	angle1 = []
	angle2 = []
	angle3 = []
	velocity1 = []
	velocity2 = []
	velocity3 = []
	count_frames = 1
	for i in range(0, len(marker_pos)):
		x, y = marker_pos[i]
		if x < 0 or y < 0:
			count_frames += 1
		else:
			# Calculate the rotation speed of the wheel
			if methods[0] == 1:
				angle1.append(math.atan2(y - coords[1], x - coords[0]))
			if methods[1] == 1:
				angle2.append(math.atan2(y - coords2[1], x - coords2[0]))
			if methods[2] == 1:
				angle3.append(math.atan2(y - coords3[1], x - coords3[0]))


			if len(angle1) > 2:
				ang_dif = angle1[-1] - angle1[-2]
				if abs(ang_dif) > math.pi:
					ang_dif = ang_dif - math.copysign(2 * math.pi, ang_dif)
				# Limit possible angle increment to 2.0 radian, this will prevent
				# huge leaps due to false marker detection
				# TODO: find a better way to reliably filter false detection spikes
				if abs(ang_dif) < 1:
					rotation_speed = abs(ang_dif) / (count_frames / fps)
					#print("velocity: {}, {}".format(rotation_speed, count_frames))
					velocity1.append(rotation_speed)

			if len(angle2) > 2:
				ang_dif = angle2[-1] - angle2[-2]
				if abs(ang_dif) > math.pi:
					ang_dif = ang_dif - math.copysign(2 * math.pi, ang_dif)
				if abs(ang_dif) < 1:
					rotation_speed = abs(ang_dif) / (count_frames / fps)
					#print("velocity: {}".format(rotation_speed))
					velocity2.append(rotation_speed)

			if len(angle3) > 2:
				ang_dif = angle3[-1] - angle3[-2]
				if abs(ang_dif) > math.pi:
					ang_dif = ang_dif - math.copysign(2 * math.pi, ang_dif)
				if abs(ang_dif) < 1:
					rotation_speed = abs(ang_dif) / (count_frames / fps)
					#print("velocity: {}".format(rotation_speed))
					velocity3.append(rotation_speed)
				
			count_frames = 1

	if methods[0] == 1:
		print("Method 1 solution for rotation speed: {:02.2f}".format(sum(velocity1) / len(velocity1)))
	if methods[1] == 1:
		print("Method 2 solution for rotation speed: {:02.2f}".format(sum(velocity2) / len(velocity2)))
	if methods[2] == 1:
		print("Method 3 solution for rotation speed: {:02.2f}".format(sum(velocity3) / len(velocity3)))




if __name__ == '__main__':
	visual_odometry()

