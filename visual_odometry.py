import cv2 # used for image proccessing
import numpy as np # used for linear algebra math
import math # used for trigonometry functions
import time # used to track time, duh





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
	for i in range(0, 50):
		# Attempt to get the frame from the video
		ret, frame = cap.read()
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

	# TODO: Find actual rotation center on the base_image

	# Save image to file
	cv2.imwrite("rotation_center.png", base_image)


	# TODO: Analyse the video frame by frame and calculate displacement
	# of the marker relative to the rotatin center

	# TODO: Calculate the rotation speed of the wheel




	# Close the video
	cap.release()





if __name__ == '__main__':
	visual_odometry()

