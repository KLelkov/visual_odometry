import numpy as np
import argparse
import cv2
import signal

from functools import wraps
import errno
import os
import copy



# Alrihgt, dont ask me how it works, but this function will attempt to
# auto-fit cv2.HoughCircles parameters to find the most likely locations
# of the desired number of circles on the image.
def find_circle_center_auto(image, number_of_circles_expected, minimum_circle_size, maximum_circle_size):
	# This may actually take over ten minutes, be patient xd

	orig_image = np.copy(image)
	output = image.copy()
	gray = image.copy()

	circles = None
	params = []

	#minimum_circle_size = 200      #this is the range of possible circle in pixels you want to find
	#maximum_circle_size = 600     #maximum possible circle size you're willing to find in pixels

	guess_dp = 1.0


	breakout = False

	max_guess_accumulator_array_threshold = 100     #minimum of 1, no maximum, (max 300?) the quantity of votes 
	                                                #needed to qualify for a circle to be found.
	circleLog = []

	guess_accumulator_array_threshold = max_guess_accumulator_array_threshold

	while guess_accumulator_array_threshold > 1 and breakout == False:
		#start out with smallest resolution possible, to find the most precise circle, then creep bigger if none found
		guess_dp = 1.0
		#print("resetting guess_dp: " + str(guess_dp))
		while guess_dp < 9 and breakout == False:
			guess_radius = maximum_circle_size
			#print("setting guess_radius: " + str(guess_radius))
			#print(circles is None)
			while True:

				#HoughCircles algorithm isn't strong enough to stand on its own if you don't
				#know EXACTLY what radius the circle in the image is, (accurate to within 3 pixels) 
				#If you don't know radius, you need lots of guess and check and lots of post-processing 
				#verification.  Luckily HoughCircles is pretty quick so we can brute force.

				#print("guessing radius: " + str(guess_radius) + 
				#		" and dp: " + str(guess_dp) + " vote threshold: " + 
				#		str(guess_accumulator_array_threshold))

				circles = cv2.HoughCircles(gray, 
					cv2.cv.CV_HOUGH_GRADIENT, 
					dp=guess_dp,               #resolution of accumulator array.
					minDist=100,                #number of pixels center of circles should be from each other, hardcode
					param1=40,
					param2=guess_accumulator_array_threshold,
					minRadius=(guess_radius-3),    #HoughCircles will look for circles at minimum this size
					maxRadius=(guess_radius+3)     #HoughCircles will look for circles at maximum this size
					)

				if circles is not None:
					if len(circles[0]) == number_of_circles_expected:
						#print("len of circles: " + str(len(circles)))
						circleLog.append(copy.copy(circles))
						params.append((guess_accumulator_array_threshold, guess_radius-3, guess_radius+3))
						#print("k1")
					break
					circles = None
				guess_radius -= 5 
				if guess_radius < 40:
					break;

			guess_dp += 1.5

		guess_accumulator_array_threshold -= 2

	#Return the circleLog with the highest accumulator threshold
	mean_center = [0, 0]
	print("Best approximations for the circle found automaticly:")
	# ensure at least some circles were found
	for i in range(0, len(circleLog)):
	#for cir in circleLog:
		cir = circleLog[i]
		# convert the (x, y) coordinates and radius of the circles to integers
		output = np.copy(orig_image)

		if (len(cir) > 1):
			print("FAIL before")
			exit()

		print(cir[0, :])
		mean_center[0] += cir[0][0][0]
		mean_center[1] += cir[0][0][1]

		cir = np.round(cir[0, :]).astype("int")

		for (x, y, r) in cir:
			cv2.circle(output, (x, y), r, (200), 2)
			cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
		print(params[i])
		#cv2.imshow("output", output)
		#cv2.waitKey()
	mean_center[0] = mean_center[0] / len(circleLog)
	mean_center[1] = mean_center[1] / len(circleLog)
	#print("mean:")
	#print(mean_center)
	return (int(mean_center[0]), int(mean_center[1]))



