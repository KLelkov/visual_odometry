# visual_odometry
### Requirements
Libraries cv2, numpy
```
sudo pip install opencv-python
sudo pip install numpy
```

### Usage
Script containt three methods for approximating rotation center of the wheel.
* The first method is based on function cv2.HoughCircles with manualy tuned parameters. It is fast and simple, but the accuracy depends on the many factors like lighting conditions, marker size etc. If testing conditions were to change drasticly, new parameters tuning would be required.
* The second method is based on finding the larges solid contour on the image with cv2.findContours. This method generally provides better accuracy, due to the ellipse approximation to account for possible image distortion. However this method is slower. And it would require to run the whole video analysis to build solid wheel contour before attempting to find rotation center.
* Third method is based on cv2.HoughCircles function with automaticly tuned parameters. This method will attempt to find best possible approximations of the rotation circle and then output the average of those best approximations. In theory this method should be the most accurate one. It is however significantly slower than the other two. It may take over ten minutes to find the solution!

When running visual_odometry.py script you can pass it console arguments to pick a approximation method (or any number of them). If you dont specify any arguments - the default methon (first one) will be used.
To select the approximation method run the command
```
python visual_odometry.py 1 1 1
```
1's specify which methods should be enabled. To disable the method - place 0 instead.
