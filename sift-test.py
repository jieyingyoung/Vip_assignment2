# pip install opencv-python==3.4.2.16
# pip install opencv-contrib-python==3.4.2.16

import cv2
import sys
import numpy as np
# imgpath = “Img001_diffuse.tif”
img = cv2.imread('Img001_diffuse.tif')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
sift = cv2.xfeatures2d.SIFT_create()
keypoints, descriptor = sift.detectAndCompute(gray,None)
img = cv2.drawKeypoints(image=img, outImage=img, keypoints =
keypoints, flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
color = (51, 163, 236))
cv2.imwrite('sift_keypoints_test.png',img)
cv2.imshow('sift_keypoints', img)
while True:
	key = cv2.waitKey(1)
	if key == 27: 
		break
