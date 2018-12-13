import numpy as np
import cv2
from matplotlib import pyplot as plt

img1 = cv2.imread('Img001_diffuse.tif',0)
img2 = cv2.imread('Img002_diffuse.tif',cv2.IMREAD_GRAYSCALE)

orb = cv2.ORB_create()   

kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)  # we should use ssd

matches = bf.match(des1,des2)

matches = sorted(matches, key = lambda x:x.distance)

img3 = cv2.drawMatches(img1,kp1,img2,kp2, matches[:40],img2,flags=2)
plt.imshow(img3),plt.show()
