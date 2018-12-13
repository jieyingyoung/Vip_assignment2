import numpy as np  
from skimage.feature import corner_harris, corner_peaks  , corner_subpix
from skimage.color import rgb2gray  
import matplotlib.pyplot as plt  
import skimage.io as io  
import cv2 as cv

# def get_harris_position(img,k = 0.04,threshold = 0.01):
#     img = rgb2gray(img)
#     height, width = np.shape(img)
#     grandfx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
#     grandfy = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)  # compute the X direction and Y direction gradient 
    
#     Ix2 = grandfx**2
#     Iy2 = grandfy**2
#     Ixy = grandfx*grandfy
    
#     Ix2 = grandfx**2
#     Iy2 = grandfy**2
#     Ixy = grandfx*grandfy
    
#     A = cv2.GaussianBlur(Ix2,(5,5),2)  # eliminate the noises
#     B = cv2.GaussianBlur(Iy2,(5,5),2) 
#     C = cv2.GaussianBlur(Ixy,(5,5),2)
    
#     M = [np.array([[A[i, j], C[i, j]], #compute M
#                [C[i, j], B[i, j]]]) for i in range(height) for j in range(width)]

#     D, T = list(map(np.linalg.det, M)), list(map(np.trace, M))  #computer det(M) and trace(M)
#     R = np.array([d-k*t**2 for d, t in zip(D, T)]) #computer R
#     R_max = np.max(R)
#     R = R.reshape(height, width)
    
#     count = 0
#     corners = []
#     for i in range(1,height-1):  #computer whether Rij is the maximum value in a certain area and it is greater than the threshold*R_max
#         for j in range(1,width-1):
#             if R[i, j] > R_max*threshold:
#                 if R[i, j] == np.max(R[i-1:i+2, j-1:j+2]):
#                     corners.append([i,j])
#                     count += 1             
#     corners = np.array(corners) #position of corners
#     return corners

# def extract_patch(img, corner,n):
#     x = corner[0]
#     y = corner[1]
#     size_n = int((n+1)/2)
#     window = img[(x-size_n):(x+size_n-1), (y-size_n):(y+size_n-1)]
#     return window 

# def show_corner(img,corners):
#     fig, ax = plt.subplots()  # show corners
#     ax.imshow(img, cmap=plt.cm.gray)
#     ax.plot(corners[:, 1], corners[:, 0], '.b', markersize=8)
#     plt.title('k = 0.04, threshold = 0.01')
#     plt.show()

# # # to take patches from image1 and image2, but didn't deal with the edges yet
# def get_patch_12(corners1,corners2,n):
# 	for each_corner1 in corners1:
# 		window1 = extract_patch(img1,each_corner1,n)
# 		np.append(windows1,window1)
# 		# print(windows1)
# 	for each_corner2 in corners2:
# 		window2 = extract_patch(img2,each_corner2,n)
# 		np.append(windows2,window2)
# 		print(np.shape(windows2))
# 	return windows1,windows2

# def ssd(f1,f2,n):
# 	for each_f1 in f1:
# 		# print(each_f1)
# 		for each_f2 in f2:
# 			score = np.sum((f1-f2)**2)
# 			# np.append(similarity,score)
# 			similarity.append(score)
# 	return sorted(similarity)

# filename1 = 'Img001_diffuse.tif'
# filename2 = 'Img002_diffuse.tif'
# img1 = cv2.imread(filename1)
# img2 = cv2.imread(filename2)
# corners1 = get_harris_position(img1)
# corners2 = get_harris_position(img2)
# print(np.shape(corners1))
# # print(corners2)
# # show_corner(img1,corners1)
# # windows1 = np.empty([11,11],np.float64)
# # windows2 = np.empty([11,11],np.float64)
# # patch1,patch2 = get_patch_12(corners1,corners2,11)
# # # print("patch1",np.shape(patch1),"patch2",np.shape(patch2))
# # similarity = []
# # ssd_score = ssd(patch1,patch2,11)
# # print(ssd_score,np.shape(ssd_score))



#----------------------------------------------
filename1 = 'Img001_diffuse.tif'
filename2 = 'Img002_diffuse.tif'
img1 = cv.imread(filename1)
img2 = cv.imread(filename2)
gray1 = cv.cvtColor(img1,cv.COLOR_BGR2GRAY)
gray2 = cv.cvtColor(img2,cv.COLOR_BGR2GRAY)
gray1 = np.float32(gray1)
gray2 = np.float32(gray2)
#dst=cv.cornerHarris(src, blockSize, ksize, k[, dst[, borderType]])
dst1 = cv.cornerHarris(gray1,2,7,0.04) 
dst2 = cv.cornerHarris(gray2,2,7,0.04) 
#result is dilated for marking the corners, not important
dst1 = cv.dilate(dst1,None)
dst2 = cv.dilate(dst2,None)
print(type(dst2))
# Threshold for an optimal value, it may vary depending on the image.
img1[dst1>0.001*dst1.max()] = [0,0,255]
img2[dst2>0.001*dst2.max()] = [0,0,255]
# to plot original image and the LoG
plt.subplot(121),plt.imshow(img1,cmap = "gray"),plt.title('image1')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(img2,cmap = "gray"),plt.title('image2')
plt.xticks([]), plt.yticks([])
plt.show()
# cv.imwrite('harris_test.png',img1)
# cv.imshow('dst',img)
# if cv.waitKey(0) & 0xff == 27:
#     cv.destroyAllWindows()
#---------------------------detecting part-----------------
# to define a rectangle of size N*N that will be used to define patches around each keypoint:







# class Point:


# for i,j in corners:
# 	p1 = Point(i - n/2, j - n/2)
# 	p2 = Point(i + n/2, j + n/2) 
# 	Rect (p1,p2)




#-----------------------------------------------
# import sys , os
# import cv
# import numpy as np


# #Window size and Thresholds
# for n in [5,7,9,13,23]:
# 	patch_size = n * n
# NCC WINDOW SIZE = 53
# SSD THRESHOLD = 1100
# SSD RATIO THRESHOLD = 0.8
# NCC THRESHOLD = 0.8
# NCC RATIO THRESHOLD = 1.2

# #Creates the feature descriptor for each interest point
# def getFeatures (R,image,features,threshold):
# global TOTAL_FEATURES
# global size
# size = patch_size
# feature number = 0
# for i in range (int(size/2),image.width−int(size/2)):
# 	for j in range(int(size/2),image.height-int(size/2):
# 		if cv.mGet(R,i,j) > threshold:
# 			if feature_number < TOTAL_FEATURES:
# 				feature_number = feature_number + 1
# 				features.append(Feature(i,j))
# 				print("number of features:",feature_number)


# i f cv .mGet(R, i , j ) > threshold : #threshold on the corner response to prune
# i n t e r e s t points
# i f feature number < TOTAL FEATURES:
# feature number = feature number + 1
# f e a tur e s . append ( Feature ( i , j ) ) #adds the f eatur e to the f eatur e l i s t
# print ”number of f eatur e s :” , feature number


# # Computes the SSD Score
# def ssd (f1,f2):
# global size
# #subtracts f2 from f1
# sub f 1 f 2 = cv . CreateMat ( s ize , s ize , cv .CV 64FC1)
# cv . Sub( f1 , f2 , s u b f 1 f 2 )
# #square and add
# f 1 f 2 s q u a r e = cv . CreateMat ( s ize , s ize , cv .CV 64FC1)
# cv .Pow( sub f 1 f 2 , f1 f 2 square , 2)
# s c o r e = cv .Sum( f 1 f 2 s q u a r e )
# return score [ 0 ] / ( s i z e ∗ s i z e )

# ------------------------------------------------------------
img1 = cv.imread('Img001_diffuse.tif',0)
img2 = cv.imread('Img002_diffuse.tif',cv.IMREAD_GRAYSCALE)
# print(img1,img2)
# Initiate SIFT detector
sift = cv.xfeatures2d.SIFT_create()
# sift = cv.SIFT()

# find the keypoints and descriptors with SIFT
keypoint1, descriptor1 = sift.detectAndCompute(img1,None)
keypoint2, descriptor2 = sift.detectAndCompute(img2,None)

# use Brute-Force matcher with parameters cv.NORM_L2
bf = cv.BFMatcher(cv.NORM_L2)
# returns k pairs of matches
# matches = bf.knnMatch(descriptor1,descriptor2,k=2)
# print(matches)

# #ssd ?
# # ssd = cv.matchTemplate(descriptor1,descriptor2,method = 0)
# # returns the best match
matches = bf.match(descriptor1,descriptor2)

# # iterate through the matches and apply a ratio test to  filter out matches that do not satisfy a condition.
matches = sorted(matches, key = lambda x:x.distance)

# # Apply ration test
# choice1 = []
# for a,b in matches:
#     if a.distance ** 2 < 0.70 * (b.distance) ** 2:
#     	# print(m.distance)
#     	choice1.append([a])
# # print(choice1,type(choice1))

# cv.drawMatchesKnn expects list of lists as matches.
img3 = cv.drawMatchesKnn(img1,keypoint1,img2,keypoint2, matches[:40],img2,flags=2)
# # print(type(img3),img3)
plt.imshow(img3),plt.show()
# # cv.imshow('matching-test', img3)
# # while True:
# # 	key = cv.waitKey(1)
# # 	if key == 27: 
# # 		break
#------------------------------

# import numpy as np
# import cv
# from matplotlib import pyplot as plt
# img1 = cv.imread('Img001_diffuse.tif',0)
# img2 = cv.imread('Img002_diffuse.tif',cv.IMREAD_GRAYSCALE)
# orb = cv2.ORB_create()
# # print(orb)
# kp1, des1 = orb.detectAndCompute(img1,None)
# # print(kp1)
# print("imagevalues{}".format(img1))
# print("descrptorvalues{}".format(des1))
# kp2, des2 = orb.detectAndCompute(img2,None)
# bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# matches = bf.match(des1,des2)
# matches = sorted(matches, key = lambda x:x.distance)
# img3 = cv2.drawMatches(img1,kp1,img2,kp2, matches[:1000],img2,flags=2)
# plt.imshow(img3),plt.show()