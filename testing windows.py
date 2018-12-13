import numpy as np
import cv2 
from matplotlib import pyplot as plt

windows1= [[[1,3,4],[5,6,6],[7,8,7]]]
windows2= [[[1,3,4],[5,4,6],[7,8,7]]]

# score = [np.sum((window1-window2)**2) for window1 in windows1 for window2 in windows2]
# print(score)

def distance(p1,p2):
	p1_sub_p2 = np.subtract(p1,p2)
	p1p2square = np.power(p1_sub_p2,2)
	score = np.sum(p1p2square)
	return score
print(distance(windows1,windows2))


Ok guys, I am running into some difficulties with my code. 
Particularly with regards to the computation of the SSD 
of the windows/patches. This is the error code that I get:

ValueError: operands could not be broadcast together with shapes (817,11,11,3) (933,)

For the list "windows1", we have (817, 11, 11, 3), which means that there are
817 windows/patches with (N) sizes 11x11 (3 is just the color channel).
For the list "windows2" however, by using the same function, I get (933,). This means
that, for some reason, there is no size information attached to the patches in this list.
I'm not sure why. 
I tried to look at the sizes of the individual windows in "windows2" separately, and
found that some have sizes 11x11, some 11x9. Per haps this could be the origin of the problem, but am not sure how to work around this problem.

ValueError: operands could not be broadcast together with shapes (817,11,11,3) (933,11,11,3)

This is the function I'm working on:

def ssd(p1,p2):
    similarity = []
    p1_sub_p2 = np.subtract(p1,p2)
    p1p2square = np.power(p1_sub_p2,2)
    score = np.sum(p1p2square)
    similarity.append(score)
    return similarity





   
    # bf = cv2.BFMatcher(cv2.NORM_L2)
    # SSD = bf ** 2
    # matches = bf.knnMatch(window1,window2,k=2) for 

# # img = np.array(range(50)).reshape(5,10)
# # corner1 = np.array([2,3])

# # def extract_window(matrix, corner,n):
# #     x = corner[0]
# #     y = corner[1]
# #     size_n = int(n/2)
# #     window = matrix[(x-n):(x+n+1), (y-n):(y+n+1)]
# #     return corner,window

# # # plt.imshow(extract_window(img,corner1,5))
# # # plt.show()

# # print(extract_window(img, corner1,5))

# #testing ssd-------------------------------------------
# def ssd(f1,f2,n):
# 	# #substracts patch2 from patch1:
# 	# sub_p1_p2 = np.array((n,n,3),np.float64)
# 	# cv2.Sub(f1,f2,sub_p1_p2)
# 	# #square and add:
# 	# square_f1_f2 = np.array((n,n,3),np.float64)
# 	# cv2.Pow(sub_f1_f2,square_f1_f2,2)
# 	# score = cv2.Sum(square_f1_f2)
# 	# return score[0]/(n*n)
# 	for each_f1 in f1:
# 		for each_f2 in f2:
# 			score = np.sum((f1-f2)**2)
# 			similarity.append(score)
# 			# similarity
# 	return sorted(similarity)

# f1 = np.array(range(363)).reshape(11,11,3)
# f2 = np.array(range(363)).reshape(11,11,3)
# similarity = []
# print(ssd(f1,f2,11))