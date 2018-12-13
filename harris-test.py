# https://docs.opencv.org/3.4/dc/d0d/tutorial_py_features_harris.html
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt 
filename = 'Img001_diffuse.tif'
img = cv.imread(filename)
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
gray = np.float32(gray)
#dst=cv.cornerHarris(src, blockSize, ksize, k[, dst[, borderType]])
dst = cv.cornerHarris(gray,2,15,0.04) 
#result is dilated for marking the corners, not important
dst = cv.dilate(dst,None)
# Threshold for an optimal value, it may vary depending on the image.
img[dst>0.001*dst.max()]=[0,0,255]
plt.imshow(img,cmap="gray")
plt.show()
# cv.imwrite('harris_test.png',img)
# cv.imshow('dst',img)
# if cv.waitKey(0) & 0xff == 27:
#     cv.destroyAllWindows()



#--------------------------------------matlab code
# from http://ros-developer.com/2017/12/14/harris-corner-detector-explained/

# im = imread('chessboard.jpg');
# im = double(rgb2gray(im))/255;

# sigma = 5;
# g = fspecial('gaussian', 2*sigma*3+1, sigma);
# dx = [-1 0 1;-1 0 1; -1 0 1];
# Ix = imfilter(im, dx, 'symmetric', 'same');
# Iy = imfilter(im, dx', 'symmetric', 'same');
# Ix2 = imfilter(Ix.^2, g, 'symmetric', 'same');
# Iy2 = imfilter(Iy.^2, g, 'symmetric', 'same');
# Ixy = imfilter(Ix.*Iy, g, 'symmetric', 'same');
# k = 0.04; % k is between 0.04 and 0.06
# % --- r = Det(M) - kTrace(M)^2 ---
# r = (Ix2.*Iy2 - Ixy.^2) - k*(Ix2 + Iy2).^2;

# norm_r = r - min(r(:));
# norm_r = norm_r ./ max(norm_r(:));
# subplot(1,2,1), imshow(im)
# subplot(1,2,2), imshow(norm_r)


# %hLocalMax = vision.LocalMaximaFinder;
# %hLocalMax.MaximumNumLocalMaxima = 100;
# %hLocalMax.NeighborhoodSize = [3 3];
# %hLocalMax.Threshold = 0.05;
# %location = step(hLocalMax, norm_r);
