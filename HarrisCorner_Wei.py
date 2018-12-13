
# coding: utf-8

# In[78]:


import numpy as np  
from skimage.feature import corner_harris, corner_peaks  , corner_subpix
from skimage.color import rgb2gray  
import matplotlib.pyplot as plt  
import skimage.io as io  
import cv2


# In[3]:


img1=io.imread('Img001_diffuse.tif')
img2=io.imread('Img002_diffuse.tif')
img9=io.imread('Img009_diffuse.tif')
img1=rgb2gray(img1)
img2=rgb2gray(img2)
img9=rgb2gray(img9)
plt.rcParams['figure.figsize'] = (10.0, 10.0)


# http://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.peak_local_max   see parameters

# In[63]:


corner1 = corner_peaks(corner_harris(img1), min_distance=1) # corner_harris: compute harris corner;   corner_peaks: find corners 
subpix1 = corner_subpix(img1, corner1, window_size=10) # corner_subpix: subpixel position of corners.

fig, ax = plt.subplots()
ax.imshow(img1, cmap=plt.cm.gray)
ax.plot(corner1[:, 1], corner1[:, 0], '.b', markersize=15)
ax.plot(subpix1[:, 1], subpix1[:, 0], '+r', markersize=15)
plt.show()


# In[47]:


corner2 = corner_peaks(corner_harris(img2), min_distance=1) # corner_harris: compute harris corner;   corner_peaks: find corners 
subpix2 = corner_subpix(img2, corner2, window_size=10) # corner_subpix: subpixel position of corners.

fig, ax = plt.subplots()
ax.imshow(img2, cmap=plt.cm.gray)
ax.plot(corner2[:, 1], corner2[:, 0], '.b', markersize=15)
ax.plot(subpix2[:, 1], subpix2[:, 0], '+r', markersize=15)
plt.show()


# In[44]:


corner9 = corner_peaks(corner_harris(img9), min_distance=1) # corner_harris: compute harris corner;   corner_peaks: find corners 
subpix9 = corner_subpix(img9, corner9, window_size=10) # corner_subpix: subpixel position of corners.

fig, ax = plt.subplots()
ax.imshow(img9, cmap=plt.cm.gray)
ax.plot(corner9[:, 1], corner9[:, 0], '.b', markersize=15)
ax.plot(subpix9[:, 1], subpix9[:, 0], '+r', markersize=15)
plt.show()


# In[74]:


def get_harris(img,k = 0.04,threshold = 0.01):
    img=rgb2gray(img)
    height, width = np.shape(img)
    grandfx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
    grandfy = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)  # compute the X direction and Y direction gradient 
    
    Ix2 = grandfx**2
    Iy2 = grandfy**2
    Ixy = grandfx*grandfy
    
    Ix2 = grandfx**2
    Iy2 = grandfy**2
    Ixy = grandfx*grandfy
    
    A = cv2.GaussianBlur(Ix2,(5,5),2)  # eliminate the noises
    B = cv2.GaussianBlur(Iy2,(5,5),2) 
    C = cv2.GaussianBlur(Ixy,(5,5),2)
    
    M = [np.array([[A[i, j], C[i, j]], #compute M
               [C[i, j], B[i, j]]]) for i in range(height) for j in range(width)]

    D, T = list(map(np.linalg.det, M)), list(map(np.trace, M))  #computer det(M) and trace(M)
    R = np.array([d-k*t**2 for d, t in zip(D, T)]) #computer R
    R_max = np.max(R)
    R = R.reshape(height, width)
    
    count = 0
    corners = []
    for i in range(1,height-1):  #computer whether Rij is the maximum value in a certain area and it is greater than the threshold*R_max
        for j in range(1,width-1):
            if R[i, j] > R_max*threshold:
                if R[i, j] == np.max(R[i-1:i+2, j-1:j+2]):
                    corners.append([i,j])
                    count += 1
                
    corners = np.array(corners) #position of corners
    
    fig, ax = plt.subplots()  # show corners
    ax.imshow(img, cmap=plt.cm.gray)
    ax.plot(corners[:, 1], corners[:, 0], '.b', markersize=8)
    plt.show()


# In[75]:


get_harris(img1)


# In[76]:


get_harris(img2)


# In[77]:


get_harris(img9)

