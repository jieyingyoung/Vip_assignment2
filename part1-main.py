
# coding: utf-8

# In[82]:


import numpy as np  
from skimage.feature import corner_harris, corner_peaks, corner_subpix
from skimage.color import rgb2gray  
import matplotlib.pyplot as plt  
import skimage.io as io  
from skimage import img_as_float
import cv2
import sklearn.preprocessing


# In[6]:


img1=io.imread('Img001_diffuse.tif')
img2=io.imread('Img002_diffuse.tif')
img9=io.imread('Img009_diffuse.tif')

img1=rgb2gray(img1)
img2=rgb2gray(img2)
img9=rgb2gray(img9)

plt.rcParams['figure.figsize'] = (10.0, 10.0)


# In[7]:


def get_harris_position(img, knsize, k, threshold):
    img = rgb2gray(img)
    height, width = np.shape(img)
    
    # compute the X direction and Y direction gradient 
    # cv2.Sobel also smoothes the image w/ .gaussianBlur()
    grandfx = cv2.Sobel(img,cv2.CV_64F,1,0,knsize)
    grandfy = cv2.Sobel(img,cv2.CV_64F,0,1,knsize)
    
    A = grandfx**2
    B = grandfy**2
    C = grandfx*grandfy
    
    M = [np.array([[A[i, j], C[i, j]], #compute M
               [C[i, j], B[i, j]]]) for i in range(height) for j in range(width)]

    D, T = list(map(np.linalg.det, M)), list(map(np.trace, M))  #compute lists of det(M) and trace(M)
    R = np.array([d-k*t**2 for d, t in zip(D, T)]) #compute array of R-values
    R_max = np.max(R) # what is the biggest R
    R = R.reshape(height, width) # reshape array back to image size
    
    #compute whether Rij is the maximum value in a certain area and it is greater than the threshold*R_max
    corners = []
    for i in range(1,height-1):
        for j in range(1,width-1):
            if R[i, j] > R_max*threshold:
                if R[i, j] == np.max(R[i-1:i+2, j-1:j+2]):
                    corners.append([i,j])
    
    #position of corners
    corners = np.array(corners)
    
    return corners


# In[46]:


position1 = get_harris_position(img1,5,k=0.05,threshold=0.01)
position2 = get_harris_position(img1,5,k=0.15,threshold=0.01)
position3 = get_harris_position(img1,5,k=0.2,threshold=0.01)


# In[9]:


position4 = get_harris_position(img1,5,k=0.05,threshold=0.1)
position5 = get_harris_position(img1,5,k=0.15,threshold=0.1)
position6 = get_harris_position(img1,5,k=0.2,threshold=0.1)


# In[50]:


len(position1)


# In[51]:


len(position2)


# In[52]:


len(position3)


# In[13]:


len(position4)


# In[14]:


len(position5)


# In[15]:


len(position6)


# In[55]:


position_max_k = get_harris_position(img1, 5, k=0.249, threshold=0.01)
len(position_max_k)


# In[41]:


position_min_k = get_harris_position(img1, 5, k=0.00001, threshold=0.01)
len(position_min_k)


# In[35]:


def show_corner(img,corners, title):
    fig, ax = plt.subplots()  # show corners
    ax.imshow(img, cmap=plt.cm.gray)
    ax.plot(corners[:, 1], corners[:, 0], '.b', markersize=8)
    plt.title(title)
    plt.show()


# In[47]:


fig, ax = plt.subplots()  # show corners
ax.imshow(img1, cmap=plt.cm.gray)
ax.plot(position1[:, 1], position1[:, 0], '.b', markersize=8)
plt.title('k=0.05,threshold=0.01')
plt.show()


# In[48]:


fig, ax = plt.subplots()  # show corners
ax.imshow(img1, cmap=plt.cm.gray)
ax.plot(position2[:, 1], position2[:, 0], '.b', markersize=8)
plt.title('k=0.15,threshold=0.01')
plt.show()


# In[49]:


fig, ax = plt.subplots()  # show corners
ax.imshow(img1, cmap=plt.cm.gray)
ax.plot(position3[:, 1], position3[:, 0], '.b', markersize=8)
plt.title('k=0.2,threshold=0.01')
plt.show()


# In[20]:


fig, ax = plt.subplots()  # show corners
ax.imshow(img1, cmap=plt.cm.gray)
ax.plot(position4[:, 1], position4[:, 0], '.b', markersize=8)
plt.title('k=0.05,threshold=0.1')
plt.show()


# In[21]:


fig, ax = plt.subplots()  # show corners
ax.imshow(img1, cmap=plt.cm.gray)
ax.plot(position5[:, 1], position5[:, 0], '.b', markersize=8)
plt.title('k=0.15,threshold=0.1')
plt.show()


# In[22]:


fig, ax = plt.subplots()  # show corners
ax.imshow(img1, cmap=plt.cm.gray)
ax.plot(position6[:, 1], position6[:, 0], '.b', markersize=8)
plt.title('k=0.2,threshold=0.1')
plt.show()


# In[56]:


show_corner(img1,position_max_k, 'k=0.249,threshold=0.01')


# In[43]:


show_corner(img1,position_min_k, 'k=0.0001,threshold=0.01')


# In[66]:


#lst = list()
#for i, el in enumerate(range(20, 1, -1)):
#    lst.append([i, el])

lst = [[1,4],[4,5],[4,3],[5,9], [2, 10], [-1, 5]]
print(lst)
print("\n")
print(sorted(lst))


# In[58]:


help(sorted)


# In[71]:


help(cv2.drawMatches)
print(position1)
lines = cv2.drawMatches(img1, position1, img1, position2, position1, img1)


# In[72]:


help(cv2.KeyPoint)


# In[80]:


lst = [1,2,3,4,5,6,7]
new_lst = []
def func(x):
    return (x*x)

new_lst = [func(a) for a in lst]
print(new_lst)


# In[84]:


C = np.array([[1,2,3,4],[5,6,7,8]])
print(np.var(C))


# In[87]:


C_norm = sklearn.preprocessing.normalize(C)
print(C_norm, np.mean(C_norm), np.var(C_norm))

