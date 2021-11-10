#!/usr/bin/env python
# coding: utf-8

# In[46]:


import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[47]:



img=cv2.imread('potato.png')
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.imshow(img)


# In[48]:


#creating binary image
lower=np.array([30,30,30])
higher=np.array([250,250,250])


# In[49]:


mask=cv2.inRange(img,lower,higher)


# In[50]:


mask.shape


# In[51]:


plt.imshow(mask,'gray')


# In[52]:


cont,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)


# In[53]:


cont_img=cv2.drawContours(img,cont,-1,255,3)


# In[54]:


plt.imshow(cont_img)


# In[55]:


c=max(cont,key=cv2.contourArea)


# In[56]:


x,y,h,w=cv2.boundingRect(c)


# In[57]:


cv2.rectangle(img,(x,y),(x+w,y+w),(0,255,0),5)
plt.imshow(img)


# In[ ]:


#import dependencies
import cv2
import numpy as np


image = cv2.imread('test1.jpg')
image= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# performing the edge detetcion
gradients_sobelx = cv2.Sobel(image, -1, 1, 0)
gradients_sobely = cv2.Sobel(image, -1, 0, 1)
gradients_sobelxy = cv2.addWeighted(gradients_sobelx, 0.5, gradients_sobely, 0.5, 0)

gradients_laplacian = cv2.Laplacian(image, -1)

canny_output = cv2.Canny(image, 80, 150)

cv2.imshow('Sobel x', gradients_sobelx)
cv2.imshow('Sobel y', gradients_sobely)
cv2.imshow('Sobel X+y', gradients_sobelxy)
cv2.imshow('laplacian', gradients_laplacian)
cv2.imshow('Canny', canny_output)
cv2.waitKey()


# In[ ]:




