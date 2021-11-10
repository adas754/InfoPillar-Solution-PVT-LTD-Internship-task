#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import pandas as pd

img_path = r'colorpic.jpg'
img = cv2.imread(img_path)

# declaring global variables 
clicked = False
r = g = b = x_pos = y_pos = 0

# Reading csv file with pandas and giving names to each column
index = ["color", "color_name", "hex", "R", "G", "B"]
csv = pd.read_csv('colors.csv', names=index, header=None)


# function to calculate minimum distance from all colors and get the most matching color
def get_color_name(R, G, B):
    minimum = 10000
    for i in range(len(csv)):
        d = abs(R - int(csv.loc[i, "R"])) + abs(G - int(csv.loc[i, "G"])) + abs(B - int(csv.loc[i, "B"]))
        if d <= minimum:
            minimum = d
            cname = csv.loc[i, "color_name"]
    return cname


# function to get x,y coordinates of mouse double click
def draw_function(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        global b, g, r, x_pos, y_pos, clicked
        clicked = True
        x_pos = x
        y_pos = y
        b, g, r = img[y, x]
        b = int(b)
        g = int(g)
        r = int(r)


cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_function)

while True:

    cv2.imshow("image", img)
    if clicked:

        # cv2.rectangle(image, start point, endpoint, color, thickness)-1 fills entire rectangle
        cv2.rectangle(img, (20, 20), (750, 60), (b, g, r), -1)

        # Creating text string to display( Color name and RGB values )
        text = get_color_name(r, g, b) + ' R=' + str(r) + ' G=' + str(g) + ' B=' + str(b)

        # cv2.putText(img,text,start,font(0-7),fontScale,color,thickness,lineType )
        cv2.putText(img, text, (50, 50), 2, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        # For very light colours we will display text in black colour
        if r + g + b >= 600:
            cv2.putText(img, text, (50, 50), 2, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

        clicked = False

    # Break the loop when user hits 'esc' key
    if cv2.waitKey(20) & 0xFF == 27:
        break

cv2.destroyAllWindows()


# In[ ]:


import cv2
import numpy


#initial function for the callin of the trackbar
def hi(x):
	#only for referece
	print("")

#initialisation of the camera
cap = cv2.VideoCapture(0)
bars = cv2.namedWindow("bars")

cv2.createTrackbar("upper_hue","bars",110,180,hi)
cv2.createTrackbar("upper_saturation","bars",255, 255, hi)
cv2.createTrackbar("upper_value","bars",255, 255, hi)
cv2.createTrackbar("lower_hue","bars",68,180, hi)
cv2.createTrackbar("lower_saturation","bars",55, 255, hi)
cv2.createTrackbar("lower_value","bars",54, 255, hi)

#Capturing the initial frame for creation of background
while(True):
	cv2.waitKey(1000)
	ret,init_frame = cap.read()
	#check if the frame is returned then brake
	if(ret):
		break

# Start capturing the frames for actual magic!!
while(True):
	ret,frame = cap.read()
	inspect = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

	#getting the HSV values for masking the cloak
	upper_hue = cv2.getTrackbarPos("upper_hue", "bars")
	upper_saturation = cv2.getTrackbarPos("upper_saturation", "bars")
	upper_value = cv2.getTrackbarPos("upper_value", "bars")
	lower_value = cv2.getTrackbarPos("lower_value","bars")
	lower_hue = cv2.getTrackbarPos("lower_hue","bars")
	lower_saturation = cv2.getTrackbarPos("lower_saturation","bars")

	#Kernel to be used for dilation
	kernel = numpy.ones((3,3),numpy.uint8)

	upper_hsv = numpy.array([upper_hue,upper_saturation,upper_value])
	lower_hsv = numpy.array([lower_hue,lower_saturation,lower_value])

	mask = cv2.inRange(inspect, lower_hsv, upper_hsv)
	mask = cv2.medianBlur(mask,3)
	mask_inv = 255-mask 
	mask = cv2.dilate(mask,kernel,5)
	
	#The mixing of frames in a combination to achieve the required frame
	b = frame[:,:,0]
	g = frame[:,:,1]
	r = frame[:,:,2]
	b = cv2.bitwise_and(mask_inv, b)
	g = cv2.bitwise_and(mask_inv, g)
	r = cv2.bitwise_and(mask_inv, r)
	frame_inv = cv2.merge((b,g,r))

	b = init_frame[:,:,0]
	g = init_frame[:,:,1]
	r = init_frame[:,:,2]
	b = cv2.bitwise_and(b,mask)
	g = cv2.bitwise_and(g,mask)
	r = cv2.bitwise_and(r,mask)
	blanket_area = cv2.merge((b,g,r))

	final = cv2.bitwise_or(frame_inv, blanket_area)

	cv2.imshow("munai's Cloak",final)

	if(cv2.waitKey(3) == ord('q')):
		break;

cv2.destroyAllWindows()
cap.release()


# In[ ]:




