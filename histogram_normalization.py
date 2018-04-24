import cv2
import numpy as np
from PIL import Image

img = cv2.imread('16_right.jpeg',1)
ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
ycrcb[:,:,0] = cv2.equalizeHist(ycrcb[:,:,0])
hst_out = cv2.cvtColor(ycrcb, cv2.COLOR_HLS2BGR)
cv2.imwrite('16_right_norm.jpeg',hst_out)

# hsv = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
# img = cv2.merge((b,g,r))
# res = np.hstack((img,equ)) #stackin images side-by-side
# equ = cv2.equalizeHist(hsv)
# print(type(equ))
# # print(type(res))
# cv2.imwrite('16_right_norm.jpeg',equ)