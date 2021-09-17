import cv2
import numpy as np

#1
src1 = cv2.imread('./exit1.jpg') 
src1 = cv2.resize(src1, dsize=(0, 0),
                 fx=0.3, fy=0.2, interpolation=cv2.INTER_AREA)
src2 = src1.copy()

#2
gray = cv2.cvtColor(src1, cv2.COLOR_RGB2GRAY)
ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
binary = cv2.bitwise_not(binary)

contours, hierarchy = cv2.findContours(binary,
                                       cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

cv2.putText(src1, "Exit", (100,100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,255), 10)
for i in range(len(contours)):
    cv2.drawContours(src1, [contours[i]], 0, (255, 0, 0), 3)

#3
dst = cv2.medianBlur(src1, ksize=5)
kernel = cv2.getStructuringElement(shape = cv2.MORPH_RECT, ksize=(3,3))
erode = cv2.erode(dst, kernel, iterations=20)
dilate = cv2.dilate(erode, kernel, iterations=10)
erode2 = cv2.erode(dilate, kernel, iterations=10)

#4
hsv = cv2.cvtColor(erode2, cv2.COLOR_BGR2HSV)
lower_green = (70, 30, 100)
upper_green = (120, 255, 255)
mask1 = cv2.inRange(hsv, lower_green, upper_green)

#5
src1[mask1>0] = (0, 255, 255)
bit_and1 = cv2.bitwise_and(src1, src1, mask = mask1)
bit_and2 = cv2.bitwise_and(src2, src2, mask = mask1)

#6
cv2.imshow('src1', src1)
cv2.imshow('src2', src2)
cv2.imshow('mask1', mask1)
cv2.imshow('bit_and1', bit_and1)
cv2.imshow('bit_and2', bit_and2)

cv2.waitKey()
cv2.destroyAllWindows()
