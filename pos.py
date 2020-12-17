#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 13:53:21 2020

@author: cory
"""
import yaml
import cv2
import numpy as np
import math

cap = cv2.VideoCapture(0)

with open("calibmat.yml", 'r') as f:
    var = yaml.safe_load(f)

mtx = var['K']['data']
dist = var['D']['data']

mtx = np.asmatrix(mtx)
mtx.resize(3,3)
print(mtx)
dist = np.asmatrix(dist)
dist.resize(1,5)
print(dist)

s1 = 400
s2 = 800

while(True):
    ret,frame = cap.read()
    #frame = cv2.imread("1.jpg")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blurred, 60, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C)[1]
  
    # findcontours 
    cnts = cv2.findContours(thresh, cv2.RETR_LIST,  
                    cv2.CHAIN_APPROX_SIMPLE)[-2]
    for cnt in cnts:
        if s1<cv2.contourArea(cnt) <s2: 
            x = []
            y = []
            for i in cnt:
                point = i[0]
                x.append(point[0])
                y.append(point[1])
            xC = math.ceil(sum(x) / len(x))
            yC = math.ceil(sum(y) / len(y))
            center_coordinates = (xC, yC)
            frame = cv2.circle(frame, center_coordinates, 2, (0, 0, 255), 2)
            
            cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 2)
    cv2.imshow("frame", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()