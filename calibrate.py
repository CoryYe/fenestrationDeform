#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 18:12:47 2020

@author: cory
"""
import cv2
import numpy as np

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((6*8,3), np.float32)
objp[:,:2] = np.mgrid[0:8,0:6].T.reshape(-1,2)

objp = objp * 2.5/100
objpoints = [] 
imgpoints = [] 

images = [1,2,3,4,5,6,7,8,9,10,11,12]

for fname in images:
    print(fname)
    img = cv2.imread(str(fname)+'.jpg')
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, (8,6),None)

    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)

        img = cv2.drawChessboardCorners(img, (8,6), corners2,ret)
        cv2.imshow('img',img)
        cv2.waitKey(500)

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
path = 'calibmat.yaml'
cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
cv_file.write("camera", mtx)
cv_file.write("distort", dist)
cv_file.release()

cv2.destroyAllWindows()