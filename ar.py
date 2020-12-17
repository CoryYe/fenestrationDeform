#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 15:41:07 2020

@author: cory
"""
import yaml
import numpy as np
import cv2
import cv2.aruco as aruco
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math


with open("calibmat.yml", 'r') as f:
    var = yaml.safe_load(f)

mtx = var['K']['data']
dist = var['D']['data']

mtx = np.asmatrix(mtx)
mtx.resize(3,3)
#print(mtx)
dist = np.asmatrix(dist)
dist.resize(1,5)
#print(dist)

def rigid(x,y,z):
    #find rigid body transform to rotate 
    #iX = [0.225, 0.225, -0.225, -0.225]
    #iZ = [0.225, -0.225, -0.225, 0.225]
    #iY = [-0.5, -0.5, -0.5, -0.5]
    iPoints = np.array([[0.225, 0.225, -0.5], [0.225, -0.225, -0.5],
                        [-0.225,-0.225,-0.5],[-0.225,0.225,-0.5]])
    
    centroidQ = np.mean(iPoints, axis=0)
    
    points =  []
    for i in range(len(x)):
        point = [x[i],y[i],z[i]]
        points.append(point)
    
    points = np.array(points)
    p = points
    q = iPoints 
    rows, cols = p.shape;
    centroidP = np.mean(points, axis=0)
    
    p = np.transpose(p)
    q = np.transpose(q)
    x = np.zeros(shape=(3,rows))
    y = np.zeros(shape=(3,rows))
    #print(x)
    for i in range(rows):
        x1 = p[:,i] - centroidP
        y1 = q[:,i] - centroidQ
        ##print(y1)
        x1 = x1.reshape(-1, 1).ravel()
        y1 = y1.reshape(-1, 1).ravel()
        x[:,i]= x1
        y[:,i]= y1
    M = np.dot(x, np.transpose(y))
    
    u, s, vh = np.linalg.svd(M)

    rot =  u * np.transpose(vh)
    #only really want x and z rotation. y rotation find in next function
    #decompose into x, y, z rotation
    thetaX = math.atan2(rot[2][1],rot[2][2])
    thetaY = math.atan2(-rot[2][0],math.sqrt(rot[2][1]**2+rot[2][2]**2))
    #thetaZ = math.atan2(rot[1][0],rot[0][0])
    
    xRot = [[1,0,0],[0,math.cos(thetaX),-math.sin(thetaX)],
            [0,math.sin(thetaX),math.cos(thetaX)]]
    yRot = [[math.cos(thetaY),0,math.sin(thetaY)],[0,1,0],
            [-math.sin(thetaY),0,math.cos(thetaY)]]
    rot = np.dot(yRot,xRot)
    return rot

def rotate(x,y,z, ids):
    #find y rotation by aligning bottom to bottom.
    line = np.array([0.45,0,0])
    linePoint1 = np.array([-0.225,-0.225,-0.5])
    linePoint2 = linePoint1+line
    #ids for bottom edges
    ind1 = np.where(ids ==0)[0][0]
    ind2 = np.where(ids ==1)[0][0]
    point1 = np.array([x[ind1],y[ind1],z[ind1]])
    point2 = np.array([x[ind2],y[ind2],z[ind2]])
    
    translate = linePoint1 - point1
    
    point1 = translate+point1
    point2 = point2+translate
    #print(linePoint1)
    #print(linePoint2)
    line0 = np.array(point2-point1)
    #print(point1)
    #print(point2)
    line1 = np.array([np.transpose(linePoint1),np.transpose(linePoint2)])
    line2 = np.array([np.transpose(point1),np.transpose(point2)])
    run = line2[0][0] - line2[1][0]
    rise = line2[0][1] - line2[1][1]
    angle = np.arctan(rise/run)
    rot = [[math.cos(angle),-math.sin(angle),0],[math.sin(angle),math.cos(angle),0],
            [0,0,1]]
    #rot = np.dot(line1,np.linalg.inv(line2))
    
    return rot, translate
    


def consolidate(mat):
    #turns matrix to X,Y,Z values cause I'm lazy to change the draw function
    X = []
    Y = []
    Z = []
    for i in range(len(mat)):
        X.append(mat[i][0])
        Y.append(mat[i][1])
        Z.append(mat[i][2])
    return X,Y,Z

def distance(p1,p2):
    return math.sqrt((p2[0]-p1[0])**2+(p2[1]-p1[1])**2+(p2[2]-p1[2])**2)

def angle (p1,p2,p3):
    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = np.array(p3)
    vec1 = p2-p1
    vec2 = p3-p2
    d1  = distance(p2,p1)
    d2 = distance(p3,p2)

    return math.atan(1.5708 - math.acos(np.dot(vec1,vec2)/(d1*d2)))

def normStrain(x,y,z,ids):
    ind0 = np.where(ids ==0)[0][0]
    ind1 = np.where(ids ==1)[0][0]
    ind2 = np.where(ids ==2)[0][0]
    ind3 = np.where(ids ==3)[0][0]
    
    point0 = [x[ind0],y[ind0],z[ind0]]
    point1 = [x[ind1],y[ind1],z[ind1]]
    point2 = [x[ind2],y[ind2],z[ind2]]
    point3 = [x[ind3],y[ind3],z[ind3]]
    #print(point0)
    #print(point1)
    #print(point2)
    #print(point3)
    yy = (0.41-distance(point3,point0))/0.41
    xx = (0.45-distance(point1,point0))/0.45
    zz = 0
    xy = angle(point1,point0,point3)
    #print(xx,yy,zz,xy)
    return xx,yy,zz,xy
    
    
def deform(x,y,z,fix,ax,ids):
    #rotate the viewed projection and find the strain tensor 
    strain = np.zeros((3,3))
    iPoints = np.array([[0.225, 0.225, -0.5], [0.225, -0.225, -0.5],
                        [-0.225,-0.225,-0.5],[-0.225,0.225,-0.5]])
    points =  []
    for i in range(len(x)):
        point = [x[i],y[i],z[i]]
        points.append(point)
    points = np.array(points)
    
    #rotate around x and y
    rotated = np.dot(points,rigid(x,y,z))
    print(rigid(x,y,z))
    x,y,z = consolidate(rotated)
    #rotate around z separately
    
    rot, trans = rotate(x,y,z,ids)
    print("Rotation")
    print(np.array(rot))
    print("Translation")
    print(np.array(trans))
    rotated = np.dot(rotated,rot)
    x,y,z = consolidate(rotated)
    xx,yy,zz,xy = normStrain(x,y,z,ids)
    strain[0][0] = xx
    strain[1][1] = yy
    strain[2][2] = zz
    strain[0][1] = 0.5*xy
    strain[1][0] = 0.5*xy
    print("Deformation")
    print(strain)
    draw(x,y,z,fig,ax,ids)
    
    

def draw(x,y,z,fig,ax, ids):
    #fig.clear()
    newIds = []
    for i in ids:
        newIds.append(i[0])
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    newIds = np.array(newIds)
    inds = newIds.argsort()
    x = x[inds]
    y = y[inds]
    z = z[inds]
    
        
    x = np.append(x,x[0])
    y = np.append(y,y[0])
    z = np.append(z,z[0])
    
    #z = [-i for i in z]
    p = ax.scatter3D(x,y,z);
    c = ax.scatter3D(0,0,0, color = 'b');
    p1 = ax.plot(x,y,z, color='r')
    
    ax.set_xlim(-0.3,0.3)
    ax.set_zlim(0,1)
    ax.set_ylim(-0.3,0.3)
    
    plt.pause(0.02)
    #plt.show()
    plt.draw()
    line = p1.pop(0)
    #p.remove()
    #line.remove()



#cap = cv2.VideoCapture(0)
plt.ion()
fig = plt.figure()

ax = plt.axes(projection="3d")


while(True):
    #ret, frame = cap.read()
    frame = cv2.imread("c6.jpg")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_1000)
    arucoParameters = aruco.DetectorParameters_create()
    corners, ids, rejectedImgPoints = aruco.detectMarkers(
        gray, aruco_dict, parameters=arucoParameters)
    frame = aruco.drawDetectedMarkers(frame, corners)
    
    rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, 0.02, mtx, dist)
    print(ids)
    print(tvec)
    if rvec is not None:
        X = []
        Y = []
        Z = []
        for i in range(len(rvec)):
            r = rvec[i]
            t = tvec[i]
            X.append(t[0][0])
            Y.append(t[0][1])
            Z.append(t[0][2])
            #aruco.drawAxis(frame,mtx,dist,r,t,0.1)    
        draw(X,Y,Z,fig,ax,ids)
        deform(X,Y,Z,fig,ax,ids)

    cv2.imshow('Display', frame)
    break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
#cap.release()
#cv2.destroyAllWindows()