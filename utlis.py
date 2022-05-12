"""
编写函数

"""

import cv2
import numpy as np


#定义一个函数找到等高线
def getContours(img, cThr=[100,100], showCanny=False, minArea=1000, filter=0, draw=False):
    """
    首先将图片转换成灰度级
    :param img:
    :return:
    """
    #首先将图片转换成灰度级
    imgGrav = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #应用一些模糊高斯模糊
    imgBlur = cv2.GaussianBlur(imgGrav,(5,5),1)
    imgCanny = cv2.Canny(imgBlur,cThr[0],cThr[1])
    kernel = np.ones((5,5))
    imgDial = cv2.dilate(imgCanny,kernel,iterations=3)  #膨胀：将前景物体变大，理解成将图像断开裂缝变小（在图片上画上黑色印记，印记越来越小）
    imgThre = cv2.erode(imgDial,kernel,iterations=2)    #腐蚀：将前景物体变小，理解成将图像断开裂缝变大（在图片上画上黑色印记，印记越来越大）
    if showCanny:
        cv2.imshow('Canny',imgThre)

    #查找等高线函数
    contoers,hiearchy = cv2.findContours(imgThre, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    finalCountours = []
    for i in contoers:
        area = cv2.contourArea(i)
        if area > minArea:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02*peri, True)
            bbox = cv2.boundingRect(approx)
            if filter > 0:
                if len(approx) == filter:
                    finalCountours.append([len(approx),area,approx,bbox,i])
            else:
                finalCountours.append([len(approx),area,approx,bbox,i])

    finalCountours = sorted(finalCountours,key=lambda x:x[1], reverse=True)

    if draw:
        for con in finalCountours:
            cv2.drawContours(img,con[4],-1,(0,0,255),3)

    return img, finalCountours

def reorder(myPoints):
    print(myPoints.shape)
    myPointsNew = np.zeros_like(myPoints)
    myPoints = myPoints.reshape((4,2))
    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints,axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    return myPointsNew
def warpImg(img,points,w,h,pad=20):
    # print(points)
    points = reorder(points)
    pts1 = np.float32(points)
    pts2 = np.float32([[0,0],[w,0],[0,h],[w,h]])
    matrix = cv2.getPerspectiveTransform(pts1,pts2)
    imgWarp = cv2.warpPerspective(img,matrix,(w,h))
    #获得更清晰的图像
    imgWarp = imgWarp[pad:imgWarp.shape[0]-pad,pad:imgWarp.shape[1]-pad]

    return imgWarp



def findDis(pts1,pts2):
    return ((pts2[0]-pts1[0])**2 + (pts2[1]-pts1[1])**2)**0.5
