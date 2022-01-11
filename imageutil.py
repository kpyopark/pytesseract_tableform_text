from __future__ import annotations
from sys import intern
from dataclasses import dataclass, field
import cv2
import numpy as np

PROGRESS_SHOW = True
MAX_PIXEL = 3508
MAX_WIDTH = 2480
MAX_HEIGHT = MAX_PIXEL
MIN_BOUNDARY_AREA = 100 * 80
MAX_BOUNDARY_AREA = 2400 * 1700 # Half of A4 size.


def debugShow(imgname, img, isShow=True):
    if isShow & PROGRESS_SHOW:
        cv2.namedWindow(imgname, cv2.WINDOW_NORMAL)
        cv2.imshow(imgname, img)
        cv2.waitKey(0)

def convertColor(img):
    tmp = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    tmp = cv2.threshold(tmp, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    return tmp

def debugShowContours(imgname, img, contours):
    if PROGRESS_SHOW:
        mask = np.copy(img)
        for c in contours:
            # get the bounding rect
            x, y, w, h = cv2.boundingRect(c)
            if w*h > MIN_BOUNDARY_AREA:
                cv2.rectangle(mask, (x, y), (x+w, y+h), (0, 0, 255), 1)
                debugShow(imgname, mask)

def drawContours(img, contours):
    copyimg = np.copy(img) * 0
    for contour in contours:
        x,y,w,h = cv2.boundingRect(contour)
        if w*h > MIN_BOUNDARY_AREA:
            cv2.rectangle(copyimg, (x,y),(x+w,y+h), (0,255,0), thickness=2)
    # debugShow('contours', copyimg)
    return copyimg

def drawLines(img, lines):
    imgcopy = np.copy(img) * 0
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(imgcopy,(x1,y1),(x2,y2),(255,255,255),2)
    # debugShow('drawlines', imgcopy)
    return imgcopy

def getHoughLinePoints(img, pixel, rho, theta, threshold):
    linesParams = cv2.HoughLines(img, pixel, rho, threshold, None, 0, 0)
    if linesParams is None:
        return []
    if linesParams.size <= 0:
        return []
    lines = []
    temp = []
    for rho, theta in linesParams[0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0+1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 -1000*(a))
        temp.append(np.asarray([x1,y1,x2,y2]))        
    lines.append(temp)


def resizeImage(img):
    height, width = img.shape
    print("height: {}, width : {}".format(height, width))
    sampling_method = cv2.INTER_LINEAR
    if height * width > MAX_WIDTH * MAX_HEIGHT :
        # This image needs downsampling
        sampling_method = cv2.INTER_LENEAR
    else :
        sampling_method = cv2.INTER_AREA
    
    if width > height :
        newheight = int(height * MAX_WIDTH / width)
        print("new h : {}, w : {}".format(newheight, MAX_WIDTH))
        tmp = cv2.resize(img, (MAX_WIDTH, newheight), interpolation=sampling_method)
    else :
        newwidth = int(width * MAX_HEIGHT / height)
        print("new h : {}, w : {}".format(newwidth, MAX_HEIGHT))
        tmp = cv2.resize(img, (newwidth, MAX_HEIGHT), interpolation=sampling_method)
    return tmp

def denoize(img):
    denoized = cv2.fastNlMeansDenoising(img, h=5, templateWindowSize=7, searchWindowSize=21)
    denoized = cv2.fastNlMeansDenoising(denoized, h=3, templateWindowSize=5, searchWindowSize=11)
    return denoized

def getHLineStretchedImage(img):
    debug = False
    blur = cv2.bilateralFilter(img, 9, 50, 50)
    debugShow('blur',blur, debug)
    kernelForStretchedLine = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    kernelForRemoveText = cv2.getStructuringElement(cv2.MORPH_RECT,(7,1))
    dilated = cv2.morphologyEx(blur, cv2.MORPH_DILATE, kernelForStretchedLine, iterations=1)
    thresh = cv2.threshold(dilated, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    debugShow('dilated', dilated, debug)
    eroded = cv2.morphologyEx(thresh, cv2.MORPH_ERODE, kernelForRemoveText, iterations=8)
    debugShow('erodedlines', eroded, debug)
    thresh = cv2.threshold(eroded, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    dilated = cv2.morphologyEx(eroded, cv2.MORPH_DILATE, kernelForRemoveText, iterations=3)
    debugShow('dilated', dilated, debug)
    thresh = cv2.threshold(eroded, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    debugShow('thresedlined', thresh,debug)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (100,1))
    detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    debugShow('detected_lines', detected_lines, debug)
    return detected_lines

def getVLineStretchedImage(img):
    debug = False
    blur = cv2.bilateralFilter(img, 9, 50, 50)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    kernelForRemoveText = cv2.getStructuringElement(cv2.MORPH_RECT,(1,5))
    dilated_img = cv2.morphologyEx(blur, cv2.MORPH_DILATE, kernel, iterations=1)
    debugShow('dilated_img', dilated_img, debug)
    thresh = cv2.threshold(dilated_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    debugShow('threshold', dilated_img, debug)
    #eroded = cv2.morphologyEx(thresh, cv2.MORPH_ERODE, kernelForRemoveText, iterations=3)
    #debugShow('eroded', dilated_img, debug)
    dilated_img = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    debugShow('MORPH_OPEN', dilated_img, debug)
    dilated_img = cv2.morphologyEx(dilated_img, cv2.MORPH_CLOSE, kernel, iterations=1)
    debugShow('MORPH_CLOSE', dilated_img, debug)
    textarea_img = getTextArea(img)
    final_img = dilated_img & ~ textarea_img
    debugShow('final_img', final_img, debug)
    return final_img

def getTextArea(img):
    debug = False
    blur = cv2.GaussianBlur(img, (9,9), 0)
    # thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    debugShow('blur',blur, debug)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    dilated_img = cv2.morphologyEx(blur, cv2.MORPH_DILATE, kernel, iterations=2)
    debugShow('dilated_img',dilated_img, debug)
    kernelForRemoveText = cv2.getStructuringElement(cv2.MORPH_RECT,(1,7))
    eroded = cv2.morphologyEx(dilated_img, cv2.MORPH_ERODE, kernelForRemoveText, iterations=2)
    debugShow('eroded',eroded, debug)
    dilated_img = cv2.morphologyEx(eroded, cv2.MORPH_DILATE, kernel, iterations=1)
    kernelForRemoveText = cv2.getStructuringElement(cv2.MORPH_RECT,(7,1))
    kernelForRemoveText2 = cv2.getStructuringElement(cv2.MORPH_RECT,(3,1))
    eroded = cv2.morphologyEx(dilated_img, cv2.MORPH_ERODE, kernelForRemoveText, iterations=2)
    eroded = cv2.morphologyEx(eroded, cv2.MORPH_ERODE, kernelForRemoveText2, iterations=2)
    thresh = cv2.threshold(eroded, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    dilated_img = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel, iterations=3)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(1,3))
    dilated_img = cv2.morphologyEx(dilated_img, cv2.MORPH_DILATE, kernel, iterations=3)
    debugShow('textarea',dilated_img, debug)
    return dilated_img


