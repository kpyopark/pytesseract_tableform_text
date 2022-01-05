from __future__ import annotations
from dataclasses import dataclass
from typing import List
from sys import intern
import cv2
import numpy as np
import math
from typing import List, Tuple
from imageutil import *

IMG_FILE = 'contract_house.png'
PROGRESS_SHOW = True
LINE_MIN_WIDTH = 3
MAX_PIXEL = 3508
MAX_WIDTH = 2480
MAX_HEIGHT = MAX_PIXEL

def convertColor(img):
    tmp = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    tmp = cv2.threshold(tmp, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    return tmp

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

def filteredAngle(angles):
    npangles = np.array(angles)
    mean = np.mean(npangles)
    std = np.std(npangles)
    dfm = abs(npangles-mean)
    max_deviation = 2
    not_outliers = dfm < max_deviation * std
    std_angles = npangles[not_outliers]
    if(len(std_angles) > 0):
        return np.median(std_angles)
    else:
        return 0

def getSkewnessFromVlines(img, vlines):
    angles = []
    print('vlines:{}'.format(vlines))
    for line in vlines:
        for x1,y1,x2,y2 in line:
            if abs(y2-y1) == 0:
                angles.append(0)
                continue
            if abs(x2-x1) == 0:
                angles.append(0)
                continue
            yd = y1 - y2
            xd = x2 - x1
            if xd == 0:
                continue
            angle = math.degrees(math.atan(yd/xd))
            angle = 90 - angle
            print(angle)
            if abs(angle) > 4:
                continue
            angles.append(angle)
    print('v angles:{}'.format(angles))
    return filteredAngle(angles)

def getSkewnessFromLines(img, lines):
    angles = []
    threshold_pixel = 12
    h_milestonpoints = []
    hlines = {}
    for line in lines:
        for x1,y1,x2,y2 in line:
            if abs(y2-y1) == 0 :
                angles.append(0)
                continue
            if abs(x2-x1) == 0 :
                angles.append(0)
                continue
            yd = y2-y1
            xd = x2-x1
            angle = math.atan(yd/xd)*180/math.pi
            if angle > 4 :
                continue
            if angle < -4 :
                continue
            angles.append(angle)
            if (abs(h_milestonpoints - y1) < threshold_pixel).sum() == 0:
                h_milestonpoints.append(y1)
                newlist = []
                newlist.append((x1,y1,x2,y2))
                hlines[y1] = newlist
            else:
                idx = [i for i,v in enumerate(abs(h_milestonpoints - y1) < threshold_pixel) if v > 0][0]
                targetlist = hlines[h_milestonpoints[idx]]
                targetlist.append((x1,y1,x2,y2))

    print('angles:{}'.format(angles))

    ## Calculate median value from the longest hline
    anglesfromlline = []
    print('horizontal lines for skewness detecting: {}'.format(hlines))
    for linepaths in hlines.values():
        linepaths.sort(key=lambda line:line[0])
        x1 = linepaths[0][0] # x1
        y1 = linepaths[0][1]
        linepaths.sort(key=lambda line:line[2])
        x2 = linepaths[-1][2]
        y2 = linepaths[-1][3]
        yd = y2-y1
        xd = x2-x1
        angle = math.atan(yd/xd)*180/math.pi
        anglesfromlline.append(angle)
    print(anglesfromlline)

    ## Calculate via HoughLine
    sorted(h_milestonpoints)
    h_milestonpoints = np.sort(h_milestonpoints)
    heights = np.diff(h_milestonpoints)
    angleFromHoughLine = None
    if(len(heights)>0):
        average_span_height = np.median(heights)
        print('avg height:{}'.format(average_span_height))
        threshold = 10
        std_line_index = int(np.argmin(abs(heights - average_span_height) < threshold, axis=0))
        std_line_ypoint = h_milestonpoints[std_line_index]
        largest_element = hlines[std_line_ypoint]
        
        print('largest elements:{}'.format(largest_element))
        x_values = np.array([])
        x_values = np.append(x_values,sorted(set([item[0] for item in largest_element])))
        x_values = np.append(x_values,sorted(set([item[2] for item in largest_element])))
        y_values = np.array([])
        y_values = np.append(y_values,sorted(set([item[1] for item in largest_element])))
        y_values = np.append(y_values,sorted(set([item[3] for item in largest_element])))
        sorted(x_values)
        sorted(y_values)
        print(x_values)
        print(y_values)
        x1 = int(x_values[0])
        x2 = int(x_values[-1])
        y1 = int(y_values[0])
        y2 = int(y_values[-1])
        if x1 > x2 :
            x1,x2 = x2,x1
        if y1 > y2 :
            y1,y2 = y2,y1
        print('largest elements ROI: {},{},{},{}'.format(x1,x2,y1,y2))        
        roi = img[y1:y2, x1:x2]
        debugShow('lineroi', roi)

        anglefromhline = []
        houghlines = cv2.HoughLines(roi,1,np.pi/180 / 10,int(abs(x2-x1)*9/10))
        if houghlines is not None :
            for oneline in houghlines:
                rho, theta = oneline[0]
                degree = math.degrees(theta)
                print('rho, theta, skewness: {}, {}'.format(rho, degree, 90-degree))
                angleFromHoughLine = (90-degree) * -1
                anglefromhline.append(angleFromHoughLine)
                angleFromHoughLine = filteredAngle(anglefromhline)

    angleFromShortPaths = filteredAngle(angles)
    angleFromLongestPaths = filteredAngle(anglesfromlline)
    if angleFromHoughLine is None:
        angleFromHoughLine = 0.0

    print('s.angle, l.angle, h.angle: {}, {}, {}'.format(angleFromShortPaths, angleFromLongestPaths, angleFromHoughLine))

    if abs(angleFromLongestPaths) > abs(angleFromShortPaths):
        if abs(angleFromHoughLine) < abs(angleFromLongestPaths):
            return angleFromHoughLine
        else:
            return angleFromLongestPaths
    else:
        return angleFromShortPaths

def get_median_angle(binary_image):
    # applying morphological transformations on the binarised image
    # to eliminate maximum noise and obtain text ares only
    # boxes = getLineDetection(binary_image)
    erode_otsu = cv2.erode(binary_image,np.ones((7,7),np.uint8),iterations=1)
    negated_erode = ~erode_otsu
    debugShow('erode_otsu', negated_erode)
    opening = cv2.morphologyEx(negated_erode,cv2.MORPH_OPEN,np.ones((5,5),np.uint8),iterations=2)
    debugShow('opening', opening)
    double_opening = cv2.morphologyEx(opening,cv2.MORPH_OPEN,np.ones((3,3),np.uint8),iterations=5)
    debugShow('double_opening', double_opening)
    double_opening_dilated_3x3 = cv2.dilate(double_opening,np.ones((3,3),np.uint8),iterations=4)
    debugShow('dilated_3x3', double_opening_dilated_3x3)
    # finding the contours in the morphologically transformed image
    contours_otsu,_ = cv2.findContours(double_opening_dilated_3x3,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    # debugShowContours('contours', double_opening_dilated_3x3, contours_otsu)
    # iniatialising the empty angles list to collet the angles of each contour
    angles = []

    # obtaining the angles of each contour using a for loop
    for cnt in range(len(contours_otsu)):
        # the last output of the cv2.minAreaRect() is the orientation of the contour
        rect = cv2.minAreaRect(contours_otsu[cnt])

        # appending the angle to the angles-list
        angles.append(rect[-1])
        
    # finding the median of the collected angles
    angles.sort()
    median_angle = np.median(angles)

    # returning the median angle
    return median_angle

# funtion to correct the median-angle to give it to the cv2.warpaffine() function
def corrected_angle(angle):
        if 0 <= angle <= 90:
            corrected_angle = angle - 90
        elif -45 <= angle < 0:
            corrected_angle = angle - 90
        elif -90 <= angle < -45:
            corrected_angle = 90 + angle
        return corrected_angle

def rotate(img, angle):
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    # print('center and radian:{}, {}', center, math.radians(angle))
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

# https://github.com/TarunChakitha/OCR/blob/master/OCR.py
# https://stackoverflow.com/questions/45322630/how-to-detect-lines-in-opencv
def getLines(img, low_threshold, min_line_length, line_gap, granulity):
    rho = 1               # distance resolution in pixels of the Hough grid
    theta = np.pi / 180 / granulity  # angular resolution in radians of the Hough grid
    lines = cv2.HoughLinesP(img, rho, theta, low_threshold, np.array([]), min_line_length, line_gap)
    return lines

def filterHVLines(lines, standard_degree):
    hlines = []
    for line in lines:
      for x1, y1, x2, y2 in line:
        degree = math.degrees(math.atan2(y1-y2, x2-x1))
        degree = degree - standard_degree
        if abs(degree) < 5:
          hlines.append(line)
    return hlines

def getHLines(img, low_threshold, min_line_length, line_gap, granulity):
    lines = getLines(img, low_threshold, min_line_length, line_gap, granulity)
    return filterHVLines(lines, 0)

def getVLines(img, low_threshold, min_line_length, line_gap, granulity):
    lines = getLines(img, low_threshold, min_line_length, line_gap, granulity)
    return filterHVLines(lines, 90)

def getAverageAngles(standard_degree, lines):
  filtered = []
  filteredlines = []
  for line in lines:
    for x1,y1,x2,y2 in line:
      # print('x1,y1,x2,y2:{},{},{},{}'.format(x1,y1,x2,y2))
      degree = math.degrees(math.atan2(y1-y2, x2-x1))
      degree = degree - standard_degree
      if abs(degree) < 5:
        filtered.append(degree)
        filteredlines.append(line)
  return filteredAngle(filtered), filteredlines

def drawLines(img, lines):
    imgcopy = np.copy(img) * 0
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(imgcopy,(x1,y1),(x2,y2),(255,255,255),2)
    # debugShow('drawlines', imgcopy)
    return imgcopy

def deskew(img):
  debug = False
  low_threshold = 30
  line_length_unit = int(img.shape[1] / 10)
  angle = 0
  filteredlines = []
  for multiple in reversed(range(9)):
    min_line_length = multiple * line_length_unit
    line_gap = 10
    hlines = getHLines(img, low_threshold, min_line_length,line_gap, 5)
    print('max line, # of lines:{},{}'.format(min_line_length, len(hlines)))
    if len(hlines) > 3:
      hlines = getHLines(img, low_threshold, min_line_length, line_gap, 10)
      angle, filteredlines = getAverageAngles(0, hlines)
      if len(filteredlines) > 3:
        break
  print(angle)
  debugShow('lines', drawLines(img, filteredlines), debug)
  angle = angle * -1
  rotatedimg = rotate(img, angle)
  return rotatedimg, angle

def deskewFromVline(img):
  low_threshold = 30
  line_length_unit = int(img.shape[0] / 10)
  angle = 0
  filteredlines = []
  for multiple in reversed(range(9)):
    min_line_length = multiple * line_length_unit
    line_gap = 10
    vlines = getVLines(img, low_threshold, min_line_length,line_gap, 5)
    print('max line, # of lines:{},{}'.format(min_line_length, len(vlines)))
    if len(vlines) > 3:
      vlines = getVLines(img, low_threshold, min_line_length, line_gap, 10)
      angle, filteredlines = getAverageAngles(90, vlines)
      if len(filteredlines) > 3:
        break
  print(angle)
  debugShow('lines', drawLines(img, filteredlines))
  angle = angle * -1
  rotatedimg = rotate(img, angle)
  return rotatedimg

def rotatePoint(point, center, angrad:float):
    point = (point[0] - center[0], point[1] - center[1])
    x = math.cos(angrad) * point[0] - math.sin(angrad) * point[1]
    y = math.sin(angrad) * point[0] + math.cos(angrad) * point[1]
    point = (x + center[0], y + center[1])
    return point

def recoverOriginalPoint(orgsize, resized, skewnessRad: float, topleft, bottomright) -> List[tuple(int, int)]:
    resizedratio = orgsize[0] / resized[0]
    resizedx1 = topleft[0] * resizedratio
    resizedx2 = bottomright[0] * resizedratio
    resizedy1 = topleft[1] * resizedratio
    resizedy2 = bottomright[1] * resizedratio
    center = (orgsize[0] // 2, orgsize[1] // 2)
    point1 = (resizedx1, resizedy1)
    point2 = (resizedx1, resizedy2)
    point3 = (resizedx2, resizedy2)
    point4 = (resizedx2, resizedy1)
    reverseang = skewnessRad * -1
    orgpoint1 = rotatePoint(point1, center, reverseang)
    orgpoint2 = rotatePoint(point2, center, reverseang)
    orgpoint3 = rotatePoint(point3, center, reverseang)
    orgpoint4 = rotatePoint(point4, center, reverseang)
    rtn = [orgpoint1, orgpoint2, orgpoint3, orgpoint4]
    return rtn

if __name__ == '__main__':

    img =  cv2.imread(IMG_FILE)

    # 0. Converting color to grey & binarization
    thresh_inv = convertColor(img)
    # 1. Resizing - Upsampling or Downsampling
    resized = resizeImage(thresh_inv)
    # debugShow('resizeImage', resized)
    # 2. deskew
    deskewed = deskew(resized)
    debugShow('deskewed', deskewed)
    deskewed = deskewFromVline(deskewed)
    debugShow('deskewed', deskewed)
