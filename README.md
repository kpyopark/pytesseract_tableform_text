# pytesseract_tableform_text

Korean Version. ([Latest](https://github.com/kpyopark/pytesseract_tableform_text/blob/main/README.kr.md))

This program supports the following features. 

 - image deskewing
 - image denozing
 - table form recognition
 - text extraction for each table cell
 - image masking for Residential ID field.
 - Key / Value pair

# Algorithm #1. Image deskewing

In many formal documents, there are many horizontal / vertical lines which are composed of a table in the document. 
I thought that to deskew a image, it is necessary to calculate the angle between straight horizontal line and long horizontal lines extracted from the image.
It uses Hough's line detection algorithm, to gather a kind of long horizontal lines. After that it merges them and calculate the skewness through arc tangent function such like below. 

```
    lines = cv2.HoughLinesP(detected_lines, 1, np.pi / 2, 10, None, 50, 2)
    for line in lines:
      for x1,y1,x2,y2 in line:
        degree = math.degrees(math.atan2(y1-y2, x2-x1))
```

Before to extract horizontal lines, to clarify line contour, it uses line stretching.

```
    kernelForStretchedLine = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    kernelForRemoveText = cv2.getStructuringElement(cv2.MORPH_RECT,(7,1))
    dilated = cv2.morphologyEx(blur, cv2.MORPH_DILATE, kernelForStretchedLine, iterations=1)
    eroded = cv2.morphologyEx(dilated, cv2.MORPH_ERODE, kernelForRemoveText, iterations=8)
```

After gathering, we can estimate the skewness of original image

# Algorithm #2. Denozing

There are many denoized algorithms. I just adopt OPENCV fastNlMean denoising algorithm

# Algorithm #3. Table recognition

Table recognition needs over to detect horizontal/vertical lines, to choose right cell boundary, merge cells and remove meaningless lines.

