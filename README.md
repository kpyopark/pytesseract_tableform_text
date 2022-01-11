# pytesseract_tableform_text

This program supports the following features. 

 - Image deskewness
 - Image denozing
 - Table Form recognition
 - Text extraction for each table cell
 - Image masking for Residential ID field.
 - Key / Value pair

# Algorithm #1. Image deskewing

In many normal document forms, there are many horizontal / vertical lines which are composed of a table in the document. 
I thought that to deskew a image, it is necessary to calculate the angle between straight horizontal line and longest horizontal line.
So, in this program, via Hough's line detection algorithm, to gather a kind of long horizontal lines and merge them, calculate the skewness through arc tangent function such like below. 

'''
    lines = cv2.HoughLinesP(detected_lines, 1, np.pi / 2, 10, None, 50, 2) # 'np.pi / 2' means to find only horizontal/vertical lines. 0' or 90' lines.
'''

Before to gather horizontal lines, I adopt horizontal/vertical line stretching to emphasize line contours such like belows

'''
    kernelForStretchedLine = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    kernelForRemoveText = cv2.getStructuringElement(cv2.MORPH_RECT,(7,1))
    dilated = cv2.morphologyEx(blur, cv2.MORPH_DILATE, kernelForStretchedLine, iterations=1)
    eroded = cv2.morphologyEx(dilated, cv2.MORPH_ERODE, kernelForRemoveText, iterations=8)
'''

After gathering, we can estimate the skewness of original image

# Algorithm #2. Denozing

There are many denoized algorithms. I just adopt OPENCV fastNlMean denoising algorithm

# Algorithm #3. Table recognition

Honestly, it's very hardest thing than other tasks. Table recognition means over to detect horizontal/vertical lines, to choose right cell boundary, merge cells and remove meaningless lines.
