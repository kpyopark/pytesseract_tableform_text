# pytesseract_tableform_text

Korean Version. ([Latest](https://github.com/kpyopark/pytesseract_tableform_text/blob/main/README.kr.md))

This application supports the following features.

 - image deskewing
 - image denozing
 - table form recognition
 - text extraction for each table cell
 - image masking for Residential ID field.
 - Key / Value pair

## Assumption

 - Input files should be scanned image files with 300 dpi and A4 size.
 - Image should contains table form documents. In this application, the document image will be analyzed into each table cells. And then, extraction of each cell will be executed. So if there is no table form in input image, OCR might be processed for whole document image.
 - In this case, Korean text extraction could be mal-functioned. And Key/Value pairs can't be extracted from the input image. 

## Installation
At first, instal latest tesseract module on development environment.
If there is no GL library on development environment, install it with tesseract. 
(Tesseract/OpenCV library uses GL Library)
```
apt install -y tesseract-ocr curl git libgl-dev
```
instal python3.8 and pip.
```
apt install -y python3.8 python3.8-distuils
curl https://bootstrap.pypa.io/get-pip.py -o /home/pytesseract_tableform_ocr/get-pip.py
python3 cr/get-pip.py
```
In macos, windows, there is no X window system at default, so some X client windows application such like Xming, VcXsrv and etc must be installed.
(How to install it. refer the next link - https://stackoverflow.com/questions/61110603/how-to-set-up-working-x11-forwarding-on-wsl2)

After that, all other pythone modules will be on pip command.
```
git clone <<this project>>
cd pytesseract_tableform_text
/usr/bin/python3.8 -m pip install -r requirements.txt
```
You can run this application as the following command line.
```
python3 ./sample.py <<file path>>
```

## Use on WSL

In WSL2, this application can be run. I tested it on Ubuntu version(16, 18) and saw it runs well.

Install VcXsrv instaed of installing x windows system on WSL itself. In that case, you should set 'No Access Control' option should be set on launching time.

In WSL2, there is no presentation layer forwarding variable(DISPLAY) at default. So you should set on bash profile such like the below.
```
cd ~
echo "export DISPLAY=`cat /etc/resolv.conf | grep nameserver | awk '{ print $2 }'`:0" >> .bashrc

```

## Use on Docker

After cloning this project, you can build the image the following command line.
```
docker build -t pytess-table-ocr:0.1 .
```
This application can accept STDIN binary image stream, so you can test this image the following command line.
```
docker run -i pytess-table-ocr:0.1 < [test image file on local file system]
```
After runningn it, JSON response will be returned as its result. (You can refer the below section to find more specific JSON body elements.)

If you want to modify and test it, you can enter bash shell mode in the container
```
docker run -it pytess-table-ocr:0.1 /bin/bash 
```
And then, you can modify source code and test it on the direction - /home/pytesseract_table_form_ocr.

## Example - Result

If you want to see the result/processes images on the screen (on the X clients), you must modify the following line to enable screen option.  
In sample.py file, modify 'debug' variable from False to True.

```
def handleFile(filepath):
    debug = True
```

After that, you can see the below image when to execute this application.
When to see the image more closely, you could noticed that there are many blue lines. 
Blue lines are analyzed by this application and these lines will be used to recognize table forms.

![image](https://user-images.githubusercontent.com/9047122/149486891-b40955d4-3f71-48a5-909c-3d9994b5d647.png)

The second screen is masked image. You can see the masked cells on residential identification fields.

![image](https://user-images.githubusercontent.com/9047122/149602385-4a1dce5e-8ab2-4db0-b99e-020363ae1c0c.png)



## Enableing intermediate images

At the latest version, only local file and STDIN could be handled on this application.
And masked image or deskewed/denoized image aren't stored on local file system at default. 
If you want to store these intermediate files on the file system, you can modify some options in 'sample.py' file.
To store masked image, modify ENBLE_MASKED_IMAGE_STORED parameter from 'False' to 'True'. After to enable this option, 
masked file with which file extension is '.masked.png' will be created on the same local direction. 
Similary, you can modify ENABLE_DENOIZED_IMAGE_STORED option from 'False' to 'True' to store denozied image into local file system.

## How to keywork-value pairs

To extract Key/value pairs, you should set default keyword list on 'sample.py' file.
To extends keywords list, you can append more keywords into 'MAJOR_KEYWORD_LIST' fields.
The following keywords are written in sample.py file at default. 

```
MAJOR_KEYWORD_LIST = [
    '주민등록번호',
    '주민번호',
    '용도구역',
    '성명',
    '용적률',
    '건축물',
    '토지',
    '면적',
    '지목',
    '거래가격',
    '소재지',
    '전화번호',
    '이름'
]
```

You can append additional keywords into the list.

After text extraction(OCR), this program analyze which cells contains keyword. At that time, 'Keyword Matching algorithm' should be needed. 
The simplest algorithm is 'exact matching'. But under OCR text, there could be mis-recognized letters such like 'Resido(e)ntial 1(I)D'. So exact matching isn't proper solution in this case. 
In this app, to convert korean letter to english letter and to extract consonants for each korean letter is a replcement of 'exact matching'. 
For example, this conversion make "주민등록번호" word to "JMDRBH". After that, matching will be processed. 

Value cell will be fixed the following rules.

  1. At first, the right cell of the keyword cell would be investigated. "Is it keyword cell or not?".
  2. If the right cell isn't detected as a keyword cell, this cell be categorized as 'value' cell. 
  3. If the right cell has keyword (so this cell also might be 'keyword' cell), the bottom of the keyword cell will be investigated.
  4. If the bottom cell has no keyword, this cell will be assigned as a 'value' cell.

** In short words, generally right or bottom of the key cell will be selected as a value cell. **

## Response Body

Lastest version of this app only accepts 'image file path' as a input parameter. In the furture, many other options will be added as input parameters.
All results of the process will be aggregated into one JSON object (Response Object)

In the Response Body,
'imagesize' attributes has size of 'original' and 'resized' image (for OCR extraction), and 'skewness' which points the angle of skewness of the input image, 
and 'tables' attribute shows all the cell information which includes texts and boundaries.
```
reponse_body = {
    'imagesize' : {
        'org' : {
            'width' : <<>>,
            'height' : <<>>
        },
        'resized' : {
            'width' : <<>>,
            'height' : <<>>
        }
    }
    'skewness' : <<skewness of original image>>
    'tables' : [
        [
            {
                'rowIndex': 0,
                'rowSpan': 1,
                'colIndex': 0,
                'colSpan': 1,
                'value': <<extracted value from tesseract>>
                'box': [ 
                    {
                        'x': <<pixel>>,
                        'y': <<pixel>>
                    },
                    {
                        'x': <<pixel>>,
                        'y': <<pixel>>
                    },
                    {
                        'x': <<pixel>>,
                        'y': <<pixel>>
                    },
                    {
                        'x': <<pixel>>,
                        'y': <<pixel>>
                    }
                ]
                'originalbox': [
                    {
                        'x': <<pixel>>,
                        'y': <<pixel>>
                    },
                    {
                        'x': <<pixel>>,
                        'y': <<pixel>>
                    },
                    {
                        'x': <<pixel>>,
                        'y': <<pixel>>
                    },
                    {
                        'x': <<pixel>>,
                        'y': <<pixel>>
                    }
                ]
            },
            ...

```

## Algorithm #1. Image deskewing

In Korea, generally a formal document template consists of one big table with many cells such like the below image.

![image](https://user-images.githubusercontent.com/9047122/150176270-3e1b8859-81e1-41fc-a2c0-a78979caec64.png)

This application uses it to deskew whole image. 
At first, this app will detects lines via 'Hough's line detection' - very common algorithm for line detection. 
After that, the skewness of lines will be calculated by atan function.
```
    lines = cv2.HoughLinesP(detected_lines, 1, np.pi / 2, 10, None, 50, 2)
    for line in lines:
      for x1,y1,x2,y2 in line:
        degree = math.degrees(math.atan2(y1-y2, x2-x1))
```
Before line detection, this app emphasizes the vertical/horizontal lines in the images.

```
    kernelForStretchedLine = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    kernelForRemoveText = cv2.getStructuringElement(cv2.MORPH_RECT,(7,1))
    dilated = cv2.morphologyEx(blur, cv2.MORPH_DILATE, kernelForStretchedLine, iterations=1)
    eroded = cv2.morphologyEx(dilated, cv2.MORPH_ERODE, kernelForRemoveText, iterations=8)
```

During line detection, it tries to find longest lines in the image. At first, it tries to find 90% length of total image width. 
If it can find the longest lines, this app uses it to calculate skewness of image. But when this app wouldn't find the line, it tries to find 80% length of image width.
Again and agian, it tries to find the longest lines in the image. 

Houghline algorithm could have many noisy small lines under small length parameter. Trying to find longest lines can prevent to find false-positive line detections.

After line detections, this app calculates median values of some lines groups.

## Algorithm #2. Denozing

This app adopts fastnlmean function which is supported on OpenCV.

## Algorithm #3. Table recognition

Before to analyze table cells in a image, I should assume that a documents can be composed of many cells such like Excel does.
Under this assumption, I need more specific standards to split cells in the image.
So the below two classes are induced. 

```
class XPointGroup:
...

class YPointGroup:
```

![image](https://user-images.githubusercontent.com/9047122/149711962-e7923008-f875-43d8-a02d-88e5b6dc065e.png)

The purpose of XPointGroup is grouping of vertical lines(pixels). As you can see the above image, logical line consists of many pixels.
XPointGroup can represent one logical vertical line. In general, original images tend to be distorted and skewed. So even one vertical line
has different x positions. XPointGroup can analyze this situation and recognize one logical line under spread pixels.

YPoingGroup class has similar purpose for logical horizontal lines. In a distorted image, one hozontal line can show various y positions. 

Using these two classes, this application analyze boundary of each cell of a whole document.

The cell bounday analyzed by these two classes can be shown the below image.

![image](https://user-images.githubusercontent.com/9047122/149712843-6ed4a800-9ed3-49bf-98eb-26a768603bfb.png)

In more specific view, position X1, X2 is a points on one line, but XPointGroup will divide two separate lines cause of its exceed of 30 pixel tolerance.

So, one logical line could be saparated into two different lines. How to handle this situation ? The answer is merging columns(colspan).

This application can handle this with rowspan/colspan. 

To detect column span, it checks whether there is a vertical line between each cells from right to left side with the following classes

```
class TableCell 

class TableMatrix
```

TableCell class represent a logical cell of a table. It has 'merged' status in it. If two cells is merged, the left top cell will become a head cell.

![image](https://user-images.githubusercontent.com/9047122/149715641-9823eada-4684-4f58-9e3e-e7a678b29546.png)

You can see the direction from left to right - C0 -> Cn. If there is no vertical line between cell, these neighboring cells would be merged. 

Detecting rowspan is progressing in a very similar way.

![image](https://user-images.githubusercontent.com/9047122/149715921-35ed9df6-9ae5-4c34-8144-4786605704fa.png)

In rowspan detection, only head cells can be applied to detect rowspan. And cells without horizontal line between them would be merged/


# To Does 

  - Tesseract best-fit korean traindata used. It might be improved more accurately based on the real data.
  - The way with the skewness be calculated is very heuristic way. This part might be improved based on ML technique.
  - Some parts of the document could be very curved. Flattening them could give us more accurate result.
  - Horizontal/Vertical line detection must be more improved.
  - Big letter could be recognized as a big lines. 
  - Code Quality is very poor
  - Currently, the entire cell with a resident registration number is masked through painting. A special masking such as blurring would be better.

# References

1.	Tesseract OCR
https://www.pyimagesearch.com/2020/09/07/ocr-a-document-form-or-invoice-with-tesseract-opencv-and-python/
2.	Find Rectangle
https://stackoverflow.com/questions/57196047/how-to-detect-all-the-rectangular-boxes-in-the-given-image
3.	Enhanced OCR recognition ratio
https://ddolcat.tistory.com/954
4.	Otsu thresholding
https://opencv-python.readthedocs.io/en/latest/doc/09.imageThresholding/imageThresholding.html
5.	Installation of python opencv
https://www.codesd.com/item/installing-opencv-python-on-amazon-linux-apache.html
https://pythonprogramming.net/loading-images-python-opencv-tutorial/
6.	Using opencv on Lambda
https://betterprogramming.pub/creating-a-python-opencv-layer-for-aws-lambda-f2d5266d3e5d
https://typless.com/tesseract-on-aws-lambda-ocr-as-a-service/
7.	Install tesseract
https://stackoverflow.com/questions/61208140/install-tesseract-for-aws-linux
8.	Korean OCR with Keras
https://github.com/Wongi-Choi1014/Korean-OCR-Model-Design-based-on-Keras-CNN
9.	Text area detection
https://d2.naver.com/helloworld/8344782
10.	Pytesseract example with openCV
https://stackoverflow.com/questions/66463697/text-blur-after-thresholding-using-opencv
11.	Curved Text flattening
http://www.daniel.umiacs.io/daniel_papersfordownload/liang-j_cvpr2005.pdf
https://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Meng_Active_Flattening_of_2014_CVPR_paper.pdf
12.	Extract ROIs from full image
https://stackoverflow.com/questions/9084609/how-to-copy-a-image-region-using-opencv-in-python
13.	Extract Korean Text with tesseract
https://github.com/tesseract-ocr/tessdata_best

14.	Detecting Overlapped areas
https://stackoverflow.com/questions/62350446/how-to-detect-overlapping-or-embedded-rectangle-in-python-opencv


