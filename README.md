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

XPointGroup 클래스의 경우, 하나의 수직선이 단 한개의 Pixel로 구성되어 있지 않기 때문에, 선을 대표하는 Pixel 집합이 필요했습니다. 
또한, deskewing을 통하여 이미지를 보정하였다고 하여도, 수평선이 정확하게 동일한 x 축에 있지 않는 경우가 많이 있었습니다. 

따라서, 어는 정도 오차 범위에 있는 동일한 수평선을 하나의 수평선으로 인식하게 해 줄 수 있는 장치가 필요했습니다. 

YPointGroup 클래스도 유사한 역할을 합니다. 즉 이미지에서 하나로 보이는, 수평선이 데이터 상으로는 여러개의 Y축에서 보일 수가 있었습니다. 
이 부분은, Skewness로 인한 문제도 포함됩니다. 즉 살짝 이미지가 뒤틀려 있다고 해도, 하나의 수평선의 제일 하단부와 상단부는 큰 차이가 날 수 있습니다. 

이 두가지 Class를 이용하여, 수직선과 수평선의 길이와 위치의 대략적인(정확하지 않은...) 경계선을 확인합니다.
Spreadsheet에서는 모든 Cell이 동일한 수평선을 유지할 경우, 같은 Row로 인식하고, 동일한 시작위치를 같게되는 cell을 column으로 인식합니다. 

마찬가지로, 위에서 추출한 XPointGroup을 활용하여, XPointGroup의 갯수만큼 Column 이 있는 큰 테이블 폼을 생각할 수 있습니다. 

위와 같이 수평선과 수직선을 그룹핑 하면, 

![image](https://user-images.githubusercontent.com/9047122/149712843-6ed4a800-9ed3-49bf-98eb-26a768603bfb.png)

와 같은 격자를 구성할 수 있게 된다. 

여기에서, 자세히 보면 X1, X2는 실제로는 같은 Line으로 인식하지만, 우리는 X Group의 한계치인 30 pixcel이 넘어서는 수직선은 X2의 별도의 선분으로 밖에는 인식을 못하게 됩니다. 

이 부분은 아래에 나오는 Rowspan / Colspan으로 해결합니다. 

먼저, YGroup과 그 다음 YGroup, 즉 두 수평선간에 있는 수직선 여부를 확인하면서 colspan을 진행합니다. 

여기에서 사용하는 주요 Class는
```
class TableCell 

class TableMatrix
```
입니다. 

여기서 TableCell은 HeadCell 여부를 가지고 있는데, Merge 되지 않은 Cell 자신이거나, Merge가 되었다면 제일 왼쪽 상단에 있는 Cell을 의미합니다.

먼저, Colspan 진행하는 것에 대해서 알아보면, 

![image](https://user-images.githubusercontent.com/9047122/149715641-9823eada-4684-4f58-9e3e-e7a678b29546.png)

와 같이, C0 -> Cn 형태로 순차적으로 넘어가면서 수직선 여부를 확인합니다. 수직선이 없으면, 그 다음 Cell과 Head Cell을 Merge 합니다. 

그 다음은 Rowspan에 확인하는 방법에 대해서 알아보면, 

![image](https://user-images.githubusercontent.com/9047122/149715921-35ed9df6-9ae5-4c34-8144-4786605704fa.png)

와 유사하게, Head Cell을 중심으로 Row by Row를 건너가면서, Cell의 Top line이 CLOSED/OPEN 상태를 확인하게 되어 있습니다. 

즉, 다음 행의 동일 Head Cell간에 Top line이 열려 있다면, 이 경우에는 Merge가 가능하므로, 두 Row간에 Merge작업이 발생합니다. 


# To Does 

  - 한글 인식을 위해서 사용한 내용은 Tesseract 기본 한글 인식 Set을 이용하였음. 이를 계선하기 위하여 궁서/나눔/굴림/나눔에 대한 Training 필요
  - 현재 Line Detection을 위해서 최대 수평선을 구하고, 평균값을 활용하여 Skewness를 구하고 있음. 이 부분에 ML 적용을 하면 더 좋을 듯
  - 일부 페이지에 대한 굴곡이 발생하였을 경우, 예를 들어 scanning 시점에 종이가 말려있거나 사직을 찍어서 보낸 경우, 이를 보완하는 로직이 추가되어야 함
  - 수평선/수직선 검출이 아직 미비하여, 명확한 Cell 인식이 안되는 부분
  - 큰 글자가 많을 경우, 글자안에 포함되어 있는 Pixel을 선분으로 인식하는 경우가 있음
  - 현재는 주민등록번호가 있는 부분 Cell 전체를 Painting으로 처리하고 있음. Blurring과 같은 특수 처리가 더 좋을 듯.


# Docker image 생성시 발생했던 오류 모음

  - tesseract / ocr이 이미 포함되어 있는 이미지는 https://tesseract-ocr.github.io/tessdoc/Docker-Containers.html 에서 구할 수 있음
  - dokcer pull 이후에, docker run -it <<image name>> /bin/bash 로 상태확인
  - 해당 이미지를 보면, python3.5로 구동되게 되어 있음
  - 필요한 python은 3.8 임
  - 해당 이미지는 apt package manager를 사용함
  - python 3.8 설치를 위하여, https://codechacha.com/ko/ubuntu-install-python39/ 링크를 이용
  - 하지만 오류 발생. WSL상에서 apt-add-repository가 잘 안된다고 함
  - 이를 우회하기 위하여, https://askubuntu.com/questions/53146/how-do-i-get-add-apt-repository-to-work-through-a-proxy 를 참고하여, 
  - /etc/apt/sources.list 에 강제로 deadsnakes/ppa 위치를 추가하고, (정확히는, deb http://ppa.launchpad.net/deadsnakes/ppa/ubuntu xenial main)
  - apt-update를 실행하여 발생한 오류 메시지에서 public key를 찾음
  - public key를 apt-key 명령어를 이용하여, 등록하려고 했으나 오류가 발생. 
  - 옵션으로 주는 --keyserver http://keyserver.ubuntu.com 을 --keyserver keyserver.ubuntu.com 으로 변경하니 PGP Key 임포트 성공
  - apt update로 python3.8 포함되어있는 repository 등록
  - 이후에는 apt install python3.8 로 설치됨


