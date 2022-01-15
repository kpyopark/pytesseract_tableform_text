# pytesseract_tableform_text

이 프로그램은 다음과 같은 동작을 수행합니다. 

 - image deskewing
 - image denozing
 - table form recognition
 - text extraction for each table cell
 - image masking for Residential ID field.
 - Key / Value pair

## 가정사항

 - 이미지는 300dpi 수준에서 Scan된 A4 용지를 기준으로 진행됩니다. 
 - 테이블 형태의 문서여야 합니다. 테이블 형태의 Cell로 구분하고 각 Cell에서 한글을 추출하기 때문에, 테이블이 존재하지 않을 경우, 전체 이미지에서 한글이 추출됩니다. 
 - 이럴 경우, 한글 추출이 정상적으로 진행되지 않으며, Key/Value 형태의 List는 추출할 수 없습니다. 

## 설치 방법
먼저, 최신 tesseract 모듈을 설치하십시요. 
gl 라이브러리가 없을 경우, libgl-dev 라이브러리도 아래와 같이 추가해 주세요.
```
apt install -y tesseract-ocr curl git libgl-dev
```
python3.8 이상과, pip를 설치하십시요.
```
apt install -y python3.8 python3.8-distuils
curl https://bootstrap.pypa.io/get-pip.py -o /home/pytesseract_tableform_ocr/get-pip.py
python3 cr/get-pip.py
```
MacOS나, Windows의 경우, X Windows 시스템이 없기 때문에, x window를 활용할 수 있는 xlaunch 프로그램 (예를 들어 Xming, VcXsrv 등)을 설치하셔야 합니다.
(자세한 사항은 이 Link를 활용하여 해결하였습니다. https://stackoverflow.com/questions/61110603/how-to-set-up-working-x11-forwarding-on-wsl2 )

이후 필요한 모듈을 pip를 통해서 설치합니다. 
```
git clone <<this project>>
cd pytesseract_tableform_text
/usr/bin/python3.8 -m pip install -r requirements.txt
```
실제 실행은 아래와 같은 형태로 동작하시면 됩니다.
```
python3 ./sample.py <<file path>>
```

## WSL 에서 사용하기

WSL2에서도 잘 동작됩니다. Ubuntu 버젼에서 정상 동작되는 것을 확인하였습니다. 

WSL2에서 직접 X Window를 구동하기 보다는, Host OS인 Windows 10에 VcXsrv를 설치하셔서 진행하시기 바랍니다. 
VcXsrv 기동 시, Access Control 부분을 해제하여야 합니다. 

또한, WSL2에서 display packet을 Host에 전달하기 위해서, 환경 변수에 display설정을 해 놓으셔야 합니다.

```
cd ~
echo "export DISPLAY=`cat /etc/resolv.conf | grep nameserver | awk '{ print $2 }'`:0" >> .bashrc

```

## Docker에서 사용하기

일단, 해당 프로젝트를 clone 한 이후
```
docker build -t pytess-table-ocr:0.1 .
```
로 build를 수행합니다. 기본적으로 STDIN에서 이미지 파일을 처리하도록 구성하였기 때문에, 
```
docker run -i pytess-table-ocr:0.1 < [local에서 테스트할 파일]
```
을 수행하면, Json 형태의 Text처리 결과가 표시됩니다. Json Object에 대한 세부 내용은 아래 부분을 참고하세요.

만약, 단순하게 실행하는게 아니고 수정이나, 기타 다른 사항을 구성하고 싶다면, 
```
docker run -it pytess-table-ocr:0.1 /bin/bash 로 접속해서 /home/pytesseract_table_form_ocr 디렉토리에서 수정해서 테스트를 해보시면 빠르게 결과 값을
```
얻어 볼 수 있습니다. 

## 실제 처리 결과 예시

화면을 보기 위해서는, 현재 주석 또는 Debug 로 막혀 있는 부분을 수정해 주어야 한다. 
예를 들어, sample.py 파일에 있는

```
def handleFile(filepath):
    debug = True
```

로 변경하고 실행하면, 아래와 같은 분석 이미지가 화면에 추가로 표시됩니다. 
자세히 보면, 파란색으로 표시된 것이, 내부에 테이블로 인식된 선들입니다. 

![image](https://user-images.githubusercontent.com/9047122/149486891-b40955d4-3f71-48a5-909c-3d9994b5d647.png)

두번째 뜨는 화면은 Masking 처리 화면입니다. 주민등록번호란을 아래와 같이 Masking 처리하여 보여줍니다. 

![image](https://user-images.githubusercontent.com/9047122/149602385-4a1dce5e-8ab2-4db0-b99e-020363ae1c0c.png)



## 중간 처리 이미지 저장 기능 활성화

현재, Local File에 있는 Image를 대상으로 처리하게 구성되어 있습니다. 
또한, 주민번호 Cell을 Mask 처리한 이미지는 별도로 저장하지 않습니다. 해당 이미지를 Local에 저장하기 위해서는 sample.py 상단에 있는 ENABLE_MASKED_IMAGE_STORED를
True로 변경할 경우, 원본 파일이 있는 위치에 '.masked.png' 라는 확장자를 가진 masked image file이 생성됩니다
유사하게 denoized 된 이미지를 저장할 수 있는 기능을 활성화 하기 위해서는 ENABLE_DENOIZED_IMAGE_STORED 기능을 활성화 해야 합니다.

## 키워드 추출 방법

Key / Pair 를 생성하기 위해서는 일단, 기준이 되는 Key가 필요합니다. 
이 부분을 정의하기 위해서는, sample.py에 있는 MAJOR_KEYWORD_LIST에 한글 키워드을 추가하시기 바랍니다. 
현재 공공문서에서 많이 활용하는 키워드를 입력해 놓았습니다. 

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

필요할 경우 해당 List에 추가하시면 됩니다.

키워드 매칭을 위하여, Exact matching을 사용할 경우, OCR 오탐오류등으로 인하여 매칭이 안될 수 있습니다. 
예를 들어, "주민등록번호"의 경우 "즈민등륵빈호" 로 익신 될 수 있습니다. 
 
이를 완화하기 위하여 영문첫글자를 추출하여, 자음비교를 수행합니다. OCR에 인식된 Text는 그대로 Cell의 Value값에 들어가 있기 때문에 그대로 활용하면 됩니다.

Value 값은, 다음과 같은 가정을 통해서 추출합니다. 
일단 Key Cell에 해당하는 오른쪽 셀을 대상으로 키워드 여부를 판단합니다. 만약 Keyword Cell이 아닌 경우, 일반 Cell로 분류하고 해당 Cell 값을 Value Cell로 지정합니다. 
만약, 오른쪽 셀이 동일한 Key Cell로 분류될 경우, 해당 셀의 하단 셀을 검사하여 Keyword가 있는 Key Cell인지 검출합니다. 

**즉, 키의 오른쪽 아니면 하단에 있는 키값이 아닌 일반 셀을 Value Cell로 맵핑합니다. **

## Response Body

현재 입력은 Input File Path만 잡게 되어 있습니다. 향후에는 serverless 상에서 다양한 옵션을 활용할 수 있도록 지원할 예정입니다.
이후, 처리 후 결과는 ResponseBody라는 클래스에 집결되어 JSON 형태로 전환됩니다. 

Response Body를 보면 먼저, 
원본이미지와 Resized 이미지의 정보를 담고 있고, 이후, 문서의 수평기울기(비틀림)을 Degree 형태로 보여주는 Skewness 항목이 있습니다. 
이후, tables 항목에서는 각 Cell 단위별, rowspancolspan및 실제 merge되어 있는 활성화되어 있는 cell 의 box 정보와 OCR에서 추출한 Text 정보가 들어 있습니다.
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

많은 공적 문서에는, 테이블을 구성하는 많은 수직/수평선이 포함되어 있습니다. 
이 부분을 최대한 활용하여, 정확한 수평라인 맞추기를 진행하도록 구성되어 있습니다. 
Hough's line detection 알고리즘을 이용하여, line을 추출하고, 이후, 이를 atan 함수를 이용하여 기울기를 구하였습니다. 

```
    lines = cv2.HoughLinesP(detected_lines, 1, np.pi / 2, 10, None, 50, 2)
    for line in lines:
      for x1,y1,x2,y2 in line:
        degree = math.degrees(math.atan2(y1-y2, x2-x1))
```

실제로는 수평선 추출전에, 약간의 이미지 조작을 통하여, 수평선이 강조된 이미지를 생성하고 이를 이용하여, 수평선을 추출하였습니다.

```
    kernelForStretchedLine = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    kernelForRemoveText = cv2.getStructuringElement(cv2.MORPH_RECT,(7,1))
    dilated = cv2.morphologyEx(blur, cv2.MORPH_DILATE, kernelForStretchedLine, iterations=1)
    eroded = cv2.morphologyEx(dilated, cv2.MORPH_ERODE, kernelForRemoveText, iterations=8)
```

Hough's line detection을 이용할 때, 초기에는 원본 이미지 횡폭의 90%에 해당하는 Line이 있는지 살펴보고, 적당한 Line이 없다면, 
10%씩 line 길이를 줄여가면서, 3개 이상의 Line이 검출될 때까지 진행합니다. 

이렇게 하는 이유는, 글자 또는 일반 이미지에 들어가 있는 점들이, Line으로 잘못 인식되는 것을 최대한 방지하기 위함입니다. 

이 절차를 끝내고, 적당한 길이의 수평선 검출이 끝나면, 이후에는 수평선의 기울기를 각각 계산한 후 median 값을 계산하여 skewness를 산출합니다.

## Algorithm #2. Denozing

Denoizing 기법은 많이 있지만, 실제로 적용한 알고리즘은 opencv에서 제공하는 fastnlmean 함수를 적용하였습니다. 

## Algorithm #3. Table recognition

테이블 식별을 위해서 착안한 아이디어는 모든 문서를 Spreadsheet 형태의 수 많은 격자로 구성되어 있다고 가정하는 부분이었습니다. 
그림 자체에 대한 분석을 위하여 아래 두가지 Class를 생성하였습니다. 

```
class XPointGroup:
...

class YPointGroup:
```

XPointGroup 클래스의 경우, 하나의 수직선이 단 한개의 Pixel로 구성되어 있지 않기 때문에, 선을 대표하는 Pixel 집합이 필요했습니다. 
또한, deskewing을 통하여 이미지를 보정하였다고 하여도, 수평선이 정확하게 동일한 x 축에 있지 않는 경우가 많이 있었습니다. 

따라서, 어는 정도 오차 범위에 있는 동일한 수평선을 하나의 수평선으로 인식하게 해 줄 수 있는 장치가 필요했습니다. 

YPointGroup 클래스도 유사한 역할을 합니다. 즉 이미지에서 하나로 보이는, 수평선이 데이터 상으로는 여러개의 Y축에서 보일 수가 있었습니다. 
이 부분은, Skewness로 인한 문제도 포함됩니다. 즉 살짝 이미지가 뒤틀려 있다고 해도, 하나의 수평선의 제일 하단부와 상단부는 큰 차이가 날 수 있습니다. 

이 두가지 Class를 이용하여, 수직선과 수평선의 길이와 위치의 대략적인(정확하지 않은...) 경계선을 확인합니다.
Spreadsheet에서는 모든 Cell이 동일한 수평선을 유지할 경우, 같은 Row로 인식하고, 동일한 시작위치를 같게되는 cell을 column으로 인식합니다. 

마찬가지로, 위에서 추출한 XPointGroup을 활용하여, XPointGroup의 갯수만큼 Column 이 있는 큰 테이블 폼을 생각할 수 있습니다. 

<<작성 진행 중>>

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
