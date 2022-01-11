# pytesseract_tableform_text

이 프로그램은 다음과 같은 동작을 수행합니다. 

 - image deskewing
 - image denozing
 - table form recognition
 - text extraction for each table cell
 - image masking for Residential ID field.
 - Key / Value pair

## Test 방법
(requirements.txt 정리중)
필요한 모듈을 pip를 통해서 설치합니다. 

(GUI 가 필요할 경우)
x window를 활용할 수 있는 xlaunch 프로그램 (예를 들어 XING, MING, VcXsrv 등)을 설치하고 기동합니다.

```
git clone <<this project>>
cd pytesseract_tableform_text
python3 ./sample.py <<file path>>
```

## Enabling hidden features
현재, Local File에 있는 Image를 대상으로 처리하게 구성되어 있습니다. 
또한, 주민번호 Cell을 Mask 처리한 이미지는 별도로 저장하지 않습니다. 해당 이미지를 Local에 저장하기 위해서는 sample.py 상단에 있는 ENABLE_MASKED_IMAGE_STORED를
True로 변경할 경우, 원본 파일이 있는 위치에 '.masked.png' 라는 확장자를 가진 masked image file이 생성됩니다
유사하게 denoized 된 이미지를 저장할 수 있는 기능을 활성화 하기 위해서는 ENABLE_DENOIZED_IMAGE_STORED 기능을 활성화 해야 합니다.

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

# Algorithm #2. Denozing

Denoizing 기법은 많이 있지만, 실제로 적용한 알고리즘은 opencv에서 제공하는 fastnlmean 함수를 적용하였습니다. 

# Algorithm #3. Table recognition

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


