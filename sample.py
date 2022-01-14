from __future__ import annotations
from dataclasses import dataclass, asdict, is_dataclass
from collections.abc import Sequence
from typing import List, Tuple
import cv2
import numpy as np
import math
import pytesseract as pt
import deskew as dk
from imageutil import *
from tablematrix import *
from random import *
import json
import sys
import io

ENABLE_MASKED_IMAGE_STORED = False
MASKED_IMAGE_FILE_EXT = '.masked.png'
ENALBE_DENOIZED_IMAGE_STORED = False
DENOIZED_IMAGE_FILE_EXT = '.denoized.png'

TEXT_MARGIN = 6
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

request = {
    'blur_resid' : True,
    'blur_original_image' : True,
    'save_blur_image' : True,
    'save_table_form_image' : True,
    'original_image_path' : '',
    'original_image_filename' : '',
    'target_image_path' : '',
    'blur_file_name' : '',
    'table_form_file_name' : '.tableform.png',
    'appended_keywords' : []
}

@dataclass(init=False)
class Boundary:
    width: float
    height: float
    def __init__(self, width:int, height:int):
        self.width = width
        self.height = height
    def __str__(self):
        rtn = {}
        rtn['width'] = self.width
        rtn['height'] = self.height
        return rtn
@dataclass(init=False)
class ImageSize:
    org : Boundary
    resized : Boundary
    def __init__(self, org:Boundary, resized:Boundary):
        self.org = org
        self.resized = resized
    def __str__(self):
        rtn = {}
        rtn['org'] = self.org.__str__()
        rtn['resized'] = self.resized.__str__()
        return rtn
@dataclass(init=False)
class ResultBody:
    imagesize: ImageSize
    skewness: float
    tables: List[List[dict]]
    keyvalues: List[tuple(dict,dict)]
    def __init__(self, imagesize:ImageSize, skewneess:float, tables:List[List[dict]], keyvalues:List[tuple(dict, dict)]):
        self.imagesize = imagesize
        self.skewness = skewneess
        self.tables = tables
        self.keyvalues = keyvalues
    def toJson(self,indent=0,sortkey=False):
        rtn = {}
        rtn['imagesize'] = self.imagesize.__str__()
        rtn['skewness'] = self.skewness
        rtn['tables'] = self.tables
        rtn['keyvalues'] = self.keyvalues
        return json.dumps(rtn,indent=indent,sort_keys=sortkey)
'''
request parameter =:
use_capital_letter_to_match = true (default)
blur_resid = true (default)
blur_original_image = true (default)
save_blur_image = true (default)
save_table_form_image = false (default)
original_image_path = <<mandatory>>
original_image_filename = <<mandatory>>
target_image_path = <<mandatory>>
blur_file_name = <<optional. If not exist, it will use same name of original file.>>
table_form_file_name = <<optional. If not set, it will use original file name + '.tableform.png'>>

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
            {
                'rowIndex' : 0,
                'rowSpan' : 2,
                'colIndex' : 1,
                'colSpan' : 1,
                'value' : <<extracted value from tesseract>>
                'box' : {
                    'x1' : <<adjusted point>>,
                    'y1' : <<adjusted point>>,
                    'x2' : <<adjusted point>>,
                    'y2' : <<adjusted point>>
                },
                'originalbox' : {
                    'x1' : <<point in original image>>,
                    'y1' : <<point in original image>>,
                    'x2' : <<point in original image>>,
                    'y2' : <<point in original image>>
                }
            },
            ...
        ],[
            {
                'rowIndex' : 1,
                'rowSpan' : 1,
                'colIndex' : 0,
                'colSpan' : 1,
                'value' : <<extracted value from tesseract>>
                'box' : {
                    'x1' : <<adjusted point>>,
                    'y1' : <<adjusted point>>,
                    'x2' : <<adjusted point>>,
                    'y2' : <<adjusted point>>
                },
                'originalbox' : {
                    'x1' : <<point in original image>>,
                    'y1' : <<point in original image>>,
                    'x2' : <<point in original image>>,
                    'y2' : <<point in original image>>
                }
            },
            {
                'rowIndex' : 0, <-- It indicates that this cell with merged with upper cell.
                'rowSpan' : 2,
                'colIndex' : 1,
                'colSpan' : 1,
                'value' : <<extracted value from tesseract>>
                'box' : {
                    'x1' : <<adjusted point>>,
                    'y1' : <<adjusted point>>,
                    'x2' : <<adjusted point>>,
                    'y2' : <<adjusted point>>
                },
                'originalbox' : {
                    'x1' : <<point in original image>>,
                    'y1' : <<point in original image>>,
                    'x2' : <<point in original image>>,
                    'y2' : <<point in original image>>
                }
            },
            ...
        ]
    ],
    keyvalues : [
        {

        }
    ]
}
'''

class MajorKeyword:
    #https://en.wikipedia.org/wiki/List_of_Hangul_jamo
    FIRST_LETTER_LIST = ['ㄱ','ㄲ','ㄴ','ㄷ','ㄸ','ㄹ','ㅁ','ㅂ','ㅃ','ㅅ','ㅆ','ㅇ','ㅈ','ㅉ','ㅊ','ㅋ','ㅌ','ㅍ','ㅎ']
    FIRST_LETTER_ENG_LIST = ['G','K','N','D','T','L','M','B','P','S','TH','O','Z','Z','Z','K','T','P','H']
    MIDDLE_LETTER_LIST = ['ᅡ','ᅢ','ᅣ','ᅤ','ᅥ','ᅦ','ᅧ','ᅨ','ᅩ','ᅪ','ᅫ','ᅬ','ᅭ','ᅮ','ᅯ','ᅰ','ᅱ','ᅲ','ᅳ','ᅴ','ᅵ']
    LAST_LETTER_LIST = [' ', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
    MAJOR_KEYWORD_CLASS_LIST:List[MajorKeyword] = []
    def __init__(self, kor):
        self.koreanKeyword = kor
        self.engCap = MajorKeyword.convertKoreanToEngCap(kor)
        self.maxLetter = len(self.engCap) + 5
    def matchEngCap(self, eng:str) -> bool:
        if (len(eng) < self.maxLetter 
            and eng.find(self.engCap) >=0):
            return True
    def __str__(self):
        rtn = {
            'koreanKeyword' : self.koreanKeyword,
            'engCap' : self.engCap,
            'maxLetter' : self.maxLetter
        }
        return json.dumps(rtn)
    def __repr__(self):
        return self.__str__()
    @staticmethod
    def matchKeyword(kor:str) -> MajorKeyword:
        if len(MajorKeyword.MAJOR_KEYWORD_CLASS_LIST) == 0:
            MajorKeyword.MAJOR_KEYWORD_CLASS_LIST = MajorKeyword.getKeywordEngCapList()
        engCap = MajorKeyword.convertKoreanToEngCap(kor)
        for keyword in MajorKeyword.MAJOR_KEYWORD_CLASS_LIST:
            if keyword.matchEngCap(engCap):
                return keyword
        return None
    @staticmethod
    def getKeywordEngCapList() -> List[MajorKeyword]:
        rtn:List[MajorKeyword] = []
        for keyword in MAJOR_KEYWORD_LIST:
            rtn.append(MajorKeyword(keyword))
        # print(rtn)
        return rtn
    @staticmethod
    def convertKoreanToEngCap(kor:str) -> str:
        rtn = ''
        for letter in list(kor):
            unicode = ord(letter)
            if 0xac00 <= unicode <= 0xd7a3:
                inx = (unicode - 0xac00) // (len(MajorKeyword.MIDDLE_LETTER_LIST) * len(MajorKeyword.LAST_LETTER_LIST))
                rtn = rtn + MajorKeyword.FIRST_LETTER_ENG_LIST[inx]
        return rtn
def getHorizontalLines(img) -> tuple(List[YPointGroup], int, int):
    debug = False
    detected_lines = getHLineStretchedImage(img)
    lines = cv2.HoughLinesP(detected_lines, 1, np.pi / 2, 10, None, 50, 2) # 'np.pi / 2' means to find only horizontal/vertical lines. 0' or 90' lines.
    debugShow('hlines', drawLines(img, lines), debug)
    ypointgroups: YPointGroups = []
    h,w = img.shape
    firstHLine = YPointGroup(0,0,w,0)
    lastHLine = YPointGroup(0,h,w,h)
    ypointgroups.append(firstHLine)
    ypointgroups.append(lastHLine)
    for line in lines:
        for x1, y1, x2, y2 in line:
            YPointGroup.checkAndAddHorizontalLine(ypointgroups, x1, y1, x2, y2)
    YPointGroup.sortHorizontalLineAndLink(ypointgroups)
    span_height, vertical_max_gap = YPointGroup.getAverageSpanHeightAndGap(ypointgroups)
    return ypointgroups, span_height, vertical_max_gap

def getPartialVerticalLineFromHoughLine(img, ypg:YPointGroup) -> None:
    if ypg.getNextYPG() is None:
        return
    height = int(ypg.height)
    maxgap = int(height * 0.095)
    # print('average height, max gap : {}, {}'.format(height, maxgap))
    if ypg.getNextYPG() is None:
        return
    y1 = ypg.y1
    y2 = ypg.getNextYPG().y1
    img_target = img[y1:y2, 0:img.shape[1]]
    debugShow('verticalimage', img_target)
    vlines = cv2.HoughLines(img,1,np.pi/2,int(height * (100 - 2 * VERTICAL_LINE_GAP_PERCENT) / 100), 0, 0, min_theta=0, max_theta=np.pi/180)
    #print('vlines:{}'.format(vlines))
    if vlines is not None :
        for oneline in vlines:
            rho, theta = oneline[0]
            x1 = math.cos(theta) * rho
            if (x1 > 0) and (theta < np.pi/180):
                ypg.checkAndAddVerticalLine(x1, 0, x1, height)
    detectedlines = []
    # print('xpg:{}'.format(ypg.xPointGroups()))
    for detectedline in ypg.xPointGroups():
        detectedlines.append([(detectedline.x, 0, detectedline.x, ypg.height)])
    debugShow('vlines', drawLines(img_target, detectedlines))

def getPartialVerticalLine(img, ypg:YPointGroup) -> None:
    debug = False
    if ypg.getNextYPG() is None:
        return
    height = int(ypg.height)
    maxgap = int(height * 0.095)
    # print('average height, max gap : {}, {}'.format(height, maxgap))
    if ypg.getNextYPG() is None:
        return
    y1 = ypg.y1
    y2 = ypg.getNextYPG().y1
    img_target = img[y1:y2, 0:img.shape[1]]
    debugShow('verticalimage', img_target, debug)
    lines = cv2.HoughLinesP(img_target, 1, np.pi / 2, 1, None,minLineLength=int(height * (100 - 2 * VERTICAL_LINE_GAP_PERCENT) / 100), maxLineGap=maxgap)
    if False:
        vlines = []
        for line in lines:
            for x1, y1, x2, y2 in line:
                angle = math.degrees(math.atan2(y2-y1,x2-x1))
                if angle > 89 and angle < 90:
                    vlines.append(line)
        debugShow('vlines', drawLines(img_target, vlines))
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                ypg.checkAndAddVerticalLine(x1, y1, x2, y2)
    detectedlines = []
    # print('xpg:{}'.format(ypg.xPointGroups()))
    for detectedline in ypg.xPointGroups():
        detectedlines.append([(detectedline.x, 0, detectedline.x, ypg.height)])
    debugShow('vlines', drawLines(img_target, detectedlines), debug)

def getVerticalLines(img, ypgs:List[YPointGroup]):
    dilated_image = getVLineStretchedImage(img)
    for ypg in ypgs:
        getPartialVerticalLine(dilated_image, ypg)
        #getPartialVerticalLineFromHoughLine(dilated_image, ypg)
    YPointGroup.calculatedAllHorizontalLineDelta(ypgs)

def showLineImageFromMatrix(img, matrix:TableMatrix, debug:bool):
    if not debug:
        return
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for inx in range(0,matrix.maxrow-1):
        row = matrix.getRow(inx)
        ypg = row.ypg
        for headCell in row.getIterable():
            cv2.rectangle(img, (headCell.x1ext, headCell.y1ext), (headCell.x2ext, headCell.y2ext), (200,255,200), 3)
        for x1, x2 in ypg.lines:
            cv2.line(img, (x1, ypg.y), (x2, ypg.y), (0,0,255), 2)
    debugShow('detectedlines', img)

# https://m.blog.naver.com/hn03049/221957851802
def extractText(img):
    # config = ('-l kor --oem 1 --psm 3')
    korean = pt.image_to_string(img, lang='kor', config='--psm 7 --oem 3')
    # print("extracted text: {}".format(korean))
    return korean

def filterForOCRTest1(img):
    kernel = np.ones((1, 1), np.uint8)
    filtered = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 41,3)
    opening = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    ret1, th1 = cv2.threshold(closing, 180, 255, cv2.THRESH_BINARY)
    ret2, th2 = cv2.threshold(th1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    blur = cv2.GaussianBlur(th2, (1, 1), 0)
    ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img = cv2.bitwise_not(th3)
    return img

def filterForOCR(img):
    blur = cv2.GaussianBlur(img, (1, 1), 0)
    ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img = cv2.bitwise_not(img)
    return img

def getROI(img, cell:TableCell):
    x1, y1, x2, y2 = cell.getEffectiveBoundary()
    roi = img[(y1+TEXT_MARGIN):(y2-TEXT_MARGIN), (x1+TEXT_MARGIN):(x2-TEXT_MARGIN)]
    w,h = roi.shape
    if w * h < 400:
        return None
    return roi

def extractTextFromCell(img, cell:TableCell) -> str:
    roi = getROI(img, cell)
    if roi is None:
        return ""
    roi = filterForOCR(roi)
    if roi is not None:
        return extractText(roi)
    else:
        return ""

def preprocessing(filename):
    if filename == 'STDIN':
        stdin = sys.stdin.buffer.read()
        # print(stdin)
        array = np.frombuffer(stdin, dtype='uint8')
        img = cv2.imdecode(array, 1)
    else:
        img =  cv2.imread(filename)
    thresh_inv = convertColor(img)   # 0. Converting color to grey & binarization
    resized = resizeImage(thresh_inv)   # 1. Resizing - Upsampling or Downsampling
    deskewed, angle = dk.deskew(resized) # 2. deskew
    denoized = denoize(deskewed) # 3. denoized
    # cv2.imwrite(IMG_FILE + ".denozied.png",denoized)
    return img, denoized, angle

def refineOCRText(text:str):
    text = text.replace('_','')
    text = text.replace('"','')
    return text

def getMatrixFromVHLine(img) -> TableMatrix:
    ypgs, _, _ = getHorizontalLines(img)
    getVerticalLines(img, ypgs)
    matrix = TableMatrix.getTableMatrixFromYPointGroups(ypgs)
    return matrix
def fillTextFromOCR(img, matrix:TableMatrix) -> TableMatrix:
    debug = False
    imgcopy = np.copy(img)
    showLineImageFromMatrix(imgcopy, matrix, debug)
    for rowinx, row in enumerate(matrix.cells):
        roi = img[row.ypg.y1:row.ypg.y1+row.ypg.height, 0:img.shape[1]]
        roicopy = np.copy(roi)
        roicopy = cv2.cvtColor(roicopy, cv2.COLOR_GRAY2RGB)
        for colinx, cell in enumerate(row.getIterable()):
            cv2.rectangle(roicopy,(cell.x1ext, cell.y1ext-row.ypg.y), (cell.x2ext, cell.y1ext-row.ypg.y), (randrange(255),randrange(255),randrange(255)),3)
            text = extractTextFromCell(img, cell)
            text = refineOCRText(text).strip()
            textns = text.replace(' ', '')
            if len(textns) > 0:
                cell.value = text
                debugShow('ROI', filterForOCR(getROI(img,cell)), debug)
                # print("roi & text : {}, {}".format(cell.getEffectiveBoundary(), textns))
                matchedKey = MajorKeyword.matchKeyword(textns)
                if matchedKey is not None:
                    # print('text:{}, matched Key:{}'.format(text, matchedKey.koreanKeyword))
                    cell.matchedKeyword = matchedKey.koreanKeyword
                residpattern = '\d{6}[-]*[1-4]\d{6}'
    return matrix
def calculateOriginalPoint(orgimg, resizedimg, skewness:float, matrix:TableMatrix):
    orgsize = (orgimg.shape[1], orgimg.shape[0])
    modsize = (resizedimg.shape[1], resizedimg.shape[0])
    linecopy = np.copy(orgimg)
    for inx in range(0,matrix.maxrow-1):
        row = matrix.getRow(inx)
        ypg = row.ypg
        snewRad = math.radians(skewness)
        for headCell in row.getIterable():
            headCell.orgbox = dk.recoverOriginalPoint(orgsize, modsize, snewRad, (headCell.x1ext, headCell.y1ext), (headCell.x2ext, headCell.y2ext))
            for linenum, line in enumerate(headCell.orgbox[:-1]):
                x1, y1 = line
                x2, y2 = headCell.orgbox[linenum+1]
                cv2.line(linecopy,(x1,y1), (x2,y2), (255,150,150), 3, 1)
            cv2.line(linecopy,(x2,y2), (headCell.orgbox[0][0],headCell.orgbox[0][1]), (255,150,150), 3, 1)
    return linecopy
def makeKeyValuePairs(matrix:TableMatrix) -> List[tuple(str,str)]:
    rtn:List[tuple(str,str)] = []
    for _, row in enumerate(matrix.cells):
        columnBasePairCount: int = 0
        rowBasePairCount:int = 0
        columns = row.getIterable()
        for _, cell in enumerate(columns[:-1]):
            keyvalue = { 'key': '' , 'value': ''}
            if cell.isKeywordCell():
                keyvalue['key'] = cell.value
                nextcell = row.getRightHeadCell(cell)
                lowercell = matrix.getLowerHeadCell(cell)
                if (nextcell is not None
                    and not nextcell.isKeywordCell()):
                    columnBasePairCount += 1
                    keyvalue['value'] = nextcell.value
                elif (lowercell is not None
                    and not lowercell.isKeywordCell()):
                    rowBasePairCount += 1
                    keyvalue['value'] = lowercell.value
            if len(keyvalue['key']) > 0:
                rtn.append((keyvalue['key'], keyvalue['value']))
    return rtn

def makeKeyValuePairWithTableCell(matrix:TableMatrix) -> List[tuple(TableCell, TableCell)]:
    rtn:List[tuple(TableCell, TableCell)] = []
    for _, row in enumerate(matrix.cells):
        columnBasePairCount: int = 0
        rowBasePairCount:int = 0
        columns = row.getIterable()
        for _, cell in enumerate(columns[:-1]):
            keycell = None
            valuecell = None
            if cell.isKeywordCell():
                keycell = cell
                nextcell = row.getRightHeadCell(cell)
                lowercell = matrix.getLowerHeadCell(cell)
                if (nextcell is not None
                    and not nextcell.isKeywordCell()):
                    columnBasePairCount += 1
                    valuecell =  nextcell
                elif (lowercell is not None
                    and not lowercell.isKeywordCell()):
                    rowBasePairCount += 1
                    valuecell = lowercell
            if keycell is not None and valuecell is not None and len(keycell.value) > 0:
                rtn.append((keycell, valuecell))
    return rtn

def makeMaskedImageFromOriginal(img, maskedCells:List[TableCell]):
    imgcopy = np.copy(img)
    for maskedCell in maskedCells:
        points = []
        for x,y in maskedCell.orgbox:
            points.append([int(x),int(y)])
        pointsnd = np.array(points, np.int32)
        # https://stackoverflow.com/questions/17241830/opencv-polylines-function-in-python-throws-exception
        pointsnd = pointsnd.reshape(-1,1,2)
        cv2.fillConvexPoly(imgcopy, pointsnd, (50,50,50), 8)
    return imgcopy

class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if is_dataclass(o):
            return asdict(o)
        return super().default(o)

def makeBoundary(img) -> Boundary:
    return Boundary(img.shape[0], img.shape[1])

def makeResultBody(orgboundary:Boundary, resizedboundary:Boundary, skewness:float, matrix:TableMatrix, keyvalues:List[tuple(TableCell, TableCell)]) -> ResultBody:
    imageSize = ImageSize(orgboundary, resizedboundary)
    kvs = []
    for key,value in keyvalues:
        obj = { 'key' : key.toDict(), 'value': value.toDict() }
        kvs.append(obj)
    result = ResultBody(imageSize, skewness, matrix.toDictHeadCells(), kvs)
    return result

def handleFile(filepath):
    debug = False
    orgimg, denoized, angle = preprocessing(filepath)
    if ENALBE_DENOIZED_IMAGE_STORED:
        cv2.imwrite(filepath + DENOIZED_IMAGE_FILE_EXT)
    matrix = getMatrixFromVHLine(denoized)
    matrix = fillTextFromOCR(denoized, matrix)
    orgline = calculateOriginalPoint(orgimg, denoized, angle, matrix)
    debugShow('orglines', orgline, debug)
    kvs = makeKeyValuePairWithTableCell(matrix)
    masked = orgimg
    maskedCells = []
    for k,v in kvs:
        # print('key:{}, value:{}'.format(k.value,v.value))
        if k.matchedKeyword == '주민등록번호':
            maskedCells.append(v)
            # print(v.__getstate__())
    masked = makeMaskedImageFromOriginal(masked, maskedCells)
    if ENABLE_MASKED_IMAGE_STORED:
        cv2.imwrite(filepath + MASKED_IMAGE_FILE_EXT)
    debugShow('maskedimage', masked, debug) 
    orgboundary = makeBoundary(orgimg)
    resizedboundary = makeBoundary(denoized)
    result = makeResultBody(orgboundary, resizedboundary, angle, matrix, kvs)
    cv2.destroyAllWindows()
    return result.toJson(4,True)

if __name__ == '__main__':
    args=sys.argv[1:]
    resultBody = handleFile(args[0])
    print(resultBody)

