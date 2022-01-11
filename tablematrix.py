from __future__ import annotations
from dataclasses import dataclass, field
from typing import List
from enum import Enum
import numpy as np
import math
import json

from numpy.lib.function_base import _calculate_shapes

VERTICAL_THRESHOLD_DEGREE = 2 # It means 88 ~ 92 degrees
VERTICAL_LINE_GAP_PERCENT = 9
X_THRESHOLD = 5
Y_THRESHOLD = 20

@dataclass(init=False)
class XPointGroup:
    x: int = 0
    minx: int = 10000
    maxx: int = -1
    sum: int = 0
    cnt: int = 0
    __xps: List[int]
    y1: int = 0
    y2: int = 0
    __prev: XPointGroup = field(repr=False)
    __next: XPointGroup = field(repr=False)
    def __init__(self) -> None:
        self.__xps = []
        self.__prev = None
        self.__next = None
    def xpoints(self) -> List[int]:
        return self.__xps
    def addVerticalLine(self, x:int, y1:int, y2:int) -> int:
        self.sum += x
        self.cnt += 1
        if x < self.minx:
            self.minx = x
        if x > self.maxx:
            self.maxx = x
        self.x = int(self.sum / self.cnt)
        self.xpoints().append(x)
        if self.y1 > y1:
            self.y1 = y1
        if self.y2 < y2:
            self.y2 = y2
        return self.x
    def isLocated(self, x:int, threshold:int, delta:int=0) -> bool:
        adjx = x+delta
        return (abs(adjx-self.x) <= threshold) # or (self.minx <= adjx and adjx <= self.maxx))
    def setUpperXPG(self,upperXpg:XPointGroup):
        self.__prev = upperXpg
        upperXpg.__next = self
    @staticmethod
    def isVerticalLine(x1, y1, x2, y2, xthreshold, marginal_space:int, partition_height:int):
        if y1 > y2:
            y1, y2 = y2, y1
        return (abs(x1-x2) <= xthreshold) and (
            (y1 <= marginal_space) and (abs(partition_height-y2) <= marginal_space)
        )
    @staticmethod
    def findClosestXPG(xpgs:List[XPointGroup], x) -> XPointGroup:
      distance = 10000
      rtn = None
      for xpg in xpgs:
        if abs(xpg.x - x) < distance:
          distance = abs(xpg.x - x)
          rtn = xpg
      return rtn

class TLProp(Enum):
  OPEN = 0
  PARTIAL_CLOSE = 1
  FULL_CLOSE = 2
  HV_FULL_CLOSE = 4
  MG_PARTIAL_CLOSE = 8
  MG_FULL_CLOSE = 16

@dataclass(init=False)
class YPointGroup:
    x1: int
    y1: int
    x2: int
    y2: int
    y: int 
    lines: List[tuple(int, int)]
    xdelta: int
    height: int
    __xpgs: List[XPointGroup]
    __prev: YPointGroup = field(repr=False)
    __next: YPointGroup = field(repr=False)
    def __init__(self, x1:int, y1:int, x2:int, y2:int):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.y = y1
        self.__prev = None
        self.__next = None
        self.__xpgs = []
        self.lines = []
        self.addHonrizontalLine(x1, x2)
    def xPointGroups(self):
        return self.__xpgs
    @staticmethod
    def checkExistingInYGs(ygs:List[YPointGroup], x1:int, y1:int, x2:int, y2:int) -> YPointGroup:
        exist = False
        if abs(y1-y2) > Y_THRESHOLD // 2:
            return None
        for oneypg in ygs:
            if oneypg.y1 > oneypg.y2:
                y1, y2 = y2, y1
            if abs(oneypg.y - y1) < Y_THRESHOLD:
                exist = True
                return oneypg
        if not exist :
            newone = YPointGroup(x1,y1,x2,y2)
            ygs.append(newone)
            return newone
    @staticmethod
    def checkAndAddHorizontalLine(ygs:List[YPointGroup], x1:int, y1:int, x2:int, y2:int) -> None:
        ypg = YPointGroup.checkExistingInYGs(ygs, x1, y1, x2, y2)
        if ypg is not None:
            ypg.addHonrizontalLine(x1, x2)
    def hasOverwrappedHLine(self, x1:int, x2:int) -> bool:
      if x1 > x2:
        x1, x2 = x2, x1
      for hlinex1, hlinex2 in self.lines:
        if hlinex1 > hlinex2:
          hlinex1, hlinex2 = hlinex2, hlinex1
        if (hlinex1 - X_THRESHOLD) <= x1 and (hlinex2 + X_THRESHOLD) >= x2:
          return True
      return False
    def hasPartiallyOverwrappedHLine(self, x1:int, x2:int) -> bool:
      if x1 > x2:
        x1, x2 = x2, x1
      for hlinex1, hlinex2 in self.lines:
        if hlinex1 > hlinex2:
          hlinex1, hlinex2 = hlinex2, hlinex1
        if (hlinex1 - X_THRESHOLD) <= x2 and (hlinex2 + X_THRESHOLD) >= x2:
          return True
      return False
    def getToplineProperty(self, x1:int, x2:int) -> TLProp:
      if self.hasOverwrappedHLine(x1, x2):
        return TLProp.FULL_CLOSE
      elif self.hasPartiallyOverwrappedHLine(x1, x2):
        return TLProp.PARTIAL_CLOSE
      else:
        return TLProp.OPEN
    def checkAndAddVerticalLine(self, x1:int, y1:int, x2:int, y2:int):
        if self.getNextYPG() is None:
            return
        # print('height, gap: {}, {}'.format(self.height, int(self.height * 0.095)))
        xthreshold = int(self.height * math.tan(math.radians(VERTICAL_THRESHOLD_DEGREE)))
        if not XPointGroup.isVerticalLine(x1, y1, x2, y2, xthreshold, self.height, int(self.height * VERTICAL_LINE_GAP_PERCENT / 100)):
            return
        exist = False
        y1 += self.y1
        y2 += self.y1
        for xpg in self.xPointGroups():
            if xpg.isLocated(x1, X_THRESHOLD):
                xpg.addVerticalLine(x1, y1, y2)
                exist = True
                break
        if not exist:
            xpg = XPointGroup()
            xpg.addVerticalLine(x1, y1, y2)
            self.xPointGroups().append(xpg)
    def setPrevYPointGroup(self,ypg:YPointGroup):
        self.__prev = ypg
        ypg.__next = self
        ypg.height = self.y1 - ypg.y1
    def getPreviousYPG(self):
        return self.__prev
    def getNextYPG(self):
        return self.__next
    def getXPointGroups(self):
        return self.__xpgs
    def haveSameHLine(self, x1:int, x2:int) -> bool:
      for orgx1, orgx2 in self.lines:
        if orgx1 > orgx2:
          orgx1, orgx2 = orgx2, orgx1
        if orgx1 == x1 and orgx2 == x2:
          return True
      return False
    def mergeOverwrappedHLine(self, x1:int, x2:int) -> tuple(bool, int, int):
        if x1 > x2:
          x1, x2 = x2, x1
        for inx, prevline in enumerate(self.lines):
          orgx1, orgx2 = prevline
          if ((x1 <= orgx1 <= x2) or
            (x1 <= orgx2 <= x2) or 
            (orgx1 <= x1 and x2 <= orgx2)):
            self.lines.remove(prevline)
            minx1 = min(orgx1, x1)
            maxx2 = max(orgx2, x2)
            if not self.haveSameHLine(minx1, maxx2):
              self.lines.append((minx1, maxx2))
            return True, minx1, maxx2
        return False, x1, x2
    def addHonrizontalLine(self, x1:int, x2:int) -> None:
        # If there is cross border lines in list, the previous line should be expended.
        if x1>x2:
            x1,x2 = x2,x1
        exist = False
        while True:
          flag, x1, x2 = self.mergeOverwrappedHLine(x1, x2)
          if self.haveSameHLine(x1, x2):
            return
          if flag == False:
            break
          exist = True
        if not exist:
          self.lines.append((x1,x2))
    def calculateDelta(self) -> int:
        if self.__prev is None:
            self.xdelta = 0
            return 0
        xdeltas = []
        for prevxpg in self.__prev.xPointGroups():
            for curxpg in self.xPointGroups():
                if prevxpg.maxx >= curxpg.x and prevxpg.minx <= curxpg.x:
                    xdeltas.append(curxpg.x - prevxpg.x)
                    curxpg.setUpperXPG(prevxpg)
        if len(xdeltas) == 0:
            self.xdelta = 0
        else:
            self.xdelta = int(np.median(xdeltas))
        return self.xdelta
    @staticmethod
    def sortHorizontalLineAndLink(ypgs:List[YPointGroup]):
        ypgs.sort(key=lambda ypg:ypg.y1)
        for inx,value in enumerate(ypgs):
            if inx == 0:
                continue
            value.setPrevYPointGroup(ypgs[inx-1])
    @staticmethod
    def calculatedAllHorizontalLineDelta(ypgs:List[YPointGroup]) -> None:
        for yg in ypgs:
            yg.calculateDelta()
    @staticmethod
    def getAverageSpanHeightAndGap(ypgs:List[YPointGroup]) -> tuple(int, int):
        horizontals = [item.y1 for item in ypgs]
        horizontals = np.sort(horizontals)
        heights = np.diff(horizontals)
        if(len(heights)>0):
            average_span_height = np.median(heights)
            vertical_max_gap = int(average_span_height * VERTICAL_LINE_GAP_PERCENT / 100)
            # print('avg height, vertical gap:{}, {}'.format(average_span_height, vertical_max_gap))
            return average_span_height, vertical_max_gap
        else:
            return 0,0

YPointGroups = List[YPointGroup]

@dataclass(init=False)
class TableCell:
  row: int = 0
  col: int = 0
  rowspan: int = 1
  colspan: int = 1
  x1: int = 0
  y1: int = 0
  x2: int = 0
  y2: int = 0
  orgbox: List[tuple(int, int)] # x, y
  topline: TLProp = TLProp.OPEN
  isMerged: bool = False
  mergeHead: TableCell = field(repr=False)
  x1ext: int = 0
  y1ext: int = 0
  x2ext: int = 0
  y2ext: int = 0
  value: str = ""
  matchedKeyword: str = ''
  tlext: TLProp = TLProp.OPEN
  xpg : XPointGroup = None
  ypg : YPointGroup = None
  __leafs: List["TableCell"] = field(repr=False)
  def __init__(self, row:int, col:int, x1:int, y1:int, x2:int, y2:int, tlprop:TLProp):
    self.row = row
    self.col = col
    self.x1ext = self.x1 = x1
    self.y1ext = self.y1 = y1
    self.x2ext = self.x2 = x2
    self.y2ext = self.y2 = y2
    self.topline = tlprop
    self.tlext = tlprop
    self.__leafs = []
    self.__leafs.append(self)
    self.orgbox = []
    self.mergeHead = self
  def checkMergedCellTopline(self):
    openedcell = []
    closedcell = []
    for leaf in self.__leafs:
      if leaf.row == self.mergeHead.row:
        if leaf.topline == TLProp.OPEN:
          openedcell.append(leaf)
        else:
          closedcell.append(leaf)
    if len(openedcell) == 0:
      self.tlext = TLProp.MG_FULL_CLOSE
    elif len(closedcell) != 0:
      self.tlext = TLProp.MG_PARTIAL_CLOSE
    else:
      self.tlext = TLProp.OPEN
  def isKeywordCell(self) -> bool:
    return len(self.matchedKeyword) > 0
  def getEffectiveBoundary(self) -> tuple(int, int, int, int):
    if self.isMerged:
      return self.mergeHead.x1ext, self.mergeHead.y1ext, self.mergeHead.x2ext, self.mergeHead.y2ext
    return self.x1, self.y1, self.x2, self.y2
  def appendLeafCell(self, newLeaf:TableCell):
    self.__leafs.append(newLeaf)
    newLeaf.mergeHead = self
    newLeaf.isMerged = True
  def mergeCol(self, cell:TableCell) -> None:
    mergeHead = self.mergeHead
    absorbedHead = cell.mergeHead
    if mergeHead == absorbedHead:
      raise Exception("Can't merge same cells. row/col:{}, {}".format(self.row, self.col))
    if mergeHead.rowspan != absorbedHead.rowspan:
      raise Exception("Can't merge two cells with different rowspan sizes. rowspan source, target:{}, {}".format(mergeHead.rowspan, absorbedHead.rowspan))
    if mergeHead.row != absorbedHead.row:
      raise Exception("Can't merge two cells on different row. row source, target:{}, {}".format(mergeHead.row, absorbedHead.row))
    if mergeHead.col > absorbedHead.col:
      absorbedHead.mergeCol(mergeHead)
      return
    for leafCell in absorbedHead.__leafs:
      mergeHead.appendLeafCell(leafCell)
    self.isMerged = True
    mergeHead.colspan += absorbedHead.colspan
    mergeHead.x2ext = absorbedHead.x2ext
    mergeHead.checkMergedCellTopline()
  def mergeRow(self, cell:TableCell) -> None:
    mergeHead = self.mergeHead
    absorbedHead = cell.mergeHead
    if mergeHead == absorbedHead:
      raise Exception("Can't merge same cells. row/col:{}, {}".format(self.row, self.col))
    if mergeHead.colspan != absorbedHead.colspan:
      raise Exception("Can't merge two cells with different colspan sizes. colspan source, target:{}, {}".format(mergeHead.colspan, absorbedHead.colspan))
    if mergeHead.col != absorbedHead.col:
      raise Exception("Can't merge two cells on different col. row source, target:{}, {}".format(mergeHead.col, absorbedHead.col))
    if mergeHead.row > absorbedHead.row:
      absorbedHead.mergeRow(mergeHead)
      return
    for leafCell in absorbedHead.__leafs:
      mergeHead.appendLeafCell(leafCell)
    self.isMerged = True
    mergeHead.rowspan += absorbedHead.rowspan
    mergeHead.y2ext = absorbedHead.y2ext
    mergeHead.checkMergedCellTopline()
  def merge(self, cell:TableCell):
    if cell == self:
      return
    mergeHead = self.mergeHead
    absorbedHead = cell.mergeHead
    if (mergeHead.row == absorbedHead.row 
      and mergeHead.col != absorbedHead.col):   # column merge
      self.mergeCol(cell)
    elif (mergeHead.col == absorbedHead.col
      and mergeHead.row != absorbedHead.row) :
      self.mergeRow(cell)
    else:
      raise Exception("Can't merge two unalignead cells. cell1(row:{}, col:{}) cell2(row:{}, col:{})."
        .format(mergeHead.row, mergeHead.col, absorbedHead.row, absorbedHead.col))
  def unmerge(self):
    if not self.isMerged:
      return
    for leaf in self.__leafs:
      leaf.isMerged = False
      leaf.colspan = 1
      leaf.rowspan = 1
      leaf.x1ext = leaf.x1
      leaf.y1ext = leaf.y1
      leaf.x2ext = leaf.x2
      leaf.y2ext = leaf.y2
      leaf.tlext = leaf.topline
      leaf.__leafs.clear()
      leaf.__leafs.append(leaf)
  def __getstate__(self):
    state=self.__dict__.copy()
    state.pop('__leafs', None)
    state.pop('mergeHead', None)
    state.pop('xpg', None)
    state.pop('ypg', None)
    state.pop('tlext', None)
    state.pop('topline', None)
    state.pop('_TableCell__leafs', None)
    return state
  def toDict(self):
    rtn = {}
    rtn['row'] = int(self.row)
    rtn['col'] = int(self.col)
    rtn['rowspan'] = int(self.rowspan)
    rtn['colspan'] = int(self.colspan)
    rtn['x1'] = int(self.x1)
    rtn['y1'] = int(self.y1)
    rtn['x2'] = int(self.x2)
    rtn['y2'] = int(self.y2)
    rtn['orgbox'] = self.orgbox
    rtn['isMerged'] = self.isMerged
    rtn['x1ext'] = int(self.x1ext)
    rtn['y1ext'] = int(self.y1ext)
    rtn['x2ext'] = int(self.x2ext)
    rtn['y2ext'] = int(self.y2ext)
    rtn['value'] = self.value
    return rtn
@dataclass(init=False)
class TableRow:
  __cols: List[TableCell]
  ypg: YPointGroup
  maxcol: int
  def __init__(self, ypg:YPointGroup, cols:List[TableCell]=[]):
    self.__cols = cols
    self.ypg = ypg
    self.maxcol = len(self.__cols)
  def getCell(self,col:int) -> TableCell:
    if (col < 0 or col >= self.maxcol):
      raise Exception("Can't retrieve out of boundary cell. max cols, requested col:{}, {}".format(self.maxcol, col))
    return self.__cols[col]
  def getRightHeadCell(self, cell:TableCell) -> TableCell:
    headerCell = cell.mergeHead
    if self.maxcol < headerCell.col+headerCell.colspan:
      return None
    return self.__cols[cell.col+cell.colspan].mergeHead
  def getLeftHeadCell(self, cell:TableCell) -> TableCell:
    headerCell = cell.mergeHead
    if headerCell.col < 1 and (headerCell.col - 1) >= self.maxcol:
      return None
    return self.__cols[headerCell.col-1].mergeHead
  def getIterable(self) -> List[TableCell]:
    rtn:List[TableCell] = []
    for cell in self.__cols:
      if cell.mergeHead == cell:
        rtn.append(cell)
    return rtn
  def __getstate__(self):
    state=self.__dict__.copy()
    state.pop('ypg', None)
    return state
  def toJson(self):
    rtn = []
    for col in self.__cols:
      rtn.append(col.__getstate__())
      print(col.__getstate__())
    return json.dumps(rtn)
  def toDict(self):
    rtn = []
    for col in self.__cols:
      rtn.append(col.__getstate__())
      print(col.__getstate__())
    return rtn
  def toDictHeadCells(self):
    rtn = []
    for col in self.getIterable():
      rtn.append(col.toDict())
      # print(col.__getstate__())
    return rtn

@dataclass(init=False)
class TableCol:
  __rows: List[TableCell]
  def __init__(self, cells:List[TableCell]):
    self.__rows = cells
  def getCell(self,row:int) -> TableCell:
    if(row < 0 or row >= len(self.__rows)):
      raise Exception("Can't retreive out of boundary cell. max row,requested row:{}, {}".format(len(self.__rows), row))
    return self.__rows[row]
  def getUpperHeadCell(self, cell:TableCell) -> TableCell:
    headCell = cell.mergeHead
    if headCell.row < 1:
      return None
    return self.getCell(headCell.row-1).mergeHead
  def getLowerHeadCell(self, cell:TableCell) -> TableCell:
    headCell = cell.mergeHead
    if headCell.row + headCell.rowspan >= len(self.__rows):
      return None
    return self.getCell(headCell.row + headCell.rowspan).mergeHead
  def getHeadList(self) -> List[TableCell]:
    rtn:List[TableCell] = []
    for cell in self.__rows:
      if cell.mergeHead == cell:
        rtn.append(cell)
    return rtn

@dataclass(init=False)
class TableMatrix:
  cells: List[TableRow]
  maxrow: int
  maxcol: int
  xpoints: List[int]
  def __init__(self, xpoints:List[int]):
    self.xpoints = xpoints
    self.maxcol = len(xpoints)
    self.maxrow = 0
    self.cells = []
  def addRow(self,ypg:YPointGroup):
    if ypg.getNextYPG() is None:
      return
    nextypg = ypg.getNextYPG()
    y1 = ypg.y1
    y2 = nextypg.y1
    oneRow:List[TableCell] = []
    mergeHeader = None
    for index, x in enumerate(self.xpoints[:-1]):
      x1 = x
      x2 = self.xpoints[index+1]
      tlprop = ypg.getToplineProperty(x1, x2)
      cell:TableCell = TableCell(self.maxrow, index, x1, y1, x2, y2, tlprop)
      cell.ypg = ypg
      if mergeHeader is None:
        mergeHeader = cell
      closestxpg = XPointGroup.findClosestXPG(ypg.getXPointGroups(), x)
      if closestxpg is not None and closestxpg.isLocated(x, X_THRESHOLD):
        cell.xpg = closestxpg
        mergeHeader = cell
      else:
        mergeHeader.merge(cell)
      oneRow.append(cell)
    self.cells.append(TableRow(ypg,oneRow))
    self.maxrow += 1
  def calculateRowspan(self) -> None:
    for col in range(self.maxcol-1):
      tablecol:TableCol = self.getCol(col)
      headlist:List[TableCell] = tablecol.getHeadList()
      for inx,headCell in enumerate(headlist[:-1]):
        if headCell != headCell.mergeHead:
          continue
        nexthead = headlist[inx+1]
        if ( headCell.colspan == nexthead.colspan 
          and nexthead.tlext == TLProp.OPEN):
          headCell.merge(nexthead)
        # if headCell.colspan == nexthead.
  def getCol(self,col:int) -> TableCol:
    cols:List[TableCell] = [item.getCell(col) for item in self.cells]
    return TableCol(cols)
  def getRow(self,row:int) -> TableRow:
    return self.cells[row]
  def getCell(self, row:int, col:int) -> TableCell:
    return self.getRow(row).getCell(col)
  def getUpperHeadCell(self,cell:TableCell) -> TableCell:
    if cell.row < 1 or (cell.row-1) > self.maxrow:
      raise None
    return self.getCell(cell.row-1,cell.col).mergeHead
  def getLowerHeadCell(self, cell:TableCell) -> TableCell:
    if cell.row+1 >= self.maxrow:
      return None
    return self.getCell(cell.row+1,cell.col).mergeHead
  @staticmethod
  def getTableMatrixFromYPointGroups(ypgs:List[YPointGroup]) -> TableMatrix:
    xpoints:List[int] = TableMatrix.calculateAllColumns(ypgs)
    matrix:TableMatrix = TableMatrix(xpoints)
    for ypg in ypgs:
      matrix.addRow(ypg) 
    matrix.calculateRowspan()
    return matrix
  @staticmethod
  def calculateAllColumns(ypgs:List[YPointGroup]) -> List[int]:
    xpoints:List[int] = [0]
    for rowinx, ypg in enumerate(ypgs):
      for xpg in ypg.getXPointGroups():
        exist:bool = False
        for compared in xpoints:
          if xpg.isLocated(compared, X_THRESHOLD):
            exist = True
            break
        if not exist:
          xpoints.append(xpg.x)
      #print('rowinx,xps:{}, {}'.format(rowinx,sorted([item.x for item in ypg.getXPointGroups()])))
      xpoints.sort()
      #print("all xpoints:{}".format(xpoints))
    return xpoints
  def __getstate__(self):
    state=self.__dict__.copy()
    state.pop('xpoints', None)
    return state
  def toJson(self):
    state=self.__dict__.copy()
    state.pop('xpoints', None)
    state.pop('cells', None)
    rows = []
    for row in self.cells:
      rowdict = []
      rowdict.append(row.toJson())
    state['tables'] = json.dumps(rows)
    return json.dumps(state)
  def toDict(self):
    state=self.__dict__.copy()
    state.pop('xpoints', None)
    state.pop('cells', None)
    rows = []
    for row in self.cells:
      rowdict = []
      rowdict.append(row.toDict())
    state['tables'] = rows
    return state
  def toDictHeadCells(self):
    state=self.__dict__.copy()
    state.pop('xpoints', None)
    state.pop('cells', None)
    rows = []
    for row in self.cells:
      rows.append(row.toDictHeadCells())
    state['tables'] = rows
    return state
if __name__ == '__main__':
  cell = TableCell(1,1,10,10,10,10,TLProp.FULL_CLOSE)
  print(cell.__getstate__())

