# -*- coding: utf-8 -*-
"""
Created on Wed Apr  10:39:52 2021

@author: CGBell
"""

from PyQt5.QtWidgets import QCheckBox, QPushButton, QLabel, QMainWindow, QSpinBox
from PyQt5.QtWidgets import QApplication, QWidget, QHBoxLayout, QComboBox, QTabWidget
from PyQt5.QtWidgets import QVBoxLayout, QSizePolicy
from PyQt5.QtGui import QPixmap, QImage, QColor, QPainter, QPalette
from PyQt5.QtCore import Qt, QPoint, QSize, QRectF
import sys
import os

import inspect
import numpy as np
import math as m
from skimage.segmentation import mark_boundaries, flood_fill, flood
from skimage.util import invert
from PIL import Image

from ParticleSpy.segptcls import process
from ParticleSpy.ParticleAnalysis import parameters, trainableParameters
from ParticleSpy.segimgs import ClusterTrained, toggle_channels

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

class ToolButton(QPushButton):

    def __init__(self, tool):
        super().__init__()
        self.setAutoExclusive(True)
        self.setCheckable(True)

class QPaletteButton(QPushButton):

    def __init__(self, color):
        super().__init__()
        self.setAutoExclusive(True)
        self.setCheckable(True)
        self.setFixedSize(QSize(24,24))
        self.color = color
        self.setStyleSheet("background-color: %s;" % color)



class Canvas(QLabel):

    def __init__(self,pixmap):
        super().__init__()
        self.OGpixmap = pixmap
        self.lastpixmap = pixmap

        self.setPixmap(pixmap)
        self.scaleFactor = 1
        #self.setBackgroundRole(QPalette.Base)
        #self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        #self.setScaledContents(True)

        self.setMouseTracking(True)
        self.first_click = None
        self.last_click  = None

        self.brush_tools = ['Freehand', 'Line', 'Polygon']
        #self.colors is ARGB
        self.colors =['#80A30015', '#806DA34D', '#8051E5FF', '#80BD2D87', '#80F5E663']
        self.color_index = 0

        self.pen_color = QColor(self.colors[0])
        self.penType = self.brush_tools[0]
        self.lineCount = 0

        self.array = np.zeros((1024,1024,3),dtype=np.uint8)

    def set_pen_color(self, c):
        self.color_index = c
        self.pen_color = QColor(self.colors[self.color_index])

    def changePen(self, brush):
        self.last_click = None
        self.lineCount = 0
        self.penType = brush

    def clearCanvas(self):

        self.last_click = None
        self.first_click = None
        self.lineCount = 0

        painter = QPainter(self.pixmap())
        painter.eraseRect(0,0,1024,1024)
        painter.drawPixmap(0,0,self.OGpixmap)
        painter.end()
        self.update()

    def clearLabels(self):
        self.array = np.zeros((1024,1024,3), dtype=np.uint8)

    def redrawLabels(self):
        array = toggle_channels(self.array)
        self.drawLabels(array)

    def drawLabels(self, thin_labels):

        shape = thin_labels.shape
        thicc_labels = np.zeros([shape[0], shape[1],4], dtype=np.uint8)

        thicc_labels[:,:,1:] = toggle_channels(thin_labels)
        thicc_labels[:,:,0] = (thin_labels > 0)*255

        thicc_labels = np.flip(thicc_labels, axis=2).copy()
        qi = QImage(thicc_labels.data, thicc_labels.shape[1], thicc_labels.shape[0], 4*thicc_labels.shape[1], QImage.Format_ARGB32_Premultiplied)
        
        pixmap = QPixmap(qi)
        pixmap = pixmap.scaled(1024, 1024, Qt.KeepAspectRatio)
        painter = QPainter(self.pixmap())
        painter.setOpacity(0.5)
        painter.drawPixmap(0,0,pixmap)
        painter.end()
        self.update()

    def lineDraw(self,pos1,pos2):
        painter = QPainter(self.pixmap())
        p = painter.pen()
        p.setWidth(3)
        p.setColor(self.pen_color)
        painter.setPen(p)
        painter.drawLine(pos1, pos2)
        painter.end()
        self.update()

    def LineTool(self,e):
        if self.lineCount == 0:
            self.last_click = QPoint(e.x(),e.y())
            self.lineCount = 1
        else:
            self.lineDraw(self.last_click,e.pos())
            self.last_click = QPoint(e.x(),e.y())
            self.lineCount = 0
            midline = (self.last_click + e.pos())/2
            self.flood(midline)

    def PolyTool(self,e):
        if self.lineCount == 0:
            self.first_click = QPoint(e.x(),e.y())
            self.last_click =  QPoint(e.x(),e.y())
            self.lineCount = 1

        elif self.lineCount == 1:
            self.lineDraw(self.last_click,e.pos())
            self.last_click = QPoint(e.x(),e.y())
            self.lineCount += 1

        elif self.lineCount > 1:
            d_x, d_y = float(self.first_click.x()-e.x()), float(self.first_click.y() - e.y())
            d_from_origin = m.sqrt((d_x)**2 + (d_y)**2)

            if d_from_origin < 10:
                
                self.lineDraw(self.last_click, self.first_click)
                self.last_click = None
                self.first_click = None
                self.lineCount = 0
            else:
                self.lineDraw(self.last_click, e.pos())
                self.last_click = QPoint(e.x(),e.y())
                self.lineCount += 1
    
    def flood(self, e):
        image = self.pixmap().toImage()
        b = image.bits()
        b.setsize(1024 * 1024 * 4)
        arr = np.frombuffer(b, np.uint8).reshape((1024, 1024, 4))
        
        OGimage = self.OGpixmap.toImage()
        b = OGimage.bits()
        b.setsize(1024 * 1024 * 4)
        OGim = np.frombuffer(b, np.uint8).reshape((1024, 1024, 4))
        OGim = np.flip(OGim, axis=2)

        arr = arr.astype(np.int32)
        arr = np.flip(arr, axis=2)

        arr_test = np.zeros_like(arr)
        arr_test[arr != OGim] = arr[arr != OGim]
        flat_arr = np.mean(arr_test[:,:,1:], axis=2)
        flooded = flood(flat_arr,(e.y(),e.x()))

        color = self.colors[self.color_index]
        rgb = [int(color[3:5], 16), int(color[5:7], 16), int(color[7:], 16)]

        #paint_arr is ARGB
        paint_arr = np.zeros_like(arr,dtype=np.uint8)
        paint_arr[:,:,0] = (flooded)*255
        #sets alpha
        paint_arr[flooded,1:] = rgb
        #fills wil pen colour
        
        #BGRA
        paint_arr = np.flip(paint_arr, axis=2).copy()
        
        qi = QImage(paint_arr.data, paint_arr.shape[1], paint_arr.shape[0], 4*paint_arr.shape[1], QImage.Format_ARGB32_Premultiplied)
        pixmap = QPixmap(qi)
        
        painter = QPainter(self.pixmap())
        painter.setOpacity(0.5)
        painter.drawPixmap(0,0,pixmap)
        painter.end()
        self.update()
        
        #self.array saves RGB values
        self.array += np.flip(paint_arr[:,:,:3], axis=2)

    def wheelEvent(self, event):
        if event.angleDelta().y() > 0:
            self.scaleImage(1.25)
        elif self.scaleFactor > 1:
            self.scaleImage(0.8)

    def scaleImage(self, factor):
        self.scaleFactor *= factor
        self.resize(self.scaleFactor * self.pixmap().size())

    def mousePressEvent(self, e):

        if e.button() == Qt.RightButton:
            self.flood(e)

        
        if e.button() ==Qt.LeftButton:

            if self.penType == 'Line':
                self.LineTool(e)
                    
            if self.penType == 'Polygon':
                self.PolyTool(e)


    def mouseMoveEvent(self, e):
        
        if e.buttons() == Qt.LeftButton:
            if self.last_click is None: # First event.
                self.last_click = QPoint(e.x(),e.y())
                return # Ignore the first time.
            
            if self.penType == 'Freehand':
                self.lineDraw(self.last_click, e.pos())
                # Update the origin for next time.
                self.last_click = QPoint(e.x(),e.y())

    def mouseReleaseEvent(self, e):
        if self.penType == 'Freehand':
            self.last_click = None
        
    def savearray(self,image):
        resized = np.array(Image.fromarray(self.array).resize((image.shape[1],image.shape[0])))
        np.save(os.path.dirname(inspect.getfile(process))+'/parameters/manual_mask',resized)