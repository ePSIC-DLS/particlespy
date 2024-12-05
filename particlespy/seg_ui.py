# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 15:50:08 2018

@author: qzo13262
"""

import inspect
import math as m
import os
import sys

import numpy as np
from PIL import Image
from PyQt5.QtCore import QPoint, QRectF, QSize, Qt
from PyQt5.QtGui import QColor, QImage, QPainter, QPalette, QPixmap
from PyQt5.QtWidgets import (QApplication, QButtonGroup, QCheckBox, QComboBox,
                             QHBoxLayout, QLabel, QMainWindow, QPushButton,
                             QSizePolicy, QSpinBox, QTabWidget, QVBoxLayout,
                             QWidget)
from skimage.segmentation import flood, flood_fill, mark_boundaries
from skimage.util import invert
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from particlespy.particle_analysis import parameters, trainable_parameters
from particlespy.segimgs import cluster_trained, toggle_channels
from particlespy.segptcls import process


class Application(QMainWindow):

    def __init__(self,im_hs,height):
        super().__init__()
        self.setWindowTitle("Segmentation UI")
        self.imflag = "Image"
        
        self.getim(im_hs)
        self.getparams()
        
        self.prev_params = parameters()
        self.prev_params.generate()
        
        offset = 50
        self.canvas_size = [height,int(self.image.shape[1]/self.image.shape[0]*height)]
        self.layout = QHBoxLayout(self)
        
        # Initialize tab screen
        self.tabs = QTabWidget()
        self.tab1 = QWidget()
        self.tab2 = QWidget()
        self.tab3 = QWidget()
        
        # Add tabs
        self.tabs.addTab(self.tab1,"Auto")
        self.tabs.addTab(self.tab2,"Manual")
        self.tabs.addTab(self.tab3,"Trainable")

        #self.central_widget = QWidget()               
        #self.setCentralWidget(self.central_widget)
        lay = QHBoxLayout()
        leftlay = QVBoxLayout()
        rightlay = QVBoxLayout()
        self.tab1.setLayout(lay)

        #tab 1
        self.label = QLabel(self)
        qi = QImage(self.image.data, self.image.shape[1], self.image.shape[0], self.image.shape[1], QImage.Format_Grayscale8)
        pixmap = QPixmap(qi)
        self.pixmap2 = pixmap.scaled(self.canvas_size[1], self.canvas_size[0], Qt.KeepAspectRatio)
        self.label.setPixmap(self.pixmap2)
        self.label.setGeometry(10,10,self.pixmap2.width(),self.pixmap2.height())
        
        height = max((self.pixmap2.height()+50,300 + offset)) #300 +50
        
        self.resize(self.pixmap2.width()+130, height)
        
        self.filt_title = QLabel(self)
        self.filt_title.setText('Pre-filtering options')
        
        self.sptxt = QLabel(self)
        self.sptxt.setText('Rolling ball size')
        
        self.sp = QSpinBox(self)
        self.sp.setMaximum(self.image.shape[0])
        self.sp.valueChanged.connect(self.rollingball)
        
        self.gausstxt = QLabel(self)
        self.gausstxt.setText('Gaussian filter kernel size')
        
        self.gauss = QSpinBox(self)
        self.gauss.setMaximum(self.image.shape[0])
        self.gauss.valueChanged.connect(self.gaussian)
        
        self.thresh_title = QLabel(self)
        self.thresh_title.setText('Thresholding options')
        
        self.comboBox = QComboBox(self)
        self.comboBox.addItem("Otsu")
        self.comboBox.addItem("Mean")
        self.comboBox.addItem("Minimum")
        self.comboBox.addItem("Yen")
        self.comboBox.addItem("Isodata")
        self.comboBox.addItem("Li")
        self.comboBox.addItem("Local")
        self.comboBox.addItem("Local Otsu")
        self.comboBox.addItem("Local+Global Otsu")
        self.comboBox.addItem("Niblack")
        self.comboBox.addItem("Sauvola")
        self.comboBox.activated[str].connect(self.threshold_choice)
        self.comboBox.activated.connect(self.updateLocalSize)
        
        self.localtxt = QLabel(self)
        self.localtxt.setText('Local filter kernel')
        
        self.local_size = QSpinBox(self)
        self.local_size.valueChanged.connect(self.local)
        self.local_size.setEnabled(False)
        
        cb = QCheckBox('Watershed', self)
        cb.stateChanged.connect(self.changeWatershed)
        
        self.ws_title = QLabel(self)
        self.ws_title.setText('Watershed Seed Separation')
        self.watershed_size = QSpinBox(self)
        self.watershed_size.setMaximum(self.image.shape[0])
        self.watershed_size.valueChanged.connect(self.watershed)
        self.watershed_size.setEnabled(False)
        
        self.wse_title = QLabel(self)
        self.wse_title.setText('Watershed Seed Erosion')
        self.watershed_erosion = QSpinBox(self)
        self.watershed_erosion.setMaximum(self.image.shape[0])
        self.watershed_erosion.valueChanged.connect(self.watershed_e)
        self.watershed_erosion.setEnabled(False)
        
        cb2 = QCheckBox('Invert', self)
        cb2.stateChanged.connect(self.changeInvert)
        
        self.minsizetxt = QLabel(self)
        self.minsizetxt.setText('Min particle size (px)')
        
        self.minsizev = QSpinBox(self)
        self.minsizev.setMaximum(self.image.shape[0]*self.image.shape[1])
        self.minsizev.valueChanged.connect(self.minsize)
        
        updateb = QPushButton('Update',self)
        updateb.clicked.connect(self.update)
        
        paramsb = QPushButton('Get Params',self)
        
        paramsb.clicked.connect(self.return_params)
        
        self.imagetxt = QLabel(self)
        self.imagetxt.setText('Display:')
        
        self.imBox = QComboBox(self)
        self.imBox.addItem("Image")
        self.imBox.addItem("Labels")
        
        self.imBox.activated[str].connect(self.changeIm)

        leftlay.addWidget(self.label)
        leftlay.addWidget(self.imagetxt)
        leftlay.addWidget(self.imBox)

        rightlay.addWidget(self.filt_title) 
        rightlay.addWidget(self.sptxt)
        rightlay.addWidget(self.sp)
        rightlay.addWidget(self.gausstxt)
        rightlay.addWidget(self.gauss)
        rightlay.addStretch(1)
        rightlay.addWidget(self.thresh_title)
        rightlay.addWidget(self.comboBox)
        rightlay.addStretch(1)
        rightlay.addWidget(self.localtxt)
        rightlay.addWidget(self.local_size)
        rightlay.addStretch(1)
        rightlay.addWidget(cb)
        rightlay.addWidget(self.ws_title)
        rightlay.addWidget(self.watershed_size)
        rightlay.addWidget(self.wse_title)
        rightlay.addWidget(self.watershed_erosion)
        rightlay.addStretch(1)
        rightlay.addWidget(cb2)
        rightlay.addStretch(1)
        rightlay.addWidget(self.minsizetxt)
        rightlay.addWidget(self.minsizev)
        rightlay.addStretch(1)
        rightlay.addWidget(updateb)
        rightlay.addWidget(paramsb)
        
        lay.addLayout(leftlay)
        lay.addLayout(rightlay)
        
        self.layout.addWidget(self.tabs)
        self.setLayout(self.layout)
        
        self.setCentralWidget(self.tabs)
        
        #Tab 2
        self.canvas = Canvas(self.pixmap2,self.canvas_size)
        #self.canvas = Drawer(self.pixmap2)
        
        self.getarrayb = QPushButton('Save Segmentation',self)
        self.getarrayb.clicked.connect(self.save_array)
        
        tab2layout = QVBoxLayout()
        tab2layout.addWidget(self.canvas)
        tab2layout.addWidget(self.getarrayb)
        tab2layout.addStretch(1)
        self.tab2.setLayout(tab2layout)

        #Tab 3
      

        self.mask = np.zeros([self.canvas_size[0],self.canvas_size[1],3])
        self.classifier = GaussianNB()
        self.tsparams = trainable_parameters()
        self.filter_kernels = ['Gaussian','Diff. Gaussians','Median','Minimum','Maximum','Sobel','Hessian','Laplacian','M-Sum','M-Mean','M-Standard Deviation','M-Median','M-Minimum','M-Maximum']

        lay3 = QHBoxLayout()
        im_lay = QVBoxLayout()
               
        self.button_lay = QVBoxLayout()

        lay3.addLayout(self.button_lay)
        lay3.addLayout(im_lay)

        self.canvas2 = Canvas(self.pixmap2,self.canvas_size)
        self.canvas2.setAlignment(Qt.AlignTop)
        
        self.tool_lay = QVBoxLayout()
        self.tool_group = QButtonGroup()
        for tool in self.canvas2.brush_tools:
            b = ToolButton(tool)
            b.pressed.connect(lambda tool=tool: self.canvas2.changePen(tool))
            b.setText(tool)
            self.tool_group.addButton(b)
            if tool == 'Freehand':
                b.setChecked(True)
            self.tool_lay.addWidget(b)
        self.button_lay.addItem(self.tool_lay)


        self.colour_lay = QHBoxLayout()
        for i in range(len(self.canvas2.colors)):
            c = self.canvas2.colors[i]
            b = QPaletteButton(c)
            b.pressed.connect(lambda i=i: self.canvas2.set_pen_color(i))
            if i== 0:
                b.setChecked(True)
            self.colour_lay.addWidget(b)
        self.button_lay.addLayout(self.colour_lay)
        im_lay.addWidget(self.canvas2)
        
        
        
        
        
        
        
        
        
        
        fk_lay = QVBoxLayout(self)
        fk_lay.setVerticalSpacing(0)
        self.kerneltxt = QLabel(self)
        self.kerneltxt.setText('Filter Kernels')     
        fk_lay.addWidget(self.kerneltxt)
        for t in range(8):
            b = QCheckBox(self.filter_kernels[t], self)
            b.pressed.connect(lambda tool=self.filter_kernels[t]: self.toggle_fk(tool))
            if t in (0,1,2,3,4,5,8):
                b.setChecked(True)
            fk_lay.addWidget(b)  
        
        self.membranetext = QLabel(self)
        self.membranetext.setText('Membrane Projections')
        fk_lay.addWidget(self.membranetext)

        for t in range(8,14):
            b = QCheckBox(self.filter_kernels[t][2:], self)
            b.pressed.connect(lambda tool=self.filter_kernels[t]: self.toggle_fk(tool))
            if t in (0,1,2,3,4,5,8):
                b.setChecked(True)
            fk_lay.addWidget(b)
            
        self.button_lay.addLayout(fk_lay)


        self.clf_lay = QVBoxLayout()
        self.kerneltxt = QLabel(self)
        self.kerneltxt.setText('Classifier')
        self.clf_lay.addWidget(self.kerneltxt)

        self.clfBox = QComboBox(self)
        self.clfBox.addItem("Random Forest")
        self.clfBox.addItem("Nearest Neighbours")
        self.clfBox.addItem("Naive Bayes")
        self.clfBox.addItem("QDA")
        self.clfBox.activated[str].connect(self.classifier_choice)
        self.clf_lay.addWidget(self.clfBox)
        
        self.button_lay.addLayout(self.clf_lay)
        



        fkp_lay = QVBoxLayout()
        self.ql1 = QLabel(self)
        self.ql1.setText('Sigma')
        fkp_lay.addWidget(self.ql1)
        
        self.spinb1 = QSpinBox(self)
        self.spinb1.valueChanged.connect(self.change_sigma)
        self.spinb1.setValue(1)
        fkp_lay.addWidget(self.spinb1)
        
        self.ql2 = QLabel(self)
        self.ql2.setText('High Sigma')
        fkp_lay.addWidget(self.ql2)
        
        self.spinb2 = QSpinBox(self)
        self.spinb2.valueChanged.connect(self.change_high_sigma)
        self.spinb2.setValue(16)
        fkp_lay.addWidget(self.spinb2)
        
        self.ql3 = QLabel(self)
        self.ql3.setText('Disk Size')
        fkp_lay.addWidget(self.ql3)
        
        self.spinb3 = QSpinBox(self)
        self.spinb3.valueChanged.connect(self.change_disk)
        self.spinb3.setValue(20)
        fkp_lay.addWidget(self.spinb3)
        
        self.button_lay.addLayout(fkp_lay)
        
             
        

        """
        self.config = QPushButton('Configure Filter Kernels', self)
        #self.config.clicked.connect()
        self.config.setToolTip('Choose individual filter kernel parameters')
        self.button_lay.addWidget(self.config)
        """
        
        self.clear = QPushButton('Clear Training Labels', self)
        self.clear.setToolTip('Removes existing training labels memory')
        self.clear.clicked.connect(self.canvas2.clearLabels)
        self.button_lay.addWidget(self.clear)

        self.redraw = QPushButton('Redraw Training Labels', self)
        self.redraw.setToolTip('draws any existing training labels in memory onto the canvas')
        self.redraw.clicked.connect(self.canvas2.redrawLabels)
        self.button_lay.addWidget(self.redraw)

        self.train = QPushButton('train classifier', self)
        self.train.pressed.connect(self.train_classifier)
        self.button_lay.addWidget(self.train)

        self.clear = QPushButton('Clear Canvas', self)
        self.clear.setToolTip('Removes any generated segmentation masks and labels from the image, does not clear training labels from memory')
        self.clear.clicked.connect(self.canvas2.clearCanvas)
        self.button_lay.addWidget(self.clear)

        self.getarrayc = QPushButton('Save and Close',self)
        self.getarrayc.clicked.connect(self.save_and_close)
        self.button_lay.addWidget(self.getarrayc)

        self.tab3.setLayout(lay3)

        self.show()

    def updateLocalSize(self):
        if self.comboBox.currentText() == 'Niblack' or self.comboBox.currentText() == 'Sauvola' or self.comboBox.currentText() == 'Local':
            self.local_size.setEnabled(True)
            self.local_size.setMinimum(1)
            self.local_size.setSingleStep(2)
            self.local_size.setMaximum(self.image.shape[0])
        elif self.comboBox.currentText() == "Local Otsu" or self.comboBox.currentText() == "Local+Global Otsu":
            self.local_size.setEnabled(True)
            self.local_size.setMaximum(self.image.shape[0])
        else:
            self.local_size.setEnabled(False)
    
    def getim(self,im_hs):
        self.im_hs = im_hs
        im = im_hs.data.astype(np.float64)
        im = im-np.min(im)
        image = np.uint8(255*im/np.max(im))
        self.image = image
        
    def getparams(self):
        self.params = parameters()
        self.params.generate()
        
    def changeIm(self):
        if str(self.imBox.currentText()) == "Image":
            self.imflag = "Image"
        if str(self.imBox.currentText()) == "Labels":
            self.imflag = "Labels"
        
    def changeWatershed(self, state):
        if state == Qt.Checked:
            self.params.segment['watershed'] = True
            self.watershed_erosion.setEnabled(True)
            self.watershed_size.setEnabled(True)
        else:
            self.params.segment['watershed'] = False
            self.watershed_erosion.setEnabled(False)
            self.watershed_size.setEnabled(False)
            
    def changeInvert(self, state):
        if state == Qt.Checked:
            self.params.segment['invert'] = True
            qi = QImage(invert(self.image).data, self.image.shape[1], self.image.shape[0], self.image.shape[1], QImage.Format_Indexed8)
            
        else:
            self.params.segment['invert'] = False
            qi = QImage(self.image.data, self.image.shape[1], self.image.shape[0], self.image.shape[1], QImage.Format_Indexed8)
        
        pixmap = QPixmap(qi)
        self.pixmap2 = pixmap.scaled(self.canvas_size, self.canvas_size, Qt.KeepAspectRatio)
        self.label.setPixmap(self.pixmap2)
            
    def rollingball(self):
        if self.sp.value() == 1:
            self.params.segment['rb_kernel'] = 0
        else:
            self.params.segment['rb_kernel'] = self.sp.value()
            
    def gaussian(self):
        self.params.segment['gaussian'] = self.gauss.value()
        
    def local(self):
        self.params.segment['local_size'] = self.local_size.value()
    
    def watershed(self):
        self.params.segment['watershed_size'] = self.watershed_size.value()
    
    def watershed_e(self):
        self.params.segment['watershed_erosion'] = self.watershed_erosion.value()
            
    def minsize(self):
        self.params.segment['min_size'] = self.minsizev.value()
            
    def update(self):
        labels = process(self.im_hs,self.params)
        labels = np.uint8(labels*(256/labels.max()))
        if self.imflag=="Image":
            #b=image
            b = np.uint8(mark_boundaries(self.image, labels, color=(1,1,1))[:,:,0]*255)
            if self.params.segment['invert'] == True:
                qi = QImage(invert(b).data, b.shape[1], b.shape[0], b.shape[1], QImage.Format_Indexed8)
            else:
                qi = QImage(b.data, b.shape[1], b.shape[0], b.shape[1], QImage.Format_Indexed8)
        if self.imflag=="Labels":
            qi = QImage(labels.data, labels.shape[1], labels.shape[0], labels.shape[1], QImage.Format_Indexed8)
        #qi = QImage(imchoice.data, imchoice.shape[1], imchoice.shape[0], imchoice.shape[1], QImage.Format_Indexed8)
        pixmap = QPixmap(qi)
        pixmap2 = pixmap.scaled(self.canvas_size[1], self.canvas_size[0], Qt.KeepAspectRatio)
        self.label.setPixmap(pixmap2)
        
        self.prev_params.load()
        self.prev_params.save(filename=os.path.dirname(inspect.getfile(process))+'/parameters/parameters_previous.hdf5')
        self.params.save()
        
    def undo(self):
        self.params.load(filename='parameters/parameters_previous.hdf5')
        
        labels = process(self.im_hs,self.params)
        labels = np.uint8(labels*(256/labels.max()))
        if self.imflag=="Image":
            #b=image
            b = np.uint8(mark_boundaries(self.image, labels, color=(1,1,1))[:,:,0]*255)
            qi = QImage(b.data, b.shape[1], b.shape[0], b.shape[1], QImage.Format_Indexed8)
        if self.imflag=="Labels":
            qi = QImage(labels.data, labels.shape[1], labels.shape[0], labels.shape[1], QImage.Format_Indexed8)
        #qi = QImage(imchoice.data, imchoice.shape[1], imchoice.shape[0], imchoice.shape[1], QImage.Format_Indexed8)
        pixmap = QPixmap(qi)
        pixmap2 = pixmap.scaled(self.canvas_size, self.canvas_size, Qt.KeepAspectRatio)
        self.label.setPixmap(pixmap2)
        
    def return_params(self,params):
        print(self.params.segment)
        
    def threshold_choice(self):
        if str(self.comboBox.currentText()) == "Otsu":
            self.params.segment['threshold'] = "otsu"
        elif str(self.comboBox.currentText()) == "Mean":
            self.params.segment['threshold'] = "mean"
        elif str(self.comboBox.currentText()) == "Minimum":
            self.params.segment['threshold'] = "minimum"
        elif str(self.comboBox.currentText()) == "Yen":
            self.params.segment['threshold'] = "yen"
        elif str(self.comboBox.currentText()) == "Isodata":
            self.params.segment['threshold'] = "isodata"
        elif str(self.comboBox.currentText()) == "Li":
            self.params.segment['threshold'] = "li"
        elif str(self.comboBox.currentText()) == "Local":
            self.params.segment['threshold'] = "local"
        elif str(self.comboBox.currentText()) == "Local Otsu":
            self.params.segment['threshold'] = "local_otsu"
        elif str(self.comboBox.currentText()) == "Local+Global Otsu":
            self.params.segment['threshold'] = "lg_otsu"
        elif str(self.comboBox.currentText()) == "Niblack":
            self.params.segment['threshold'] = "niblack"
        elif str(self.comboBox.currentText()) == "Sauvola":
            self.params.segment['threshold'] = "sauvola"

    def toggle_fk(self, tool):
        if tool == 'Gaussian':
            self.tsparams.gaussian[0] = not self.tsparams.gaussian[0]
        elif tool == 'Diff. Gaussians':
            self.tsparams.diff_gaussian[0] = not self.tsparams.diff_gaussian[0]
        elif tool == 'Median':
            self.tsparams.median[0] = not self.tsparams.median[0]
        elif tool == 'Minimum':
            self.tsparams.minimum[0] = not self.tsparams.minimum[0]
        elif tool == 'Maximum':
            self.tsparams.maximum[0] = not self.tsparams.maximum[0]
        elif tool == 'Sobel':
            self.tsparams.sobel[0] = not self.tsparams.sobel[0]
        elif tool == 'Hessian':
            self.tsparams.hessian[0] = not self.tsparams.hessian[0]
        elif tool == 'Laplacian':
            self.tsparams.laplacian[0] = not self.tsparams.laplacian[0]
        elif tool == 'M-Sum':
            self.tsparams.membrane[1] = not self.tsparams.membrane[1]
        elif tool == 'M-Mean':
            self.tsparams.membrane[2] = not self.tsparams.membrane[2]
        elif tool == 'M-Standard Deviation':
            self.tsparams.membrane[3] = not self.tsparams.membrane[3]
        elif tool == 'M-Median':
            self.tsparams.membrane[4] = not self.tsparams.membrane[4]
        elif tool == 'M-Minimum':
            self.tsparams.membrane[5] = not self.tsparams.membrane[5]
        elif tool == 'M-Maximum':
            self.tsparams.membrane[6] = not self.tsparams.membrane[6]

    def change_sigma(self):
        self.tsparams.set_global_sigma(self.spinb1.value())
    def change_high_sigma(self):
        self.tsparams.diff_gaussian[3] = self.spinb2.value()
    def change_disk(self):
        self.tsparams.set_global_disk_size(self.spinb3.value())

    def classifier_choice(self):
        if str(self.comboBox.currentText()) == "Random Forest":
            self.classifier = RandomForestClassifier(n_estimators=200)
        elif str(self.comboBox.currentText()) == "Nearest Neighbours":
            self.classifier = KNeighborsClassifier()
        elif str(self.comboBox.currentText()) == "Naive Bayes":
            self.classifier = GaussianNB()
        elif str(self.comboBox.currentText()) == "QDA":
            self.classifier = QuadraticDiscriminantAnalysis()
    
    def train_classifier(self):
        
        array = self.canvas2.array
        self.mask = np.array(Image.fromarray(array).resize((self.image.shape[1],self.image.shape[0])))
        self.trained_mask, self.classifier = cluster_trained(self.im_hs, self.mask, self.classifier, self.tsparams)
        
        self.canvas2.clearCanvas()
        if self.trained_mask.any() != 0:
            self.canvas2.drawLabels(self.trained_mask)

    def save_array(self):
        self.canvas.savearray(self.image)

    def save_and_close(self):
        array = self.canvas2.array
        self.mask = np.array(Image.fromarray(array).resize((self.image.shape[1],self.image.shape[0])))
        self.canvas2.savearray(self.image)
        self.close()


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

    def __init__(self,pixmap,canvas_size):
        super().__init__()
        self.OGpixmap = pixmap
        self.lastpixmap = pixmap

        self.canvas_size = canvas_size
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

        self.array = np.zeros((self.canvas_size[0],self.canvas_size[1],3),dtype=np.uint8)

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
        painter.eraseRect(0,0,self.canvas_size[0],self.canvas_size[1])
        painter.drawPixmap(0,0,self.OGpixmap)
        painter.end()
        self.update()

    def clearLabels(self):
        self.array = np.zeros((self.canvas_size[0],self.canvas_size[1],3), dtype=np.uint8)

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
        pixmap = pixmap.scaled(self.canvas_size[1], self.canvas_size[0], Qt.KeepAspectRatio)
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
        b.setsize(self.canvas_size[0] * self.canvas_size[1] * 4)
        arr = np.frombuffer(b, np.uint8)
        arr = arr.reshape((self.canvas_size[0], self.canvas_size[1], 4)).copy()

        OGimage = self.OGpixmap.toImage()
        c = OGimage.bits()
        c.setsize(self.canvas_size[0] * self.canvas_size[1] * 4)
        
        c = np.frombuffer(c, np.uint8).copy()
        OGim = np.reshape(c, (self.canvas_size[0], self.canvas_size[1], 4))
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

def main(image,height):
    
    ex = Application(image,height)
    
    return(ex)
    
def seg_ui(image):
    """
    Function to launch the Segmentation User Interface.
    """
    app = QApplication(sys.argv)
    app.aboutToQuit.connect(app.deleteLater)
    
    screen = app.primaryScreen()
    size = screen.size()
    height = int(0.8*size.height())

    if 1024 < height:
        height = 1024

    #params = ParticleAnalysis.param_generator()
    ex = main(image,height)
    
    #ex.show()
    app.exec_()
    
    return(ex)
    
if __name__ == '__main__':
    import hyperspy.api as hs
    filename = "data/JEOL HAADF Image.dm4"
    haadf = hs.load(filename)
    
    image_out = np.zeros_like(haadf)
    
    app = QApplication(sys.argv)
    app.aboutToQuit.connect(app.deleteLater)
    
    screen = app.primaryScreen()
    print('Screen: %s' % screen.name())
    size = screen.size()
    print('Size: %d x %d' % (size.width(), size.height()))
    rect = screen.availableGeometry()
    print('Available: %d x %d' % (rect.width(), rect.height()))
    
    #params = ParticleAnalysis.param_generator()
    ex = main(haadf)
    
    #ex.show()
    app.exec_()
    