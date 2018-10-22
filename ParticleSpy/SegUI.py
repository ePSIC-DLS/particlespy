# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 12:15:47 2018

@author: qzo13262
"""

from PyQt5.QtWidgets import QCheckBox, QPushButton, QLabel, QMainWindow, QApplication, QWidget, QVBoxLayout, QComboBox
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import sys

import numpy as np
from skimage.segmentation import mark_boundaries

import segptcls
import ParticleAnalysis

def segui(im_hs):
  
    im = im_hs.data
    
    params = ParticleAnalysis.param_generator()
    
    image = np.uint8(255*(im-np.min(im))/(np.max(im)-np.min(im)))
    labels = segptcls.process(im_hs,params)
    labels = np.uint8(labels*(255/labels.max()))
    
    class Menu(QMainWindow):
    
        def __init__(self):
            super().__init__()
            self.setWindowTitle("Segmentation UI")
            self.imflag = "Image"
    
            #self.central_widget = QWidget()               
            #self.setCentralWidget(self.central_widget)
            lay = QVBoxLayout()
    
            self.label = QLabel(self)
            qi = QImage(image.data, image.shape[1], image.shape[0], image.shape[1], QImage.Format_Indexed8)
            pixmap = QPixmap(qi)
            self.label.setPixmap(pixmap)
            self.label.setGeometry(10,10,image.shape[1],image.shape[0])
            
            self.resize(pixmap.width()+130, pixmap.height()+50)
            
            cb = QCheckBox('Watershed', self)
            cb.move(image.shape[1]+20, 40)
            #cb.toggle()
            cb.stateChanged.connect(self.changeWatershed)
            
            updateb = QPushButton('Update',self)
            updateb.move(image.shape[1]+20,image.shape[0]-20)
            
            updateb.clicked.connect(self.update)
            
            self.imBox = QComboBox(self)
            self.imBox.addItem("Image")
            self.imBox.addItem("Labels")
            self.imBox.move(15, image.shape[0]+15)
            
            self.imBox.activated[str].connect(self.changeIm)
            
            self.comboBox = QComboBox(self)
            self.comboBox.addItem("Otsu")
            self.comboBox.addItem("Mean")
            self.comboBox.addItem("Minimum")
            self.comboBox.addItem("Yen")
            self.comboBox.addItem("Isodata")
            self.comboBox.addItem("Li")
            self.comboBox.addItem("Local")
            self.comboBox.move(image.shape[1]+20, 10)
    
            self.comboBox.activated[str].connect(self.threshold_choice)
    
            lay.addWidget(self.label)
            lay.addWidget(self.comboBox)
            self.show()
            
        def changeIm(self):
            if str(self.imBox.currentText()) == "Image":
                self.imflag = "Image"
            if str(self.imBox.currentText()) == "Labels":
                self.imflag = "Labels"
            
        def changeWatershed(self, state):
            if state == Qt.Checked:
                params['watershed'] = True
            else:
                params['watershed'] = None
                
        def update(self):
            labels = segptcls.process(im_hs,params)
            labels = np.uint8(labels*(256/labels.max()))
            if self.imflag=="Image":
                #b=image
                b = np.uint8(mark_boundaries(image, labels, color=(1,1,1))[:,:,0]*255)
                qi = QImage(b.data, b.shape[1], b.shape[0], b.shape[1], QImage.Format_Indexed8)
            if self.imflag=="Labels":
                qi = QImage(labels.data, labels.shape[1], labels.shape[0], labels.shape[1], QImage.Format_Indexed8)
            #qi = QImage(imchoice.data, imchoice.shape[1], imchoice.shape[0], imchoice.shape[1], QImage.Format_Indexed8)
            pixmap = QPixmap(qi)
            self.label.setPixmap(pixmap)
            
        def threshold_choice(self):
            if str(self.comboBox.currentText()) == "Otsu":
                params['threshold'] = "otsu"
            if str(self.comboBox.currentText()) == "Mean":
                params['threshold'] = "mean"
            if str(self.comboBox.currentText()) == "Minimum":
                params['threshold'] = "minimum"
            if str(self.comboBox.currentText()) == "Yen":
                params['threshold'] = "yen"
            if str(self.comboBox.currentText()) == "Isodata":
                params['threshold'] = "isodata"
            if str(self.comboBox.currentText()) == "Li":
                params['threshold'] = "li"
            if str(self.comboBox.currentText()) == "Local":
                params['threshold'] = "local"
    
    app = QApplication(sys.argv)
    app.aboutToQuit.connect(app.deleteLater)
    #ex = Menu()
    app.exec_()
    
    return(params)