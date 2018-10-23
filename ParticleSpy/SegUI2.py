# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 15:50:08 2018

@author: qzo13262
"""

from PyQt5.QtWidgets import QCheckBox, QPushButton, QLabel, QMainWindow, QSpinBox
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QComboBox
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import sys

import numpy as np
from skimage.segmentation import mark_boundaries

import segptcls
import ParticleAnalysis

class Application(QMainWindow):

    def __init__(self,im_hs):
        super().__init__()
        self.setWindowTitle("Segmentation UI")
        self.imflag = "Image"
        
        self.getim(im_hs)
        self.getparams()

        #self.central_widget = QWidget()               
        #self.setCentralWidget(self.central_widget)
        lay = QVBoxLayout()

        self.label = QLabel(self)
        qi = QImage(self.image.data, self.image.shape[1], self.image.shape[0], self.image.shape[1], QImage.Format_Indexed8)
        pixmap = QPixmap(qi)
        self.label.setPixmap(pixmap)
        self.label.setGeometry(10,10,self.image.shape[1],self.image.shape[0])
        
        self.resize(pixmap.width()+130, pixmap.height()+70)
        
        cb = QCheckBox('Watershed', self)
        cb.move(self.image.shape[1]+20, 40)
        #cb.toggle()
        cb.stateChanged.connect(self.changeWatershed)
        
        self.sp = QSpinBox(self)
        self.sp.valueChanged.connect(self.rollingball)
        self.sp.move(self.image.shape[1]+20, 70)
        
        updateb = QPushButton('Update',self)
        updateb.move(self.image.shape[1]+20,110)
        
        updateb.clicked.connect(self.update)
        
        paramsb = QPushButton('Get Params',self)
        paramsb.move(self.image.shape[1]+20,140)
        
        paramsb.clicked.connect(self.return_params)
        
        self.imBox = QComboBox(self)
        self.imBox.addItem("Image")
        self.imBox.addItem("Labels")
        self.imBox.move(15, self.image.shape[0]+15)
        
        self.imBox.activated[str].connect(self.changeIm)
        
        self.comboBox = QComboBox(self)
        self.comboBox.addItem("Otsu")
        self.comboBox.addItem("Mean")
        self.comboBox.addItem("Minimum")
        self.comboBox.addItem("Yen")
        self.comboBox.addItem("Isodata")
        self.comboBox.addItem("Li")
        self.comboBox.addItem("Local")
        self.comboBox.move(self.image.shape[1]+20, 10)

        self.comboBox.activated[str].connect(self.threshold_choice)

        lay.addWidget(self.label)
        lay.addWidget(self.comboBox)
        lay.addWidget(self.sp)
        self.show()
        
    def getim(self,im_hs):
        self.im_hs = im_hs
        im = im_hs.data
        image = np.uint8(255*(im-np.min(im))/(np.max(im)-np.min(im)))
        self.image = image
        
    def getparams(self):
        self.params = ParticleAnalysis.param_generator()
        
    def changeIm(self):
        if str(self.imBox.currentText()) == "Image":
            self.imflag = "Image"
        if str(self.imBox.currentText()) == "Labels":
            self.imflag = "Labels"
        
    def changeWatershed(self, state):
        if state == Qt.Checked:
            self.params['watershed'] = True
        else:
            self.params['watershed'] = None
            
    def rollingball(self):
        if self.sp.value() == 1:
            self.params['rb_kernel'] = 0
        else:
            self.params['rb_kernel'] = self.sp.value()
            
    def update(self):
        labels = segptcls.process(self.im_hs,self.params)
        labels = np.uint8(labels*(256/labels.max()))
        if self.imflag=="Image":
            #b=image
            b = np.uint8(mark_boundaries(self.image, labels, color=(1,1,1))[:,:,0]*255)
            qi = QImage(b.data, b.shape[1], b.shape[0], b.shape[1], QImage.Format_Indexed8)
        if self.imflag=="Labels":
            qi = QImage(labels.data, labels.shape[1], labels.shape[0], labels.shape[1], QImage.Format_Indexed8)
        #qi = QImage(imchoice.data, imchoice.shape[1], imchoice.shape[0], imchoice.shape[1], QImage.Format_Indexed8)
        pixmap = QPixmap(qi)
        self.label.setPixmap(pixmap)
        
    def return_params(self,params):
        print(self.params)
        
    def threshold_choice(self):
        if str(self.comboBox.currentText()) == "Otsu":
            self.params['threshold'] = "otsu"
        if str(self.comboBox.currentText()) == "Mean":
            self.params['threshold'] = "mean"
        if str(self.comboBox.currentText()) == "Minimum":
            self.params['threshold'] = "minimum"
        if str(self.comboBox.currentText()) == "Yen":
            self.params['threshold'] = "yen"
        if str(self.comboBox.currentText()) == "Isodata":
            self.params['threshold'] = "isodata"
        if str(self.comboBox.currentText()) == "Li":
            self.params['threshold'] = "li"
        if str(self.comboBox.currentText()) == "Local":
            self.params['threshold'] = "local"
    
def main(haadf):
    
    ex = Application(haadf)
    
    return(ex)
    
if __name__ == '__main__':
    import hyperspy.api as hs
    haadf_file = "JEOL HAADF IMAGE.dm4"
    ac_folder = r"Z:\data\2018\cm19688-7\raw\DM\SI data (9)/"
    haadf = hs.load(ac_folder+haadf_file)
    app = QApplication(sys.argv)
    app.aboutToQuit.connect(app.deleteLater)
    
    #params = ParticleAnalysis.param_generator()
    ex = main(haadf)
    
    #ex.show()
    app.exec_()
    