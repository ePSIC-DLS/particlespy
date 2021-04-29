# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 15:50:08 2018

@author: qzo13262
"""

from PyQt5.QtWidgets import QCheckBox, QPushButton, QLabel, QMainWindow, QSpinBox, QGroupBox, QGridLayout
from PyQt5.QtWidgets import QApplication, QWidget, QHBoxLayout, QComboBox, QTabWidget
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QSizePolicy
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
from ParticleSpy.Canvas import *

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

class FKConfig(QWidget):
    def __init__(self, params, parent=None):
        super(FKConfig, self).__init__(parent)
        self.params = params
        self.initUI()

    def initUI(self):   
        grid = QGridLayout()  
        self.setLayout(grid)

        headings = ['Filter Kernel','Enabled','Prefilter', 'Prefilter Size','High Sigma','Disk Size']

        for i in range(len(headings)):
            button = QLabel(self)
            button.setText(headings[i])
            grid.addWidget(button, 0,i)

        names = ['Gaussian','Difference of Gaussians','Median','Minimum','Maximum','Sobel','Hessian','Laplacian','Membrane Projection']
        self.enableds = []
        self.prefilters = []
        self.pfsizes = []
        for i in range(len(names)):
            button = QLabel(self)
            button.setText(names[i])
            grid.addWidget(button, i+1,0)

            self.enableds.append(QCheckBox(self))
            grid.addWidget(self.enableds[i], i+1,1)

            self.prefilters.append(QCheckBox(self))
            self.prefilters[i].stateChanged.connect(lambda i=i: togglePFSize(i))
            grid.addWidget(self.prefilters[i], i+1,2)

            self.pfsizes.append(QSpinBox(self))
            self.pfsizes[i].valueChanged.connect(lambda i=i: changePFsize(i))
            grid.addWidget(self.pfsizes[i], i+1,3)
            
            if i == 1:
                hs = QSpinBox(self)
                grid.addWidget(hs, i+1,4)

        self.move(300, 150)
        self.setWindowTitle('Filter Kernel Configuration')  
        self.show()

        def update_FK(self, i):
            if i == 0:
                pr = self.params.gaussian
            elif i == 1:
                pr = self.params.diff_gaussian            
            elif i == 2:
                pr = self.params.median
            elif i == 3:
                pr = self.params.minimum
            elif i == 4:
                pr = self.params.maximum
            elif i == 5:
                pr = self.params.sobel
            elif i == 6:
                pr = self.params.hessian            
            elif i == 7:
                pr = self.params.laplacian 

            
        def togglePFSize(self, i):
            if self.prefilters[i].isChecked():
                self.pfsizes[i].setEnabled(True)
            else:
                self.pfsizes[i].setEnabled(False)


        #def changePFsize(self, i):

            
        
        


class Application(QMainWindow):

    def __init__(self,im_hs):
        super().__init__()
        self.setWindowTitle("Segmentation UI")
        self.imflag = "Image"
        
        self.getim(im_hs)
        self.getparams()
        
        self.prev_params = parameters()
        self.prev_params.generate()
        
        offset = 50
        
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
        self.pixmap2 = pixmap.scaled(1024, 1024, Qt.KeepAspectRatio)
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
        self.canvas = Canvas(self.pixmap2)
        #self.canvas = Drawer(self.pixmap2)
        
        self.getarrayb = QPushButton('Save Segmentation',self)
        self.getarrayb.clicked.connect(self.save_array)
        
        tab2layout = QVBoxLayout()
        tab2layout.addWidget(self.canvas)
        tab2layout.addWidget(self.getarrayb)
        tab2layout.addStretch(1)
        self.tab2.setLayout(tab2layout)

        #Tab 3

        

        self.mask = np.zeros([1024,1024,3])
        self.classifier = GaussianNB()
        self.tsparams = trainableParameters()
        self.filter_kernels = ['Gaussian','Diff. Gaussians','Median','Minimum','Maximum','Sobel','Hessian','Laplacian','M-Sum','M-Mean','M-Standard Deviation','M-Median','M-Minimum','M-Maximum']

        lay3 = QHBoxLayout()
        im_lay = QVBoxLayout()
        button_lay = QVBoxLayout()
        colour_lay = QHBoxLayout()


        lay3.addLayout(button_lay)
        lay3.addLayout(im_lay)

        self.canvas2 = Canvas(self.pixmap2)
        self.canvas2.setAlignment(Qt.AlignTop)
        
        for tool in self.canvas2.brush_tools:
            b = ToolButton(tool)
            b.pressed.connect(lambda tool=tool: self.canvas2.changePen(tool))
            b.setText(tool)
            if tool == 'Freehand':
                b.setChecked(True)
            button_lay.addWidget(b)


        for i in range(len(self.canvas2.colors)):
            c = self.canvas2.colors[i]
            b = QPaletteButton(c)
            b.pressed.connect(lambda i=i: self.canvas2.set_pen_color(i))
            if i== 0:
                b.setChecked(True)
            colour_lay.addWidget(b)
        button_lay.addLayout(colour_lay)

        im_lay.addWidget(self.canvas2)
        
        self.clftxt = QLabel(self)
        self.clftxt.setText('Classifier')
        button_lay.addWidget(self.clftxt)

        self.clfBox = QComboBox(self)
        self.clfBox.addItem("Random Forest")
        self.clfBox.addItem("Nearest Neighbours")
        self.clfBox.addItem("Naive Bayes")
        self.clfBox.activated[str].connect(self.classifier_choice)

        self.kerneltxt = QLabel(self)
        self.kerneltxt.setText('Filter Kernels')

        
        button_lay.addWidget(self.clfBox)
        button_lay.addWidget(self.kerneltxt)

        for t in range(8):
            b = QCheckBox(self.filter_kernels[t], self)
            b.pressed.connect(lambda tool=self.filter_kernels[t]: self.toggle_fk(tool))
            if t in (0,1,2,3,4,5,8):
                b.setChecked(True)
            button_lay.addWidget(b)  

        self.membranetext = QLabel(self)
        self.membranetext.setText('Membrane Projections')
        button_lay.addWidget(self.membranetext)

        for t in range(8,14):
            b = QCheckBox(self.filter_kernels[t][2:], self)
            b.pressed.connect(lambda tool=self.filter_kernels[t]: self.toggle_fk(tool))
            if t in (0,1,2,3,4,5,8):
                b.setChecked(True)
            
            button_lay.addWidget(b)



        self.ql1 = QLabel(self)
        self.ql1.setText('Sigma')
        self.spinb1 = QSpinBox(self)
        self.spinb1.valueChanged.connect(self.change_sigma)
        self.spinb1.setValue(1)
        self.ql2 = QLabel(self)
        self.ql2.setText('High Sigma')
        self.spinb2 = QSpinBox(self)
        self.spinb2.valueChanged.connect(self.change_high_sigma)
        self.spinb2.setValue(16)
        self.ql3 = QLabel(self)
        self.ql3.setText('Disk Size')
        self.spinb3 = QSpinBox(self)
        self.spinb3.valueChanged.connect(self.change_disk)
        self.spinb3.setValue(20)

        button_lay.addWidget(self.ql1)
        button_lay.addWidget(self.spinb1)
        button_lay.addWidget(self.ql2)
        button_lay.addWidget(self.spinb2)
        button_lay.addWidget(self.ql3)
        button_lay.addWidget(self.spinb3)

        self.config = QPushButton('Configure Filter Kernels', self)
        self.config.setToolTip('Choose individual filter kernel parameters')
        self.config.clicked.connect(self.ShowFK)
        button_lay.addWidget(self.config)

        self.clear = QPushButton('Clear Training Labels', self)
        self.clear.setToolTip('Removes existing training labels memory')
        self.clear.clicked.connect(self.canvas2.clearLabels)
        button_lay.addWidget(self.clear)

        self.redraw = QPushButton('Redraw Training Labels', self)
        self.redraw.setToolTip('draws any existing training labels in memory onto the canvas')
        self.redraw.clicked.connect(self.canvas2.redrawLabels)
        button_lay.addWidget(self.redraw)

        self.train = QPushButton('train classifier', self)
        self.train.pressed.connect(self.train_classifier)
        button_lay.addWidget(self.train)

        self.clear = QPushButton('Clear Canvas', self)
        self.clear.setToolTip('Removes any generated segmentation masks and labels from the image, does not clear training labels from memory')
        self.clear.clicked.connect(self.canvas2.clearCanvas)
        button_lay.addWidget(self.clear)

        self.getarrayc = QPushButton('Save and Close',self)
        self.getarrayc.clicked.connect(self.save_and_close)
        button_lay.addWidget(self.getarrayc)

        self.tab3.setLayout(lay3)

        self.show()
    
    def ShowFK(self):
        self.fk = FKConfig(self.tsparams)
        self.fk.show()

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
        self.pixmap2 = pixmap.scaled(512, 512, Qt.KeepAspectRatio)
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
        pixmap2 = pixmap.scaled(512, 512, Qt.KeepAspectRatio)
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
        pixmap2 = pixmap.scaled(512, 512, Qt.KeepAspectRatio)
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
        self.tsparams.setGlobalSigma(self.spinb1.value())
    def change_high_sigma(self):
        self.tsparams.diff_gaussian[3] = self.spinb2.value()
    def change_disk(self):
        self.tsparams.setGlobalDiskSize(self.spinb3.value())

    def classifier_choice(self):
        if str(self.comboBox.currentText()) == "Random Forest":
            self.classifier = RandomForestClassifier(n_estimators=200)
        elif str(self.comboBox.currentText()) == "Nearest Neighbours":
            self.classifier = KNeighborsClassifier()
        elif str(self.comboBox.currentText()) == "Naive Bayes":
            self.classifier = GaussianNB()
    
    def train_classifier(self):
        
        array = self.canvas2.array
        self.mask = np.array(Image.fromarray(array).resize((self.image.shape[1],self.image.shape[0])))
        self.trained_mask, self.classifier = ClusterTrained(self.im_hs, self.mask, self.classifier, self.tsparams)
        
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




def main(haadf):
    
    ex = Application(haadf)
    
    return(ex)
    
def SegUI(image):
    """
    Function to launch the Segmentation User Interface.
    """
    app = QApplication(sys.argv)
    app.aboutToQuit.connect(app.deleteLater)
    
    #params = ParticleAnalysis.param_generator()
    ex = main(image)
    
    #ex.show()
    app.exec_()
    
    return(ex)
    
if __name__ == '__main__':
    import hyperspy.api as hs
    filename = "Data/JEOL HAADF Image.dm4"
    haadf = hs.load(filename)
    
    image_out = np.zeros_like(haadf)
    
    app = QApplication(sys.argv)
    app.aboutToQuit.connect(app.deleteLater)
    
    #params = ParticleAnalysis.param_generator()
    ex = main(haadf)
    
    #ex.show()
    app.exec_()
    