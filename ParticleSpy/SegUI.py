# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 15:50:08 2018

@author: qzo13262
"""

from PyQt5.QtWidgets import QCheckBox, QPushButton, QLabel, QMainWindow, QSpinBox
from PyQt5.QtWidgets import QApplication, QWidget, QHBoxLayout, QComboBox, QTabWidget
from PyQt5.QtWidgets import QVBoxLayout
from PyQt5.QtGui import QPixmap, QImage, QColor, QPainter
from PyQt5.QtCore import Qt
import sys
import os

import inspect
import numpy as np
from skimage.segmentation import mark_boundaries, flood_fill
from skimage.util import invert
from PIL import Image

from ParticleSpy.segptcls import process
from ParticleSpy.ParticleAnalysis import parameters

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
        self.tabs.addTab(self.tab3,"WIP Training")

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
        self.pixmap2 = pixmap.scaled(512, 512, Qt.KeepAspectRatio)
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
        
        self.wse_title = QLabel(self)
        self.wse_title.setText('Watershed Seed Erosion')
        self.watershed_erosion = QSpinBox(self)
        self.watershed_erosion.setMaximum(self.image.shape[0])
        self.watershed_erosion.valueChanged.connect(self.watershed_e)
        
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

        lay3 = QHBoxLayout()
        im_lay = QVBoxLayout()
        button_lay = QVBoxLayout()

        lay3.addLayout(button_lay)
        lay3.addLayout(im_lay)

        self.canvas2 = Canvas(self.pixmap2)
        self.canvas2.setAlignment(Qt.AlignTop)
        
        for tool in brush_tools:
            b = ToolButton(tool)
            b.clicked.connect(lambda tool=tool: self.change_pen(tool))
            b.setText(tool)
            button_lay.addWidget(b)


        im_lay.addWidget(self.canvas2)

        self.getarrayc = QPushButton('Save and Close',self)
        self.getarrayc.clicked.connect(self.save_and_close)
        
        button_lay.addWidget(self.getarrayc)
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
        else:
            self.params.segment['watershed'] = False
            
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
        if str(self.comboBox.currentText()) == "Mean":
            self.params.segment['threshold'] = "mean"
        if str(self.comboBox.currentText()) == "Minimum":
            self.params.segment['threshold'] = "minimum"
        if str(self.comboBox.currentText()) == "Yen":
            self.params.segment['threshold'] = "yen"
        if str(self.comboBox.currentText()) == "Isodata":
            self.params.segment['threshold'] = "isodata"
        if str(self.comboBox.currentText()) == "Li":
            self.params.segment['threshold'] = "li"
        if str(self.comboBox.currentText()) == "Local":
            self.params.segment['threshold'] = "local"
        if str(self.comboBox.currentText()) == "Local Otsu":
            self.params.segment['threshold'] = "local_otsu"
        if str(self.comboBox.currentText()) == "Local+Global Otsu":
            self.params.segment['threshold'] = "lg_otsu"
        if str(self.comboBox.currentText()) == "Niblack":
            self.params.segment['threshold'] = "niblack"
        if str(self.comboBox.currentText()) == "Sauvola":
            self.params.segment['threshold'] = "sauvola"

    def change_pen(self, pen):
        self.canvas.changePen(pen)

    
    def save_array(self):
        self.canvas.savearray(self.image)

    def save_and_close(self):
        self.canvas.savearray(self.image)
        self.closeEvent

brush_tools = ['Freehand', 'Line', 'Polygon']

class ToolButton(QPushButton):

    def __init__(self, tool):
        super().__init__()
        self.setAutoExclusive(True)
        self.setCheckable(True)


class Canvas(QLabel):

    def __init__(self,pixmap):
        super().__init__()
        self.setPixmap(pixmap)
        """"
        self.tool_pixmap = QLabel()
        tool_layer = QPixmap(self.size())
        self.tool_pixmap.setPixmap(tool_layer)"""
        
        self.setMouseTracking(True)
        self.last_x, self.last_y = None, None
        self.last_x_clicked , self.last_y_clicked = None, None
        self.pen_color = QColor(255, 0, 0, 20)
        self.penType = brush_tools[0]
        self.lineCount = 0

    def set_pen_color(self, c):
        self.pen_color = QColor(c)

    def changePen(self, brush):
        self.penType = brush
        
    def mousePressEvent(self, e):
        #this bit is probably broken
        if e.button() == Qt.RightButton:
            image = self.pixmap().toImage()
            b = image.bits()
            b.setsize(512 * 512 * 4)
            arr = np.frombuffer(b, np.uint8).reshape((512, 512, 4))
            
            arr_test = arr[:,:,0]/arr[:,:,2]
            
            painted_arr = np.zeros_like(arr[:,:,0:3])
            painted_arr[:,:,2][arr_test!=1] = 255
            
            painted_arr[:,:,2] = flood_fill(painted_arr[:,:,2],(e.y(),e.x()),255)
            
            qi = QImage(painted_arr.data, painted_arr.shape[1], painted_arr.shape[0], 3*painted_arr.shape[1], QImage.Format_RGB888)
            pixmap = QPixmap(qi)
            
            painter = QPainter(self.pixmap())
            painter.setOpacity(0.1)

            painter.drawPixmap(0, 0, pixmap)
            painter.end()
            self.update()
            
            self.array = painted_arr[:,:,2]
        
        if e.button() ==Qt.LeftButton:
            print("click")
            if self.penType == 'Line':
                if self.lineCount == 0:
                    self.last_x_clicked, self.last_y_clicked = e.x(), e.y()
                    self.lineCount = 1
                else:
                    painter = QPainter(self.pixmap())
                    p = painter.pen()
                    p.setWidth(5)
                    p.setColor(self.pen_color)
                    painter.setPen(p)
                    painter.drawLine(self.last_x_clicked, self.last_y_clicked, e.x(), e.y())
                    self.last_x_clicked, self.last_y_clicked = e.x(), e.y()
                    self.lineCount = 0
                    painter.end()
                self.update()



    def mouseMoveEvent(self, e):
        if self.last_x is None: # First event.
            self.last_x = e.x()
            self.last_y = e.y()
            return # Ignore the first time.
        
        if e.buttons() == Qt.LeftButton:
            if self.penType == 'Freehand':
                painter = QPainter(self.pixmap())
                p = painter.pen()
                p.setWidth(5)
                p.setColor(self.pen_color)
                painter.setPen(p)
                painter.drawLine(self.last_x, self.last_y, e.x(), e.y())
                painter.end()
                self.update()
        
                # Update the origin for next time.
                self.last_x = e.x()
                self.last_y = e.y()
        """
        if e.buttons() != Qt.LeftButton and (self.penType == 'Line' or self.penType == 'Polygon'):
            if self.lineCount == 1:
                painter = QPainter(self.tool_pixmap)
                

                #this just paints over the line behind with black ideally should isolate to new canvas

                #draw new temp line
                p = painter.pen()
                p.setWidth(2)
                p.pen_color = QColor(255, 255, 255, 255)
                painter.setPen(p)

                painter.drawLine(self.last_x_clicked, self.last_y_clicked, e.x(), e.y())
                painter.end()
                self.update()

                self.last_x = e.x()
                self.last_y = e.y()
                """



    def mouseReleaseEvent(self, e):
        if self.penType == 'Freehand':
            self.last_x = None
            self.last_y = None
        
    def savearray(self,image):
        resized = np.array(Image.fromarray(self.array).resize((image.shape[1],image.shape[0])))
        np.save(os.path.dirname(inspect.getfile(process))+'/parameters/manual_mask',resized)

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
    