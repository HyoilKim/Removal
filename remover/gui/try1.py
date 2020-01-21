from PIL import Image
from gui.ui_window import Ui_Form
from gui.ui_draw import *
from PIL import Image, ImageQt
import random, io, os
import numpy as np
import torch
import torchvision.transforms as transforms
from util import task, util
from dataloader.image_folder import make_dataset
from model import create_model
from util.visualizer import Visualizer
import sys
from options.test_options import TestOptions
from gui.ui_model import ui_model
from PyQt5 import QtWidgets, QtGui,QtCore


class painter(QtWidgets.QWidget):
    def __init__(self, parent, image=None):
        super(painter, self).__init__()
        self.ParentLink = parent
        self.setPalette(QtGui.QPalette(QtCore.Qt.white))
        self.setAutoFillBackground(True)
        self.setMaximumSize(self.ParentLink.opt.loadSize[0], self.ParentLink.opt.loadSize[1])
        self.map = QtGui.QImage(self.ParentLink.opt.loadSize[0], self.ParentLink.opt.loadSize[1], QtGui.QImage.Format_RGB32)
        self.map.fill(QtCore.Qt.black)
        self.image = image
        self.shape = self.ParentLink.shape
        self.CurrentWidth = self.ParentLink.CurrentWidth
        self.MouseLoc = point(0, 0)
        self.LastPos = point(0, 0)
        self.Brush = True
        self.DrawingShapes_free = shapes()
        self.DrawingShapes_rec = shapes()
        self.IsPainting = True
        self.IsEraseing = False
        self.iteration = 0
        self.CurrentColor = colour3(255, 255, 255)
        self.ShapeNum = 0
        self.IsMouseing = False
        self.PaintPanel = 0


    def saveDraw(self):
        painter = QtGui.QPainter()
        painter.begin(self.map)
        self.drawRectangle(painter)
        painter.end()
        #self.map.save('./test.png')

    def drawRectangle(self, painter):
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        for i in range(self.DrawingShapes_rec.NumberOfShapes()-1):

            T = self.DrawingShapes_rec.GetShape(i)
            T1 = self.DrawingShapes_rec.GetShape(i+1)

            if T.ShapeNumber == T1.ShapeNumber:
                pen = QtGui.QPen(QtGui.QColor(T.Color.R, T.Color.G, T.Color.B), T.Width/2, QtCore.Qt.SolidLine)
                painter.setPen(pen)
                painter.setBrush(QtGui.QColor(T.Color.R, T.Color.G, T.Color.B))
                painter.drawRects(QtCore.QRect(QtCore.QPoint(T.Location.X, T.Location.Y),
                                               QtCore.QPoint(T1.Location.X, T1.Location.Y)))


class ui_model(QtWidgets.QWidget, Ui_Form):
    shape = 'line'
    CurrentWidth = 1

    def __init__(self, opt):
        super(ui_model, self).__init__()
        self.setupUi(self)
        self.opt = opt
        # self.show_image = None
        self.show_result_flag = False
        self.opt.loadSize = [256, 256]
        self.visualizer = Visualizer(opt)
        self.model_name = ['celeba_center', 'paris_center', 'imagenet_center', 'place2_center',
                           'celeba_random', 'paris_random','imagenet_random', 'place2_random']
        self.img_root = './datasets/'
        self.img_files = ['celeba-hq', 'paris', 'imagenet', 'place2']
        self.graphicsView_2.setMaximumSize(self.opt.loadSize[0]+30, self.opt.loadSize[1]+30)

    def showImage(self, fname):
        """Show the masked images"""
        value = self.comboBox.currentIndex()
        img = Image.open(fname).convert('RGB')
        self.img_original = img.resize(self.opt.loadSize)
        if value > 4:
            self.img = self.img_original
        else:
            self.img = self.img_original
            sub_img = Image.fromarray(np.uint8(255*np.ones((128, 128, 3))))
            mask = Image.fromarray(np.uint8(255*np.ones((128, 128))))
            self.img.paste(sub_img, box=(64, 64), mask=mask)
        self.show_image = ImageQt.ImageQt(self.img)
        self.new_painter(self.show_image)

    def show_result(self):
        """Show the results and original image"""
        if self.show_result_flag:
            self.show_result_flag = False
            new_pil_image = Image.fromarray(util.tensor2im(self.img_out.detach()))
            new_qt_image = ImageQt.ImageQt(new_pil_image)
        else:
            self.show_result_flag = True
            new_qt_image = ImageQt.ImageQt(self.img_original)
        self.graphicsView_2.scene = QtWidgets.QGraphicsScene()
        item = QtWidgets.QGraphicsPixmapItem(QtGui.QPixmap.fromImage(new_qt_image))
        self.graphicsView_2.scene.addItem(item)
        self.graphicsView_2.setScene(self.graphicsView_2.scene)

    def load_model(self):
        """Load different kind models for different datasets and mask types"""
        self.opt.name = 'place2_random'
        self.opt.img_file = './'
        self.model = create_model(self.opt)

    def load_image(self):
        """Load the image"""
        self.fname, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'select the image', self.opt.img_file, 'Image files(*.jpg *.png)')
        img = Image.open(fname).convert('RGB')
        self.img_original = img.resize(self.opt.loadSize)
        if value > 4:
            self.img = self.img_original
        else:
            self.img = self.img_original
            sub_img = Image.fromarray(np.uint8(255*np.ones((128, 128, 3))))
            mask = Image.fromarray(np.uint8(255*np.ones((128, 128))))
            self.img.paste(sub_img, box=(64, 64), mask=mask)
        self.show_image = ImageQt.ImageQt(self.img)
        self.new_painter(self.show_image)

    def save_result(self):
        """Save the results to the disk"""
        util.mkdir(self.opt.results_dir)
        img_name = self.fname.split('/')[-1]
        data_name = self.opt.img_file.split('/')[-1].split('.')[0]

        # save the original image
        original_name = '%s_%s_%s' % ('original', data_name, img_name)
        original_path = os.path.join(self.opt.results_dir, original_name)
        img_original = util.tensor2im(self.img_truth)
        util.save_image(img_original, original_path)

        # save the mask
        mask_name = '%s_%s_%d_%s' % ('mask', data_name, self.PaintPanel.iteration, img_name)
        mask_path = os.path.join(self.opt.results_dir, mask_name)
        img_mask = util.tensor2im(self.img_m)
        util.save_image(img_mask, mask_path)

        # save the results
        result_name = '%s_%s_%d_%s' % ('result', data_name, self.PaintPanel.iteration, img_name)
        result_path = os.path.join(self.opt.results_dir, result_name)
        img_result = util.tensor2im(self.img_out)
        util.save_image(img_result, result_path)

    def new_painter(self, image=None):
        """Build a painter to load and process the image"""
        # painter
        self.PaintPanel = painter(self, image)
        self.PaintPanel.close()
        self.stackedWidget.insertWidget(0, self.PaintPanel)
        self.stackedWidget.setCurrentWidget(self.PaintPanel)

    def draw_mask(self, maskStype):
        """Draw the mask"""
        self.shape = maskStype
        self.PaintPanel.shape = maskStype

    def set_input(self):
        """Set the input for the network"""
        # get the test mask from painter
        self.PaintPanel.saveDraw()
        buffer = QtCore.QBuffer()
        buffer.open(QtCore.QBuffer.ReadWrite)
        self.PaintPanel.map.save(buffer, 'PNG')
        pil_im = Image.open(io.BytesIO(buffer.data()))

        # transform the image to the tensor
        img = self.transform(self.img)
        mask = torch.autograd.Variable(self.transform(pil_im)).unsqueeze(0)
            # mask from the random mask
            # mask = Image.open(self.mname)
            # mask = torch.autograd.Variable(self.transform(mask)).unsqueeze(0)
        mask = (mask < 1).float()

        if len(self.opt.gpu_ids) > 0:
            img = img.unsqueeze(0).cuda(self.opt.gpu_ids[0])
            mask = mask.cuda(self.opt.gpu_ids[0])

        # get I_m and I_c for image with mask and complement regions for training
        mask = mask
        self.img_truth = img * 2 - 1
        self.img_m = mask * self.img_truth
        self.img_c = (1 - mask) * self.img_truth

        return self.img_m, self.img_c, self.img_truth, mask

    def fill_mask(self):
        """Forward to get the generation results"""
        img_m, img_c, img_truth, mask = self.set_input()
        if self.PaintPanel.iteration < 100:
            with torch.no_grad():
                # encoder process
                distributions, f = self.model.net_E(img_m)
                q_distribution = torch.distributions.Normal(distributions[-1][0], distributions[-1][1])
                #q_distribution = torch.distributions.Normal( torch.zeros_like(distributions[-1][0]), torch.ones_like(distributions[-1][1]))
                z = q_distribution.sample()

                # decoder process
                scale_mask = task.scale_pyramid(mask, 4)
                self.img_g, self.atten = self.model.net_G(z, f_m=f[-1], f_e=f[2], mask=scale_mask[0].chunk(3, dim=1)[0])
                self.img_out = (1 - mask) * self.img_g[-1].detach() + mask * img_m

                # get score
                score =self.model.net_D(self.img_out).mean()
                self.label_6.setText(str(round(score.item(),3)))
                self.PaintPanel.iteration += 1
    
        self.show_result_flag = True
        self.show_result()


if __name__ == "__main__":

    #인식된 사람 (x1,y1,x2,y2)을 element로 갖는 리스트
    square = list()

    #모델 Places2-random 으로 설정
    app = QtWidgets.QApplication(sys.argv)
    opt = TestOptions().parse()
    opt.name = 'place2_random'
    opt.image = './'
    model = create_model(opt)

    #사진에 사람 수대로 직사각형 그리기
    for i in len(square):

    #fill

    #이미지 저장
