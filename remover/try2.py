# -*- coding: utf-8 -*- 
from PIL import Image
from gui.ui_window import Ui_Form
from gui.ui_draw import *
from PIL import Image, ImageQt
import random, io, os
import numpy as np
import torch
import torchvision.transforms as transforms
from dataloader.image_folder import make_dataset
from model import create_model
import sys
from options.test_options import TestOptions
from util import task, util
from gui.ui_model import ui_model
from util.visualizer import Visualizer
from PyQt5 import QtWidgets, QtGui,QtCore

def main(opt, img, boxlist):
    transform = transforms.Compose([transforms.ToTensor()])
    #model 설정
    opt.name = 'place2_random'
    opt.img_file = './'
    model = create_model(opt)

    #mask 만들기
    width, height = img.size
    mask = Image.new("RGB", (width, height))
    
    #[x1, y1, x2, y2]
    for i in range(len(boxlist)):
        element = boxlist[i]
        im = Image.new("RGB", (abs(int(element[2])-int(element[0])), abs(int(element[3])-int(element[1]))), (255,255,255))
        mask.paste(im, (int(element[0]), int(element[1]), int(element[2]), int(element[3])))
    
    pil_im = mask

    img = transform(img)
    value = 8

    mask = torch.autograd.Variable(transform(pil_im)).unsqueeze(0)
    mask = (mask < 1).float()

    if len(opt.gpu_ids) > 0:
        img = img.unsqueeze(0).cuda(opt.gpu_ids[0])
        mask = mask.cuda(opt.gpu_ids[0])
    
    img_truth = img * 2 - 1
    img_m = mask * img_truth
    img_c = (1 - mask) * img_truth

    #fill_mask
    #with torch.no_grad():
    with torch.no_grad():
        distributions, f = model.net_E(img_m)
        q_distribution = torch.distributions.Normal(distributions[-1][0], distributions[-1][1])
        z = q_distribution.sample()

        scale_mask = task.scale_pyramid(mask, 4)
        img_g, atten = model.net_G(z, f_m=f[-1], f_e=f[2], mask=scale_mask[0].chunk(3, dim=1)[0])
        img_out = (1 - mask) * img_g[-1].detach() + mask * img_m
    
    #save the result
    img_result = util.tensor2im(img_out)
    img_final = Image.fromarray(img_result, 'RGB')
    #util.save_image(img_result, './')

    return img_final

if __name__=="__main__":
    image = Image.open('06.jpg')
    app = QtWidgets.QApplication(sys.argv)
    opt = TestOptions().parse()
    main(opt, image, [[ 151 , 194 , 174 , 254 ],
[ 94 , 169 , 102 , 177 ],
[ 147 , 232 , 156 , 248 ],
[ 70 , 169 , 96 , 181 ],
[ 105 , 168 , 116 , 176 ],
[ 102 , 169 , 107 , 176 ],
[ 119 , 168 , 127 , 174 ],
[ 112 , 168 , 118 , 175 ],
[ 99 , 169 , 105 , 177 ],
[ 96 , 169 , 104 , 177 ],
[ 104 , 169 , 110 , 176 ]
])
    sys.exit(app.exec_())