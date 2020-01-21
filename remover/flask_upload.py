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
from flask import Flask, render_template, request
from werkzeug import secure_filename
import try2
from PIL import Image, ImageQt
import time

import os
import pathlib
import numpy as np
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from IPython.display import display

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

app = Flask(__name__)

#업로드 HTML 렌더링
@app.route('/upload')
def render_file():
   return render_template('upload.html')

#파일 업로드 처리
@app.route('/fileUpload', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      f = request.files['file']
      image = Image.open(f)
      image.save('./static/before.jpg','JPEG')
      # C:\Users\q\Desktop\tensorflow\models\research\object_detection\test_images
      app = QtWidgets.QApplication(sys.argv)
      opt = TestOptions().parse()
      
      # patch tf1 into `utils.ops`
      utils_ops.tf = tf.compat.v1

      # Patch the location of gfile
      tf.gfile = tf.io.gfile

      def load_model(model_name):
         base_url = 'http://download.tensorflow.org/models/object_detection/'
         model_file = model_name + '.tar.gz'
         model_dir = tf.keras.utils.get_file(
            fname=model_name, 
            origin=base_url + model_file,
            untar=True)

         model_dir = pathlib.Path(model_dir)/"saved_model"

         model = tf.saved_model.load(str(model_dir))
         model = model.signatures['serving_default']

         return model

      # List of the strings that is used to add correct label for each box.
      PATH_TO_LABELS = 'models/research/object_detection/data/mscoco_label_map.pbtxt'
      category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

      # If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
      PATH_TO_TEST_IMAGES_DIR = pathlib.Path("C:/Users/q/Desktop/Pluralistic-Inpainting/static")
      TEST_IMAGE_PATHS = sorted(list(PATH_TO_TEST_IMAGES_DIR.glob("*.jpg")))
      #TEST_IMAGE_PATHS = "C:/Users/q/Desktop/tensorflow/models/research/object_detection/test_images"

      print("")
      print("~~~~~~~~~~~~~~~~~~~")
      print(TEST_IMAGE_PATHS)
      print("")

      model_name = 'ssd_mobilenet_v1_coco_2017_11_17'
      detection_model = load_model(model_name)

      print(detection_model.inputs)
      detection_model.output_dtypes
      detection_model.output_shapes

      def run_inference_for_single_image(model, image):
         image = np.asarray(image)
         # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
         input_tensor = tf.convert_to_tensor(image)
         # The model expects a batch of images, so add an axis with `tf.newaxis`.
         input_tensor = input_tensor[tf.newaxis,...]

         # Run inference
         output_dict = model(input_tensor)
         # All outputs are batches tensors.
         # Convert to numpy arrays, and take index [0] to remove the batch dimension.
         # We're only interested in the first num_detections.
         num_detections = int(output_dict.pop('num_detections'))
         output_dict = {key:value[0, :num_detections].numpy() 
                        for key,value in output_dict.items()}
         output_dict['num_detections'] = num_detections

         # detection_classes should be ints.
         output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
         # Handle models with masks:
         if 'detection_masks' in output_dict:
            # Reframe the the bbox mask to the image size.
            detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                     output_dict['detection_masks'], output_dict['detection_boxes'],
                        image.shape[0], image.shape[1])      
            detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                                tf.uint8)
            output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()   
         return output_dict

      def show_inference(model, image_path):
         image_np = np.array(Image.open(image_path))
         output_dict = run_inference_for_single_image(model, image_np)
         # print("ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ")
         # print(output_dict['detection_boxes'])
         # print("ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ")
         # Visualization of the results of a detection.
         vis_util.visualize_boxes_and_labels_on_image_array(
               image_np,
               output_dict['detection_boxes'],
               output_dict['detection_classes'],
               output_dict['detection_scores'],
               category_index,
               instance_masks=output_dict.get('detection_masks_reframed', None),
               use_normalized_coordinates=True,
               line_thickness=8)
         width, height = image_np.shape[:2]
         box = np.squeeze(output_dict['detection_boxes'])
         res = []
         for i in range(len(output_dict['detection_boxes'])):
               ymin = (int(box[i, 0]*height))
               xmin = (int(box[i, 1]*width))
               ymax = (int(box[i, 2]*height))
               xmax = (int(box[i, 3]*width))
               #print('[', xmin, "," ,ymin, ",", xmax, ",", ymax, ']')
               tmp = []
               tmp.append(str(xmin))
               tmp.append(str(ymin))
               tmp.append(str(xmax))
               tmp.append(str(ymax))
               #tmp = '[' + str(xmin) + "," + str(ymin) + "," + str(xmax) + "," + str(ymax) + '],'
               res.append(tmp)
         #        coordinateList.append(str)
         #print(tmp)
         display(Image.fromarray(image_np))
         return res

      model_name = "mask_rcnn_inception_resnet_v2_atrous_coco_2018_01_28"
      masking_model = load_model("mask_rcnn_inception_resnet_v2_atrous_coco_2018_01_28")
      masking_model.output_shapes
      coordinate = ""
      for image_path in TEST_IMAGE_PATHS:
         coordinate = show_inference(masking_model, image_path)
   
      
      img_final = try2.main(opt, image, coordinate)
      img_final.save('./static/final.jpg','JPEG')
      #저장할 경로 + 파일명
      #f.save("./img/" + secure_filename(f.filename))
      return render_template('img_test.html', before = "../static/before.jpg", final = "../static/final.jpg")

if __name__ == '__main__':
   #  서버 실행
  app.run(debug = True)
