#!/usr/bin/env python
# coding: utf-8

# # Mask R-CNN - Inspect Ballon Trained Model
#
# Code and visualizations to test, debug, and evaluate the Mask R-CNN model.


from data import COCODetection, get_label_map, MEANS, COLORS
from yolact import Yolact
from utils.augmentations import BaseTransform, FastBaseTransform, Resize
from utils.functions import MovingAverage, ProgressBar
from layers.box_utils import jaccard, center_size, mask_iou
from utils import timer
from utils.functions import SavePath
from layers.output_utils import postprocess, undo_image_transformation
import pycocotools

from data import cfg, set_cfg, set_dataset

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import argparse
import time
import random
import cProfile
import pickle
import json
import os
from collections import defaultdict
from pathlib import Path
from collections import OrderedDict
from PIL import Image

import matplotlib.pyplot as plt
import cv2


  

# Ros libraries
import roslib
import rospy
# Ros Messages
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from skimage.util import img_as_float
from collections import defaultdict

import time
 
print('Loading model...', end='')
with torch.no_grad():
    cudnn.fastest = True
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    net = Yolact()
    net.load_weights('weights/yolact_plus_base_3_88_interrupt.pth')
    net.eval()
    net = net.cuda()
    net.detect.use_fast_nms = True
    net.detect.use_cross_class_nms = False
    cfg.mask_proto_debug = False

color_cache = defaultdict(lambda: {})

top_k = 15
score_threshold = 0.2
crop = False
display_lincomb = False
display_masks = True
eval_mask_branch = True
display_fps = True
display_bboxes = True
display_text =  True
display_scores = True

def prep_display(dets_out, img, h, w, undo_transform=True, class_color=False, mask_alpha=0.45, fps_str=''):
    """
    Note: If undo_transform=False then im_h and im_w are allowed to be None.
    """
    if undo_transform:
        img_numpy = undo_image_transformation(img, w, h)
        img_gpu = torch.Tensor(img_numpy).cuda()
    else:
        img_gpu = img / 255.0
        h, w, _ = img.shape
    
    with timer.env('Postprocess'):
        save = cfg.rescore_bbox
        cfg.rescore_bbox = True
        t = postprocess(dets_out, w, h, visualize_lincomb = display_lincomb,
                                        crop_masks        = crop,
                                        score_threshold   = score_threshold)
        cfg.rescore_bbox = save

    with timer.env('Copy'):
        idx = t[1].argsort(0, descending=True)[:top_k]
        
        if cfg.eval_mask_branch:
            # Masks are drawn on the GPU, so don't copy
            masks = t[3][idx]
        classes, scores, boxes = [x[idx].cpu().numpy() for x in t[:3]]

    num_dets_to_consider = min(top_k, classes.shape[0])
    for j in range(num_dets_to_consider):
        if scores[j] < score_threshold:
            num_dets_to_consider = j
            break

    # Quick and dirty lambda for selecting the color for a particular index
    # Also keeps track of a per-gpu color cache for maximum speed
    def get_color(j, on_gpu=None):
        global color_cache
        color_idx = (classes[j] * 5 if class_color else j * 5) % len(COLORS)
        
        if on_gpu is not None and color_idx in color_cache[on_gpu]:
            return color_cache[on_gpu][color_idx]
        else:
            color = COLORS[color_idx]
            if not undo_transform:
                # The image might come in as RGB or BRG, depending
                color = (color[2], color[1], color[0])
            if on_gpu is not None:
                color = torch.Tensor(color).to(on_gpu).float() / 255.
                color_cache[on_gpu][color_idx] = color
            return color

    # First, draw the masks on the GPU where we can do it really fast
    # Beware: very fast but possibly unintelligible mask-drawing code ahead
    # I wish I had access to OpenGL or Vulkan but alas, I guess Pytorch tensor operations will have to suffice
    if display_masks and eval_mask_branch and num_dets_to_consider > 0:
        # After this, mask is of size [num_dets, h, w, 1]
        masks = masks[:num_dets_to_consider, :, :, None]
        
        # Prepare the RGB images for each mask given their color (size [num_dets, h, w, 1])
        colors = torch.cat([get_color(j, on_gpu=img_gpu.device.index).view(1, 1, 1, 3) for j in range(num_dets_to_consider)], dim=0)
        masks_color = masks.repeat(1, 1, 1, 3) * colors * mask_alpha

        # This is 1 everywhere except for 1-mask_alpha where the mask is
        inv_alph_masks = masks * (-mask_alpha) + 1
        
        # I did the math for this on pen and paper. This whole block should be equivalent to:
        #for j in range(num_dets_to_consider):
            #img_gpu = img_gpu * inv_alph_masks[j] + masks_color[j]

        masks_color_summand = masks_color[0]
        if num_dets_to_consider > 1:
            inv_alph_cumul = inv_alph_masks[:(num_dets_to_consider-1)].cumprod(dim=0)
            masks_color_cumul = masks_color[1:] * inv_alph_cumul
            masks_color_summand += masks_color_cumul.sum(dim=0)

        img_gpu = img_gpu * inv_alph_masks.prod(dim=0) + masks_color_summand
        masks = masks.byte().cpu().numpy()
 

    # Then draw the stuff that needs to be done on the cpu
    # Note, make sure this is a uint8 tensor or opencv will not anti alias text for whatever reason
    img_numpy = (img_gpu * 255).byte().cpu().numpy()

 
    if num_dets_to_consider == 0:
        return img_numpy

    white_image = np.zeros((540, 960, 3), np.uint8)
    white_image[:] = (255, 255, 255)
    object_mask_image = np.zeros((540, 960,1), np.uint8)
    kernel = np.ones((21,21),np.uint8)

    if display_text or display_bboxes:
        for j in reversed(range(num_dets_to_consider)):
            x1, y1, x2, y2 = boxes[j, :]
            color = get_color(j)
            score = scores[j]

            if display_bboxes:
                cv2.rectangle(img_numpy, (x1, y1), (x2, y2), color, 1)

            if display_text:
                _class = cfg.dataset.class_names[classes[j]]
                text_str = '%s: %.2f' % (_class, score) if display_scores else _class

                font_face = cv2.FONT_HERSHEY_DUPLEX
                font_scale = 0.6
                font_thickness = 1

                text_w, text_h = cv2.getTextSize(text_str, font_face, font_scale, font_thickness)[0]

                text_pt = (x1, y1 - 3)
                text_color = [255, 255, 255]

                cv2.rectangle(img_numpy, (x1, y1), (x1 + text_w, y1 - text_h - 4), color, -1)
                cv2.putText(img_numpy, text_str, text_pt, font_face, font_scale, text_color, font_thickness, cv2.LINE_AA)

            object_mask_image = masks[j]

            contours, hierarchy = cv2.findContours(
                object_mask_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.fillPoly(white_image, contours, (random.randint(
                0, 255), random.randint(0, 255), random.randint(0, 255)))
               
 
    return img_numpy

class Yolact_ROS_Node:

    def __init__(self):
        '''Initialize ros publisher, ros subscriber'''
        # topic where we publish
        self.image_pub = rospy.Publisher("/output/maskrcnn/segmented",
                                         Image)

        self.subscriber = rospy.Subscriber("/camera/color/image_raw",
                                           Image, self.callback, queue_size=1, buff_size=2002428800)
        self.bridge = CvBridge()                                                         
        self.counter = 0
        ########################
        # your fancy code here #
        ########################
        self.start_time = time.time()
        self.x = 1 # displays the frame rate every 1 second
 



    def callback(self, ros_data):
        '''Callback function of subscribed topic. 
        Here images get converted and OBJECTS detected'''
        #### direct conversion to CV2 ####
        cv_image = self.bridge.imgmsg_to_cv2(ros_data, desired_encoding="bgr8")

        # Uncomment thefollowing block in order to collect training data
         
        '''
        self.counter = self.counter +1 
        '''
 
        frame = torch.from_numpy(cv_image).cuda().float()
        batch = FastBaseTransform()(frame.unsqueeze(0))

        try:
            batch = batch.cuda()
            preds = net(batch)

            img_numpy = prep_display(preds, frame, None, None, undo_transform=False)
            # Run object detection

            #### PUBLISH SEGMENTED IMAGE ####
            msg = self.bridge.cv2_to_imgmsg(img_numpy, "bgr8")
            msg.header.stamp = rospy.Time.now()
            self.image_pub.publish(msg)
        except KeyboardInterrupt:
            print('Stopping...') 
        self.counter+=1
        if (time.time() - self.start_time) > self.x :
            print("FPS: ", self.counter / (time.time() - self.start_time))
            self.counter = 0
            self.start_time = time.time()    
  

    def segment_objects_on_white_image(self,image, boxes, masks, class_ids,
                                       scores=None,):
        """Apply color splash effect.
        image: RGB image [height, width, 3]
        mask: instance segmentation mask [height, width, instance count]
        Returns result image.
        """
        # Make a grayscale copy of the image. The grayscale copy still
        # has 3 RGB channels, though.
        #xyz = rgb2xyz(image)
        height = image.shape[0]
        width = image.shape[1]
        channels = image.shape[2]

        N = boxes.shape[0]

        white_image = np.zeros((height, width, channels), np.uint8)
        white_image[:] = (255, 255, 255)
        object_mask_image = np.zeros((height, width), np.uint8)
        kernel = np.ones((21,21),np.uint8)

        for i in range(N):

            # Bounding box
            if not np.any(boxes[i]):
                # Skip this instance. Has no bbox. Likely lost in image cropping.
                continue

            class_id = class_ids[i]
            score = scores[i] if scores is not None else None
            if(score < 0.99):
                break
            # Mask
            object_mask_image[:,:] = masks[:, :, i]
    
            contours, hierarchy = cv2.findContours(
                object_mask_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.fillPoly(white_image, contours, (random.randint(
                0, 255), random.randint(0, 255), random.randint(0, 255)))
               
  
        #white_image = cv2.erode(white_image,kernel,iterations = 1)
        return white_image


# Run Node
if __name__ == '__main__':
    '''Initializes and cleanup ros node'''
    ic = Yolact_ROS_Node()
    rospy.init_node('Yolact_ROS_Node', anonymous=True)
    rospy.Rate(10)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down ROS Image feature detector module")
