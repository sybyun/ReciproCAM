import os
import sys
import time
import math
from tkinter.tix import IMAGE
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image

import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.transforms.functional import to_pil_image, pil_to_tensor, normalize, resize
from torchvision.io.image import read_image

import torch.optim as optim
import torch.onnx as torch_onnx

from torchcam.methods import GradCAM, SmoothGradCAMpp, GradCAMpp, ScoreCAM
from torchcam.utils import overlay_mask

from util.util import *
from ReciproCam.util.recipro_cam_surgery import *
from util.recipro_cam_hook import *
from util.recipro_cam_blackbox import *
from util.recipro_cam_filter import *
from util.recipro_cam_loc import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Test with: {device}")

model = models.resnet50(pretrained=True).to(device)
#model = models.vgg16(pretrained=True)
model_g = models.resnet50(pretrained=True).to(device)
model_g_pp = models.resnet50(pretrained=True).to(device)
model_sg_pp = models.resnet50(pretrained=True).to(device)
model_score = models.resnet50(pretrained=True).to(device)

backbone = torch.nn.Sequential(*list(model.children())[:-2]).to(device)
cl_head = torch.nn.Sequential(list(model.children())[-2], 
                                torch.nn.Flatten(), 
                                list(model.children())[-1]).to(device)


#-------------------- single object --------------
#input_file_name = '/mnt/SSD_1T/ILSVRC2012_cls/val/n06794110/ILSVRC2012_val_00011160.JPEG' #Trapic sign
#input_file_name = '/mnt/SSD_1T/ILSVRC2012_cls/val/n01806143/ILSVRC2012_val_00039299.JPEG' #Peacock
#input_file_name = '/mnt/SSD_1T/ILSVRC2012_cls/val/n02236044/ILSVRC2012_val_00007130.JPEG' #Mantis
#input_file_name = './data/CUB-200-2011_seagul.png'
#input_file_name = './data/CUB-200-2011_seagul2.png'

#-------------------- same objects -------------
#input_file_name = '/mnt/SSD_1T/ILSVRC2012_cls/val/n01443537/ILSVRC2012_val_00028713.JPEG' #Two gold fishs
#input_file_name = '/mnt/SSD_1T/ILSVRC2012_cls/val/n04612504/ILSVRC2012_val_00025780.JPEG' #Four yachts
#input_file_name = '/mnt/SSD_1T/ILSVRC2012_cls/val/n02607072/ILSVRC2012_val_00045825.JPEG' #Two fishs
#input_file_name = '/mnt/SSD_1T/ILSVRC2012_cls/val/n01530575/ILSVRC2012_val_00030224.JPEG' #Four sparrows

#-------------------- multiple objects ---------
#input_file_name = '/mnt/SSD_1T/ILSVRC2012_cls/val/n04039381/ILSVRC2012_val_00011663.JPEG' #Tennis player
#input_file_name = '/mnt/SSD_1T/ILSVRC2012_cls/val/n04039381/ILSVRC2012_val_00006517.JPEG' #Tennis player2
#input_file_name = '/mnt/SSD_1T/ILSVRC2012_cls/val/n02514041/ILSVRC2012_val_00005684.JPEG' #Person+Fish
#input_file_name = '/mnt/SSD_1T/ILSVRC2012_cls/val/n01440764/ILSVRC2012_val_00011233.JPEG' #Person+Trout
#input_file_name = '/mnt/SSD_1T/ILSVRC2012_cls/val/n02389026/ILSVRC2012_val_00017532.JPEG' #Person+Horse
#input_file_name = './data/Dog_cat_image.png'
#input_file_name = './data/Dog_person_image.png'
#input_file_name = '/mnt/SSD_1T/ILSVRC2012_cls/val/n04552348/ILSVRC2012_val_00000369.JPEG' #Two planes and statue
#input_file_name = '/mnt/SSD_1T/ILSVRC2012_cls/val/n02006656/ILSVRC2012_val_00000774.JPEG' #Two Crane and four mallard
#input_file_name = '/mnt/SSD_1T/ILSVRC2012_cls/val/n02085620/ILSVRC2012_val_00017702.JPEG' #Three dogs
#input_file_name = '/mnt/SSD_1T/ILSVRC2012_cls/val/n02087394/ILSVRC2012_val_00004992.JPEG' #Rhodesian + French bull
#input_file_name = '/mnt/SSD_1T/ILSVRC2012_cls/val/n02088094/ILSVRC2012_val_00032668.JPEG' #Two dogs Shepherd
#input_file_name = '/mnt/SSD_1T/ILSVRC2012_cls/val/n04285008/ILSVRC2012_val_00033293.JPEG' #Four cars sport car
#input_file_name = '/mnt/SSD_1T/ILSVRC2012_cls/val/n07753592/ILSVRC2012_val_00004614.JPEG' #Fruits
#input_file_name = './data/Bird_image2.png' #Kite and other


#input_file_name = '/mnt/SSD_1T/ILSVRC2012_cls/val/n01534433/ILSVRC2012_val_00003903.JPEG' #Bird
#input_file_name = '/mnt/SSD_1T/ILSVRC2012_cls/val/n02002556/ILSVRC2012_val_00006959.JPEG' #Bird
#input_file_name = '/mnt/SSD_1T/ILSVRC2012_cls/val/n02091635/ILSVRC2012_val_00039569.JPEG' #Bird
#input_file_name = '/mnt/SSD_1T/ILSVRC2012_cls/val/n02951358/ILSVRC2012_val_00023574.JPEG' #kayak
#input_file_name = '/mnt/SSD_1T/ILSVRC2012_cls/val/n02066245/ILSVRC2012_val_00009440.JPEG' #Whale
#input_file_name = '/mnt/SSD_1T/ILSVRC2012_cls/val/n03376595/ILSVRC2012_val_00008903.JPEG' #Pants
#input_file_name = '/mnt/SSD_1T/ILSVRC2012_cls/val/n03769881/ILSVRC2012_val_00005606.JPEG' #Ven
#input_file_name = './data/bird_image1.png'
#input_file_name = './data/STL10_plane_image.png'
#input_file_name = './data/Car_image.png'
#input_file_name = './data/Dog_image.png'
#input_file_name = './data/Dog_image2.png'
#input_file_name = './data/Bird_image.png'
#input_file_name = './data/Bird_image2.png'

#------------------MS COCO------------------------
#input_file_name = '/mnt/SSD_1T/coco/images/val2017/000000001584.jpg' #Bus
#input_file_name = '/mnt/SSD_1T/coco/images/val2017/000000002149.jpg' #apple
#input_file_name = '/mnt/SSD_1T/coco/images/val2017/000000023781.jpg' #Fruits
#input_file_name = '/mnt/SSD_1T/coco/images/val2017/000000027620.jpg' #desk
#input_file_name = '/mnt/SSD_1T/coco/images/val2017/000000029675.jpg' #hotdog
#input_file_name = '/mnt/SSD_1T/coco/images/val2017/000000045472.jpg' #pineapple
#input_file_name = '/mnt/SSD_1T/coco/images/val2017/000000048153.jpg' #shoe
#input_file_name = '/mnt/SSD_1T/coco/images/val2017/000000052413.jpg' #phone
input_file_name = '/mnt/SSD_1T/coco/images/val2017/000000054605.jpg' #ice cream + cake
#input_file_name = '/mnt/SSD_1T/coco/images/val2017/000000066926.jpg' #bagel
#input_file_name = '/mnt/SSD_1T/coco/images/val2017/000000070254.jpg' #tram


img = read_image(input_file_name)

Height = 224
Width = 224
Mean = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
input_tensor = normalize(resize(img, (Height, Width)) / 255., Mean, STD).to(device)

def recipro_cam_test():

    begin_time = time.time()
    
    recipro_cam = ReciproCamLoc(model, Mean, STD, 0, 3, device=device)
    cam, class_id = recipro_cam(img, Height, Width)

    last_time = time.time()
    print('Recipro CAM Execution Time: ', last_time - begin_time, ' : ', class_id)

    result = overlay_mask(to_pil_image(img), to_pil_image(cam.detach(), mode='F'), alpha=0.5)
    #plt.imshow(result); plt.axis('off'); plt.tight_layout(); plt.show()

    cam_f = torch.where(cam > 0.5, torch.tensor(255, dtype=torch.uint8).to(device), torch.tensor(0, dtype=torch.uint8).to(device))
    cam_i = to_pil_image(cam_f.detach(), mode='L').resize((result.size), Image.Resampling.BILINEAR)

    plt.imshow(result)
    ax = plt.gca()
    
    im = cv2.cvtColor(np.array(cam_i), cv2.COLOR_RGB2BGR)
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    contours, hierarchy = cv2.findContours(gray,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[-2:]
    idx =0
    for cnt in contours:
        idx += 1
        x,y,w,h = cv2.boundingRect(cnt)
        #print(idx, ' = (', x, ',', y, ') (', x+w, ',', y+h, ')')
        rect = plt.Rectangle((x,y),w,h,linewidth=2,edgecolor='g',facecolor='none')
        ax.add_patch(rect)
    
    plt.show()

    return 
    

def get_concat_h(im1, im2, center_space=20, margin=20):
    dst = Image.new('RGB', (im1.width + im2.width + center_space + margin, im1.height), color = (255, 255, 255))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width + center_space, 0))
    return dst

def get_concat_v(im1, im2):
    dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst


if __name__ == '__main__':

    recipro_cam_test()

