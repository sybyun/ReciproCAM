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
from util.recipro_cam import *
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

path2data="/mnt/SSD_1T/coco/images/val2017"
path2json="/mnt/SSD_1T/coco/annotations/instances_val2017.json"
coco_val = datasets.CocoDetection(root = path2data,annFile = path2json)


Height = 224
Width = 224
Mean = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

import operator
from collections import defaultdict
from nltk.corpus import wordnet as wn

def common_synset(words):
    synsets = []
    for word in words:
        synsets.extend(wn.synsets(word))
    count_dict = defaultdict(lambda: 0)
    for synset in synsets:
        count_dict[synset.name()] += 1
    try:
        common_synset_name = sorted(count_dict.items(), key=operator.itemgetter(1), reverse=True)[0][0]
    except:
        print(words, count_dict)
    return wn.synset(common_synset_name)

def get_hypernyms(synset):
    return [s[0].name().split('.')[0] for s in list(synset.hypernym_distances())]

def create_imagenet_to_coco():
    imagenet_ids = []
    imagenet_labels = []
    imagenet_labels_words = []
    with open('dataset/imagenet_1000_labels.txt', 'r') as file:
        for i, line in enumerate(file):
            id_, label = line.strip().split(': ')
            label_words = ['_'.join(s.split()).lower() for s in label.split(', ') if s]
            imagenet_ids.append(id_)
            imagenet_labels.append(label)
            imagenet_labels_words.append(label_words)

    coco_labels = []
    with open('dataset/coco_labels.txt', 'r') as file:
        for line in file:
            coco_labels.append('_'.join(line.strip().split()).lower())

    imagenet2coco = [None for _ in range(len(imagenet_labels))]
    for i, imagenet_label_words in enumerate(imagenet_labels_words):
        imagenet_label_hypernyms = get_hypernyms(common_synset(imagenet_label_words))
        coco_id = 1
        for coco_label in coco_labels:
            if coco_label in imagenet_label_hypernyms:
                imagenet2coco[i] = coco_id
            coco_id += 1

    with open('dataset/imagenet2coco3.txt', 'w') as file:
        for i, (id_, label, coco_label) in enumerate(zip(imagenet_ids, imagenet_labels, imagenet2coco)):
            if i != len(imagenet_labels) - 1:
                file.write(f'{i}\t{coco_label}\n')
            else:
                file.write(f'{i}\t{coco_label}')

coco_to_imagenet = {}
def load_map_data():
    with open('dataset/imagenet2coco3.txt', 'r') as file:
        for i, line in enumerate(file):
            imgnet_id, coco_id = line.strip().split('\t')
            if coco_id != 'None':
                coco_to_imagenet.setdefault(coco_id,[]).append(int(imgnet_id))

colors = ['r', 'b', 'pink', 'gray', 'yellow', 'purple', 'magenta', 'black']
def recipro_cam_loc_metric():

    load_map_data()

    begin_time = time.time()
    
    recipro_cam = ReciproCamLoc(model, Mean, STD, 0, 3, device=device)

    #print('Number of samples: ', len(coco_val))
    #for i, img, ann in enumerate(coco_val):
    #    img, ann = coco_val.__getitem__(0)
    #    img = pil_to_tensor(img)
    #    input_tensor = normalize(resize(img, (Height, Width)) / 255., Mean, STD).to(device)

    #    cam, class_id = recipro_cam(input_tensor, Height, Width)

    img, ann = coco_val.__getitem__(1600)
    img = pil_to_tensor(img)
    dict_indexes = {}
    bboxes = []
    for i in range(len(ann)):
        index = ann[i].get('category_id')
        if coco_to_imagenet.__contains__(str(index)):
            dict_indexes[str(index)] = coco_to_imagenet[str(index)]
            bboxes.append(np.array(ann[i].get('bbox')))
    if len(bboxes) == 0:
        return
    
    cam, class_id, scores = recipro_cam(img, Height, Width, dict_indexes)
    if len(scores) == 0:
        return
    n_scores = scores / np.array(scores).max()

    last_time = time.time()
    print('Recipro CAM Execution Time: ', last_time - begin_time, ' : ', class_id, ' : ', scores)

    for i in range(len(cam)):
        if i == 0:
            cam_sum = cam[i]*n_scores[i]
        else:
            cam_sum = torch.where(cam_sum > cam[i]*n_scores[i], cam_sum, cam[i]*n_scores[i])
    
    result = overlay_mask(to_pil_image(img), to_pil_image(cam_sum.detach(), mode='F'), alpha=0.5)
    plt.imshow(result)
    ax = plt.gca()        

    for i in range(len(cam)):
        cam_f = torch.where(cam[i] > (1 - (0.5*n_scores[i])), torch.tensor(255, dtype=torch.uint8).to(device), torch.tensor(0, dtype=torch.uint8).to(device))
        cam_i = to_pil_image(cam_f.detach(), mode='L').resize((result.size), Image.Resampling.BILINEAR)
        im = cv2.cvtColor(np.array(cam_i), cv2.COLOR_RGB2BGR)
        gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        contours, hierarchy = cv2.findContours(gray,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[-2:]
        idx =0
        full_boxes = []
        for cnt in contours:
            idx += 1
            x,y,w,h = cv2.boundingRect(cnt)
            full_boxes.append([x,y,x+w,y+h])
        full_boxes = np.array(full_boxes)
        nms_boxes = NMS(full_boxes)
        for box in nms_boxes:
            x,y,x2,y2 = box
            w = x2-x
            h = y2-y
            rect = plt.Rectangle((x,y),w,h,linewidth=2,edgecolor=colors[i%8],facecolor='none')
            ax.add_patch(rect)
    
    for i in range(len(bboxes)):    
        x, y, w, h = bboxes[i]
        rect = plt.Rectangle((x,y),w,h,linewidth=2,edgecolor='g',facecolor='none')
        ax.add_patch(rect)
    plt.show()
    
    return 
    

if __name__ == '__main__':

    #create_imagenet_to_coco()
    recipro_cam_loc_metric()

