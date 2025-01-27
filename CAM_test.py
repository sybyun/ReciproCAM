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

#backbone = torch.nn.Sequential(*list(model.children())[:-3]).to(device)
#neck = torch.nn.Sequential(list(model.children())[-3]).to(device)
#cl_head = torch.nn.Sequential(list(model.children())[-2], 
#                                torch.nn.Flatten(), 
#                                list(model.children())[-1]).to(device)

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
input_file_name = '/mnt/SSD_1T/ILSVRC2012_cls/val/n01443537/ILSVRC2012_val_00028713.JPEG' #Two gold fishs
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
#input_file_name = '/mnt/SSD_1T/coco/images/val2017/000000054605.jpg' #ice cream + cake
#input_file_name = '/mnt/SSD_1T/coco/images/val2017/000000066926.jpg' #bagel
#input_file_name = '/mnt/SSD_1T/coco/images/val2017/000000070254.jpg' #tram

'''
img = read_image(input_file_name)
Height = 224
Width = 224
Mean = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
input_tensor = normalize(resize(img, (Height, Width)) / 255., Mean, STD).to(device)
'''
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
img = Image.open(input_file_name)
img = img.resize((224, 224))
input_tensor = transform(img).to(device)
img = np.asarray(img)

def recipro_cam_test():

    begin_time = time.time()
    
    #recipro_cam = ReciproCamFilter(backbone, neck, cl_head, device=device)
    recipro_cam = ReciproCam(backbone, cl_head, device=device)
    #recipro_cam = ReciproCamHook(model, device=device, target_layer_name='layer3')
    #recipro_cam = ReciproCamBB(model, device=device)
    cam, class_id = recipro_cam(input_tensor.unsqueeze(0), [0,1])
    #am = recipro_cam.get_feature_map()
    #cam, class_id = recipro_cam(input_tensor.unsqueeze(0), 19)

    #recipro_cam = ReciproCamLoc(backbone, cl_head, Mean, STD, device=device)
    #recipro_cam = ReciproCamLoc(model, Mean, STD, 0, 3, device=device)
    #cam, class_id = recipro_cam(img, Height, Width)
    #cam = torch.where(cam > 0.5, cam, torch.tensor(0.0, dtype=torch.float32).to(device))

    last_time = time.time()
    print('Recipro CAM Execution Time: ', last_time - begin_time, ' : ', class_id)

    #result = overlay_mask(to_pil_image(img), to_pil_image(am.detach(), mode='F'), alpha=0.5)
    #plt.imshow(result); plt.axis('off'); plt.tight_layout(); plt.show()

    result = overlay_mask(to_pil_image(img), to_pil_image(cam.detach(), mode='F'), alpha=0.5)
    plt.imshow(result); plt.axis('off'); plt.tight_layout(); plt.show()

    filename = './cam_results/test.png'
    result.save(filename)

    return 
    
def torchcam_gradCAMpp():
    
    model.eval()
    begin_time = time.time()
    cam_extractor = GradCAMpp(model_g)
    out = model_g(input_tensor.unsqueeze(0))
    print('class = ', out.squeeze(0).argmax().item())
    activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)
    #activation_map = cam_extractor(19, out)
    last_time = time.time()
    print('GradCAM++ Execution Time: ', last_time - begin_time, ' : ', out.squeeze(0).argmax().item())

    result = overlay_mask(to_pil_image(img), to_pil_image(activation_map[0].squeeze(0), mode='F'), alpha=0.5)
    plt.imshow(result); plt.axis('off'); plt.tight_layout(); plt.show()

    return

def torchcam_ScoreCAM():
    
    model_g.eval()
    begin_time = time.time()
    cam_extractor = ScoreCAM(model_g)
    out = model_g(input_tensor.unsqueeze(0).to(device))
    activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)
    #activation_map = cam_extractor(19, out)
    last_time = time.time()
    print('ScoreCAM Execution Time: ', last_time - begin_time, ' : ', out.squeeze(0).argmax().item())

    result = overlay_mask(to_pil_image(img), to_pil_image(activation_map[0].squeeze(0), mode='F'), alpha=0.5)
    plt.imshow(result); plt.axis('off'); plt.tight_layout(); plt.show()

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

def all_cam_test(given_class_id = None):

    begin_time = time.time()
    recipro_cam = ReciproCam(backbone, cl_head, device=device)
    if given_class_id is None:
        recipro_cam_out, class_id = recipro_cam(input_tensor.unsqueeze(0))
    else:
        recipro_cam_out, class_id = recipro_cam(input_tensor.unsqueeze(0), given_class_id)
    last_time = time.time()
    print('Target CAM Execution Time: ', last_time - begin_time, ' : ', class_id)
    recipro_cam_result = overlay_mask(to_pil_image(img), to_pil_image(recipro_cam_out.detach(), mode='F'), alpha=0.5)
    #plt.imshow(recipro_cam_result); plt.axis('off'); plt.tight_layout(); plt.show()

    model.eval()
    begin_time = time.time()
    cam_extractor = GradCAM(model)
    out = model(input_tensor.unsqueeze(0))
    if given_class_id is None:
        grad_cam_out = cam_extractor(out.squeeze(0).argmax().item(), out)
    else:
        grad_cam_out = cam_extractor(given_class_id, out)
    last_time = time.time()
    print('GradCAM Execution Time: ', last_time - begin_time, ' : ', out.squeeze(0).argmax().item())
    grad_cam_result = overlay_mask(to_pil_image(img), to_pil_image(grad_cam_out[0].squeeze(0), mode='F'), alpha=0.5)
    #plt.imshow(grad_cam_result); plt.axis('off'); plt.tight_layout(); plt.show()
    
    #softmax = torch.nn.Softmax(dim=1)
    #prediction = softmax(out)
    #print('Prediction: ', prediction.squeeze(0).topk(5))
    

    model_g_pp.eval()
    begin_time = time.time()
    cam_extractor = GradCAMpp(model_g_pp)
    out = model_g_pp(input_tensor.unsqueeze(0))
    if given_class_id is None:
        grad_cam_pp_out = cam_extractor(out.squeeze(0).argmax().item(), out)
    else:
        grad_cam_pp_out = cam_extractor(given_class_id, out)
    last_time = time.time()
    print('GradCAM++ Execution Time: ', last_time - begin_time, ' : ', out.squeeze(0).argmax().item())
    grad_cam_pp_result = overlay_mask(to_pil_image(img), to_pil_image(grad_cam_pp_out[0].squeeze(0), mode='F'), alpha=0.5)
    #plt.imshow(grad_cam_result); plt.axis('off'); plt.tight_layout(); plt.show()

    model_sg_pp.eval()
    begin_time = time.time()
    cam_extractor = SmoothGradCAMpp(model_sg_pp)
    out = model_sg_pp(input_tensor.unsqueeze(0))
    if given_class_id is None:
        sgrad_cam_pp_out = cam_extractor(out.squeeze(0).argmax().item(), out)
    else:
        sgrad_cam_pp_out = cam_extractor(given_class_id, out)
    last_time = time.time()
    print('Smooth GradCAM++ Execution Time: ', last_time - begin_time, ' : ', out.squeeze(0).argmax().item())
    sgrad_cam_pp_result = overlay_mask(to_pil_image(img), to_pil_image(sgrad_cam_pp_out[0].squeeze(0), mode='F'), alpha=0.5)
    #plt.imshow(grad_cam_result); plt.axis('off'); plt.tight_layout(); plt.show()

    model_score.eval()
    begin_time = time.time()
    cam_extractor = ScoreCAM(model_score)
    out = model_score(input_tensor.unsqueeze(0).to(device))
    if given_class_id is None:
        score_cam_out = cam_extractor(out.squeeze(0).argmax().item(), out)
    else:
        score_cam_out = cam_extractor(given_class_id, out)
    last_time = time.time()
    print('ScoreCAM Execution Time: ', last_time - begin_time, ' : ', out.squeeze(0).argmax().item())
    score_cam_result = overlay_mask(to_pil_image(img), to_pil_image(score_cam_out[0].squeeze(0), mode='F'), alpha=0.5)
    #plt.imshow(score_cam_result); plt.axis('off'); plt.tight_layout(); plt.show()

    
    filename = './cam_results/multi_objects_two_spoonbills.png'
    img1 = get_concat_h(to_pil_image(img), grad_cam_result, center_space=20, margin=0)
    img2 = get_concat_h(img1, grad_cam_pp_result, center_space=20, margin=0)
    img3 = get_concat_h(img2, sgrad_cam_pp_result, center_space=20, margin=0)
    img4 = get_concat_h(img3, score_cam_result, center_space=20, margin=0)
    concated_image = get_concat_h(img4, recipro_cam_result, center_space=20, margin=0)
    
    return concated_image
    

def three_cam_test(given_class_id = None):

    begin_time = time.time()
    recipro_cam = ReciproCam(backbone, cl_head, device=device)
    if given_class_id is None:
        recipro_cam_out, class_id = recipro_cam(input_tensor.unsqueeze(0))
    else:
        recipro_cam_out, class_id = recipro_cam(input_tensor.unsqueeze(0), given_class_id)
    last_time = time.time()
    print('Target CAM Execution Time: ', last_time - begin_time, ' : ', class_id)
    recipro_cam_result = overlay_mask(to_pil_image(img), to_pil_image(recipro_cam_out.detach(), mode='F'), alpha=0.5)
    #plt.imshow(recipro_cam_result); plt.axis('off'); plt.tight_layout(); plt.show()

    
    model_g_pp.eval()
    begin_time = time.time()
    cam_extractor = GradCAMpp(model_g_pp)
    out = model_g_pp(input_tensor.unsqueeze(0))
    if given_class_id is None:
        grad_cam_pp_out = cam_extractor(out.squeeze(0).argmax().item(), out)
    else:
        grad_cam_pp_out = cam_extractor(given_class_id, out)
    last_time = time.time()
    print('GradCAM++ Execution Time: ', last_time - begin_time, ' : ', out.squeeze(0).argmax().item())
    grad_cam_pp_result = overlay_mask(to_pil_image(img), to_pil_image(grad_cam_pp_out[0].squeeze(0), mode='F'), alpha=0.5)
    #plt.imshow(grad_cam_result); plt.axis('off'); plt.tight_layout(); plt.show()
    

    #softmax = torch.nn.Softmax(dim=1)
    #prediction = softmax(out)
    #print('Prediction: ', prediction.squeeze(0).topk(5))

    model_score.eval()
    begin_time = time.time()
    cam_extractor = ScoreCAM(model_score)
    out = model_score(input_tensor.unsqueeze(0).to(device))
    if given_class_id is None:
        score_cam_out = cam_extractor(out.squeeze(0).argmax().item(), out)
    else:
        score_cam_out = cam_extractor(given_class_id, out)
    last_time = time.time()
    print('ScoreCAM Execution Time: ', last_time - begin_time, ' : ', out.squeeze(0).argmax().item())
    score_cam_result = overlay_mask(to_pil_image(img), to_pil_image(score_cam_out[0].squeeze(0), mode='F'), alpha=0.5)
    #plt.imshow(score_cam_result); plt.axis('off'); plt.tight_layout(); plt.show()


    img1 = get_concat_h(to_pil_image(img), grad_cam_pp_result, center_space=20, margin=0)
    img2 = get_concat_h(img1, score_cam_result, center_space=20, margin=0)
    concated_image = get_concat_h(img2, recipro_cam_result, center_space=20, margin=0)
    
    return concated_image
    

if __name__ == '__main__':

    #torchcam_gradCAMpp()
    recipro_cam_test()
    #torchcam_ScoreCAM()
    #concated_image = all_cam_test()
    #concated_image = three_cam_test()

    #filename = './cam_results/layer3_same_multiple_objects_four_sparrows.png'
    #concated_image.save(filename)

