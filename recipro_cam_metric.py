import time
from tkinter.tix import IMAGE
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.transforms.functional import to_pil_image, pil_to_tensor, normalize
import torch.optim as optim
import torch.onnx as torch_onnx

from sklearn.metrics import auc
from torchmetrics import PearsonCorrCoef

from torchcam.methods import GradCAM, SmoothGradCAMpp, GradCAMpp, ScoreCAM

from util.util import *
from util.recipro_cam import *
from util.recipro_cam_hook import *
from util.recipro_cam_blackbox import *
from util.recipro_cam_filter import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Test with: {device}")

#model = models.vgg16(pretrained=True).to(device)
#eval_model = models.vgg16(pretrained=True).to(device)
#model = models.resnet18(pretrained=True).to(device)
#eval_model = models.resnet18(pretrained=True).to(device)
#model = models.resnet50(pretrained=True).to(device)
#eval_model = models.resnet50(pretrained=True).to(device)
model = models.resnet101(pretrained=True).to(device)
eval_model = models.resnet101(pretrained=True).to(device)
#model = torch.hub.load('pytorch/vision:v0.10.0', 'resnext50_32x4d', pretrained=True).to(device)
#eval_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnext50_32x4d', pretrained=True).to(device)
#model = torch.hub.load('pytorch/vision:v0.10.0', 'resnext101_32x8d', pretrained=True).to(device)
#eval_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnext101_32x8d', pretrained=True).to(device)

model_g_pp = models.resnet50(pretrained=True).to(device)
model_sg_pp = models.resnet50(pretrained=True).to(device)
model_score = models.resnet50(pretrained=True).to(device)

feature_net = torch.nn.Sequential(*list(model.children())[:-2]).to(device)
head_net = torch.nn.Sequential(list(model.children())[-2], 
                                torch.nn.Flatten(), 
                                list(model.children())[-1]).to(device)


Height = 224
Width = 224
MOD = 50

batch_size = 250

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder('/mnt/SSD_1T/ILSVRC2012_cls/val', transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.CenterCrop((Height, Width)),
        transforms.ToTensor(),
        normalize,
    ])),
    batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
softmax = torch.nn.Softmax(dim=1)


def average_drop_increase(type='white'):

    avg_drop = 0.0
    avg_inc = 0.0

    explanation = torch.zeros(batch_size, 3, Height, Width)
    yc = torch.zeros(batch_size)
    oc = torch.zeros(batch_size)

    N = 0

    eval_model.eval()
    if type == 'white':
        recipro_cam = ReciproCam(feature_net, head_net, device=device)
        feature_net.eval()
        head_net.eval()
    elif type == 'hook':
        recipro_cam = ReciproCamHook(model, device=device)
    else:
        recipro_cam = ReciproCamBB(model, device=device)

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(val_loader, 0):
            images, labels = images.to(device), labels.to(device)
            if type == 'white':
                features = feature_net(images)
                predictions = head_net(features)
                predictions = softmax(predictions)
            else:
                predictions = model(images)
                predictions = softmax(predictions)
            for i in range(batch_size):
                class_id = labels[i].item()
                yc[i] = predictions[i][class_id]
                cam, _ = recipro_cam(images[i].unsqueeze(0), class_id)
                cam = to_pil_image(cam, mode='F')
                cam = cam.resize((Height,Width), resample=Image.Resampling.BICUBIC)
                cam = pil_to_tensor(cam)
                explanation[i] = torch.mul(images[i], cam.repeat(3,1,1).to(device))
            explanation = explanation.to(device)
            o_predictions = eval_model(explanation)        
            o_predictions = softmax(o_predictions)
            for i in range(batch_size):
                class_id = labels[i].item()
                oc[i] = o_predictions[i][class_id]
            #oc = o_predictions[:,class_id]
            drop = torch.nn.functional.relu(yc - oc) / yc
            inc = torch.count_nonzero(torch.nn.functional.relu(oc - yc))
            avg_drop += drop.sum()
            avg_inc += inc

            N += batch_size

            #if batch_idx*batch_size >= 2000:
            #    break
            if batch_idx % MOD == MOD-1:
                print('Batch ID: ', batch_idx + 1)

        avg_drop /= N * 0.01
        avg_inc /= N * 0.01

    return avg_drop, avg_inc


def dauc_iauc(type='white'):

    DAUC_score = 0.0
    IAUC_score = 0.0
    N = 0

    eval_model.eval()
    if type == 'white':
        recipro_cam = ReciproCam(feature_net, head_net, device=device)
        feature_net.eval()
        head_net.eval()
    elif type == 'hook':
        recipro_cam = ReciproCamHook(model, device=device)
    else:
        recipro_cam = ReciproCamBB(model, device=device)

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(val_loader, 0):
            images, labels = images.to(device), labels.to(device)
            if type == 'white':
                features = feature_net(images)
                predictions = head_net(features)
                predictions = softmax(predictions)
                bs, ch, h, w = features.shape
            elif type == 'hook':
                h = int(Height/32)
                w = int(Width/32)
                predictions = model(images)
                predictions = softmax(predictions)
            else:
                h = int(Height/recipro_cam.patch_h)
                w = int(Width/recipro_cam.patch_w)
                predictions = model(images)
                predictions = softmax(predictions)

            dx = int(Width/w)
            dy = int(Height/h)
            ck = torch.zeros(h*w+1).to(device)
            for i in range(batch_size):
                class_id = labels[i].item()
                cam, _ = recipro_cam(images[i].unsqueeze(0), class_id)
                cam = cam.reshape(-1,)
                _, s_index = cam.sort(dim=0, descending=True)
                del_mask = torch.ones(3, Height, Width).to(device)
                inc_mask = torch.zeros(3, Height, Width).to(device)
                base_color = torch.mean(images[i], dim=(1, 2)).unsqueeze(0)
                base_color = base_color.reshape(1,3,1,1)
                auc_images = base_color.repeat(2*h*w,1,Height, Width).to(device)
                for j in range(h*w):
                    s_idx = int(s_index[j].cpu().item())
                    ci = s_idx//w
                    cj = s_idx - ci*w
                    xs = int(cj*dx)
                    xe = int(min((cj+1)*dx, Width-1))
                    ys = int(ci*dy)
                    ye = int(min((ci+1)*dy, Height-1))
                    del_mask[:,ys:ye+1,xs:xe+1] = 0.0
                    inc_mask[:,ys:ye+1,xs:xe+1] = 1.0
                    auc_images[j] = images[i]*del_mask
                    #auc_images[h*w + j] = images[i]*inc_mask
                    #auc_images[j] = auc_images[j]*inc_mask + images[i]*del_mask
                    auc_images[h*w + j] = auc_images[h*w +j]*del_mask + images[i]*inc_mask
                o_predictions = eval_model(auc_images)        
                o_predictions = softmax(o_predictions)
                ck[0] = predictions[i][class_id]
                ck[1:] = o_predictions[:h*w,class_id]
                #ck /= ck.max()
                x = np.arange(0, len(ck))
                y = ck.detach().cpu().numpy()
                DAUC_score += auc(x, y) / len(ck)
                ck[h*w] = predictions[i][class_id]
                ck[:h*w] = o_predictions[h*w:,class_id]
                #ck /= ck.max()
                y = ck.detach().cpu().numpy()
                IAUC_score += auc(x, y) / len(ck)
            N += batch_size

            #if batch_idx*batch_size >= 2000:
            #    break
            if batch_idx % MOD == MOD-1:
                print('Batch ID: ', batch_idx + 1)

        DAUC_score = DAUC_score / (N * 0.01)
        IAUC_score = IAUC_score / (N * 0.01)

    return DAUC_score, IAUC_score


def ADCC(type='white'):

    adcc = 0.0
    coherency = 0.0
    complexity = 0.0
    avg_drop = 0.0
    avg_inc = 0.0

    pearson = PearsonCorrCoef().to(device)

    explanation = torch.zeros(batch_size, 3, Height, Width).to(device)
    yc = torch.zeros(batch_size)
    oc = torch.zeros(batch_size)

    N = 0

    eval_model.eval()
    if type == 'white':
        recipro_cam = ReciproCam(feature_net, head_net, device=device)
        feature_net.eval()
        head_net.eval()
    elif type == 'hook':
        recipro_cam = ReciproCamHook(model, device=device)
    else:
        recipro_cam = ReciproCamBB(model, device=device)

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(val_loader, 0):
            images, labels = images.to(device), labels.to(device)

            if type == 'white':
                features = feature_net(images)
                predictions = head_net(features)
                predictions = softmax(predictions)
                bs, ch, h, w = features.shape
            elif type == 'hook':
                h = int(Height/32)
                w = int(Width/32)
                predictions = model(images)
                predictions = softmax(predictions)
            else:
                h = int(Height/recipro_cam.patch_h)
                w = int(Width/recipro_cam.patch_w)
                predictions = model(images)
                predictions = softmax(predictions)


            pre_cam = torch.zeros(batch_size, h*w).to(device)

            for i in range(batch_size):
                class_id = labels[i].item()
                yc[i] = predictions[i][class_id]
                cam, _ = recipro_cam(images[i].unsqueeze(0), class_id)
                pre_cam[i] = cam.reshape(-1,).to(device)
                complexity += cam.sum()/(h*w)
                cam = to_pil_image(cam, mode='F')
                cam = cam.resize((Height,Width), resample=Image.Resampling.BICUBIC)
                cam = pil_to_tensor(cam)
                explanation[i] = torch.mul(images[i], cam.repeat(3,1,1).to(device))
            o_predictions = eval_model(explanation)        
            o_predictions = softmax(o_predictions)
            for i in range(batch_size):
                class_id = labels[i].item()
                oc[i] = o_predictions[i][class_id]
                cam, _ = recipro_cam(explanation[i].unsqueeze(0), class_id)
                cam = cam.reshape(-1,).to(device)
                coherency += 0.5 * (1.0 + pearson(cam, pre_cam[i]))
            drop = torch.nn.functional.relu(yc - oc) / yc
            inc = torch.count_nonzero(torch.nn.functional.relu(oc - yc))
            avg_drop += drop.sum()
            avg_inc += inc

            N += batch_size

            #if batch_idx*batch_size >= 2000:
            #    break
            if batch_idx % MOD == MOD-1:
                print('Batch ID: ', batch_idx + 1)

        avg_drop = avg_drop.cpu().item() / N
        avg_inc = avg_inc.cpu().item() / N
        coherency = coherency.cpu().item() / N
        complexity = complexity.cpu().item() / N
        adcc = 3.0/(1.0/coherency + 1.0/(1-complexity) + 1.0/(1-avg_drop))

        avg_drop *= 100.0
        avg_inc *= 100.0
        coherency *= 100.0
        complexity *= 100.0
        adcc *= 100.0

    return avg_drop, avg_inc, coherency, complexity, adcc


def all_metrics(type='white'):

    adcc = 0.0
    coherency = 0.0
    complexity = 0.0
    avg_drop = 0.0
    avg_inc = 0.0
    dauc = 0.0
    iauc = 0.0
    N = 0

    pearson = PearsonCorrCoef().to(device)

    explanation = torch.zeros(batch_size, 3, Height, Width).to(device)
    yc = torch.zeros(batch_size)
    oc = torch.zeros(batch_size)

    eval_model.eval()
    if type == 'white':
        recipro_cam = ReciproCam(feature_net, head_net, device=device)
        feature_net.eval()
        head_net.eval()
    elif type == 'hook':
        recipro_cam = ReciproCamHook(model, device=device)
    else:
        recipro_cam = ReciproCamBB(model, device=device)

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(val_loader, 0):
            images, labels = images.to(device), labels.to(device)
            if type == 'white':
                features = feature_net(images)
                predictions = head_net(features)
                predictions = softmax(predictions)
                bs, ch, h, w = features.shape
            elif type == 'hook':
                h = int(Height/32)
                w = int(Width/32)
                predictions = model(images)
                predictions = softmax(predictions)
            else:
                h = int(Height/recipro_cam.patch_h)
                w = int(Width/recipro_cam.patch_w)
                predictions = model(images)
                predictions = softmax(predictions)

            pre_cam = torch.zeros(batch_size, h*w).to(device)
            dx = int(Width/w)
            dy = int(Height/h)
            ck = torch.zeros(h*w+1).to(device)

            for i in range(batch_size):
                class_id = labels[i].item()
                yc[i] = predictions[i][class_id]
                cam, _ = recipro_cam(images[i].unsqueeze(0), class_id)
 
                pre_cam[i] = cam.reshape(-1,).to(device)

                auc_cam = pre_cam[i]
                _, s_index = auc_cam.sort(dim=0, descending=True)
                del_mask = torch.ones(3, Height, Width).to(device)
                inc_mask = torch.zeros(3, Height, Width).to(device)
                base_color = torch.mean(images[i], dim=(1, 2)).unsqueeze(0)
                base_color = base_color.reshape(1,3,1,1)
                auc_images = base_color.repeat(2*h*w,1,Height, Width).to(device)
                #auc_images[:h*w] = images[i].repeat(h*w, 1, 1, 1)
                for j in range(h*w):
                    s_idx = int(s_index[j].cpu().item())
                    ci = s_idx//w
                    cj = s_idx - ci*w
                    xs = int(cj*dx)
                    xe = int(min((cj+1)*dx, Width-1))
                    ys = int(ci*dy)
                    ye = int(min((ci+1)*dy, Height-1))
                    del_mask[:,ys:ye+1,xs:xe+1] = 0.0
                    inc_mask[:,ys:ye+1,xs:xe+1] = 1.0
                    auc_images[j] = images[i]*del_mask
                    auc_images[h*w + j] = auc_images[h*w +j]*del_mask + images[i]*inc_mask
                    #auc_images[h*w + j] = images[i]*inc_mask
                    #auc_images[j] = auc_images[j]*inc_mask + images[i]*del_mask
                o_predictions = eval_model(auc_images)        
                o_predictions = softmax(o_predictions)
                ck[0] = predictions[i][class_id]
                ck[1:] = o_predictions[:h*w,class_id]
                #ck /= ck.max()
                x = np.arange(0, len(ck))
                y = ck.detach().cpu().numpy()
                dauc += auc(x, y) / len(ck)
                ck[h*w] = predictions[i][class_id]
                ck[:h*w] = o_predictions[h*w:,class_id]
                #ck /= ck.max()
                y = ck.detach().cpu().numpy()
                iauc += auc(x, y) / len(ck)

                complexity += cam.sum()/(h*w)
                cam = to_pil_image(cam, mode='F')
                cam = cam.resize((Height,Width), resample=Image.Resampling.BICUBIC)
                cam = pil_to_tensor(cam)
                explanation[i] = torch.mul(images[i], cam.repeat(3,1,1).to(device))
            o_predictions = eval_model(explanation)        
            o_predictions = softmax(o_predictions)
            for i in range(batch_size):
                class_id = labels[i].item()
                oc[i] = o_predictions[i][class_id]
                cam, _ = recipro_cam(explanation[i].unsqueeze(0), class_id)
                cam = cam.reshape(-1,).to(device)
                coherency += 0.5 * (1.0 + pearson(cam, pre_cam[i]))
            drop = torch.nn.functional.relu(yc - oc) / yc
            inc = torch.count_nonzero(torch.nn.functional.relu(oc - yc))
            avg_drop += drop.sum()
            avg_inc += inc

            N += batch_size

            #if batch_idx*batch_size >= 2000:
            #    break
            if batch_idx % MOD == MOD-1:
                print('Batch ID: ', batch_idx + 1)

        avg_drop = avg_drop.cpu().item() / N
        avg_inc = avg_inc.cpu().item() / N
        coherency = coherency.cpu().item() / N
        complexity = complexity.cpu().item() / N
        adcc = 3.0/(1.0/coherency + 1.0/(1-complexity) + 1.0/(1-avg_drop))
        dauc /= N * 0.01
        iauc /= N * 0.01

        avg_drop *= 100.0
        avg_inc *= 100.0
        coherency *= 100.0
        complexity *= 100.0
        adcc *= 100.0

    return avg_drop, avg_inc, dauc, iauc, coherency, complexity, adcc


def performance_test():
    
    N = 4

    feature_net.eval()
    head_net.eval()
    begin_time = time.time()
    recipro_cam = ReciproCam(feature_net, head_net, device=device)
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(val_loader, 0):
            images, labels = images.to(device), labels.to(device)
            for i in range(batch_size):
                class_id = labels[i].item()
                cam, _ = recipro_cam(images[i].unsqueeze(0), class_id)
            if batch_idx >= N:
                break
    last_time = time.time()
    recipro_cam_time = (last_time - begin_time)/(batch_size*N)

    begin_time = time.time()
    recipro_cam_bb = ReciproCamBB(model, device=device)
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(val_loader, 0):
            images, labels = images.to(device), labels.to(device)
            for i in range(batch_size):
                class_id = labels[i].item()
                cam, _ = recipro_cam_bb(images[i].unsqueeze(0), class_id)
            if batch_idx >= N:
                break
    last_time = time.time()
    recipro_cam_bb_time = (last_time - begin_time)/(batch_size*N)
    
    model.eval()
    begin_time = time.time()
    cam_extractor = GradCAM(model)
    for batch_idx, (images, labels) in enumerate(val_loader, 0):
        images, labels = images.to(device), labels.to(device)
        for i in range(batch_size):
            class_id = labels[i].item()
            out = model(images[i].unsqueeze(0))
            grad_cam_out = cam_extractor(class_id, out)
        if batch_idx >= N:
            break
    last_time = time.time()
    grad_cam_time = (last_time - begin_time)/(batch_size*N)

    model_g_pp.eval()
    begin_time = time.time()
    cam_extractor = GradCAMpp(model_g_pp)
    for batch_idx, (images, labels) in enumerate(val_loader, 0):
        images, labels = images.to(device), labels.to(device)
        for i in range(batch_size):
            class_id = labels[i].item()
            out = model_g_pp(images[i].unsqueeze(0))
            grad_cam_pp_out = cam_extractor(class_id, out)
        if batch_idx >= N:
            break
    last_time = time.time()
    grad_cam_pp_time = (last_time - begin_time)/(batch_size*N)

    model_sg_pp.eval()
    begin_time = time.time()
    cam_extractor = SmoothGradCAMpp(model_sg_pp)
    for batch_idx, (images, labels) in enumerate(val_loader, 0):
        images, labels = images.to(device), labels.to(device)
        for i in range(batch_size):
            class_id = labels[i].item()
            out = model_sg_pp(images[i].unsqueeze(0))
            sgrad_cam_pp_out = cam_extractor(class_id, out)
        if batch_idx >= N:
            break
    last_time = time.time()
    smooth_grad_cam_pp_time = (last_time - begin_time)/(batch_size*N)

    model_score.eval()
    begin_time = time.time()
    cam_extractor = ScoreCAM(model_score)
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(val_loader, 0):
            images, labels = images.to(device), labels.to(device)
            for i in range(batch_size):
                class_id = labels[i].item()
                out = model_score(images[i].unsqueeze(0))
                score_cam_out = cam_extractor(class_id, out)
            if batch_idx >= N:
                break
    last_time = time.time()
    score_cam_time = (last_time - begin_time)/(batch_size*N)
    
    print('Recipro-CAM      : ', recipro_cam_time)
    print('Recipro-CAM-BB   : ', recipro_cam_bb_time)

    print('Grad-CAM         : ', grad_cam_time)
    print('Grad-CAM++       : ', grad_cam_pp_time)
    print('Smooth Grad-CAM++: ', smooth_grad_cam_pp_time)
    print('Score-CAM        : ', score_cam_time)
    
    
if __name__ == '__main__':

    #avg_drop, avg_inc = average_drop_increase('white')
    #print('AVG_Drop = ', avg_drop.cpu().item(), ' : AVG_Inc = ', avg_inc.cpu().item())

    '''
    dauc, iauc = dauc_iauc('white')    
    print('======= Model: ResNxt50 ======')
    print('Delition  = ', dauc)
    print('Insertion = ', iauc)
    
    '''
    avg_drop, avg_inc, coherency, complexity, adcc = ADCC('white')    
    print('======= Model: ResNet101 ======')
    print('Agv Drop   = ', avg_drop)
    print('Avg Inc    = ', avg_inc)
    print('Coherency  = ', coherency)
    print('Complexity = ', complexity)
    print('ADCC       = ', adcc)
    '''
    
    
    avg_drop, avg_inc, delition, insertion, coherency, complexity, adcc = all_metrics('white')    
    print('======= Gaussian Model: ResNet-50 ======')
    print('Agv Drop   = ', avg_drop)
    print('Avg Inc    = ', avg_inc)
    print('Delition   = ', delition)
    print('Insertion  = ', insertion)
    print('Coherency  = ', coherency)
    print('Complexity = ', complexity)
    print('ADCC       = ', adcc)
    '''
    #performance_test()
    
    
