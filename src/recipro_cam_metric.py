import copy
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models

from torchvision.transforms.functional import to_pil_image, pil_to_tensor, normalize
from PIL import Image
from sklearn.metrics import auc
from torchmetrics import PearsonCorrCoef

from src.recipro_cam import ReciproCam

MOD = 50

def average_drop_increase(model, data_loader, Height, Width, batch_size, device='cpu'):

    avg_drop = 0.0
    avg_inc = 0.0

    explanation = torch.zeros(batch_size, 3, Height, Width)
    yc = torch.zeros(batch_size)
    oc = torch.zeros(batch_size)

    N = 0

    eval_model = copy.deepcopy(model)
    eval_model.eval()
    softmax = torch.nn.Softmax(dim=1)
    recipro_cam = ReciproCam(model, device=device)

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(data_loader, 0):
            images, labels = images.to(device), labels.to(device)
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
            drop = torch.nn.functional.relu(yc - oc) / yc
            inc = torch.count_nonzero(torch.nn.functional.relu(oc - yc))
            avg_drop += drop.sum()
            avg_inc += inc

            N += batch_size

            if batch_idx % MOD == MOD-1:
                print('Batch ID: ', batch_idx + 1)

        avg_drop /= N * 0.01
        avg_inc /= N * 0.01

    return avg_drop, avg_inc


def dauc_iauc(model, data_loader, Height, Width, batch_size, device='cpu'):

    DAUC_score = 0.0
    IAUC_score = 0.0

    N = 0
    eval_model = copy.deepcopy(model)
    eval_model.eval()
    softmax = torch.nn.Softmax(dim=1)
    recipro_cam = ReciproCam(model, device=device)

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(data_loader, 0):
            images, labels = images.to(device), labels.to(device)
            h = int(Height/32)
            w = int(Width/32)
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
                    auc_images[h*w + j] = auc_images[h*w +j]*del_mask + images[i]*inc_mask
                o_predictions = eval_model(auc_images)        
                o_predictions = softmax(o_predictions)
                ck[0] = predictions[i][class_id]
                ck[1:] = o_predictions[:h*w,class_id]
                x = np.arange(0, len(ck))
                y = ck.detach().cpu().numpy()
                DAUC_score += auc(x, y) / len(ck)
                ck[h*w] = predictions[i][class_id]
                ck[:h*w] = o_predictions[h*w:,class_id]
                y = ck.detach().cpu().numpy()
                IAUC_score += auc(x, y) / len(ck)
            N += batch_size

            if batch_idx % MOD == MOD-1:
                print('Batch ID: ', batch_idx + 1)

        DAUC_score = DAUC_score / (N * 0.01)
        IAUC_score = IAUC_score / (N * 0.01)

    return DAUC_score, IAUC_score


def ADCC(model, data_loader, Height, Width, batch_size, device='cpu'):

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
    eval_model = copy.deepcopy(model)
    eval_model.eval()
    softmax = torch.nn.Softmax(dim=1)
    recipro_cam = ReciproCam(model, device=device)

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(data_loader, 0):
            images, labels = images.to(device), labels.to(device)

            h = int(Height/32)
            w = int(Width/32)
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

