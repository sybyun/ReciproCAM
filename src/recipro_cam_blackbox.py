from typing import OrderedDict
import torch
import numpy as np


class ReciproCamBB:
    def __init__(self, model, device, patch_h=32, patch_w=32, overlap=16):
        self.model = model
        self.model.eval()
        self.softmax = torch.nn.Softmax(dim=1)
        self.device = device
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.overlap = overlap
        self.h = 0
        self.w = 0


    def patch_images(self, input, nc, H, W):
        self.h = h = int(H / self.patch_h)
        self.w = w = int(W / self.patch_w)
        k_h = self.patch_h
        k_w = self.patch_w
        pf = int(self.overlap/2)
        pf2 = self.overlap
        new_inputs = torch.zeros(h*w, nc, H, W).to(self.device)
        for b in range(h*w):
            for i in range(h):
                ky_s = max(i*k_h - pf2, 0)
                ky_e = min((i+1)*k_h + pf2, H-1)
                for j in range(w):
                    if b == i*w + j:
                        kx_s = max(j*k_w - pf2, 0)
                        kx_e = min((j+1)*k_w + pf2, W-1)
                        new_inputs[b,:,ky_s:ky_e+1,kx_s:kx_e+1] = (1/16.0) * input[0,:,ky_s:ky_e+1,kx_s:kx_e+1]
            for i in range(h):
                ky_s = max(i*k_h - pf, 0)
                ky_e = min((i+1)*k_h + pf, H-1)
                for j in range(w):
                    if b == i*w + j:
                        kx_s = max(j*k_w - pf, 0)
                        kx_e = min((j+1)*k_w + pf, W-1)
                        new_inputs[b,:,ky_s:ky_e+1,kx_s:kx_e+1] = (1/8.0) * input[0,:,ky_s:ky_e+1,kx_s:kx_e+1]
            for i in range(h):
                ky_s = max(i*k_h, 0)
                ky_e = min((i+1)*k_h, H-1)
                for j in range(w):
                    if b == i*w + j:
                        kx_s = max(j*k_w, 0)
                        kx_e = min((j+1)*k_w, W-1)
                        new_inputs[b,:,ky_s:ky_e+1,kx_s:kx_e+1] = (1/4.0) * input[0,:,ky_s:ky_e+1,kx_s:kx_e+1]
        new_inputs = torch.cat((input, new_inputs), dim=0)

        return new_inputs


    def get_class_activaton_map(self, predictions, index, h, w):
        cam = (predictions[:,index]).reshape((h, w))
        cam_min = cam.min()
        cam = (cam - cam_min) / (cam.max() - cam_min)
    
        return cam


    def __call__(self, input, index=None):
        with torch.no_grad():

            bs, nc, H, W = input.shape
            new_inputs = self.patch_images(input, nc, H, W)
            predictions = self.model(new_inputs)
            predictions = self.softmax(predictions)

            if index == None:
                index = predictions[0].argmax().item()
                print('Probability: ', predictions[0].max().item())

            cam = self.get_class_activaton_map(predictions[1:, :], index, int(H/32), int(W/32))

        return cam, index

