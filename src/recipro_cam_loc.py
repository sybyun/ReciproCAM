from typing import OrderedDict
import torch
import numpy as np
from PIL import Image
from torchvision.transforms.functional import to_pil_image, pil_to_tensor, normalize, resize


class ReciproCamLoc:
    def __init__(self, model, mean, std, step_size, num_step, device, target_layer_name=None):
        self.model = model
        self.model.eval()
        self.mean = mean
        self.std = std
        if step_size != 0:
            assert step_size % 32 == 0, 'Step size should be multiple of 32!'
        self.step_size = step_size
        self.num_step = num_step
        self.target_layer_name = target_layer_name
        self.device = device
        self.feature = None
        filter = [[1/16.0, 1/8.0, 1/16.0],
                    [1/8.0, 1/2.0, 1/8.0],
                    [1/16.0, 1/8.0, 1/16.0]]
        self.gaussian = torch.tensor(filter).to(device)
        self.softmax = torch.nn.Softmax(dim=1)
        self.target_layers = []
        self.conv_depth = 0
        if self.target_layer_name is not None:
            children = dict(self.model.named_children())
            target = children[self.target_layer_name]
            self.find_target_layer(target)
        else:
            self.find_target_layer(self.model)
        self.target_layers[-1].register_forward_hook(self.cam_hook())


    def find_target_layer(self, m, depth=0):
        children = dict(m.named_children())
        if children == {}:
            if isinstance(m, torch.nn.Conv2d):
                self.target_layers.clear()
                self.target_layers.append(m)
                self.conv_depth = depth
            elif self.conv_depth == depth and len(self.target_layers) > 0 and isinstance(m, torch.nn.BatchNorm2d):
                self.target_layers.append(m)
            elif self.conv_depth == depth and len(self.target_layers) > 0 and isinstance(m, torch.nn.ReLU):
                self.target_layers.append(m)
        else:
            for name, child in children.items():
                self.find_target_layer(child, depth+1)


    def cam_hook(self):
        def fn(_, input, output):
            self.feature = output[0].unsqueeze(0)
            bs, nc, h, w = self.feature.shape
            new_features = self.mosaic_feature(self.feature, nc, h, w)
            new_features = torch.cat((self.feature, new_features), dim = 0)
            return new_features

        return fn


    def mosaic_feature(self, feature_map, nc, h, w):
        new_features = torch.zeros(h*w, nc, h, w).to(self.device)
        for b in range(h*w):
            for i in range(h):
                for j in range(w):
                    if b == i*w + j:
                        new_features[b,:,i,j] = feature_map[0,:,i,j]

        return new_features


    def get_class_activaton_map(self, mosaic_predic, index, h, w):
        cams = []
        scores = []
        if type(index) != int:
            for key, val in index.items():
                indexes = torch.as_tensor(val).to(self.device)
                score = torch.index_select(mosaic_predic[0,:], 0, indexes).max(dim=0).values
                cam = torch.index_select(mosaic_predic[1:,:], 1, indexes)
                cam = cam.max(dim=1).values                
                cam = cam.reshape((h, w))
                cam_min = cam.min()
                cam = (cam - cam_min) / (cam.max() - cam_min)
                cams.append(cam)
                scores.append(score)
        else:
            cam = mosaic_predic[1:,index]
            score = mosaic_predic[0:,index]
            cam = cam.reshape((h, w))
            cam_min = cam.min()
            cam = (cam - cam_min) / (cam.max() - cam_min)
            cams.append(cam)
            scores.append(score)

        return cams, scores


    def __call__(self, img, height, width, index=None):
        H = height
        W = width

        t_h = None
        t_w = None
        cams = None
        scores = None
        with torch.no_grad():
            input = normalize(resize(img, (H, W)) / 255., self.mean, self.std).to(self.device)
            output = self.model(input.unsqueeze(0))
            #prediction = self.softmax(output)
            prediction = torch.sigmoid(output)
            _, _, h, w = self.feature.shape
            if index is None:
                index = prediction[0].argmax().item()
            #print('Predicted objects: ', (torch.topk(prediction[0][80:], 10)[0]).cpu().detach().numpy())
            print('Predicted objects: ', torch.topk(output[0][80:], 10))
            cams_org, scores_org = self.get_class_activaton_map(prediction, index, h, w)

            for n in range(self.num_step, 1, -1):
                if self.step_size > 0:
                    nth_size = (H + (n-1)*self.step_size, W + (n-1)*self.step_size)
                else:
                    nth_size = (n * H, n * W)
                input = normalize(resize(img, nth_size) / 255., self.mean, self.std).to(self.device)
                #prediction = self.softmax(self.model(input.unsqueeze(0)))
                prediction = torch.sigmoid(self.model(input.unsqueeze(0)))
                _, _, h_n, w_n = self.feature.shape
                if t_h == None or t_w == None:
                    t_h = h_n
                    t_w = w_n

                nth_cams, nth_scores = self.get_class_activaton_map(prediction, index, h_n, w_n)
                for i in range(len(nth_cams)):
                    nth_cams[i] = pil_to_tensor(to_pil_image(nth_cams[i].detach(), mode='F').resize((t_h, t_w), Image.BILINEAR)).squeeze(0).to(self.device)
                    if cams != None:
                        cams[i] += nth_cams[i]
                        if scores[i] < nth_scores[i]:
                            scores[i] = nth_scores[i]
                if cams == None:
                    cams = nth_cams
                    scores = nth_scores

            for i in range(len(cams_org)):
                cams_org[i] = pil_to_tensor(to_pil_image(cams_org[i].detach(), mode='F').resize((t_h, t_w), Image.BILINEAR)).squeeze(0).to(self.device)
                cams[i] += cams_org[i]
                if scores[i] < scores_org[i]:
                    scores[i] = scores_org[i]
            
            for i in range(len(cams)):
                cam_min = cams[i].min()
                cams[i] = (cams[i] - cam_min) / (cams[i].max() - cam_min)
                scores[i] *= 100.0
                scores[i] = scores[i].item()

        return cams, index.keys(), scores

