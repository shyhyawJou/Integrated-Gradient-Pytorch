import torch

import cv2
import numpy as np



class Integrated_Gradient:
    def __init__(self, model, device, preprocess, steps=20):     
        self.model = model.to(device)
        self.device = device
        self.prep = preprocess
        self.steps = steps

    def get_heatmap(self, img, baseline=None):        
        # check whether the input shape is the same as the baseline shape
        x = self.prep(img)[None].to(self.device)
        if baseline is None:
            baseline = torch.zeros_like(x, requires_grad=False)
        self._check(x, baseline)
        
        # get predict label
        output = self.model(x)
        pred_label = output.max(1)[1]
        
        # compute integrated gradients
        x.requires_grad_(True)
        X, delta_X = self._get_X_and_delta(x, baseline, self.steps)
        ig_grad = torch.autograd.grad(self.model(X)[:, pred_label].mean(), X)[0]
        ig_grad = ig_grad.cpu().numpy()
        ig_grad = delta_X.cpu().numpy() * (ig_grad[:-1] + ig_grad[1:]) / 2.
        ig_grad = ig_grad.sum(axis=0, keepdims=True)
        
        # plot
        cam = np.clip(ig_grad, 0, None)
        cam = cam / cam.max() * 255.
        cam = cam.squeeze().transpose(1, 2, 0).astype(np.uint8)
        cam = cv2.resize(cam, img.size[:2])
        cam = cv2.applyColorMap(cam, cv2.COLORMAP_TURBO)[..., ::-1]      
        
        return self.model(x.detach()), cam
                                         
    def _get_X_and_delta(self, x, baseline, steps):
        alphas = torch.linspace(0, 1, steps + 1).view(-1, 1, 1, 1)
        delta = (x - baseline)
        x = baseline + alphas * delta
        return x, (delta / steps).detach()
        
    def _check(self, x, baseline):
        if x.shape != baseline.shape:
            raise ValueError(f'input shape should equal to baseline shape. '
                             f'Got input shape: {x.shape}, '
                             f'baseline shape: {baseline.shape}') 

