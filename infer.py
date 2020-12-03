import cv2
import torch
import numpy as np
from torch.nn import functional as F

from net import Net


cuda = True and torch.cuda.is_available()
model_path = "model/batch.pth"
threshold = 0.5

with torch.no_grad():
    net = Net(512, 1)
    net.load_state_dict(torch.load(model_path))
    if cuda:
        net = net.cuda()
        
    while True:
        grid = torch.zeros((160, 160, 2))
        for h in range(160):
            for w in range(160):
                grid[h, w] = torch.Tensor([w, h])
                
        msg = input("msg:")
        if msg == "q":
            break
        elif msg == "load":
            net.load_state_dict(torch.load(model_path))
            continue
            
        img_path = msg.split(":")[0]
        
        img_cv = cv2.imread(img_path)
        img_cv = cv2.resize(img_cv, dsize=(640, 640))
        
        img = img_cv.transpose(2, 0, 1) / 255.
        img = (img - 0.5) / 0.5
        img = torch.Tensor(img).unsqueeze(0)
        if cuda:
            img = img.cuda()
            grid = grid.cuda()
            
        hm_preds, xy_preds, wh_preds = net(img)
        h_max = F.max_pool2d(hm_preds, kernel_size=5, padding=2, stride=1)
        
        hm_preds = (hm_preds > threshold) & (hm_preds==h_max)
        hm_preds.squeeze_()
        
        wh_preds = wh_preds.squeeze().permute(1, 2, 0)
        wh_preds = wh_preds[hm_preds]
        wh_preds = torch.exp(wh_preds)
        
        xy_preds = xy_preds.squeeze().permute(1, 2, 0)
        xy_preds = xy_preds[hm_preds]
        
        grid = grid[hm_preds] * 4
        grid_min = grid + (xy_preds-wh_preds/2) * 4
        grid_max = grid + (xy_preds+wh_preds/2) * 4
        
        grid_min = grid_min.cpu().numpy()
        grid_max = grid_max.cpu().numpy()
        for i in range(len(grid_max)):
            img_cv = cv2.circle(img_cv, (grid[i, 0], grid[i, 1]), 2, (0, 255, 0), 1)
            img_cv = cv2.rectangle(img_cv,
                                (int(grid_min[i, 0]), int(grid_min[i, 1])),
                                (int(grid_max[i, 0]), int(grid_max[i, 1])), (0, 0, 255), 2)
        cv2.imwrite("a.jpg", img_cv)
        