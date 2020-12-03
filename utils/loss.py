import torch
import torch.nn as nn


class headmap_loss:
    def __init__(self, alpha=2, beta=4):
        self.alpha = alpha
        self.beta = beta
    
    def __call__(self, hm_preds, hm_gts):
        # pos_mask = hm_gts == 1
        # neg_mask = ~pos_mask
        
        # pos_loss = -torch.log(hm_preds+1e-14) * torch.pow(1-hm_preds, self.alpha) * pos_mask
        # neg_loss = -torch.log(1-hm_preds+1e-14) * torch.pow(hm_preds, self.alpha) * torch.pow(1-hm_gts, self.beta) * neg_mask
        # pos_mask = hm_gts == 1
        pos_mask = hm_gts > 1-1e-6
        neg_mask = ~pos_mask
        
        hm_preds_pos = hm_preds[pos_mask]
        hm_preds_neg = hm_preds[neg_mask]
        hm_gts_neg = hm_gts[neg_mask]
        
        pos_loss = -torch.log(hm_preds_pos+1e-14) * torch.pow(1-hm_preds_pos, self.alpha)# * pos_mask
        neg_loss = -torch.log(1-hm_preds_neg+1e-14) * torch.pow(hm_preds_neg, self.alpha) * torch.pow(1-hm_gts_neg, self.beta)# * neg_mask
        return pos_loss.sum() + neg_loss.sum()
    
    
class loss:
    def __init__(self, xy_weight=1, wh_weight=0.1, hm_weight=1):
        self.hm_weight = hm_weight
        self.xy_weight = xy_weight
        self.wh_weight = wh_weight
        
        self.headmap_loss = headmap_loss()
        self.txty_loss = nn.SmoothL1Loss(reduction="none")
        self.wh_loss = nn.SmoothL1Loss(reduction="none")
        
    def __call__(self, hm_preds, xy_preds, wh_preds, hm_gts, txtys, twths, xywh_masks):
        b, _, h, w = hm_preds.shape
        hm_preds = hm_preds.permute(0, 2, 3, 1).reshape(b*h*w, -1)
        xy_preds = xy_preds.permute(0, 2, 3, 1).reshape(b*h*w, -1)
        wh_preds = wh_preds.permute(0, 2, 3, 1).reshape(b*h*w, -1)
        
        hm_gts = hm_gts.reshape(b*h*w, -1)
        txtys = txtys.reshape(b*h*w, -1)
        twths = twths.reshape(b*h*w, -1)
        if not xywh_masks is None:
            xywh_masks = xywh_masks.reshape(b*h*w)
            xy_preds = xy_preds[xywh_masks]
            txtys = txtys[xywh_masks]
            wh_preds = wh_preds[xywh_masks]
            twths = twths[xywh_masks]
        txty_loss = self.txty_loss(xy_preds, txtys).sum() / b
        wh_loss = self.wh_loss(wh_preds, twths).sum() / b
        
        hm_loss = self.headmap_loss(hm_preds, hm_gts) / b
        total_loss = self.hm_weight * hm_loss + self.xy_weight * txty_loss + self.wh_weight * wh_loss
        return total_loss, hm_loss, txty_loss, wh_loss
    