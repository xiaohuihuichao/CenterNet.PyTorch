import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset


def bright_adjust(img, alpha=1.2, beta=127):
    img_mean = img.mean()
    img_adjust = (img - img_mean) * alpha + beta
    img_adjust = img_adjust.clip(0, 255)
    return np.asarray(img_adjust, np.uint8)


# img_path:xmin,ymin,xmax,ymax,label_idx;xmin,ymin,xmax,ymax,label_idx;...
def parse_data_file(data_file_path):
    with open(data_file_path, "r") as f:
        messages = f.readlines()
    messages = [i.strip().split(":") for i in messages]
    img_paths = [i[0] for i in messages]
    gts = [i[1].split(";") for i in messages]
    labels = []
    for gt in gts:
        label = []
        for box in gt:
            # xmin,ymin,xmax,ymax,label_id
            # label += [[float(i) for i in box.split(",")[0:5]]]
            label += [[float(i) for i in box.split(",")]]
        labels += [label]
    return img_paths, labels


class dataset(Dataset):
    def __init__(self, data_file_path, num_classes, wh=(640, 640), stride=4.):
        super().__init__()
        assert os.path.isfile(data_file_path)
        self.img_paths, self.labels = parse_data_file(data_file_path)
        self.num_classes = num_classes
        self.wh = wh
        self.stride = stride
        
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, index):
        img_path = self.img_paths[index]
        label = np.asarray(self.labels[index])
        
        img = cv2.imread(img_path)
        img = cv2.resize(img, dsize=self.wh)
        h, w = img.shape[0:2]
        img = bright_adjust(img)
        
        img = img.transpose(2, 0, 1) / 255.
        img = (img - 0.5) / 0.5
        img = torch.Tensor(img)
        
        hm_gt = torch.zeros(h//self.stride, w//self.stride, self.num_classes)
        txty = torch.zeros(h//self.stride, w//self.stride, 2)
        twth = torch.zeros(h//self.stride, w//self.stride, 2)
        xywh_masks = torch.zeros(h//self.stride, w//self.stride, dtype=torch.bool)
        for i in label:
            xmin, ymin, xmax, ymax, label_idx = i
            xmin *= (self.wh[0]/self.stride)
            xmax *= (self.wh[0]/self.stride)
            ymin *= (self.wh[1]/self.stride)
            ymax *= (self.wh[1]/self.stride)
            label_idx = int(label_idx)
            
            c_x_int, c_y_int, tx, ty, tw, th = get_txtytwth(xmin, ymin, xmax, ymax)
            r = gaussian_radius(th)
            hm_tensor = gaussian((h//self.stride, w//self.stride), c_x_int, c_y_int, r/3)
            hm_gt[:, :, label_idx] = torch.where(hm_gt[:, :, label_idx]>hm_tensor, hm_gt[:, :, label_idx], hm_tensor)
            txty[c_y_int, c_x_int, 0] = tx
            txty[c_y_int, c_x_int, 1] = ty
            twth[c_y_int, c_x_int, 0] = tw
            twth[c_y_int, c_x_int, 1] = th
            xywh_masks[c_y_int, c_x_int] = True
        return img, hm_gt, txty, twth, xywh_masks
    
    
def get_txtytwth(xmin, ymin, xmax, ymax):
    c_x = (xmin + xmax) / 2
    c_y = (ymin + ymax) / 2
    c_x_int = int(c_x)
    c_y_int = int(c_y)
    tx = c_x - c_x_int
    ty = c_y - c_y_int
    
    tw = np.log(xmax - xmin)
    th = np.log(ymax - ymin)
    return c_x_int, c_y_int, tx, ty, tw, th


def gaussian_radius(box_h, min_overlap=0.7):
    a1 = 1
    b1 = (box_h + box_h)
    c1 = box_h * box_h * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2 #(2*a1)

    a2 = 4
    b2 = 2 * (box_h + box_h)
    c2 = (1 - min_overlap) * box_h * box_h
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2 #(2*a2)

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (box_h + box_h)
    c3 = (min_overlap - 1) * box_h * box_h
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2 #(2*a3)
    return min(r1, r2, r3)

def gaussian_fn(x, y, c_x_int, c_y_int, sigma_h, sigma_w):
    return np.exp(- (x - c_x_int)**2 / (2*sigma_w**2) - (y - c_y_int)**2 / (2*sigma_h**2))

def gaussian(tensor_shape, c_x_int, c_y_int, sigma_h, sigma_w=-1):
    if sigma_w < 0:
        sigma_w = sigma_h
    h, w = tensor_shape
    tensor = torch.zeros(size=(tensor_shape))
    for x in range(c_x_int-3*int(sigma_w), c_x_int+3*int(sigma_w)+1):
        for y in range(c_y_int-3*int(sigma_h), c_y_int+3*int(sigma_h)+1):
            if 0<=x<w and 0<=y<h:
                v = gaussian_fn(x, y, c_x_int, c_y_int, sigma_h, sigma_w)
                tensor[y, x] = v
    return tensor


if __name__ == "__main__":
    h, w = 101, 101
    t = gaussian((h, w), 51, 51, 10, 25)
    # t = gaussian((h, w), 51, 51, 25)
    t = np.asarray(t*255, dtype=np.uint8)
    
    cv2.imwrite("a.jpg", t)
    # cv2.namedWindow("img", cv2.WINDOW_NORMAL)
    # cv2.imshow("img", t)
    # cv2.waitKey(0)
