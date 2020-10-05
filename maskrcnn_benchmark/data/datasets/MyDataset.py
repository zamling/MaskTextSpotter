import numpy as np
import torch
from torch.utils import data
import scipy.io as sio
import os
from tqdm import tqdm
import cv2
import json
from PIL import Image

from maskrcnn_benchmark.structures.bounding_box import BoxList


class MyDataset_train(object):
    def __init__(self, json_dir, transforms=None):
        self.json_dir = json_dir
        with open(json_dir,'r') as f:
            data = json.load(f)
        self.data = data
        self.transforms = transforms

    def __getitem__(self, item):
        img = Image.open(self.data[item]['image']).convert("RGB")

        # dummy target
        boxes = self.data[item]['box']
        boxes = np.array(boxes)
        target = BoxList(boxes, img.size, mode="xyxy")
        classes = torch.ones(len(boxes))
        target.add_field("labels",classes)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target, self.data[item]['image']

    def __len__(self):
        return len(self.data)

class MyDataset_test(object):
    def __init__(self, json_dir, transforms=None):
        self.json_dir = json_dir
        with open(json_dir,'r') as f:
            data = json.load(f)
        self.data = data
        self.transforms = transforms

    def __getitem__(self, item):
        img = Image.open(self.data[item]['image']).convert("RGB")

        # dummy target
        boxes = self.data[item]['box']
        boxes = np.array(boxes)
        target = BoxList(boxes, img.size, mode="xyxy")
        classes = torch.ones(len(boxes))
        target.add_field("labels",classes)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target, self.data[item]['image']

    def __len__(self):
        return 3000




def mat_to_json():
    matfile = '/data1/zem/Resnet.CRNN/data/SynthText/gt.mat'
    img_root = '/data1/zem/Resnet.CRNN/data/SynthText'
    data = sio.loadmat(matfile)

    num_img = len(data['wordBB'][0])
    # words = []
    # for val in data['txt'][0][0]:
    #     v = [x.split('\n') for x in val.strip().split(' ')]
    #     v = [[vv_ for vv_ in vv if len(vv_) > 0] for vv in v]
    #     words.extend(sum(v, []))
    json_data = []
    tbar = tqdm(range(num_img))
    for i in tbar:
        img_info = {}
        boxes = []
        wordsBB = data['wordBB'][0][i]
        if len(wordsBB.shape) == 2:
            wordsBB = np.expand_dims(wordsBB,axis=2)
        assert len(wordsBB.shape)==3,'the shape is {}'.format(wordsBB.shape)
        wordsBB = np.around(np.array(wordsBB), decimals=2).transpose(2, 1, 0)
        # print(words[0])
        for j in range(wordsBB.shape[0]):
            x1 = wordsBB[j][0][0]
            y1 = wordsBB[j][0][1]
            x2 = wordsBB[j][1][0]
            y2 = wordsBB[j][1][1]
            x3 = wordsBB[j][2][0]
            y3 = wordsBB[j][2][1]
            x4 = wordsBB[j][3][0]
            y4 = wordsBB[j][3][1]
            x_min = min([x1, x2, x3, x4])
            x_max = max([x1, x2, x3, x4])
            y_min = min([y1, y2, y3, y4])
            y_max = max([y1, y2, y3, y4])
            box = [round(float(x_min),2), round(float(y_min),2), round(float(x_max),2), round(float(y_max),2)]
            boxes.append(box)
        img_path = data['imnames'][0][i][0]
        img_abs_path = os.path.join(img_root, img_path)
        img_info['image'] = img_abs_path
        img_info['box'] = boxes
        json_data.append(dict(img_info))
    with open('/data1/zem/Resnet.CRNN/data/SynthText/my_gt.json','w') as f:
        json.dump(json_data,f)
    print('done')

if __name__ == "__main__":
    with open('/data1/zem/Resnet.CRNN/data/SynthText/my_gt.json','r') as f:
        data = json.load(f)
    print('finish')
