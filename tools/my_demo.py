import os
from cv2 import cv2
import torch
from torchvision import transforms as T

from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.utils.chars import getstr_grid, get_tight_rect

from PIL import Image
import numpy as np
import argparse
import json

class BoxDetection(object):
    def __init__(
            self,
            cfg,
            min_image_size = 224
    ):
        self.cfg = cfg.clone()
        self.model = build_detection_model(cfg)
        self.model.eval()
        self.device = torch.device(cfg.MODEL.DEVICE)

        self.min_image_size = min_image_size
        checkpointer = DetectronCheckpointer(cfg,self.model)
        _ = checkpointer.load(cfg.MODEL.WEIGHT)
        self.model.to(self.device)
        self.transforms = self.build_trainsform()
        self.cpu_device = torch.device("cpu")

    def build_trainsform(self):
        cfg = self.cfg
        # we are loading images with OpenCV, so we don't need to convert them
        # to BGR, they are already! So all we need to do is to normalize
        # by 255 if we want to convert to BGR255 format, or flip the channels
        # if we want it to be in RGB in [0-1] range.
        if cfg.INPUT.TO_BGR255:
            to_bgr_transform = T.Lambda(lambda x: x * 255)
        else:
            to_bgr_transform = T.Lambda(lambda x: x[[2, 1, 0]])

        normalize_transform = T.Normalize(
            mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD
        )

        transform = T.Compose(
            [
                # T.ToPILImage(),
                # T.Resize(self.min_image_size),
                T.ToTensor(),
                to_bgr_transform,
                normalize_transform,
            ]
        )
        return transform


    def detection_box(self,original_image):
        image = original_image
        image_list = to_image_list(image, self.cfg.DATALOADER.SIZE_DIVISIBILITY)
        image_list = image_list.to(self.device)
        with torch.no_grad():
            predictions = self.model(image_list)
        return predictions

    def visualization(self,original_image,output_dir):
        image = self.transforms(original_image)
        image_list = to_image_list(image, self.cfg.DATALOADER.SIZE_DIVISIBILITY)
        image_list = image_list.to(self.device)
        with torch.no_grad():
            predictions = self.model(image_list)
        for box in predictions[0].bbox.cpu().numpy().tolist():
            cv2.rectangle(original_image,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),(0,0,255),2)
        cv2.imwrite(output_dir,original_image)
        print('done')







def main(args):
    cfg.merge_from_file(args.config_file)

    detector = BoxDetection(cfg,min_image_size=800)
    image = cv2.imread(args.image_path)
    detector.visualization(image,args.output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='parameters for demo')
    parser.add_argument("--config-file",type=str,default='configs/finetune.yaml')
    parser.add_argument("--output_file", type=str, default='./outputs/output.jpg')
    parser.add_argument("--image_path", type=str, default='/data1/zem/Resnet.CRNN/data/SynthText/8/ballet_106_0.jpg')
    args = parser.parse_args()
    main(args)





