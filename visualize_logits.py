from YOLOSegPlusPlus import YOLOSegPlusPlus
from custom_yolo_predictor.custom_detseg_predictor import CustomSegmentationPredictor
from custom_yolo_trainer.custom_trainer import CustomSegmentationTrainer
from dataset import CustomDataset
import torch

import os
import time
from typing import Tuple, List
from itertools import cycle

import torch
from torch.amp import GradScaler
from torch import optim
from torch.utils.data import DataLoader

from torchinfo import summary
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from monai.losses import DiceLoss
from monai.metrics import DiceMetric

import cv2
from torchvision import transforms
import matplotlib.pyplot as plt
from torchvision.transforms import GaussianBlur
import torchvision
import torch

import torch.nn.functional as F


def spatial_confidence(logits, k_frac=0.05):
    """
    Calculates confidence spatially (from the logits)
    """
    probs = torch.sigmoid(logits).flatten()
    k = max(1, int(k_frac * probs.numel()))
    topk_mean = probs.topk(k).values.mean()
    return topk_mean


def argmax_conf(detect_branch):
    """
    Calculates confidence by obtaining max
    """

    class_probs = detect_branch[0, 4, :]        # [525]
    confidence_scores = torch.sigmoid(class_probs)
    idx = confidence_scores.argmax()
    # return detect_branch[0, :4, idx]
    return confidence_scores[idx]


def test_predictor():
    """
    Visualize Logits using the YOLO Predictor custom class
    """
    # Create trainer and predictor instances
    p_args = dict(model="pretrained_detect_yolo/best_yolo12n_det/weights/best.pt",
                  data=f"data/data.yaml",
                  verbose=True,
                  imgsz=160,
                  save=False)

    # Create predictor and Load checkpoint
    YOLO_predictor = CustomSegmentationPredictor(overrides=p_args)
    YOLO_predictor.setup_model(p_args["model"])

    x = cv2.imread("/home/jun/Desktop/inspirit/YOLOSeg++/archive/BraTS-SSA-00002-00030-t1c_image.png",
                   cv2.IMREAD_UNCHANGED)

    x = transforms.ToTensor()(x)
    x = transforms.Resize(size=(160, 160))(x)
    x = x.to("cuda")
    x = x.unsqueeze(0)

    model = YOLO_predictor.model.model
    model.to("cuda")

    x = model(x)

    detect_branch, cls_branch = x
    a, b, c = cls_branch

    """SPLIT CHANNELS
    B, G, R, A = cv2.split(x)
    channels = [B, G, R, A]
    names = ['channel_B_data', 'channel_G_data',
        'channel_R_data', 'channel_A_data']

    for channel, name in zip(channels, names):
        filename = f'{name}.png'
        success = cv2.imwrite(filename, channel)
    """

    heatmap = a[:, -1:]
    heatmap = torch.sigmoid(heatmap)
    confidence = argmax_conf(detect_branch)
    space_confidence = spatial_confidence(heatmap)

    print(confidence, space_confidence)
    plt.imshow(heatmap.squeeze(0).squeeze(0).cpu().numpy())
    plt.show()


def test_trainer1():
    """
    Visualize logits using the YOLO Trainer custom class
    """

    # Create trainer and predictor instances
    p_args = dict(model="pretrained_detect_yolo/best_yolo12n_det/weights/best.pt",
                  data=f"data/data.yaml",
                  verbose=True,
                  imgsz=160,
                  save=False)

    # Create trainer and Load checkpoint
    YOLO_trainer = CustomSegmentationTrainer(overrides=p_args)
    YOLO_trainer.setup_model()

    x = cv2.imread("archive/BraTS-SSA-00041-0007-t1c_image.png",
                   cv2.IMREAD_UNCHANGED)
    # x = cv2.imread("/home/jun/Desktop/inspirit/YOLOSeg++/archive/BraTS-SSA-00002-00030-t1c_image.png",
    # cv2.IMREAD_UNCHANGED)

    x = transforms.ToTensor()(x)
    x = transforms.Resize(size=(160, 160))(x)
    x = x.to("cuda")
    x = x.unsqueeze(0)

    # x = torch.zeros(1, 4, 160, 160).to('cuda')

    model = YOLO_trainer.model
    model.to("cuda")
    # model.eval()

    x = model.predict(x)
    # detect_branch, cls_branch = x
    twenty, ten, five = x
    logits = twenty[:, -1:]
    logits = torch.sigmoid(logits)

    plt.imshow(logits.squeeze(0).squeeze(0).cpu().detach().numpy())
    # plt.imshow(logits.squeeze(0).squeeze(0).cpu().detach().numpy())
    plt.show()


def test_trainer2():
    """
    Visualize logits using the YOLO Trainer custom class
    """

    # Create trainer and predictor instances
    p_args = dict(model="pretrained_detect_yolo/best_yolo12n_det/weights/best.pt",
                  data=f"data/data.yaml",
                  verbose=True,
                  imgsz=160,
                  save=False)

    # Create trainer and Load checkpoint
    YOLO_trainer = CustomSegmentationTrainer(overrides=p_args)
    YOLO_trainer.setup_model()

    x = cv2.imread("archive/BraTS-SSA-00041-0007-t1c_image.png",
                   cv2.IMREAD_UNCHANGED)
    # x = cv2.imread("/home/jun/Desktop/inspirit/YOLOSeg++/archive/BraTS-SSA-00002-00030-t1c_image.png",
    # cv2.IMREAD_UNCHANGED)

    x = transforms.ToTensor()(x)
    x = transforms.Resize(size=(160, 160))(x)
    x = x.to("cuda")
    x = x.unsqueeze(0)

    # x = torch.zeros(1, 4, 160, 160).to('cuda')

    model = YOLO_trainer.model
    model.to("cuda")
    model.eval()

    x = model.predict(x)
    detect_branch, cls_branch = x
    twenty, ten, five = cls_branch
    logits = twenty[:, -1:]
    logits = torch.sigmoid(logits)

    plt.imshow(logits.squeeze(0).squeeze(
        0).cpu().detach().numpy())
    plt.show()


if __name__ == "__main__":
    test_trainer2()
    test_trainer1()
    # test_predictor()
