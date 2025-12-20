from YOLOSegPlusPlus import YOLOSegPlusPlus
from custom_yolo_predictor.custom_detseg_predictor import CustomSegmentationPredictor
from dataset import CustomDataset

import os
import time
from typing import Tuple, List, Union
from itertools import cycle
from nms import non_max_suppression

import torch
from torch.amp import GradScaler
from torch import optim
from torch.utils.data import DataLoader

from torchinfo import summary
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from monai.losses import DiceLoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric

import cv2
from torchvision import transforms  
import matplotlib.pyplot as plt
from torchvision.transforms import GaussianBlur  
import torchvision
import torch
from torch import nn

import torch.nn.functional as F

class Evaluator: 
    def __init__(self,
                model: YOLOSegPlusPlus,
                data_path: str, 
                model_path: str = None,
                device: str = "cuda",
                image_size: int = 160, 
                batch_size: int = 1, ):
        """
        
        """
    
        self.model = model
        self.device = device
        self.data_path = data_path
        self.model_path = model_path

        self.metric = DiceMetric(
            include_background = False, 
            reduction="mean_batch",     
            get_not_nans = False, 
            ignore_empty = False, 
            num_classes = 2,            # 2 stands for [0, 1], technically single class
            return_with_label = False
        )

        self.hd95 = HausdorffDistanceMetric(
            include_background = False, 
            percentile=95, 
            reduction = "none",
            get_not_nans = True, 
        )

        self.image_size = image_size
        self.batch_size = batch_size

    
    def spatial_confidence(self, logits: torch.tensor, k_frac: float = 0.20):
        """

        """
        probs = torch.sigmoid(logits).flatten()
        k = max(1, int(k_frac * probs.numel()))
        topk_mean = probs.topk(k).values.mean()
        return topk_mean
        
    def get_current_time(self) -> str: 
        """
        Get current time in YMD | HMS format
        Used for creating non-conflicting result dirs
        Returns
            (str) Time in Ymd | HMs format
        """
        current_time = time.localtime()
        return time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
            
    def create_dataloader(self, data_path: str, split: str) -> Tuple[DataLoader, DataLoader]: 
        """
        Create dataloader from CustomDataset
        Depends on SegmentationDataset

        Args:
            data_path (str): root directory of dataset
        Returns:
            (Tuple[Dataloader]): train_dataloader and val_dataloader
        """                            
        test_dataset = CustomDataset(
                            root_path = data_path, 
                            image_path = f"images/{split}", 
                            mask_path = f"masks/{split}", 
                            objectmap_path = f"objectmap/{split}",
                            image_size = self.image_size, 
                            objectmap_sizes = [20])

        test_dataloader = DataLoader(dataset=test_dataset,
                                    batch_size=self.batch_size,
                                    shuffle=False, 
                                    num_workers=8)
        return test_dataloader
    
    def evaluate(self) -> None: 
        """

        """
        # Add model to device
        self.model.to(self.device)  
        self.model.eval()

        # Creates the dataloader
        test_dataloader = self.create_dataloader(data_path=self.data_path, split="test")

        # Set PyTorch CPU and GPU seeds
        SEED = 42
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            # Sets seed for all available GPUs
            torch.cuda.manual_seed_all(SEED)

        self.metric.reset()
        total_TP = total_FP = total_FN = 0
        start_time = time.time()
        
        with torch.no_grad():
            for idx, img_mask_heatmap in enumerate(tqdm(test_dataloader)):
                img = img_mask_heatmap[0].float().to(self.device)
                mask = img_mask_heatmap[1].float().to(self.device)
                heatmap = img_mask_heatmap[2][0].float().to(self.device).squeeze(0)

                ### YOLO inference
                yolo_out = YOLO_predictor.model(img)
                detect_branch, cls_branch = yolo_out
                a, b, c = cls_branch

                ## Sigmoid
                logits = torch.sigmoid(a[:, -1:]) # <- Extract last item

                ## Raw
                # logits = a[:, -1:] # <- Extract last item

                ## Z-Score Norm
                # objectmap_tensor = a[:, -1:] # <- Extract last item
                # mean = objectmap_tensor.mean()
                # std = objectmap_tensor.std()
                # if std > 0: objectmap_tensor = (objectmap_tensor - mean) / std
                # else: objectmap_tensor = objectmap_tensor - mean # Avoid NaN if std = 0
                # logits = objectmap_tensor

                ### Calculate the confidence    
                out = non_max_suppression(detect_branch)[0]

                if len(out) == 0: 
                    pred_binary = torch.zeros(1, 1, self.image_size, self.image_size).to(self.device)
                else: 
                    conf = out[0][4]
                    if conf <= 0: 
                        pred_binary = torch.zeros(1, 1, self.image_size, self.image_size).to(self.device)
                    else:
                        pred = self.model(img, logits)
                        pred_sigmoid = torch.nn.functional.sigmoid(pred)
                        pred_binary  = (pred_sigmoid > 0.5).float()

                # Calculate HD95
                pred_hot_encoded = torch.cat([1 - pred_binary, pred_binary], dim=1)
                mask_hot_encoded = torch.cat([1 - mask, mask], dim=1)                    
                self.hd95(pred_hot_encoded, mask_hot_encoded)

                # Calculate Precision and Recall
                TP = (pred_binary * mask).sum().float()
                FP = (pred_binary * (1 - mask)).sum().float()
                FN = ((1 - pred_binary) * mask).sum().float()
                total_TP += TP.item()
                total_FP += FP.item()
                total_FN += FN.item()

                # Calculate Dice
                self.metric(pred_binary, mask)          

            # Aggregate Precision & Recall
            val_precision_metric = total_TP / (total_TP + total_FP + 1e-6)
            val_recall_metric = total_TP / (total_TP + total_FN + 1e-6)

            # Aggregate HD95
            hd95_raw_results, hd95_not_nans_count = self.hd95.aggregate()
            is_valid = hd95_not_nans_count.bool()
            successful_hd95_values = hd95_raw_results[is_valid]
            val_hd95_metric = torch.mean(successful_hd95_values).item()

            # Aggregate Dice   
            val_dice_metric = self.metric.aggregate().item()

        print(f"Final Dice Metric: {val_dice_metric}...")
        print(f"Final HD95 Metric: {val_hd95_metric}...")
        print(f"Final Precision Metric: {val_precision_metric}...")
        print(f"Final Recall Metric: {val_recall_metric}...")

def count_parameters(model: torch.nn.Module, only_trainable: bool = True) -> Union[int, List[int]]:
    """
    Counts the total number of parameters in a PyTorch model.

    Args:
        model (nn.Module): The PyTorch model instance (e.g., a loaded model or a custom nn.Module).
        only_trainable (bool): If True, counts only parameters that require gradients (trainable).
                               If False, returns a list: [trainable_params, total_params].

    Returns:
        Union[int, List[int]]: The total count of parameters (or a list if only_trainable is False).
    """

    # Generator expression to count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if only_trainable:
        return trainable_params
    else:
        # Generator expression to count ALL parameters (trainable + fixed)
        all_params = sum(p.numel() for p in model.parameters())
        return [trainable_params, all_params]

def modify_YOLO(model): 
    old_conv_module = model.model.model.model[0]
    print(old_conv_module)

    # 3. Get the inner nn.Conv2d layer and its weights
    old_nn_conv = old_conv_module.conv
    old_conv_weights = old_nn_conv.weight.data 

    # --- Setup for the new Conv module ---

    # 5. Determine the new nn.Conv2d layer parameters
    # c2 (output channels) comes from the old nn.Conv2d layer
    out_channels = old_nn_conv.out_channels
    # Other parameters from the old nn.Conv2d layer
    kernel_size = old_nn_conv.kernel_size
    stride = old_nn_conv.stride
    padding = old_nn_conv.padding
    dilation = old_nn_conv.dilation
    groups = old_nn_conv.groups
    # Note: YOLO's Conv sets bias=False, so we skip transferring it.

    # 6. Create the new 4-channel nn.Conv2d layer
    new_nn_conv = nn.Conv2d(
        in_channels=4,  # Change from 3 to 4
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=False # Must match the original YOLO structure
    )

    # 7. Initialize the new Conv weights tensor
    # Shape: [out_channels, in_channels (4), kernel_height, kernel_width]
    new_conv_weights = new_nn_conv.weight.data

    # 8. Transfer the original 3-channel weights (RGB)
    new_conv_weights[:, 0:3, :, :] = old_conv_weights[:, 0:3, :, :]

    # 9. Calculate and set the average for the 4th channel
    # Calculate the mean across the input channel dimension (dim=1) of the old weights
    avg_weights = old_conv_weights.mean(dim=1, keepdim=True) 
    # Copy the averaged 3-channel weights to the 4th channel (index 3)
    new_conv_weights[:, 3:4, :, :] = avg_weights

    # 10. Reconstruct the entire Conv module (and keep the original BN and ACT)
    # We can't easily instantiate the original 'Conv' class without the source file 
    # and 'autopad' function, so we modify the existing module's components.

    # Set the new nn.Conv2d layer into the existing Conv module
    old_conv_module.conv = new_nn_conv

    # The BN and ACT layers remain unchanged, which is correct because they operate 
    # on the 'out_channels', which hasn't changed (it's still 16 in your example).
    # You must ensure the BN's num_features matches the Conv's out_channels.

    print("✅ Model's first 'Conv' module successfully modified.")
    print(f"   - New input channels: 4")
    print(f"   - Original output channels: {out_channels}")

if __name__ == "__main__": 
    # Create trainer and predictor instances
    p_args = dict(model="pretrained_detect_yolo/yolo12n_det/weights/best.pt",
                data=f"data/data.yaml", 
                verbose=True,
                imgsz=160, 
                save=False)

    # Create predictor and Load checkpoint
    YOLO_predictor = CustomSegmentationPredictor(overrides=p_args)
    YOLO_predictor.setup_model(p_args["model"])

    # Create YOLOU instance
    model = YOLOSegPlusPlus(predictor=YOLO_predictor)
    # modify_YOLO(YOLO_predictor)
    
    # Load the checkpoint
    checkpoint = torch.load("fair_ablation_study/soft_gating/weights/best.pth", map_location=torch.device('cuda'))  # or 'cuda' if using GPU
        
    # If checkpoint is a state_dict directly:
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
        
    # Load weights into the model
    model.load_state_dict(state_dict)

    trainable_count = count_parameters(model, only_trainable=True)
    all_counts = count_parameters(model, only_trainable=False)
    print(f"Total Trainable Parameters: {trainable_count:,}")
    print(f"Total All Parameters (Trainable + Fixed): {all_counts[1]:,}")
    print("-" * 30)
            
    eval = Evaluator(model=model, 
                    data_path="data/stacked_segmentation", 
                    model_path=None, 
                    image_size = 160,
                    batch_size = 1,
                    device = "cuda")
    eval.evaluate()
