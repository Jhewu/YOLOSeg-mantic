# Local
from custom_yolo_trainer.custom_trainer import CustomSegmentationTrainer

# External Libs
import torch
import torch.nn as nn
from ultralytics.nn.modules import C3Ghost, LightConv
from torch.nn import Sequential, Module, Upsample, Conv2d, Conv1d, Identity, AdaptiveAvgPool2d

class BoundaryRefinementModule(nn.Module):
    def __init__(self, in_channels: int, simple: bool = True):
        """
        Custom Boundary Refinement Module for YOLOSegmantic. 
        Refines segmentation boundaries using edge-aware feature enhancement

        (Key Insight): HD95 errors occur at boundaries where predictions are uncertain.
        This module detects boundaries and sharpens features in those regions.

        Args:
            in_channels (int): Number of input channels from the decoder
            simple (bool): If True, uses a lightweight edge detection and refinement (Basically a separated DWConv + Sigmoid).
                           If False, uses a more complex edge detection with additional convolutions.

        """
        super().__init__()

        if simple:
            self.edge_detector = nn.Sequential(
                # Depthwise only (Spatial only)
                nn.Conv2d(in_channels, in_channels, 3, padding=1,
                          groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels),

                # HardSigmoid is much faster on CPU than standard Sigmoid
                nn.Hardsigmoid(inplace=True)
            )
            self.refine_conv = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 1, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True)
            )

        else:
            # Edge detection path (learns to detect boundaries)
            self.edge_detector = nn.Sequential(
                # Depthwise: detects spatial patterns per channel
                nn.Conv2d(in_channels, in_channels, 3,
                          padding=1, groups=in_channels),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),

                # Pointwise: combines channel info for edge map
                nn.Conv2d(in_channels, in_channels, 1),
                nn.BatchNorm2d(in_channels),
                nn.Sigmoid()  # Produces edge attention map [0, 1]
            )

            # Feature refinement path
            self.refine_conv = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, padding=1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        """
        Args:
            x: Feature map [B, C, H, W]

        Returns:
            refined: Boundary-refined features [B, C, H, W]
        """
        # Detect boundary regions (high gradient areas)
        edge_attention = self.edge_detector(x)  # [B, C, H, W], values in [0,1]

        # Refine features
        refined = self.refine_conv(x)

        # Apply edge-weighted residual connection
        # - In boundary regions (edge_attention u2248 1): use refined features more
        # - In flat regions (edge_attention u2248 0): keep original features
        output = x + refined * edge_attention

        return output


class ResDWCOnv(Module):
    def __init__(self, in_channels: int, out_channels: int, k1: int = 3):
        """
        ResDWCOnv consists of a single YOLO ULtralytics LightConv 
        (i.e., Depthwise Separable Convolution) layer with a residual connection.

        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            k1 (int): Kernel size for the LightConv layer
        """
        super().__init__()
        self.conv = LightConv(
            c1=in_channels,
            c2=out_channels,
            k=k1,
            act=True)

        # 1x1 conv to match channels if needed
        self.residual_conv = (
            Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x):
        """
        Forward step for ResDWCOnv
        Args:
            x (torch.tensor): Input tensor [B, in_channels, H, W]
        """
        residual = self.residual_conv(x)
        out = self.conv(x)
        out += residual
        return out


class DoubleResDSConv(Module):
    def __init__(self, in_channels: int, out_channels: int, k1: int = 3, k2: int = 3):
        """
        DoubleResDSConv consists of two sequential YOLO ULtralytics LightConv (i.e., Depthwise Separable Convolutions) 
        layers with a residual connection.

        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            k1 (int): Kernel size for the first LightConv layer
            k2 (int): Kernel size for the second LightConv layer
        """
        super().__init__()
        self.conv = Sequential(
            LightConv(
                c1=in_channels,
                c2=out_channels,
                k=k1,
                act=True),
            LightConv(
                c1=out_channels,
                c2=out_channels,
                k=k2,
                act=True))

        # 1x1 conv to match channels if needed
        self.residual_conv = (
            Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels else Identity()
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Forward step for DoubleResDSConv

        Args:
            x (torch.tensor): Input tensor [B, in_channels, H, W]
        """
        residual = self.residual_conv(x)
        out = self.conv(x)
        out += residual
        return out


class ECA(Module):
    def __init__(self, k_size: int = 3):
        super(ECA, self).__init__()
        """
        Constructs a ECA module (Efficient Channel Attention). 
        Reference - [https://arxiv.org/abs/1910.03151]

        Args: 
            k_size (int): kernel size for Conv1d
        """
        self.avg_pool = AdaptiveAvgPool2d(1)
        self.conv = Conv1d(1, 1, kernel_size=k_size,
                           padding=(k_size - 1) // 2, bias=False)

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Args: 
            x (torch.tensor): input tensor (after mask + x concat)
        Returns:
            (torch.tensor)  : output tensor (same dimensions as input tensor but with attention applied)
        """
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)
                      ).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = torch.sigmoid(y)

        return x * y.expand_as(x)


class YOLOSegmantic(Module):
    def __init__(self,
                 predictor: CustomSegmentationTrainer,
                 config: dict):
        """
        Creates a YOLOSegmantic Network with Pretrained YOLOv12 (detection) model

        Args: 
            predictor (CustomSegmentationTrainer): Pretrained YOLOv12 model wrapped in CustomSegmentationTrainer
            refinement (str): Type of boundary refinement to apply ("simple", "basic", or "")
            verbose (bool): Whether to print detailed architecture information during initialization

        Attributes: 
            [WORK IN PROGRESS]

        Methods: 
            [WORK IN PROGRESS]

        -------------------------------------------------------------------------------------------------------------
        YOLOv12 backbone
        -------------------------------------------------------------------------------------------------------------
                                   from  n    params  module                                       arguments
          0                  -1  1       608  ultralytics.nn.modules.conv.Conv             [4, 16, 3, 2]
          1                  -1  1      4672  ultralytics.nn.modules.conv.Conv             [16, 32, 3, 2]
          2                  -1  1      6640  ultralytics.nn.modules.block.C3k2            [32, 64, 1, False, 0.25]
          3                  -1  1     36992  ultralytics.nn.modules.conv.Conv             [64, 64, 3, 2]
          4                  -1  1     26080  ultralytics.nn.modules.block.C3k2            [64, 128, 1, False, 0.25]
          5                  -1  1    147712  ultralytics.nn.modules.conv.Conv             [128, 128, 3, 2]
          6                  -1  2    180864  ultralytics.nn.modules.block.A2C2f           [128, 128, 2, True, 4]
          7                  -1  1    295424  ultralytics.nn.modules.conv.Conv             [128, 256, 3, 2]
          8                  -1  2    689408  ultralytics.nn.modules.block.A2C2f           [256, 256, 2, True, 1]
        -------------------------------------------------------------------------------------------------------------

        """

        super().__init__()
        # ---Config Unpacking--- #

        self._refinement = config.get("refinement", "simple")
        self._verbose    = config.get("verbose", False)
        self._layers     = config.get("channels", [
                                            [128, 64, 2],
                                            [64, 64],
                                            [64, 32, 1],
                                            [32, 16],
                                            [16, 8], 
                                            [8, 1]
                                        ])
        self.num_classes = config.get("num_classes", 1)

        # ---YOLO predictor and backbone--- #
        self.yolo = predictor.model
        for param in self.yolo.parameters():  # <- Frozen
            param.requires_grad = False
        self.yolo.eval()
        self.encoder = self.yolo.model[:5]

        # ---Decoder Body--- #
        self.bilinear = Upsample(
            scale_factor=2, mode="bilinear", align_corners=False)
        self.nearest = Upsample(
            scale_factor=2, mode="nearest")
        
        self.decoder = nn.ModuleList([
            Sequential(  # <- Mixing (128 Skip) + 
                C3Ghost(self._layers[0][0], self._layers[0][1], n=self._layers[0][2]),
                ECA(),
            ),
            Sequential(  # <- Assume Upsample Here 20x20 -> 40x40
                self.nearest,
                DoubleResDSConv(self._layers[1][0], self._layers[1][1]),
            ),
            Sequential(  # <- Mixing (64 Input) + (64 Skip)
                C3Ghost(self._layers[2][0], self._layers[2][1], n=1),
                ECA(),
            ),
            Sequential(  # <- Assume Upsample Here 40x40 -> 80x80
                self.nearest,
                ResDWCOnv(self._layers[3][0], self._layers[3][1]),

            ),
            Sequential(  # <- Assume Upsample Here 80x80 -> 160x160
                self.bilinear,
                ResDWCOnv(self._layers[4][0], self._layers[4][1]),
            ),
        ])
        self.output = nn.Conv2d(in_channels=self._layers[5][0], out_channels=self._layers[5][1], kernel_size=self.num_classes)
        # ---Decoder Body--- #

        # --- Boundary Refinement Section --- #
        if self._refinement == "simple":
            self.boundary_refine = BoundaryRefinementModule(8, simple=True)
        elif self._refinement == "basic":
            self.boundary_refine = BoundaryRefinementModule(8, simple=False)
        else:
            self.boundary_refine = None

        # --- Indices --- #
        self.upsample_idx = {2, 5, 6}
        self.encoder_skip_idx = {0, 1, 2, 4}  # <- Including ALL layers
        self.decoder_skip_idx = {2, 3, 4}
        # --- Indices --- #

    def inference(self, x: torch.tensor) -> torch.tensor:
        """
        (Inference ONLY) Inference step for YOLOSegmantic
        Run YOLO Detect forward, and YOLOSegmantic Segmentator Head sequentially

        Args:
            x (torch.tensor): Input tensor [B, 4, H, W]

        Returns:
            x (torch.tensor): Output tensor [B, 1, H, W]
        """

        with torch.no_grad():
            out = self.forward(x)
        return out

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        (Training ONLY) Forward step for YOLOSegmantic

        Args:
            x (torch.tensor): Input tensor [B, 4, H, W]
            logits (torch.tensor): YOLO Detect logits to concatenate at skips [B, 1, 20, 20]

        Returns:
            x (torch.tensor): Output tensor [B, 1, H, W]
        """
        # --- YOLO detect forward --- #
        with torch.no_grad():
            x, features, logits = self.yolo.predict(
                x, return_features=True, seg_features_idxs=self.encoder_skip_idx)
        # --- YOLO detect forward --- #

        i = -1  # <- Start from last index

        # --- Decoder Semantic Bottleneck --- # 
        skip = torch.sigmoid(features[i])
        x = skip * (logits + 1)
        i -= 1
        # --- Decoder Semantic Bottleneck --- #

        # --- Decoder Body --- #
        for idx, module in enumerate(self.decoder):
            if idx in self.decoder_skip_idx:
                skip = features[i]
                x = x * (skip + 1)
                i -= 1
            x = module(x)
        # --- Decoder Body --- #

        # --- Boundary Refinement --- #
        if self.boundary_refine:
            x = self.boundary_refine(x)
        # --- Boundary Refinement --- #

        return self.output(x)
