# Local
from custom_yolo_trainer.custom_trainer import CustomSegmentationTrainer

# External libs
import torch
import torch.nn as nn
from torch.nn import Sequential, Module, Upsample, Conv2d, Conv1d, Identity, AdaptiveAvgPool2d
from ultralytics.nn.modules import C3Ghost, LightConv


class BoundaryRefinementModule(nn.Module):
    """
    Refines segmentation boundaries using edge-aware feature enhancement

    Key Insight: HD95 errors occur at boundaries where predictions are uncertain.
    This module detects boundaries and sharpens features in those regions.
    """

    def __init__(self, in_channels):
        super().__init__()

        # Edge detection path (learns to detect boundaries)
        # Uses depthwise separable conv for efficiency
#         self.edge_detector = nn.Sequential(
#             # Depthwise: detects spatial patterns per channel
#             nn.Conv2d(in_channels, in_channels, 3,
#                       padding=1, groups=in_channels),
#             nn.BatchNorm2d(in_channels),
#             nn.ReLU(inplace=True),
# 
#             # Pointwise: combines channel info for edge map
#             nn.Conv2d(in_channels, in_channels, 1),
#             nn.BatchNorm2d(in_channels),
#             nn.Sigmoid()  # Produces edge attention map [0, 1]
#         )
        self.edge_detector = nn.Sequential(
            # Depthwise only (Spatial only)
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            # HardSigmoid is much faster on CPU than standard Sigmoid
            nn.Hardsigmoid(inplace=True) 
        )

        # Feature refinement path
        # self.refine_conv = nn.Sequential(
        #     nn.Conv2d(in_channels, in_channels, 3, padding=1),
        #     nn.BatchNorm2d(in_channels),
        #     nn.ReLU(inplace=True)
        # )
        self.refine_conv = nn.Sequential(
                    nn.Conv2d(in_channels, in_channels, 1, bias=False),
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


class AdvancedBoundaryRefinement(nn.Module):
    """
    More sophisticated version using explicit gradient-based edge detection
    """

    def __init__(self, in_channels):
        super().__init__()

        # Sobel-like edge detection (learns edge kernels)
        self.horizontal_edge = nn.Conv2d(in_channels, in_channels,
                                         kernel_size=3, padding=1,
                                         groups=in_channels, bias=False)
        self.vertical_edge = nn.Conv2d(in_channels, in_channels,
                                       kernel_size=3, padding=1,
                                       groups=in_channels, bias=False)

        # Feature processing
        self.edge_fusion = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        self.boundary_refine = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3,
                      padding=1, groups=in_channels),
            nn.Conv2d(in_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Compute gradients in horizontal and vertical directions
        h_edges = self.horizontal_edge(x)
        v_edges = self.vertical_edge(x)

        # Combine edge information
        edges = torch.cat([h_edges, v_edges], dim=1)
        edge_features = self.edge_fusion(edges)

        # Generate boundary attention
        boundary_attn = self.boundary_refine(edge_features)

        # Weighted combination: emphasize boundaries
        return x + x * boundary_attn


class SingleLightConv(Module):
    def __init__(self, in_channels, out_channels, k1=3):
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
        residual = self.residual_conv(x)
        out = self.conv(x)
        out += residual
        return out


class DoubleLightConv(Module):
    def __init__(self, in_channels, out_channels, k1=3, k2=3):
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

    def forward(self, x):
        residual = self.residual_conv(x)
        out = self.conv(x)
        out += residual
        return out


class ECA(Module):
    def __init__(self, k_size: int = 3):
        super(ECA, self).__init__()
        """
        Constructs a ECA module. Efficient Channel Attention for Conv

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


class YOLOSegPlusPlus(Module):
    def __init__(self,
                 predictor: CustomSegmentationTrainer,
                 refinement: str = "basic",
                 training: bool = True,
                 verbose: bool = False):
        """
        WARNING: DOCUMENTATION NOT UPDATED

        Creates a YOLOSeg++ Network with Pretrained YOLOv12 (detection) model

        Args: 
            WORK IN PROGRESS

        Attributes: 
            WORK IN PROGRESS

        Methods: 
            WORK IN PROGRESS

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
        # ---YOLO predictor and backbone---
        self.yolo = predictor.model
        for param in self.yolo.parameters():  # <- Frozen
            param.requires_grad = False
        self.yolo.eval()
        self.encoder = self.yolo.model[:5]

        # ---Decoder Body---
        self.bilinear = Upsample(
            scale_factor=2, mode="bilinear", align_corners=False)
        self.nearest = Upsample(
            scale_factor=2, mode="nearest")
        self.decoder = nn.ModuleList([
            Sequential(  # <- Mixing (128 Skip) + (1 Logits)
                C3Ghost(128, 64, n=1),
                ECA(),
            ),
            Sequential(  # <- Assume Upsample Here 20x20 -> 40x40
                self.nearest,
                DoubleLightConv(64, 64),
            ),
            Sequential(  # <- Mixing (64 Input) + (64 Skip)
                C3Ghost(64, 32),
                ECA(),
            ),
            Sequential(  # <- Assume Upsample Here 40x40 -> 80x80
                self.nearest,
                SingleLightConv(32, 16),

            ),
            Sequential(  # <- Assume Upsample Here 80x80 -> 160x160
                self.bilinear,
                SingleLightConv(16, 8),
            ),
        ])
        self.output = nn.Conv2d(in_channels=8, out_channels=1, kernel_size=1)
        
        # ---Auxiliary Output Heads Section---
        self.aux_out = nn.ModuleList([
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Conv2d(8, 1, kernel_size=1)
        ])

        # ---Boundary Refinement Section---
        if refinement == "basic":
            self.boundary_refine = BoundaryRefinementModule(8)
        elif refinement == "advanced":
            self.boundary_refine = AdvancedBoundaryRefinement(8)
        else:
            self.boundary_refine = None

        # ---Miscellaneous Section---
        self.verbose = verbose

        # ---Indices---
        self.upsample_idx = {2, 5, 6}
        # self.encoder_skip_idx = {2, 4}  # <- Must be at respective resolution
        self.encoder_skip_idx = {0, 1, 2, 4}
        self.decoder_skip_idx = {2, 3, 4}
        self.aux_decoder_idx = {3, 4}  # Example indices

    def inference(self, x: torch.tensor) -> torch.tensor:
        """
        Inference forward step for YOLOSeg++
        Run YOLO forward, and YOLOSeg++ Segmentator Head sequentially

        Args:
            x (torch.tensor): Input tensor [B, 4, H, W]

        Returns:
            x (torch.tensor): Output tensor [B, 1, H, W]
        """

        with torch.no_grad():
            out = self.forward(x)
        return out

    def forward(self, x: torch.tensor, return_aux=False) -> torch.tensor:
        """
        Training ONLY forward step for YOLOSeg++
        YOLO logits are precomputed to reduce training time (check generate_objectmaps.py)
        Use self.inference() for inference

        Args:
            x (torch.tensor): Input tensor [B, 4, H, W]
            logits (torch.tensor): YOLO Detect logits to concatenate at skips [B, 1, 20, 20]

        Returns:
            x (torch.tensor): Output tensor [B, 1, H, W]
        """
        # ---YOLO detect forward---
        with torch.no_grad():
            x, features, logits = self.yolo.predict(
                x, return_features=True, seg_features_idxs=self.encoder_skip_idx)

        i = -1  # <- Start from last index

        # ---Decoder "Semantic Bottleneck"---
        skip = torch.sigmoid(features[i])
        x = skip * (logits + 1)
        i -= 1

        # ---Decoder Body---
        aux_out, aux_idx = [], 0
        for idx, module in enumerate(self.decoder):
            if idx in self.decoder_skip_idx:
                skip = features[i]
                x = x * (skip + 1)
                i -= 1
            x = module(x)
            # ---Aux Output (Deep Supervision)---
            if return_aux and idx in self.aux_decoder_idx:
                aux_pred = self.aux_out[aux_idx](x)
                aux_out.append(aux_pred)
                aux_idx += 1

        # -- Boundary Refinement ---
        if self.boundary_refine:
            x = self.boundary_refine(x)

        out = self.output(x)
        if return_aux:
            return out, aux_out
        return out
