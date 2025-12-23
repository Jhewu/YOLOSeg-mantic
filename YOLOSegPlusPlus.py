from typing import List
from ultralytics.nn.modules import C3Ghost, DWConv, C2f, DWConvTranspose2d, Conv, C3k2, ConvTranspose, CBAM, LightConv

# local/custom scripts
from custom_yolo_predictor.custom_detseg_predictor import CustomDetectionPredictor

import torch
import torch.nn as nn
from torch.nn import Sequential, Module, Upsample, Conv2d, Conv1d, Identity, AdaptiveAvgPool2d
import torch.nn.functional as F


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
                 predictor: CustomDetectionPredictor,
                 verbose: bool = False):
        """
        WARNING: DOCUMENTATION NOT UPDATED

        Creates a YOLOU-Seg++ Network with Pretrained YOLOv12 (detection) model
        Main Idea: Using YOLOv12 bbox as guidance in UNet skip connections and recycling YOLOv12 backbone as the encoder

        Args: 
            predictor (CustomSegmentationTrainer): Custom YOLO segmentation predictor allowing 4-channels

        Attributes: 
            yolo_predictor (CustomSegmentationTrainer): Custom YOLO segmentation predictor instance  
            encoder (nn.Sequential): Encoder portion extracted from YOLOv12-Seg backbone (first 9 layers)
            decoder (nn.Sequential): Decoder portion constructed from encoder modules
            bottleneck (nn.Sequential): First bottleneck layer with BottleneckCSP block

        Methods: 
            _hook_fn: Forward hook function for caching activations (mainly used for YOLOv12-Seg forward pass)
            _assign_hooks: Registers forward hooks on specified modules
            _create_concat_block: Creates concatenation blocks for skip connections
            YOLO_forward: Performs YOLOv12-Seg forward pass to generate initial masks
            _STN_forward: Applies spatial transformer network for affine transformation
            forward: Main forward pass implementation
            _reverse_module_channels: Converts encoder modules to decoder-compatible modules
            _construct_decoder_from_encoder: Builds decoder from encoder modules
            check_encoder_decoder_symmetry: Utility method to verify encoder-decoder symmetry
            print_yolo_named_modules: Debug utility to print all YOLO modules

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
        self.yolo = predictor.model.model
        for param in self.yolo.parameters():  # <- Frozen
            param.requires_grad = False
        self.yolo.eval()

        self.encoder = self.yolo.model[:5]
        
        self.upsample = Upsample(
            scale_factor=2, mode="bilinear", align_corners=False)

        # ---Decoder Body---
        self.decoder = nn.ModuleList([
            Sequential(  # <- Mixing (128 Skip) + (1 Logits)
                # C3Ghost(128+1, 96, n=1),
                C3Ghost(128, 96, n=1),
                ECA(),
            ),
            Sequential(  # <- Assume Upsample Here 20x20 -> 40x40
                self.upsample,
                SingleLightConv(96, 64),
                ECA()

                # ---PREVIOUS---
                # DoubleLightConv(96, 64),
                # ---PREVIOUS---
            ),
            Sequential(  # <- Mixing (64 Input) + (64 Skip)
                C3Ghost(64+64, 64),
                ECA(),
            ),
            Sequential(  # <- Assume Upsample Here 40x40 -> 80x80
                self.upsample,

                SingleLightConv(64, 32),
                ECA()

                # ---PREVIOUS---
                # DoubleLightConv(64, 32)
                # ---PREVIOUS---
            ),
            Sequential(  # <- Assume Upsample Here 80x80 -> 160x160
                self.upsample,

                SingleLightConv(32, 16),
                ECA()

                # ---PREVIOUS---
                # DoubleLightConv(32, 16)
                # ---PREVIOUS---
            ),
        ])
        self.output = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1)

        # ---Miscellaneous Section---
        self.verbose = verbose

        # ---Indices--- 
        self.upsample_idx = {2, 5, 6}
        self.encoder_skip_idx = {2, 4}  # <- Must be at respective resolution
        self.decoder_skip_idx = {2}
        
        if torch.cuda.is_available():
            print(f"\nATTENTION: CUDA {torch.cuda.get_device_name(
                0)} is available, forwarding YOLOv12 backbone twice is faster than forward hooks...\n")
        else:
            print(
                f"\nATTENTION: CUDA is not available (CPU), using forward hooks to save on compute...\n")
            self.activation_cache = []
            self._assign_hooks(modules=list(self.decoder_skip_idx))

    def _hook_fn(self, module, input, output):
        """
        Forward hook, once activate appends the output to
        self.activation_cache
        """
        self.activation_cache.append(output)
        if self.verbose:
            print(f"\nSuccessfully cached the output {module}\n")

    def _assign_hooks(self, modules: list[str] = [2, 4]):
        """
        Assigns forward hooks for YOLOv12-Seg forward
        Depends on self._hook_fn()

        Args:
            modules (list[str]): List containing the names of the modules
        """
        found = []
        for name, module in self.encoder.named_modules():
            if name in modules:
                module.register_forward_hook(self._hook_fn)
                if verbose:
                    print(f"Hook registered on: {name} -> {module}")
                found.append(name)

        if not found:
            raise ValueError(f"Modules not found in YOLO")
            
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
            # ---YOLO detect forward---
            x = self.yolo(x)
            detect_branch, cls_branch = x
            twenty, ten, five = cls_branch  # <- Resolution-wise
            logits = twenty[:, -1:]
            # ---YOLO detect forward---

            i = len(self.activation_cache) - 1  # <- Start from last index

            # ---Decoder "Semantic Bottleneck"---
            skip = self.activation_cache[-1]
            x = (skip * logits) + skip
            i -= 1
            # ---Decoder "Semantic Bottleneck"---

            # ---Decoder Body---
            for idx, module in enumerate(self.decoder):
                if idx in self._indices.get("skip_connections_decoder"):
                    skip = self.activation_cache[i]
                    x = torch.concat([x, skip], dim=1)
                    i -= 1
                x = module(x)
            out = self.output(x)
            # ---Decoder Body---
        self.activation_cache.clear()
        return out

    def forward(self, x: torch.tensor, logits: torch.tensor) -> torch.tensor:
        """
        Training ONLY forward step for YOLOSeg++
        YOLO logits are precomputed to reduce training time (check generate_objectmaps.py)
        Use self.inference() for inference

        Args:
            x        (torch.tensor):  Input tensor [B, 4, H, W]
            heatmaps (torch.tensor):  List of resized heatmaps tensors to concatenate at skips [1, 1, h, w], where h and w are resized heights and weights (< H and W)

        Returns:
            x (torch.tensor): Output tensor [B, 1, H, W]
        """
        #---Encoder (weights frozen in training loop)---    
        skip_connections = []
        for idx, module in enumerate(self.encoder):
            x = module(x)
            if idx in self.encoder_skip_idx: 
                # Manually cache tensors for skips
                skip_connections.append(x)

        #---Decoder "Semantic Bottleneck" (trainable)---
        i = len(skip_connections) - 1  # <- Start from last index
        skip = skip_connections[i]
        i -= 1
        
        # ---CONCATENATION (MUST MODIFY ARCHITECTURE)---
        # x = torch.concat([skip, logits], dim=1)
        # ---CONCATENATION (MUST MODIFY ARCHITECTURE)---

        # ---SOFT GATING---
        x = (skip * logits) + skip
        # ---SOFT GATING---

        # ---NO LOGITS---
        # x = skip
        # ---NO LOGITS---

        # ---Decoder Body (Trainable)---
        
        for idx, module in enumerate(self.decoder):
            if idx in self.decoder_skip_idx: 
                skip = skip_connections[i]
                x = torch.concat([x, skip], dim=1)
                i -=1
            x = module(x)
        out = self.output(x)
        return out
