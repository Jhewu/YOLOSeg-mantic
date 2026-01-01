# Internal Libs
from typing import List, Tuple
import os

# External Libs
import torch
import cv2
from torchvision import transforms
from torch.utils.data.dataset import Dataset


class CustomDataset(Dataset):
    def __init__(self,
                 root_path: str,
                 image_path: str,
                 mask_path: str,
                 image_size: int = 160,
                 augmentation: bool = True,
                 seed: int = 42,
                 subsample: float = 1.0):

        # Paths
        self.root_path = root_path
        self.image_dir = root_path + f"/{image_path}/"
        self.mask_dir = root_path + f"/{mask_path}/"

        image_filenames = sorted(os.listdir(self.image_dir))
        self.basenames = [os.path.splitext(f)[0] for f in image_filenames]
        self.basenames = self.basenames[:int(len(self.basenames) * subsample)]

        # Basic validation check (for mask/image pairs)
        for basename in self.basenames:
            if not os.path.exists(self.mask_dir + basename + ".png"):
                raise FileNotFoundError(f"Mask file not found for {basename}")

        # Loading
        self.image_size = image_size
        self.to_tensor = transforms.ToTensor()

        # Augmentation
        self.do_augmentation = augmentation
        self.augmentation = transforms.Compose([
            transforms.RandomAffine(
                degrees=10,
                translate=(5/self.image_size, 5 /
                           self.image_size),  # <- ~5 pixels
                # <- Mild zoom in/out
                scale=(0.8, 1.2)),
            transforms.RandomHorizontalFlip(p=0.5)
        ])

    def __getitem__(self, index) -> Tuple[torch.tensor, torch.tensor]:
        basename = self.basenames[index]

        # Construct Imag/Masks Paths
        img_path = self.image_dir + basename + ".png"
        mask_path = self.mask_dir + basename + ".png"

        # Load Image (RGBA -> BGRA in cv2.IMREAD_UNCHANGED)
        img_rgb_a = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

        # Load Mask (L/Grayscale)
        mask_np = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Resize Image & Masks
        img_resized = cv2.resize(
            img_rgb_a, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        mask_resized = cv2.resize(
            mask_np, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

        # Convert to PyTorch Tensor (HWC -> CWH, /255)
        img_tensor = self.to_tensor(img_resized)
        mask_tensor = self.to_tensor(mask_resized)

        return img_tensor, mask_tensor

    def __len__(self) -> int:
        return len(self.basenames)
