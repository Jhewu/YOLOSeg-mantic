# Local
from YOLOSegPlusPlus import YOLOSegPlusPlus
from custom_yolo_predictor.custom_detseg_predictor import CustomSegmentationPredictor
from dataset import CustomDataset

# Internal Lib
from typing import Tuple

# External Lib
from torch.utils.data import DataLoader


def create_dataloader(data_path: str) -> Tuple[DataLoader]:
    """
        Create dataloader from CustomDataset

        Args:
            data_path (str): root directory of dataset

        Returns:
            (Tuple[Dataloader]): train_dataloader and val_dataloader
        """
    train_dataset = CustomDataset(
        root_path=data_path,
        image_path="images/train",
        objectmap_path="objectmap/train",
        mask_path="masks/train",
        image_size=self.image_size,
        objectmap_sizes=[20])

    val_dataset = CustomDataset(
        root_path=data_path,
        image_path="images/val",
        objectmap_path="objectmap/val",
        mask_path="masks/val",
        image_size=self.image_size,
        objectmap_sizes=[20])

    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=self.batch_size,
                                  shuffle=False,
                                  num_workers=10)

    val_dataloader = DataLoader(dataset=val_dataset,
                                batch_size=self.batch_size,
                                shuffle=False,
                                num_workers=10)  # <- do not shuffle

    return train_dataloader, val_dataloader


def profile_model(model: YOLOSegPlusPlus,
                  mode: str = "gpu"
                  ) -> None:

    return


if __name__ == "__main__":
    # Create predictor and load checkpoint
    p_args = dict(model="pretrained_detect_yolo/best_yolo12n_det/weights/best.pt",
                  data=f"data/data.yaml",
                  verbose=True,
                  imgsz=160,
                  save=False)

    YOLO_predictor = CustomSegmentationPredictor(overrides=p_args)
    YOLO_predictor.setup_model(p_args["model"])

    # Create model instance
    model = YOLOSegPlusPlus(predictor=YOLO_predictor)

    profile_model()
