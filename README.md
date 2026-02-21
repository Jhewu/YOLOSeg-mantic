# YOLOSeg-mantic: Leveraging Detection-Driven Semantic Priors for Image-Level Imbalanced Clinical Tomograms on Embedded Edge Hardware(BraTS-Africa)

[**REPOSITORY WORK IN PROGRESS**]

Clinical tomograms present image-level imbalance with a high-level of negative slices in relation to positive slices. Automated segmentation of tumor tissue generally needs supervision, requiring radiologists to identify target pathology prior to model deployment. Recent designs achieved great performance and applicability in validation studies while avoiding inter-radiologist variability but require high computational resources. We have developed a method called YOLOSeg-mantic to be a lightweight, high-throughput edge semantic segmentation inference framework which can address the gap between high performance and high efficiency. YOLOSeg-mantic reuses classification head logits which represent the spatial priors of object locations and passes them through residual linear gating to form a compact UNet-style decoder which fuses the respective structural skip features. We compared YOLOSeg-mantic on the 2D BraTS-Africa dataset for whole tumor segmentation to several leading methods and found that YOLOSeg-matic had the highest combined accuracy and efficiency performance. Our work shows that a semantic segmentation model can be run in resource limited environments with the right understanding of computational resources. While we demonstrate the effectives in the tomogram tumor data, the method may be broadly applicable to other medical vision applications where real-time segmentation may be needed such as video-assisted surgery.

## 📁 Structure [WORK IN PROGRESS]
```
├── archive/ : irrelevant files but not yet to be deleted
├── custom_yolo_predictor/ : Ultralytics YOLO custom predictor (for 4-channel images)
├── custom_yolo_trainer/ : Ultralytics YOLO custom trainer (for 4-channel images)
├── data/ : Dataset location
├── runs/ : Run storage
├── samples : sample BraTS-Africa slices
├── tools : Stores "utils" scripts that are not used often
├── checkpoints/ : Pre-trained YOLOv12n detect model
├── dataset.py : Custom dataset class for YOLOSeg-mantic
├── evaluate_model.py : Evaluate YOLOSeg-mantic (with YOLO Detect gating)
├── YOLOSegmantic.py : YOLOSegmantic constructor
├── train.py : YOLOSegmantic trainer
```

## Documentation

[FILL HERE]

## Remaining ToDo

1. 
