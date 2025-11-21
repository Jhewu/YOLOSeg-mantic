# YOLOSeg-mantic: Leveraging Detection-Derived Semantics for Lightweight Brain Tumor Segmentation on Edge Devices (BraTS-SSA)
[**REPOSITORY WORK IN PROGRESS**]
Current state-of-the-art brain tumor segmentation often depends on computationally expensive 3D tensors and transformer-based self-attention mechanisms. Although these approaches yield strong performance, they remain impractical for deployment in low-resource or edge environments. Conversely, existing low-parameter 2D segmentation architectures reduce computational cost but sacrifice segmentation fidelity (e.g., YOLO-based variants) or suffer from image-level class imbalance (i.e., a majority of 2D slices lacking tumor regions), where optimizing for both detection and segmentation tasks is challenging with limited parameters (e.g., TinyUNet). 
To address this challenge, we developed YOLOSeg-mantic, a low-parameter (2.6M + 80K) 2D semantic segmentation framework that leverages detection-derived confidence signals from a pretrained YOLOv12n backbone. By using YOLO’s class logit as a semantic bottleneck and forming a compact UNet-style decoder using backbone skip features, our method explicitly transfers YOLO’s objectness prior into the segmentation task, improving robustness to image-level class imbalance while maintaining fine-grained boundary precision. YOLOSeg-mantic further serves as a transfer-level framework to adapt lightweight YOLO detectors into accurate semantic segmentation with minimal additional parameters, achieving 16 FPS in theoretical CPU inference. 
On our validation set (BraTS-SSA), YOLOSeg-mantic achieves a Dice Similarity Coefficient of 0.87 for whole-tumor segmentation, surpassing Vanilla UNet (7M), TinyUNet (0.48M), YOLO12n-Seg (2.9M), and YOLO12x-Seg (64M), while maintaining a reasonable balance in Precision and Recall, indicating reduced sensitivity to slice-level class imbalance. The proposed decoupled architecture also enables task-specific optimization beyond YOLO’s coupled training objective, demonstrating that efficient and accurate brain-tumor segmentation is feasible through detection-informed design, paving the way for future peer-reviewed validation towards a clinical edge deployment solution. 

## 📁 Structure [WORK IN PROGRESS]
```
├── custom_yolo_predictor/ 	# Ultralytics YOLO Custom Predictor (for 4-channel images)
├── custom_yolo_trainer/	  # Ulatralytics YOLO Custom Trainer (for 4-channel images)
├── data/  					        # YOLOU dataset
├── yolo_checkpoint/ 		    # Pre-trained YOLOv12-Seg model
├── YOLOSegmantic.py 	      # Creates dataset for YOLOU
├── dataset.py 				      # Creates dataset for YOLOU
├── train.py				        # YOLOU trainer
```
