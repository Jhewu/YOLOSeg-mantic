# Local
from custom_yolo_trainer.custom_trainer import CustomSegmentationTrainer
from YOLOSegmantic import YOLOSegmantic

from trainer import SegmentationTrainer
from metrics import SegmentationMetrics
from loss import SegmentationLoss
from dataset import SegmentationDataLoader # <- import later

from tools.count_parameters import print_trainable_parameters

# External
import yaml 
import torch
import numpy as np

# Internal 
import random
import argparse

def set_seed(seed: int = 42): 
    # Set Seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        # Sets seed for all available GPUs
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

if __name__ == "__main__": 
    # -------------------------------------------------------------
    des="""
    Run YOLOSegmantic evaluation with the specified parameters in parameters.yaml
    and with confidence gating. Batch_size is set to 1. 
    """
    # -------------------------------------------------------------

    parser = argparse.ArgumentParser(description=des.lstrip(" "), formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-p", "--param_dir", type=str, help='directory of YAML training parameter configuration file\t[parameters.yaml]')
    parser.add_argument("-w", "--weight_dir", type=str, help='directory of pretrained weights\t[runs/segmantic_0/best.pt]')
    parser.add_argument("-c", "--conf", type=float, help='confidence threshold for YOLO detector\t[0.25]')
    args = parser.parse_args()

    ## --- Parse ----------------- ##
    if args.param_dir is not None:
        PARAM_DIR = args.param_dir
    else: 
        PARAM_DIR = "parameters.yaml"
    if args.weight_dir is not None: 
        WEIGHT_DIR = args.weight_dir
    else: 
        WEIGHT_DIR = "runs/segmantic_0/best.pt"
    if args.conf is not None: 
        CONF = args.conf
    else: 
        CONF = 0.25

    ## --- Load Parameters ----------------- ##
    with open(f"{PARAM_DIR}", "r") as f:
        params = yaml.safe_load(f)
    
    # Set configuration to load pretrained model weights
    params['trainer']['training']['is_load_and_train'] = True
    params['trainer']['training']['load_and_train_path'] = WEIGHT_DIR

    # Create predictor and load checkpoint
    yolo_cfg = params['yolo']
    p_args = dict(model=yolo_cfg['weights'],
                data=yolo_cfg['data_dir'],
                verbose=yolo_cfg['verbose'],
                imgsz=yolo_cfg['image_size'],
                save=yolo_cfg['save'])     
    YOLO_trainer = CustomSegmentationTrainer(overrides=p_args)
    YOLO_trainer.setup_model()

    # Create YOLOSegmantic instance
    model_cfg = params['model']
    model = YOLOSegmantic(predictor=YOLO_trainer, 
                          config=model_cfg)
    print_trainable_parameters(model)

    metrics = SegmentationMetrics()
    loss = SegmentationLoss()

    d_cfg = params['dataloader']
    dataloader = SegmentationDataLoader(
        root_path= d_cfg['root_path'],
        image_dir=d_cfg['image_dir'],
        mask_dir=d_cfg['mask_dir'],
        image_size=d_cfg['image_size'],
        augmentation=False,
        subsample=d_cfg['subsample'],
        batch_size=1,
        num_workers=d_cfg['num_workers'],
        shuffle=d_cfg['use_shuffle'],
        persistent_workers=d_cfg['use_persistent_workers'],
        pin_memory=d_cfg['use_pin_memory'],
    )

    set_seed()

    trainer = SegmentationTrainer(model=model, 
            loss_fn=loss, 
            metrics=metrics, 
            dataloader=dataloader, 
            params=params['trainer'], 
            param_dir=PARAM_DIR
            )

    # TODO: add parameters to evaluate
    trainer.evaluate(
        split = "test", 
        use_conf_thres = True, 
        conf_thres = CONF)