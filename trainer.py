# Local
from base import BaseTrainer
from tools.nms import non_max_suppression

# External 
import torch
from torch.utils.data import DataLoader

from tqdm import tqdm

# Internal
import os
import contextlib

class SegmentationTrainer(BaseTrainer):
    """
    Subclass of BaseTrainer
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def eval_step(self, inputs, targets, use_conf_thres: bool = False, conf_thres: float = 0.25): 
        """
        eval_step with use_conf_thres only works if batch_size is 1
        
        """
        if use_conf_thres and self.batch_size == 1: 
            preds, yolo_out = self.model.forward(x, return_yolo_out=True)
            pred_sigmoid = torch.nn.functional.sigmoid(preds)
            preds_binary = (pred_sigmoid > 0.5).float()  
            loss = self.loss_fn(preds, targets)

            detect_branch, cls_branch = yolo_out

            nms_out = non_max_suppression(detect_branch, conf_thres=conf_thres)[0]

            if len(nms_out) == 0: 
                preds_binary = torch.zeros(1, 1, self.image_size, self.image_size).to(self.device)
            else: 
                conf = nms_out[0][4]
                if conf <= conf_thres: 
                    preds_binary = torch.zeros(1, 1, self.image_size, self.image_size).to(self.device)
                    if self.verbose: 
                        print("confidence is less than the threshold and still passed")

            return preds_binary, loss
        else: 
            preds = self.model(inputs)
            pred_sigmoid = torch.nn.functional.sigmoid(preds)
            preds_binary = (pred_sigmoid > 0.5).float()  
            loss = self.loss_fn(preds, targets)

            return preds_binary, loss

    def set_model_to_train(self): 
        """
        For overriding in case of models with different train/eval modes for submodules (e.g. BatchNorm, Dropout).
        """
        self.model.train()
        self.model.yolo.eval()

    def build_checkpoint_path(self): 
        return os.path.join("runs", self._get_current_time())

    @torch.no_grad()
    def _eval_epoch(self, loader: DataLoader, use_conf_thres: bool = False, conf_thres: float = 0.25) -> tuple[float, dict]:
        self.model.eval()
        self.metrics.reset()
        total_loss, n = 0.0, 0

        autocast_ctx = (
                torch.amp.autocast(device_type=self.device)
                if self.mixed_precision
                else contextlib.nullcontext()
        )

        for raw_batch in tqdm(loader):
            batch           = self._move(raw_batch)
            inputs, targets = self._unpack(batch)

            with autocast_ctx:
                preds, loss = self.eval_step(inputs, targets, use_conf_thres, conf_thres)

            self.metrics.update(preds, targets)

            total_loss += loss.item() * (targets.size(0) if hasattr(targets, "size") else 1)
            n          += (targets.size(0) if hasattr(targets, "size") else 1)

        return total_loss / max(n, 1), self.metrics.compute()

    def evaluate(self, split: str = "test", use_conf_thres: bool = False, conf_thres: float = 0.25) -> tuple[float, dict]:
        train_loader, val_loader = self.dataloader.get_dataloader(split) 
        loader = val_loader if split == "test" else train_loader
        loss, metrics = self._eval_epoch(loader, use_conf_thres, conf_thres) 
        print(f"\n── {split.upper()} RESULTS ──")
        print(f"  loss: {loss:.4f}")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
        return loss, metrics