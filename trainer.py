# Local
from base import BaseTrainer

# External 
import torch

# Internal
import os

class SegmentationTrainer(BaseTrainer):
    """
    Subclass of BaseTrainer
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def eval_step(self, inputs, targets):    
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
