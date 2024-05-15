import torch
import pytorch_lightning as pl
from metrics import score
import pandas as pd
import torch.nn as nn
import numpy as np

class BirdModule(pl.LightningModule):
    
    def __init__(self, model, loss=None, opt=None, scheduler=None, df=None):
        super().__init__()
        self.optimizer = opt
        self.scheduler = scheduler
        self.backbone = model
        self.loss_fn = loss
        if isinstance(df, pd.DataFrame):
            self.label_list = sorted(df['primary_label'].unique())
        self.validation_step_outputs = []
        
    def forward(self, images):
        return self.backbone(images)
    
    def configure_optimizers(self):
        model_optimizer = self.optimizer
        lr_scheduler = self.scheduler
        
        return {
            'optimizer': model_optimizer,
            'lr_scheduler': {
                'scheduler': lr_scheduler,
                'interval': 'epoch',
                'monitor': 'val_loss',
                'frequency': 1
            }
        }
    
    def training_step(self, batch, batch_idx):
        feature, target = batch
        feature = feature.to(self.device)
        target = target.to(self.device)
        y_pred = self(feature)
        train_loss = self.loss_fn(y_pred, target)
        self.log('train_loss', train_loss, True)
        return train_loss
    
    def validation_step(self, batch, batch_idx):
        feature, target = batch
        feature = feature.to(self.device)
        target = target.to(self.device)
        with torch.no_grad():
            y_pred = self(feature)
        self.validation_step_outputs.append({"logits": y_pred, "targets": target})
        
    def train_dataloader(self):
        return self._train_dataloader

    def validation_dataloader(self):
        return self._validation_dataloader
    
    def on_validation_epoch_end(self):
        
        outputs = self.validation_step_outputs
        output_val = nn.Softmax(dim=1)(torch.cat([x['logits'] for x in outputs], dim=0)).cpu().detach()
        target_val = torch.cat([x['targets'] for x in outputs], dim=0).cpu().detach()
        
        val_loss = self.loss_fn(output_val, target_val)
        target_val = torch.nn.functional.one_hot(target_val, len(self.label_list))
        
        gt_df = pd.DataFrame(target_val.numpy().astype(np.float32), columns=self.label_list)
        pred_df = pd.DataFrame(output_val.numpy().astype(np.float32), columns=self.label_list)
        gt_df['id'] = [f'id_{i}' for i in range(len(gt_df))]
        pred_df['id'] = [f'id_{i}' for i in range(len(pred_df))]
        val_score = score(gt_df, pred_df, row_id_column_name='id')
        self.log("val_score", val_score, True)
        self.validation_step_outputs = list()
        
        return {'val_loss': val_loss, 'val_score': val_score}