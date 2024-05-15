import torch
from metrics import score
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger
from dataset import BirdDataset
import pandas as pd
from holdout import holdout_infer
import numpy as np
import os
from model import predict
from lightning import BirdModule

def run_training(model, optimizer, scheduler, loss, config_dict, trainval_df, holdout_df):

    torch.set_float32_matmul_precision('high')
    holdout_args = config_dict['data_args']['holdout_args']
    label_list = sorted(trainval_df['primary_label'].unique())

    # Save Validation Scores
    val_scores = list()

    # Save Validation Prediction
    oof_df = trainval_df.copy()
    pred_cols = [f'pred_{t}' for t in label_list]
    oof_df = pd.concat([oof_df, pd.DataFrame(np.zeros((len(oof_df), len(pred_cols)*2)).astype(np.float32), columns=label_list+pred_cols)], axis=1)

    if holdout_args['use_holdout']:

        # Save Holdout Scores
        fold_test_score_list = list()

        # Save Holdout Prediction
        holdout_df.index = range(len(holdout_df))
        ood_df = holdout_df.copy()
        pred_cols = [f'pred_{t}' for t in label_list]
        hold_idx = holdout_df.index
        ood_df = pd.concat([ood_df, pd.DataFrame(np.zeros((len(ood_df), len(pred_cols)*2)).astype(np.float32), columns=label_list+pred_cols)], axis=1)
        hold_predictions = list()

    for f in range(config_dict['data_args']['trainval_split_args']['n_folds']):
        
        # Train and Validate             
        bird_model = BirdModule(model, loss, optimizer, scheduler, trainval_df)
        val_preds, val_gts, val_score = train_fold(bird_model, f, trainval_df, config_dict, label_list)

        # Save Validation Predictions and Scores
        val_idx = list(trainval_df[trainval_df['fold'] == f].index)
        oof_df.loc[val_idx, label_list] = val_gts
        oof_df.loc[val_idx, pred_cols] = val_preds
        val_scores.append(val_score)

        if holdout_args['use_holdout']:

            # Holdout Fold Inference
            hold_preds, hold_gts, hold_score = holdout_infer(f, holdout_df, config_dict, label_list)

            # Save Holdout Predictions and Scores
            ood_df.loc[hold_idx, label_list] = hold_gts
            ood_df.loc[hold_idx, pred_cols] = hold_preds
            fold_test_score_list.append(hold_score)
            hold_predictions.append(hold_preds)

    # Print Fold Validation Scores
    for idx, val_score in enumerate(val_scores):
        print(f'Fold {idx} Val Score: {val_score:.5f}')
    oof_df.to_csv(f"{config_dict['output_folder']}/oof_pred.csv", index=False)

    if holdout_args['use_holdout']:

        # Print Fold Holdout Scores
        for idx, hold_score in enumerate(fold_test_score_list):
            print(f'Fold {idx} Holdout Score: {hold_score:.5f}')
        ood_df.to_csv(f"{config_dict['output_folder']}/ood_pred.csv", index=False)

        # Ensemble Holdout Scores
        ensemble_df = holdout_df.copy()
        pred_cols = [f'pred_{t}' for t in label_list]
        ensemble_df = pd.concat([ensemble_df, pd.DataFrame(np.zeros((len(ensemble_df), len(pred_cols)*2)).astype(np.float32), columns=label_list+pred_cols)], axis=1)
        ensemble_predictions = np.mean(hold_predictions, axis=0)
        pred_df = pd.DataFrame(ensemble_predictions, columns=label_list)
        pred_df['id'] = np.arange(len(pred_df))
        gt_df = pd.DataFrame(hold_gts, columns=label_list)
        gt_df['id'] = np.arange(len(gt_df))
        holdout_score = score(gt_df, pred_df, row_id_column_name='id')
        print('Ensemble Score', holdout_score)
        ensemble_df.loc[hold_idx, label_list] = hold_gts
        ensemble_df.loc[hold_idx, pred_cols] = ensemble_predictions
        ensemble_df.to_csv(f"{config_dict['output_folder']}/holdout_ensemble_result.csv", index=False)

    
def train_fold(bird_model, fold_id, df, config_dict, label_list):
    training_args = config_dict['training_args']
    
    print('================================================================')
    print(f"==== Running training for fold {fold_id} ====")
    
    # Dataframes
    train_df = df[df['fold'] != fold_id].copy()
    valid_df = df[df['fold'] == fold_id].copy()
    print(f'Train Samples: {len(train_df)}')
    print(f'Valid Samples: {len(valid_df)}')
    
    # Datasets
    train_ds = BirdDataset(train_df, 'Train')
    val_ds = BirdDataset(valid_df, 'Val')

    # Dataloaders
    train_dl = torch.utils.data.DataLoader(
        train_ds,
        batch_size=training_args['batch_size'],
        shuffle=True,
        num_workers=training_args['n_workers'],
        pin_memory=True,
        persistent_workers=True
    )
    val_dl = torch.utils.data.DataLoader(
        val_ds,
        batch_size=training_args['val_batch_size'],
        shuffle=False,
        num_workers=training_args['n_workers'],
        pin_memory=True,
        persistent_workers=True
    )

    # Checkpointing
    checkpoint_callback = ModelCheckpoint(monitor='val_score',
                                          dirpath=os.path.join(config_dict['output_folder'], 'ckpts'),
                                          save_top_k=1,
                                          save_last=False,
                                          save_weights_only=True,
                                          filename=f"fold_{fold_id}",
                                          mode='max')
    callbacks_to_use = [checkpoint_callback, TQDMProgressBar(refresh_rate=1)]
    
    # == Trainer ==
    os.makedirs(os.path.join(config_dict['output_folder'], 'lightning_logs'), exist_ok=True)
    logger = CSVLogger(save_dir=config_dict['output_folder'], name='lightning_logs')
    trainer = pl.Trainer(
        max_epochs=training_args['epochs'],
        val_check_interval=0.5,
        callbacks=callbacks_to_use,
        enable_model_summary=False,
        accelerator="gpu",
        deterministic=True,
        precision='16-mixed' if config_dict['mixed_precision'] else 32,
        logger=logger
    )
    
    # == Training ==
    trainer.fit(bird_model, train_dataloaders=train_dl, val_dataloaders=val_dl)
    
    # == Validation ==
    best_model_path = checkpoint_callback.best_model_path
    weights = torch.load(best_model_path, map_location=config_dict['device'])['state_dict']
    bird_model.load_state_dict(weights)
    preds, gts = predict(val_dl, bird_model, label_list, config_dict['device'])
    pred_df = pd.DataFrame(preds, columns=label_list)
    pred_df['id'] = np.arange(len(pred_df))
    gt_df = pd.DataFrame(gts, columns=label_list)
    gt_df['id'] = np.arange(len(gt_df))
    val_score = score(gt_df, pred_df, row_id_column_name='id')
    
    # == Save Val Stuff ==
    pred_cols = [f'pred_{t}' for t in label_list]
    valid_df = pd.concat([valid_df.reset_index(), pd.DataFrame(np.zeros((len(valid_df), len(label_list)*2)).astype(np.float32), columns=label_list+pred_cols)], axis=1)
    valid_df[label_list] = gts
    valid_df[pred_cols] = preds
    valid_df.to_csv(f"{config_dict['output_folder']}/pred_df_f{fold_id}.csv", index=False)
    
    return preds, gts, val_score


