import argparse
import yaml
import os 
import pytorch_lightning as pl
from dataframes import generate_dataframes, locate_training_files
from dataset import BirdDataset
from lightning import BirdModule
from loss import get_loss
from model import get_model
from optimizer import get_optimizer
from preprocess import preprocess_train
from splits import add_split_info
from scheduler import get_scheduler
from trainer import run_training
from visualize import save_example_batch

def main(args):
    
    # LOAD CONFIG
    with open(args.config_path, 'r') as file:
        config_dict = yaml.safe_load(file)

    # PARSE ARGS
    data_args = config_dict['data_args']
    preprocess_train_args = config_dict['preprocess_train_args']
    training_args = config_dict['training_args']
    split_args = data_args['trainval_split_args']
    model_args = config_dict['model_args']

    # Set Seed
    pl.seed_everything(config_dict['seed'], workers=True)

    # Create Save Dir
    os.makedirs(config_dict['output_folder'])
    with open(os.path.join(config_dict['output_folder'], 'config.yaml'), "w") as file:
        yaml.dump(config_dict, file, default_flow_style=False)        
        
    # Dataframes
    trainval_df, holdout_df = generate_dataframes(config_dict)
    
    # Preprocess/Load
    if preprocess_train_args['run_preprocess']: 
        trainval_feat_path = preprocess_train(trainval_df, preprocess_train_args, config_dict['output_folder'])
    else:
        trainval_feat_path = preprocess_train_args['loading_trainval_feats_path']
    trainval_df = locate_training_files(trainval_df, trainval_feat_path, training_args['input_feature'])

    # Splits
    if trainval_df['filename'].str.endswith('SEG1').any():
        assert  'group' in split_args['type'], 'Use grouped split to avoid audio segment leakage.'
    trainval_df = add_split_info(trainval_df, split_args)
    trainval_df.to_csv(path_or_buf = os.path.join(config_dict['output_folder'], 'trainval_df.csv'), index=False)

    # View Batch
    if training_args['input_feature'] == 'images' and config_dict['save_example_batch']:
        save_example_batch(BirdDataset(trainval_df, 'Train'), config_dict['output_folder'])
    
    # Model
    model = get_model(model_args, n_classes=trainval_df['primary_label'].nunique())

    # Optimizer
    optimizer = get_optimizer(model, training_args)

    # Learning Rate Scheduler
    scheduler = get_scheduler(optimizer, training_args)
    
    # Loss
    loss = get_loss(training_args)

    # Run  
    run_training(model, optimizer, scheduler, loss, config_dict, trainval_df, holdout_df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, help="A path to a .json containing all experiment parameters", required=True)
    args = parser.parse_args()
    main(args)