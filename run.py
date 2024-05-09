import argparse
import json
import os 
import torch
from metrics import score
from preprocess import preprocess_train
from dataframes import generate_dataframes, save_holdout, locate_training_files

def main(args):
    
    # LOAD CONFIG FROM .JSON
    with open(args.config_path, 'r') as file:
        config_dict = json.load(file)

    # PARSE SPECIFIC ARGS
    data_args = config_dict['data_args']
    preprocess_train_args = config_dict['preprocess_train_args']
    training_args = config_dict['training_args']

    # SET DEVICE
    device = torch.device("cuda:0" if (config_dict['device'] == 'cuda' and torch.cuda.is_available()) else "cpu")

    # CREATE OUTPUT FOLDER
    os.makedirs(config_dict['output_folder'])

    # DATAFRAMES
    trainval_df, holdout_df = generate_dataframes(data_args)
    if data_args['holdout_args']['use_holdout']:
        print('> Saving Holdout...')
        save_holdout(holdout_df, data_args, config_dict['output_folder'])
    
    # LOCATE TRAIN FILES (Updates trainval dataframe, in case of multiple slices per audio)
    if preprocess_train_args['run_preprocess']:
        # Preprocess and Locate Train Files 
        trainval_feat_path = preprocess_train(trainval_df, preprocess_train_args, config_dict['output_folder'])
        trainval_df = locate_training_files(trainval_df, trainval_feat_path, training_args['features'])
    else:
        # Locate Train Files    
        trainval_df = locate_training_files(trainval_df, preprocess_train_args['loading_trainval_feats_path'], training_args['features']) 

    print(trainval_df)
    # TRAINVAL SPLITS
    # if windowed assert grouped or grouped kfold to avoid vaazemtno
    # trainval_split_args = data_dict['trainval_split_args']
    # trainval_df = add_split_info(trainval_df, trainval_split_args)

    # DATASET

    # DATALOADER
    
    # MODEL

    # TRAINING, VALIDATION, HOLDOUT LOOP

    # FINAL MODEL / ENSEMBLE HOLDOUT


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, help="A path to a .json containing all experiment parameters", required=True)
    args = parser.parse_args()
    main(args)