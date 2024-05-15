import pandas as pd
import librosa 
import os 
import numpy as np
from tqdm import tqdm
from holdout import create_holdout, save_holdout, holdoudf_2_files
from dupes_config import dupes
pd.options.mode.chained_assignment = None  # default='warn'

def generate_dataframes(config_dict):

    data_dict = config_dict['data_args']

    ##################
    #### METADATA ####
    ##################

    # Read Raw Info
    metadata = pd.read_csv(data_dict['metadata_path'])

    # Create Targets
    metadata['target'] = generate_ordinal_targets(metadata)

    # Add Filepaths
    metadata['filepath'] = data_dict['metadata_path'].split('/train')[0] + '/train_audio/' + metadata.filename
    metadata['filename'] = metadata['filename'].apply(lambda x: x.split('.')[0].replace('/', '-'))

    # Add Durations
    print('> Computing All Audio Durations...')
    metadata['Duration'] = metadata.apply(lambda x: librosa.get_duration(path=x['filepath']), axis=1)

    # Reorder 
    metadata = metadata[['filename', 'primary_label', 'secondary_labels', 'target', 'filepath', 'Duration', 'rating']]

    # Remove Duplicates
    files_to_remove = [file_tuple[0].replace('/', '-').split('.')[0] for file_tuple in dupes]
    print('> Removing', len(files_to_remove), 'duplicates...')
    mask = metadata['filename'].map(lambda x: all(word not in x for word in files_to_remove))
    metadata = metadata[mask].reset_index(drop=True)

    ################
    ### HOLDOUT ####
    ################

    holdout_args = data_dict['holdout_args']
    if holdout_args['use_holdout']:
        print('> Using Holdout!')

        if holdout_args['load_holdout']:

            # Load Holdout
            print('> Loading Holdout at Path:', holdout_args['loading_holdout_path'])
            holdout_list = pd.read_csv(holdout_args['loading_holdout_path'])

        else:
            # Create Holdout
            holdout_list = create_holdout(metadata, holdout_args['k'], holdout_args['percent'], holdout_args['rating_leq_than'])
     
        # Create Dfs
        holdout_df = metadata[metadata['filename'].isin(holdout_list['filename'])].reset_index(drop=True)
        trainval_df = metadata[~metadata['filename'].isin(holdout_list['filename'])].reset_index(drop=True)

        # Preprocess Features
        print('> Processing Holdout...')
        holdout_df = holdoudf_2_files(holdout_df, os.path.join(config_dict['output_folder'], 'holdout_features'))

        # Save DF
        print('> Saving Holdout...')
        save_holdout(holdout_df, data_dict, config_dict['output_folder'])
        print('> #Holdout Samples: ', len(holdout_df))
        print('> #Training Samples:', len(trainval_df))
    else:
        print('> NOT Using Holdout:')
        trainval_df = metadata
        holdout_df = None
        print('> #Training Samples:', len(trainval_df))

    return trainval_df, holdout_df


def generate_ordinal_targets(raw_metadata):
    classes = sorted(raw_metadata['primary_label'].unique())
    label_ids = list(range(len(classes)))
    label2id = dict(zip(classes, label_ids))
    labels = raw_metadata.primary_label.map(label2id)
    return labels

def locate_training_files(trainval_df, trainval_feat_path, features):
    assert features in ['images', 'audios', 'spectrograms']
    print('> Locating Training Files that are on TrainVal Dataframe...')
    filelist = os.listdir(trainval_feat_path + '/' + features)
    fls = pd.DataFrame({'filename': filelist })
    cols = {col: [] for col in trainval_df.columns}

    # Iterate through the list of files on path and get info from trainval dataframe
    for idx in tqdm(range(len(fls))):
        info = fls.iloc[idx]['filename'].split("_SEG")[0]
        row = trainval_df[trainval_df['filename'] == info]
        for coluna in cols.keys():
            if coluna == 'filename':
                cols[coluna].append(fls.iloc[idx]['filename'].split('.')[0])
                continue
            if coluna == 'filepath':
                cols[coluna].append(trainval_feat_path + '/' + features + '/' + fls.iloc[idx]['filename'])
                continue
            if coluna == 'Duration':
                wav = np.load(trainval_feat_path + '/audios/' + fls.iloc[idx]['filename'])
                cols[coluna].append(librosa.get_duration(y=wav, sr=32000))
                continue
            cols[coluna].append(row[coluna].item())
    return pd.DataFrame(cols)

