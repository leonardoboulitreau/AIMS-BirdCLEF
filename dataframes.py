import pandas as pd
import librosa 
import os 
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import KFold, StratifiedKFold
pd.options.mode.chained_assignment = None  # default='warn'

def generate_dataframes(data_dict):

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
    files_to_remove = [file_tuple[0] for file_tuple in dupes]
    mask = metadata['filename'].map(lambda x: all(word not in x for word in files_to_remove))
    metadata = metadata[mask]

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

def create_holdout(metadata, k, percent, rating_leq_than):
    print('> Computing Holdout for the parameters:', (k, percent, rating_leq_than))
    total_duration_per_species = metadata.groupby('primary_label')['Duration'].sum().sort_values()
    bottom_k_species = total_duration_per_species.sort_values().head(k)
    holdout_bottom_kspecies_vol = bottom_k_species * percent
    holdout_list = pd.DataFrame(columns=metadata.columns)
    n_audios = []
    n_audios_taken = []
    for species, total_duration in holdout_bottom_kspecies_vol.items():
        species_audios = metadata[metadata['primary_label'] == species].sort_values(by='Duration')
        n_audios.append(len(species_audios))
        specie_dur = 0
        n = 0
        for idx, row in species_audios.iterrows():
            if specie_dur >= total_duration:
                break
            else:
                if row['rating'] <= rating_leq_than:
                    holdout_list = pd.concat([holdout_list, row.to_frame().transpose()])
                    specie_dur += librosa.get_duration(path=row.filepath)
                    n+=1
                else:
                    pass
        n_audios_taken.append(n)
    holdout_list.reset_index(drop=True, inplace=True)
    holdout_list = holdout_list[['filename']]
    return holdout_list

def add_split_info(trainval_df, trainval_split_args):
    if trainval_split_args['type'] == 'kfold':
        kf = KFold(n_splits=trainval_split_args['n_folds'], shuffle=trainval_split_args['shuffle'], random_state = trainval_split_args['seed'])
        trainval_df['fold'] = 0
        for fold, (train_idx, val_idx) in enumerate(kf.split(trainval_df)):
            trainval_df.loc[val_idx, 'fold'] = fold
        for fold in range(trainval_split_args['n_folds']):
            print('> Fold', fold, 'has #', len(trainval_df[trainval_df['fold'] == fold]), 'samples.')

    elif trainval_split_args['type'] == 'stratifiedkfold':
        kf = StratifiedKFold(n_splits=trainval_split_args['n_folds'], shuffle=trainval_split_args['shuffle'], random_state = trainval_split_args['seed'])
        trainval_df['fold'] = 0
        for fold, (train_idx, val_idx) in enumerate(kf.split(trainval_df, y=trainval_df['primary_label'])):
            for idx in val_idx:
                trainval_df.iloc[idx, trainval_df.columns.get_loc('fold') ] = fold
        for fold in range(trainval_split_args['n_folds']):
            print('> Fold', fold, 'has #', len(trainval_df[trainval_df['fold'] == fold]), 'samples.')
    else:
        raise NotImplementedError
    return trainval_df
    
def save_holdout(df, data_args, path):
    holdout_args = data_args['holdout_args']
    df.to_csv(path_or_buf=  path + '/holdout_'+str(holdout_args['k'])+'bottom_'+str(holdout_args['percent']*100)+'%_' + 'ratleq' + str(holdout_args['rating_leq_than']) + '_.csv')
    return None

def locate_training_files(trainval_df, trainval_feat_path, features):
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
