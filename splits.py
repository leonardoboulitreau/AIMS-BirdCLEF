from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold, StratifiedGroupKFold


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

    elif trainval_split_args['type'] == 'groupkfold':

        # Initial group and prefix
        generate_group.group = 0
        generate_group.prefix = ''
        trainval_df['group'] = trainval_df.apply(generate_group, axis=1)
        print(trainval_df)

        kf = GroupKFold(n_splits=trainval_split_args['n_folds'])
        
        trainval_df['fold'] = 0
        for fold, (train_idx, val_idx) in enumerate(kf.split(trainval_df, y=trainval_df['primary_label'], groups=trainval_df['group'].to_list())):
            for idx in val_idx:
                trainval_df.iloc[idx, trainval_df.columns.get_loc('fold') ] = fold
        for fold in range(trainval_split_args['n_folds']):
            print('> Fold', fold, 'has #', len(trainval_df[trainval_df['fold'] == fold]), 'samples.')
    
    elif trainval_split_args['type'] == 'groupstratifiedkfold':
        # Initial group and prefix
        generate_group.group = 0
        generate_group.prefix = ''
        trainval_df['group'] = trainval_df.apply(generate_group, axis=1)

        kf = StratifiedGroupKFold(n_splits=trainval_split_args['n_folds'], shuffle=trainval_split_args['shuffle'], random_state = trainval_split_args['seed'])
        trainval_df['fold'] = 0
        for fold, (train_idx, val_idx) in enumerate(kf.split(trainval_df, y=trainval_df['primary_label'], groups=trainval_df['group'].to_list())):
            for idx in val_idx:
                trainval_df.iloc[idx, trainval_df.columns.get_loc('fold') ] = fold
        for fold in range(trainval_split_args['n_folds']):
            print('> Fold', fold, 'has #', len(trainval_df[trainval_df['fold'] == fold]), 'samples.')
    else:
        raise NotImplementedError
    return trainval_df
    
def generate_group(row):
    prefix = row['filename'].split('_SEG')[0]
    if generate_group.prefix != prefix:
        generate_group.prefix = prefix
        generate_group.group += 1
    return generate_group.group