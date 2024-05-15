import pandas as pd
import librosa
import numpy as np
import time
import cv2
import soundfile as sf
from scipy import signal as sci_signal
from lightning import BirdModule
from model import get_model
import torch
from dataset import BirdDataset
from model import predict
from metrics import score
import os

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

def save_holdout(df, data_args, path):
    holdout_args = data_args['holdout_args']
    df.to_csv(path_or_buf=  path + '/holdout_'+str(holdout_args['k'])+'bottom_'+str(holdout_args['percent']*100)+'%_' + 'ratleq' + str(holdout_args['rating_leq_than']) + '_.csv')
    return None

def holdoudf_2_files(holdout_df, folder):
    os.makedirs(folder, exist_ok=True)
    times = []
    df = pd.DataFrame(columns=holdout_df.columns)
    for idx in range(len(holdout_df)):
        start_time = time.time()
        row = holdout_df.iloc[idx]
        filepath = row['filepath']
        audio_data, sr = librosa.load(filepath, sr=None)
        spec = oog2spec_via_scipy(audio_data, sr)
        pad = 512 - (spec.shape[1] % 512)
        if pad > 0:
            spec = np.pad(spec, ((0,0), (0,pad)))
        spec = spec.reshape(512,-1,512).transpose([0, 2, 1])
        spec = cv2.resize(spec, (256, 256), interpolation=cv2.INTER_AREA)
        end_time = time.time()
        times.append(end_time - start_time)
        
        # Save Files and Updated Df
        if len(spec.shape) == 3:
            for j in range(spec.shape[-1]):
                np.save( folder + '/' + row['filename'].split('_')[0] + '_' + f'{(j+1)*5}' + '.npy', spec[:, :, j])
                new_row = pd.DataFrame([{
                    'filename': row['filename'],
                    'primary_label': row['primary_label'],
                    'secondary_labels': row['secondary_labels'],
                    'target': row['target'],
                    'filepath': folder + '/' + row['filename'].split('_')[0] + '_' + f'{(j+1)*5}' + '.npy',
                    'Duration': row['Duration'],
                    'rating': row['rating']
                }])
                df = pd.concat([df, new_row], ignore_index=True)
        else:
            j=0
            np.save(folder + '/' + row['filename'].split('_')[0] + '_' + f'{(j+1)*5}' + '.npy', spec[:, :])
            new_row = pd.DataFrame([{
                    'filename': row['filename'],
                    'primary_label': row['primary_label'],
                    'secondary_labels': row['secondary_labels'],
                    'target': row['target'],
                    'filepath': folder + '/' + row['filename'].split('_')[0] + '_' + f'{(j+1)*5}' + '.npy',
                    'Duration': row['Duration'],
                    'rating': row['rating']
                }])
            df = pd.concat([df, new_row], ignore_index=True)
    print('> Average time to process a holdout file is', np.mean(times))
    return df

def holdout_infer(f, holdout_df, config_dict, label_list):
    bird_model = BirdModule(get_model(config_dict['model_args'], n_classes=len(label_list)))
    weights = torch.load(os.path.join(config_dict['output_folder'], 'ckpts', 'fold_' + str(f) + '.ckpt'), map_location=torch.device('cpu'))['state_dict']
    bird_model.load_state_dict(weights)    
    test_dataset = BirdDataset(holdout_df, 'Holdout')
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config_dict['training_args']['val_batch_size'],
        shuffle=False,
        num_workers=config_dict['training_args']['n_workers'],
        pin_memory=True,
        persistent_workers=True
    )
    preds, gts = predict(test_loader, bird_model, label_list, config_dict['device'])
    pred_df = pd.DataFrame(preds, columns=label_list)
    pred_df['id'] = np.arange(len(pred_df))
    gt_df = pd.DataFrame(gts, columns=label_list)
    gt_df['id'] = np.arange(len(gt_df))
    holdout_score = score(gt_df, pred_df, row_id_column_name='id')
    pred_cols = [f'pred_{t}' for t in label_list]
    holdout_df = pd.concat([holdout_df.reset_index(), pd.DataFrame(np.zeros((len(holdout_df), len(label_list)*2)).astype(np.float32), columns=label_list+pred_cols)], axis=1)
    holdout_df[label_list] = gts
    holdout_df[pred_cols] = preds
    holdout_df.to_csv(f"{config_dict['output_folder']}/holdout_pred_df_f{f}.csv", index=False)
    return preds, gts, holdout_score

def oog2spec_via_scipy(audio_data, sr):
    # handles NaNs
    mean_signal = np.nanmean(audio_data)
    audio_data = np.nan_to_num(audio_data, nan=mean_signal) if np.isnan(audio_data).mean() < 1 else np.zeros_like(audio_data)
    
    # to spec.
    frequencies, times, spec_data = sci_signal.spectrogram(
        audio_data, 
        fs=sr, 
        nfft=1095, 
        nperseg=412, 
        noverlap=100, 
        window='hann'
    )
    
    # Filter frequency range
    valid_freq = (frequencies >= 40) & (frequencies <= 15000)
    spec_data = spec_data[valid_freq, :]
    
    # Log
    spec_data = np.log10(spec_data + 1e-20)
    
    # min/max normalize
    spec_data = spec_data - spec_data.min()
    spec_data = spec_data / spec_data.max()
    
    return spec_data