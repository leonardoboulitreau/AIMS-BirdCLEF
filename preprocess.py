from tqdm import tqdm
import librosa
import numpy as np
import os 
from scipy import signal as sci_signal
import cv2
import math

def preprocess_train(trainval_df, preprocess_train_args, output_folder):
    print('> Running Preprocessing of Train Files')
    trainval_feat_path = os.path.join(output_folder, 'trainval_feats')
    os.makedirs(trainval_feat_path, exist_ok=True)

    for i, row_metadata in tqdm(trainval_df.iterrows()):

        ###############
        #### Audio ####
        ###############

        # Load
        audio_data, sr = librosa.load(row_metadata['filepath'], sr=None)

        # Compute Windows
        file_windows = []
        if preprocess_train_args['audio_window'] == 'center':
            n_copy = math.ceil(5 * sr / len(audio_data))
            if n_copy > 1: audio_data = np.concatenate([audio_data]*n_copy)
            start_idx = int(len(audio_data) / 2 - 2.5 * sr)
            end_idx = int(start_idx + 5.0 * sr)
            audio_data = audio_data[start_idx:end_idx]
            file_windows.append(audio_data)
        elif preprocess_train_args['audio_window'] == 'slicing':
            raise NotImplementedError
        else:
            raise NotImplementedError

        # Audio Processing (For each window: filtering, denoising, augmentation, etc)
        
        # Save Audios if wanted
        if 'audio' in preprocess_train_args['output_feature_types']:
            audios_path = os.path.join(trainval_feat_path, 'audios')
            os.makedirs(audios_path, exist_ok=True)
            for idx in range(len(file_windows)):
                np.save(audios_path + '/' + row_metadata['filename'] + '_SEG' + str(idx) + '.npy', file_windows[idx])


        #####################
        #### Spectrogram ####
        #####################

        # Spectrogram Processing
        if 'spectrogram' in preprocess_train_args['output_feature_types']:
            spectrograms = []
            for window in file_windows:
                mean_signal = np.nanmean(window)
                window = np.nan_to_num(window, nan=mean_signal) if np.isnan(window).mean() < 1 else np.zeros_like(window)
                
                # Compute Spec
                frequencies, times, spec_data = sci_signal.spectrogram(
                    window, 
                    fs=sr, 
                    nfft= preprocess_train_args['spec_n_fft'], 
                    nperseg= preprocess_train_args['spec_win_len'], 
                    noverlap= preprocess_train_args['spec_hop_len'], 
                    window= preprocess_train_args['spec_win_type']
                )
                
                # Cut Frequencies
                valid_freq = (frequencies >= preprocess_train_args['spec_cut_min_freq']) & (frequencies <= preprocess_train_args['spec_cut_max_freq'])
                spec_data = spec_data[valid_freq, :]
                
                # Turn to Log
                spec_data = np.log10(spec_data + 1e-20)
                
                # Normalize
                spec_data = spec_data - spec_data.min()
                spec_data = spec_data / spec_data.max()
                spectrograms.append(spec_data)

            # Save Spectrograms if wanted
            specs_path = os.path.join(trainval_feat_path, 'spectrograms')
            os.makedirs(specs_path, exist_ok=True)
            for idx in range(len(spectrograms)):
                np.save(specs_path + '/' + row_metadata['filename'] + '_SEG' + str(idx) + '.npy', spectrograms[idx])

        ###############
        #### Image ####
        ###############

        # Image Processing
        if 'image' in preprocess_train_args['output_feature_types']:
            images = []
            for spec in spectrograms:
                spec_data = cv2.resize(spec, (256, 256), interpolation=cv2.INTER_AREA)
                images.append(spec_data)

            # Save Images if wanted
            
            images_path = os.path.join(trainval_feat_path, 'images')
            os.makedirs(images_path, exist_ok=True)
            for idx in range(len(images)):
                np.save(images_path + '/' + row_metadata['filename'] + '_SEG' + str(idx) + '.npy', images[idx])

    return trainval_feat_path

