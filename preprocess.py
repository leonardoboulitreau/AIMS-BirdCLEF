from tqdm import tqdm
import librosa
import numpy as np
import os 
from scipy import signal as sci_signal
import cv2
import math
import pandas as pd

def preprocess_train(trainval_df, preprocess_train_args, output_folder):
    print('> Running Preprocessing of Train Files')
    trainval_feat_path = os.path.join(output_folder, 'trainval_feats')
    os.makedirs(trainval_feat_path, exist_ok=True)

    for i, row_metadata in tqdm(trainval_df.iterrows()):

        ###############
        #### Audio ####
        ###############

        # Compute Windows
        audio_windows = generate_audio_windows(row_metadata['filepath'], preprocess_train_args)

        # Audio Processing 
        audio_windows = process_audio_windows(audio_windows, preprocess_train_args)
        
        # Save Audios
        if 'audio' in preprocess_train_args['output_feature_types']:
            audios_path = os.path.join(trainval_feat_path, 'audios')
            os.makedirs(audios_path, exist_ok=True)
            for idx in range(len(audio_windows)):
                np.save(audios_path + '/' + row_metadata['filename'] + '_SEG' + str(idx) + '.npy', audio_windows[idx])


        #####################
        #### Spectrogram ####
        #####################

        # Spectrogram Processing
        if 'spectrogram' in preprocess_train_args['output_feature_types']:
            spectrograms = generate_spectrograms(audio_windows, preprocess_train_args)

            # Save Spectrograms
            specs_path = os.path.join(trainval_feat_path, 'spectrograms')
            os.makedirs(specs_path, exist_ok=True)
            for idx in range(len(spectrograms)):
                np.save(specs_path + '/' + row_metadata['filename'] + '_SEG' + str(idx) + '.npy', spectrograms[idx])

        ###############
        #### Image ####
        ###############

        if 'image' in preprocess_train_args['output_feature_types']:

            # Image Processing
            images = generate_images(spectrograms, preprocess_train_args)            

            # Save Images
            images_path = os.path.join(trainval_feat_path, 'images')
            os.makedirs(images_path, exist_ok=True)
            for idx in range(len(images)):
                np.save(images_path + '/' + row_metadata['filename'] + '_SEG' + str(idx) + '.npy', images[idx])

    return trainval_feat_path


def generate_audio_windows(path, preprocess_train_args):
    audio_data, sr = librosa.load(path, sr=preprocess_train_args['sampling_rate'])
    win_dur = preprocess_train_args['window_dur']
    file_windows = []
    if 'center' in preprocess_train_args['audio_window']:
        n_copy = math.ceil(win_dur * sr / len(audio_data))
        if n_copy > 1: audio_data = np.concatenate([audio_data]*n_copy)
        start_idx = int(len(audio_data) / 2 - 2.5 * sr)
        end_idx = int(start_idx + win_dur * sr)
        audio_data = audio_data[start_idx:end_idx]
        file_windows.append(audio_data)
    if 'left' in preprocess_train_args['audio_window']:
        n_copy = math.ceil(win_dur * sr / len(audio_data))
        if n_copy > 1: audio_data = np.concatenate([audio_data]*n_copy)
        start_idx = 0
        end_idx = int(start_idx + win_dur * sr)
        audio_data = audio_data[start_idx:end_idx]
        file_windows.append(audio_data)
    if 'right' in preprocess_train_args['audio_window']:
        n_copy = math.ceil(win_dur * sr / len(audio_data))
        if n_copy > 1: audio_data = np.concatenate([audio_data]*n_copy)
        start_idx = int(len(audio_data) - win_dur * sr)
        end_idx = int(start_idx + win_dur * sr)
        audio_data = audio_data[start_idx:end_idx]
        file_windows.append(audio_data)
    if 'slicing' in preprocess_train_args['audio_window']:
        raise NotImplementedError
    
    return file_windows

def process_audio_windows(audio_windows, preprocess_train_args):
    return audio_windows

def generate_spectrograms(audio_windows, preprocess_train_args):
    spectrograms = []
    for window in audio_windows:
        mean_signal = np.nanmean(window)
        window = np.nan_to_num(window, nan=mean_signal) if np.isnan(window).mean() < 1 else np.zeros_like(window)
        
        # Compute Spec
        frequencies, times, spec_data = sci_signal.spectrogram(
            window, 
            fs=preprocess_train_args['sampling_rate'], 
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
    return spectrograms

def generate_images(spectrograms, preprocess_train_args):
    images = []
    for spec in spectrograms:
        spec_data = cv2.resize(spec, (preprocess_train_args['img_size'], preprocess_train_args['img_size']), interpolation=cv2.INTER_AREA)
        images.append(spec_data)
    return images