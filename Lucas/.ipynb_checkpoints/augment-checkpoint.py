from torch import nn
import torch
from torch.distributions import Beta
import numpy as np
import random

# Sum an background soung to the wave
class BackgroundAugmentation(object):
    def __init__(self, min_scale, max_scale, back_df, mode = 'train'):
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.backgrounds = back_df
        self.mode = mode

    def __call__(self, waveform):
        numero_aleatorio = random.randint(0, len(self.backgrounds)-1)
        noise_path = self.backgrounds.wav_path.values[numero_aleatorio]
        noise, sample_rate = librosa.load(noise_path, sr = None)
        # noise = np.concatenate([noise]*3) #if training with 15sec
            
        rand_scale = random.uniform(self.min_scale, self.max_scale)
#         print(rand_scale)
        noisy_speech = (rand_scale * waveform + noise) / (1+rand_scale)
#         print(waveform.mean(), noisy_speech.mean())
        return noisy_speech
        
# Get random wav from df with labels and return the sumixup wave and labels
class SumMixUp(object):
    def __init__(self, df, labels, min_pct= 0.3, max_pct=1):
        self.df = df
        self.labels = labels
        self.min_pct = min_pct
        self.max_pct = max_pct

    def __call__(self, waveform, label):
        idx = random.randint(0, len(self.df)-1)
        noise_path = self.df.wav_path.values[idx]
        random_wav, sample_rate = librosa.load(noise_path, sr = None)
        random_label = self.labels[idx]

        rand1 = random.uniform(0, self.max_pct - self.min_pct) + self.min_pct
        rand2 = random.uniform(0, self.max_pct - self.min_pct) + self.min_pct


        ridx = random.randint(0, 5) # Works because of 6 * 5sec chunks
        waveform = rand1*waveform + rand2*random_wav[ridx*32000:(ridx+5)*32000]
        
        if rand1 >= 0.5:
            rand1 = 1
        else:
            rand1 = 1 - 2*(0.5 - rand1)

        if rand2 >= 0.5:
            rand2 = 1
        else:
            rand2 = 1 - 2*(0.5 - rand2)

        # print(rand1, rand2)
        
        label = rand1*label + rand2*random_label

        # print(label)
        return waveform , np.clip(label, 0 ,1)

# mixup in spec level
class Mixup(nn.Module):
    def __init__(self, mix_beta):

        super(Mixup, self).__init__()
        self.beta_distribution = Beta(mix_beta, mix_beta)

    def forward(self, X, Y, weight=None):

        bs = X.shape[0]
        n_dims = len(X.shape)
        perm = torch.randperm(bs)
        coeffs = self.beta_distribution.rsample(torch.Size((bs,))).to(X.device)

        if n_dims == 2:
            X = coeffs.view(-1, 1) * X + (1 - coeffs.view(-1, 1)) * X[perm]
        elif n_dims == 3:
            X = coeffs.view(-1, 1, 1) * X + (1 - coeffs.view(-1, 1, 1)) * X[perm]
        else:
            X = coeffs.view(-1, 1, 1, 1) * X + (1 - coeffs.view(-1, 1, 1, 1)) * X[perm]

        Y = coeffs.view(-1, 1) * Y + (1 - coeffs.view(-1, 1)) * Y[perm]

#         print(coeffs, perm)
        if weight is None:
            return X, Y
        else:
            weight = coeffs.view(-1) * weight + (1 - coeffs.view(-1)) * weight[perm]
            return X, Y, weight

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

# cut mix in spec level
class Cutmix(nn.Module):
    def __init__(self, mix_beta):

        super(Cutmix, self).__init__()
        self.beta_distribution = Beta(mix_beta, mix_beta)

    def forward(self, X, Y, weight=None):

        bs = X.shape[0]
        n_dims = len(X.shape)
        perm = torch.randperm(bs)
        coeffs = self.beta_distribution.rsample((1, )).to(X.device)

        assert n_dims == 4

        bbx1, bby1, bbx2, bby2 = rand_bbox(X.size(), coeffs[0].item())

        X[:, :, bbx1:bbx2, bby1:bby2] = X[perm, :, bbx1:bbx2, bby1:bby2]

        coeffs = torch.Tensor([1 - ((bbx2 - bbx1) * (bby2 - bby1) / (X.size()[-1] * X.size()[-2]))]).to(X.device)

        Y = coeffs.view(-1, 1) * Y + (1 - coeffs.view(-1, 1)) * Y[perm]

        if weight is None:
            return X, Y
        else:
            weight = coeffs.view(-1) * weight + (1 - coeffs.view(-1)) * weight[perm]
            return X, Y, weight