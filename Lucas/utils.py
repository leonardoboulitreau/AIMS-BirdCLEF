import torch
import numpy as np
from torch.utils.data import DataLoader,random_split
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
import librosa

def _pad_tensor(x, length):
    _pad = 0.0
    assert x.ndim == 2
    x = np.pad(x, [[0, 0], [0, length - x.shape[1]]], mode="constant", constant_values=_pad)
    return x


def prepare_tensor(inputs, out_steps=1):
    max_len = max((x.shape[1] for x in inputs))
    remainder = max_len % out_steps
    pad_len = max_len + (out_steps - remainder) if remainder > 0 else max_len
    return np.stack([_pad_tensor(x, pad_len) for x in inputs])


class SimpleDataset(Dataset):
    def __init__(self, df, segment_size = 5, 
                 root_path= '/kaggle/input/birdclef-2024/train_audio/',
                 n_mels = 80,
                 n_fft = 1024,
                 hop_len = 256,
                 fmin = 0,
                 fmax = 16000):
        self.df = df
        self.segment_size = segment_size # Size of random slices in seconds
        self.root_data_path = root_path
        self.n_mels = n_mels       
        self.n_fft = n_fft
        self.hop_len = hop_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        y, sr = librosa.load(self.root_data_path + self.df.filename.values[idx], sr=None)
        y = np.asarray(y, dtype=np.float32)
        
        L_max = len(y)
        if(L_max > self.segment_size*sr):
            R_start = np.random.randint(0,L_max - self.segment_size*sr)
            y_ = y[R_start:R_start+self.segment_size*sr]
        else:
            y_ = y
            
        class_id = self.df.labels.values[idx]

        melspec = librosa.feature.melspectrogram(y=y_,sr=sr,n_fft=self.n_fft,win_length=self.n_fft,
                                                 hop_length=self.hop_len,n_mels=self.n_mels).astype("float32")
        melspec = librosa.power_to_db(melspec ** 2, ref=np.max)
        
        return melspec, class_id

def my_collate_fn(batch):

    melspec = [x[0] for x in batch]
    label = [x[1] for x in batch]

    melspec = prepare_tensor(melspec)

    return torch.FloatTensor(melspec), torch.LongTensor(label)


class ReferenceEncoderClassifier(nn.Module):
    """NN module creating a fixed size prosody embedding from a spectrogram.

    inputs: mel spectrograms [batch_size, num_spec_frames, num_mel]
    outputs: [batch_size, embedding_dim]
    """

    def __init__(self, num_mel, embedding_dim, num_classes, use_nonlinear_proj = False):

        super().__init__()
        self.num_mel = num_mel
        filters = [1] + [32, 32, 64, 64, 128, 128]
        num_layers = len(filters) - 1
        convs = [
            nn.Conv2d(
                in_channels=filters[i], out_channels=filters[i + 1], kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)
            )
            for i in range(num_layers)
        ]
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList([nn.BatchNorm2d(num_features=filter_size) for filter_size in filters[1:]])

        post_conv_height = self.calculate_post_conv_height(num_mel, 3, 2, 1, num_layers)
        self.recurrence = nn.GRU(
            input_size=filters[-1] * post_conv_height, hidden_size=embedding_dim, batch_first=True
        )

        self.dropout = nn.Dropout(p=0.5)

        self.use_nonlinear_proj = use_nonlinear_proj

        if(self.use_nonlinear_proj):
            self.proj = nn.Linear(embedding_dim, embedding_dim)
            nn.init.xavier_normal_(self.proj.weight) # Good init for projection
            # self.proj.bias.data.zero_() # Not random bias to "move" z

        self.classifier_layer = nn.Linear(embedding_dim, num_classes)

    def forward(self, inputs):
        batch_size = inputs.size(0)
        x = inputs.view(batch_size, 1, -1, self.num_mel)
        # x: 4D tensor [batch_size, num_channels==1, num_frames, num_mel]
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x)
            x = bn(x)
            x = F.relu(x)

        x = x.transpose(1, 2)
        # x: 4D tensor [batch_size, post_conv_width,
        #               num_channels==128, post_conv_height]
        post_conv_width = x.size(1)
        x = x.contiguous().view(batch_size, post_conv_width, -1)
        # x: 3D tensor [batch_size, post_conv_width,
        #               num_channels*post_conv_height]
        self.recurrence.flatten_parameters()
        _, out = self.recurrence(x)
        # out: 3D tensor [seq_len==1, batch_size, encoding_size=128]

        if(self.use_nonlinear_proj):
            out = torch.tanh(self.proj(out))
            out = self.dropout(out)

        out = self.classifier_layer(out)

        return out.squeeze(0)

    @staticmethod
    def calculate_post_conv_height(height, kernel_size, stride, pad, n_convs):
        """Height of spec after n convolutions with fixed kernel/stride/pad."""
        for _ in range(n_convs):
            height = (height - kernel_size + 2 * pad) // stride + 1
        return height
    

from tqdm import tqdm
def train(model, train_loader, val_loader, epochs = 100, eval_epochs = 5, lr = 0.001, use_cuda = True, save_model_path = 'baseline'):
    if(use_cuda):
        model_ = model.cuda()
    else:
        return 'If you are really wanting to train it on CPU code it itself!'

    optimizer = optim.Adam(model_.parameters(), lr=lr)

    # Defining losses
    ce = nn.CrossEntropyLoss()
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []

    best_loss = 9999

    niter = epochs

    print(f"Start training with {niter} epochs: \n")
    iter_ = 0
    for _ in range(0, niter):
        model_.train()
        print(f'EPOCH {_}/{niter}')
        acc_ = 0
        acc_b = 0
        acc_gd = 0
        acc_gb = 0
        batch_len = 0
        count_upt = 0

        for batch in tqdm(train_loader):

            melspec = batch[0].cuda()
            labels = batch[1].cuda()
            optimizer.zero_grad()
            predictions = model_(melspec)

            loss = ce(predictions, labels)
            train_loss.append(loss.item())
            acc_ += (predictions.argmax(axis = -1).cpu() == labels.cpu()).sum()

            # Total loss and opt step
            loss.backward()
            optimizer.step()

            batch_len += len(labels)
            iter_+=1

            if iter_ % 5000 == 0:
            #       if(loss < best_loss):
            #         print("Saving best model")
                torch.save(model_.state_dict(), f'{save_model_path}/checkpoint_{iter_}.pth')
            #         best_loss = loss
            #         count_upt += 1



        train_acc.append(acc_/batch_len)

        if(_%eval_epochs == 0):
            model_.eval()
            acc_ = 0
            batch_len_ = 0
            eval_loss = 0
            for batch in tqdm(val_loader):

                melspec = batch[0].cuda()
                labels = batch[1].cuda()
                optimizer.zero_grad()
                predictions = model_(melspec)
                batch_len_ += len(labels)
                loss = ce(predictions, labels)
                val_loss.append(loss.item())
                acc_ += (predictions.argmax(axis = -1).cpu() == labels.cpu()).sum()
                eval_loss += loss.item()

            eval_avg_loss = eval_loss / len(val_loader)

            if(eval_avg_loss < best_loss):
                print("Saving best model")
                torch.save(model_.state_dict(), f'{save_model_path}/best_model_{iter_}.pth')
                best_loss = eval_avg_loss
                count_upt += 1

            val_acc.append(acc_/batch_len_)


    return model_, train_loss, val_loss, train_acc, val_acc
