import torch
import numpy as np
class BirdDataset(torch.utils.data.Dataset):
    
    def __init__(
        self,
        metadata,
        set,
    ):
        super().__init__()
        self.metadata = metadata
        self.set = set

    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, index):
        if self.set == 'Holdout':
            row_metadata = self.metadata.iloc[index]
            input_feature = np.load(row_metadata['filepath'])
            target = row_metadata['target']     
        elif self.set == 'Train':
            row_metadata = self.metadata.iloc[index]
            input_feature = np.load(row_metadata['filepath'])
            target = row_metadata['target']     
        elif self.set == 'Val':
            row_metadata = self.metadata.iloc[index]
            input_feature = np.load(row_metadata['filepath'])
            target = row_metadata['target']    
        return torch.tensor(input_feature, dtype=torch.float32), torch.tensor(target, dtype=torch.long)
