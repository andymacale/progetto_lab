import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd

class RawLuminositaDataset(Dataset):

    # Costruttore
    def __init__(self, path, transform=None):
        self.data = pd.read_csv(path)
        self.transform = transform or transform.Compose([transforms.ToTensor(),])

    # Numero di elementi del dataset
    def __len__(self):
        return len(self.data)
    
    # Input e target all'indice ind
    def __getitem__(self, ind):
        input = self.data.iloc[ind, 0]
        target = self.data.iloc[ind, 1]

        input = Image.open(input).convert("RGB")
        target = Image.open(target).convert("RGB")

        input_transform = self.transform(input)
        target_transform = self.transform(target)

        return input_transform, target_transform