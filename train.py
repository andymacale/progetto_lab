import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import os

from rete import UNet  # Importa la rete dal file rete.py

# === Dataset personalizzato che carica immagini input e target da un CSV ===
class ImagePairDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.df = pd.read_csv(csv_path)   # Legge il CSV con input_path, target_path, flag
        self.transform = transform        # Trasformazioni (es. resize, to tensor)

    def __len__(self):
        return len(self.df)               # Numero totale di immagini

    def __getitem__(self, idx):
        input_img = Image.open(self.df.iloc[idx]['input_path']).convert("RGB")
        target_img = Image.open(self.df.iloc[idx]['target_path']).convert("RGB")
        flag = self.df.iloc[idx]['flag']  # 0 o 1

        if self.transform:
            input_img = self.transform(input_img)
            target_img = self.transform(target_img)

        # Crea una mappa costante 1xHxW con valore 0 o 1
        flag_tensor = torch.full((1, input_img.shape[1], input_img.shape[2]), float(flag))

        # Concatena input (3xHxW) + flag (1xHxW) → 4xHxW
        input_with_flag = torch.cat([input_img, flag_tensor], dim=0)

        return input_with_flag, target_img

# === Hyperparametri ===
batch_size = 4
lr = 1e-4
epochs = 10
csv_path = '/content/drive/My Drive/dataset/dataset.csv'  # Path al CSV su Drive

# === Trasformazioni da applicare a tutte le immagini ===
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resizing standard
    transforms.ToTensor()           # Conversione a tensore PyTorch
])

# === Creazione del dataset e dataloader ===
dataset = ImagePairDataset(csv_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# === Inizializzazione della rete e spostamento su GPU (se disponibile) ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet().to(device)

# === Ottimizzatore e funzione di perdita ===
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()  # Errore medio quadratico tra output e immagine target

# === Training loop ===
for epoch in range(epochs):
    model.train()         # Modalità training
    epoch_loss = 0        # Accumula la loss totale dell'epoca

    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)  # Sposta su GPU/CPU
        optimizer.zero_grad()           # Azzeramento dei gradienti
        outputs = model(inputs)         # Forward pass
        loss = criterion(outputs, targets)  # Calcolo della loss
        loss.backward()                 # Backpropagation
        optimizer.step()                # Aggiornamento pesi
        epoch_loss += loss.item()       # Somma la loss

    print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}")

# === Salvataggio del modello addestrato ===
torch.save(model.state_dict(), 'unet_trained.pth')
