import torch
import torch.nn as nn
import torch.nn.functional as F

# Classe che permette l'esecuzione di due convoluzioni con l'unità
# logica rettificata (ReLu), alla base delle reti neurali UNet

class DoubleConv(nn.Module):

    # Costruttore
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Costruzione delle rete neurale
        self.block = nn.Sequential(
            # DoubleConv RGB quindi 3 canali di out_channels
            nn.Conv2d(in_channels, out_channels = 3, padding = 1), # conv 1
            nn.ReLU(inplace = True), # attivatore 1
            nn.Conv2d(in_channels, out_channels = 3, padding = 1), # conv 2
            nn.ReLU(inplace = True), # attivatore 2
        )

    # Avvio
    def forward(self, elem):
        return self.block(elem)
    

# Classe che definisce la rete neurale UNet vera e propria
class UNet(nn.Module):

    # Costruttore:
    # 4 canali di in_channels (RGB + flag) e 3 di out_channels (RGB)
    def __init__(self, in_channels = 4, out_channels = 3):
        super().__init__()

        # Encoder della rete neurale (discesa)
        # Definizione: due blocchi che effettuano sottocampionamento
        self.down1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2) # dimezza la dimensione
        self.down2 = DoubleConv(in_channels, 64)
        self.pool2 = nn.MaxPool2d(2) # dimezza la dimensione

        # Punto più in profondità della rete
        self.bottleneck = DoubleConv(128, 256)

        # Decoder della rete neurale (salita)
        # Definizione: due blocchi che effettuano
        # sovracampionamento, concatenazione e DoubleConv
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2) # da 64 a 128
        self.con2 = DoubleConv(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2) # da 64 a 128
        self.con2 = DoubleConv(128, 64)

        # Conversione nei tre canali con una DoubleConv 1x1
        self.out = nn.Conv2d(64, out_channels, kernel_size = 1)

    
    def forward(self, elem):
    
        # Encoder
        d1 = self.down1(elem)               
        d2 = self.down2(self.pool1(d1))

        # Nodo in fondo
        bn = self.bottleneck(self.pool2(d2))  

        # Decoder
        up2 = self.up2(bn)                    
        up2 = torch.cat([up2, d2], dim=1)     
        up2 = self.up_conv2(up2)              

        up1 = self.up1(up2)                  
        up1 = torch.cat([up1, d1], dim=1)     
        up1 = self.up_conv1(up1)              

        # Output finale
        return self.out(up1)

        
 




