import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
import math
from pytorch_lightning.callbacks import Callback


# --------- Dataset ---------
# Wrap a list/array of sequences
# Return tensors of shape [seq_len]
class FeatureDataset(Dataset):
    def __init__(self, events):
        self.data = np.array(events, dtype=np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx])

# --------- Positional Encoding ---------
# Implements sinusoidal positional encoding as in the original Transformer paper
# Adds positional info to each timestep
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return x

# --------- Transformer Autoencoder ---------
class TransformerAutoencoder(pl.LightningModule):

    def __init__(self, seq_len=6, d_model=64, nhead=2, num_layers=3, lr=2e-4):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.lr = lr

        self.input_proj = nn.Linear(1, d_model) # projects scalar input to a vector space
        self.pos_encoder = PositionalEncoding(d_model=d_model)

        # Embedding for missing positions
        self.mask_pos_embedding = nn.Embedding(seq_len, 16)
        self.mask_proj = nn.Linear(d_model + 16, d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1)
        )

        self.criterion = nn.MSELoss()

    def corrupt_input(self, x, mask_idx=None):
        batch_size, seq_len = x.shape[0], x.shape[1]
        if mask_idx is None:
            mask_idx = torch.randint(0, seq_len, (batch_size,), device=x.device)

        x_corrupted = []
        for i in range(batch_size):
            idx = mask_idx[i]
            # Remove the masked feature
            xi = torch.cat([x[i, :idx], x[i, idx+1:]], dim=0)
            x_corrupted.append(xi.unsqueeze(-1))
        x_corrupted = torch.stack(x_corrupted)  # [batch, seq_len-1, 1]
        return x_corrupted, mask_idx

    def forward(self, x, mask_idx):
        """
        x: [batch, seq_len-1, 1]  Input sequence (one feature missing)
        mask_idx: [batch]         Index of the missing feature
        """
        if x.ndim == 2:
            x = x.unsqueeze(-1)  # → [batch, seq_len, 1]

        batch_size = x.size(0)
        x = self.input_proj(x)                # [batch, seq_len-1, d_model]
        x = self.pos_encoder(x)

        mask_embed = self.mask_pos_embedding(mask_idx)  # [batch, 16]
        mask_embed_expanded = mask_embed.unsqueeze(1).expand(-1, x.size(1), -1)  # [batch, seq_len-1, 16]

        x = torch.cat([x, mask_embed_expanded], dim=-1)  # [batch, seq_len-1, d_model+16]
        x = self.mask_proj(x)  # [batch, seq_len-1, d_model]

        x = x.permute(1, 0, 2)  # [seq_len-1, batch, d_model]
        x = self.transformer_encoder(x)

        x = x.permute(1, 0, 2) # → [batch, seq_len-1, d_model]
        x_selected = x.mean(dim=1)  # → [batch, d_model]

        y_pred = self.fc(x_selected).squeeze(-1)  # [batch]

        return y_pred

    def training_step(self, batch, batch_idx):
        x_full = batch
        x_corrupted, mask_idx = self.corrupt_input(x_full)
        y_true = x_full[torch.arange(x_full.size(0)), mask_idx]
        y_pred = self(x_corrupted, mask_idx)
        loss = self.criterion(y_pred, y_true)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x_full = batch
        x_corrupted, mask_idx = self.corrupt_input(x_full)
        y_true = x_full[torch.arange(x_full.size(0)), mask_idx]
        y_pred = self(x_corrupted, mask_idx)
        loss = self.criterion(y_pred, y_true)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

class LossTracker(Callback):
    def __init__(self):
        self.train_losses = []
        self.val_losses = []

    def on_train_epoch_end(self, trainer, pl_module):
        self.train_losses.append(trainer.callback_metrics["train_loss"].item())

    def on_validation_epoch_end(self, trainer, pl_module):
        self.val_losses.append(trainer.callback_metrics["val_loss"].item())
