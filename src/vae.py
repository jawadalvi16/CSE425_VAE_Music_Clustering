import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


# -----------------------------
# Load features
# -----------------------------
AUDIO_FEATS = "data/audio_mfcc.npy"
LYRICS_FEATS = "data/lyrics_emb.npy"

audio = np.load(AUDIO_FEATS)      # (N, 40, 1300)
lyrics = np.load(LYRICS_FEATS)    # (N, 384)

N = audio.shape[0]

# Flatten MFCC
audio_flat = audio.reshape(N, -1)  # (N, 52000)

# Concatenate audio + lyrics
X_np = np.concatenate([audio_flat, lyrics], axis=1).astype(np.float32)

# -----------------------------
# IMPORTANT: Standardize features (prevents NaN)
# -----------------------------
mean = X_np.mean(axis=0, keepdims=True)
std = X_np.std(axis=0, keepdims=True) + 1e-6
X_np = (X_np - mean) / std

# Save scaler (useful for report / reproducibility)
np.save("data/x_mean.npy", mean.astype(np.float32))
np.save("data/x_std.npy", std.astype(np.float32))

X = torch.tensor(X_np, dtype=torch.float32)

INPUT_DIM = X.shape[1]
LATENT_DIM = 32
BATCH_SIZE = 32
EPOCHS = 30
LR = 1e-4          # smaller LR
BETA = 0.001       # KL weight (beta-VAE style)


# -----------------------------
# VAE Model
# -----------------------------
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

        self.dec = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1024),
            nn.ReLU(),
            nn.Linear(1024, input_dim),
        )

    def encode(self, x):
        h = self.enc(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.dec(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


def vae_loss(recon_x, x, mu, logvar, beta=BETA):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction="mean")
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl, recon_loss, kl


# -----------------------------
# Training
# -----------------------------
dataset = TensorDataset(X)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = VAE(INPUT_DIM, LATENT_DIM).to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)

print(f"Training VAE on {device} | input_dim={INPUT_DIM} | N={N}")

for epoch in range(EPOCHS):
    model.train()
    total, total_recon, total_kl = 0.0, 0.0, 0.0

    for (batch_x,) in tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        batch_x = batch_x.to(device)

        recon, mu, logvar = model(batch_x)
        loss, recon_l, kl_l = vae_loss(recon, batch_x, mu, logvar)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)  # extra stability
        optimizer.step()

        total += loss.item()
        total_recon += recon_l.item()
        total_kl += kl_l.item()

    print(
        f"Epoch {epoch+1}: loss={total/len(loader):.6f} "
        f"recon={total_recon/len(loader):.6f} kl={total_kl/len(loader):.6f}"
    )

# -----------------------------
# Save latent vectors
# -----------------------------
model.eval()
with torch.no_grad():
    mu, _ = model.encode(X.to(device))
    Z = mu.cpu().numpy().astype(np.float32)

np.save("data/latent_z.npy", Z)
print("Saved latent vectors to data/latent_z.npy")
