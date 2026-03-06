"""
train_rom.py
============
Trains a dynamic Reduced Order Model (ROM) of the enginespeed Simulink model
using PyTorch LSTM. The ROM maps:

    Inputs (per timestep): [AirCharge, Speed, SparkAdvance]
    Output (per timestep): [Torque]

Architecture:
    - 1-layer LSTM, hidden_size=32  (ECU-deployable)
    - Linear output head
    - ~4700 parameters total

Outputs:
    models/rom_model.pth       - trained model
    models/normalization.json  - normalization statistics
    models/weights_export.json - raw weights for C code generation
    plots/training_loss.png    - training/validation loss curves
    plots/sample_predictions.png
"""

import os
import json
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ── Paths ────────────────────────────────────────────────────────────────────
PROJ  = '/Users/arkadiyturevskiy/Documents/Claude/Claude_ROM_session4'
DATA  = os.path.join(PROJ, 'data',   'training_data.csv')
MDIR  = os.path.join(PROJ, 'models')
PDIR  = os.path.join(PROJ, 'plots')
os.makedirs(MDIR, exist_ok=True)
os.makedirs(PDIR, exist_ok=True)

# ── Hyper-parameters ─────────────────────────────────────────────────────────
SEQ_LEN    = 100      # timesteps per training window  (5 s @ dt=0.05)
STRIDE     = 10       # window stride for data augmentation
BATCH_SIZE = 32
EPOCHS     = 400
LR         = 1e-3
HIDDEN     = 32
LAYERS     = 1
VAL_FRAC   = 0.2      # fraction of simulations held out for val

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


# ── Dataset ──────────────────────────────────────────────────────────────────
class WindowDataset(Dataset):
    """Sliding-window dataset from concatenated simulation data."""

    def __init__(self, windows_X, windows_y):
        self.X = torch.tensor(windows_X, dtype=torch.float32)
        self.y = torch.tensor(windows_y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ── Model ────────────────────────────────────────────────────────────────────
class EngineROM(nn.Module):
    """Single-layer LSTM ROM for engine torque prediction."""

    def __init__(self, input_size=3, hidden_size=HIDDEN,
                 num_layers=LAYERS, output_size=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=0.0)
        self.fc   = nn.Linear(hidden_size, output_size)

    def forward(self, x, state=None):
        out, state = self.lstm(x, state)
        return self.fc(out), state

    def step(self, x_t, state=None):
        """Single-timestep inference for real-time ECU deployment."""
        x_t = x_t.unsqueeze(1)          # (batch, 1, input_size)
        out, state = self.lstm(x_t, state)
        return self.fc(out.squeeze(1)), state


# ── Utilities ────────────────────────────────────────────────────────────────
def load_data():
    df = pd.read_csv(DATA)
    print(f"Loaded {len(df):,} samples from {len(df['SimID'].unique())} simulations")
    return df


def compute_norm(df):
    cols = ['AirCharge', 'Speed', 'SparkAdvance', 'Torque']
    stats = {}
    for c in cols:
        stats[c] = {'mean': float(df[c].mean()), 'std': float(df[c].std())}
    return stats


def normalize(arr, stats):
    return (arr - stats['mean']) / (stats['std'] + 1e-8)


def denormalize(arr, stats):
    return arr * (stats['std'] + 1e-8) + stats['mean']


def make_windows(df, stats, sim_ids):
    """Extract sliding-window sequences from selected simulations."""
    windows_X, windows_y = [], []
    for sid in sim_ids:
        sub = df[df['SimID'] == sid].sort_values('Time').reset_index(drop=True)
        ac  = normalize(sub['AirCharge'].values,   stats['AirCharge'])
        spd = normalize(sub['Speed'].values,       stats['Speed'])
        sa  = normalize(sub['SparkAdvance'].values, stats['SparkAdvance'])
        tq  = normalize(sub['Torque'].values,      stats['Torque'])

        X   = np.stack([ac, spd, sa], axis=1)   # (T, 3)
        y   = tq[:, None]                         # (T, 1)

        for start in range(0, len(X) - SEQ_LEN, STRIDE):
            windows_X.append(X[start:start + SEQ_LEN])
            windows_y.append(y[start:start + SEQ_LEN])

    return np.array(windows_X), np.array(windows_y)


# ── Training ─────────────────────────────────────────────────────────────────
def train():
    # ── Load & split data
    df  = load_data()
    all_ids = sorted(df['SimID'].unique())
    n_val   = max(1, int(len(all_ids) * VAL_FRAC))
    val_ids  = all_ids[-n_val:]
    train_ids = all_ids[:-n_val]
    print(f"Train sims: {train_ids}  |  Val sims: {val_ids}")

    # ── Normalization (fit on training only)
    train_df = df[df['SimID'].isin(train_ids)]
    stats    = compute_norm(train_df)
    with open(os.path.join(MDIR, 'normalization.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Normalization saved.")

    # ── Build datasets
    X_tr, y_tr = make_windows(df, stats, train_ids)
    X_val,y_val = make_windows(df, stats, val_ids)
    print(f"Training windows: {len(X_tr)}  |  Val windows: {len(X_val)}")

    train_loader = DataLoader(WindowDataset(X_tr, y_tr),
                              batch_size=BATCH_SIZE, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(WindowDataset(X_val, y_val),
                              batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    # ── Model, optimizer, scheduler
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = EngineROM().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}  |  Device: {device}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
    criterion = nn.MSELoss()

    # ── Training loop
    train_losses, val_losses = [], []
    best_val  = float('inf')
    best_state = None

    print(f"\nTraining for {EPOCHS} epochs...\n{'─'*55}")
    for epoch in range(1, EPOCHS + 1):
        # Train
        model.train()
        ep_loss = 0.0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred, _ = model(Xb)
            loss     = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_loss += loss.item()
        ep_loss /= len(train_loader)

        # Validate
        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                pred, _ = model(Xb)
                v_loss  += criterion(pred, yb).item()
        v_loss /= len(val_loader)

        train_losses.append(ep_loss)
        val_losses.append(v_loss)
        scheduler.step()

        if v_loss < best_val:
            best_val   = v_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 50 == 0 or epoch == 1:
            print(f"Epoch {epoch:4d}/{EPOCHS}  |  "
                  f"Train MSE: {ep_loss:.6f}  |  "
                  f"Val MSE: {v_loss:.6f}  |  "
                  f"LR: {scheduler.get_last_lr()[0]:.2e}")

    # ── Load best model
    model.load_state_dict(best_state)
    print(f"\nBest validation MSE: {best_val:.6f}")

    # ── Compute RMSE in physical units
    model.eval()
    tq_std = stats['Torque']['std']
    tr_rmse = math.sqrt(sum(train_losses) / len(train_losses)) * tq_std
    val_rmse = math.sqrt(best_val) * tq_std
    print(f"Training RMSE (physical): {tr_rmse:.3f} N·m")
    print(f"Validation RMSE (physical): {val_rmse:.3f} N·m")

    # ── Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'input_size':  3,
            'hidden_size': HIDDEN,
            'num_layers':  LAYERS,
            'output_size': 1
        },
        'train_losses': train_losses,
        'val_losses':   val_losses,
        'best_val_mse': best_val,
        'stats':        stats
    }, os.path.join(MDIR, 'rom_model.pth'))
    print(f"Model saved to models/rom_model.pth")

    # ── Export weights for C implementation
    export_weights_for_c(model, stats)

    # ── Plots
    plot_training_curves(train_losses, val_losses, EPOCHS)
    plot_sample_predictions(model, device, df, stats, val_ids)

    print("\nTraining complete!")
    return model, stats, train_losses, val_losses


def export_weights_for_c(model, stats):
    """Export LSTM + FC weights as JSON (used by C generator)."""
    sd = model.state_dict()

    def to_list(t):
        return t.detach().cpu().numpy().tolist()

    weights = {
        'lstm': {
            'weight_ih': to_list(sd['lstm.weight_ih_l0']),  # (4H, I)
            'weight_hh': to_list(sd['lstm.weight_hh_l0']),  # (4H, H)
            'bias_ih':   to_list(sd['lstm.bias_ih_l0']),    # (4H,)
            'bias_hh':   to_list(sd['lstm.bias_hh_l0']),    # (4H,)
        },
        'fc': {
            'weight': to_list(sd['fc.weight']),  # (O, H)
            'bias':   to_list(sd['fc.bias']),    # (O,)
        },
        'config': {
            'input_size':  3,
            'hidden_size': HIDDEN,
            'output_size': 1
        },
        'normalization': stats
    }

    path = os.path.join(MDIR, 'weights_export.json')
    with open(path, 'w') as f:
        json.dump(weights, f)
    print(f"Weights exported to models/weights_export.json")


def plot_training_curves(train_losses, val_losses, epochs):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.semilogy(range(1, epochs+1), train_losses, label='Training', color='royalblue', lw=1.5)
    ax.semilogy(range(1, epochs+1), val_losses,   label='Validation', color='tomato', lw=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss (normalized)')
    ax.set_title('Training & Validation Loss (log scale)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    start = epochs // 4
    ax.semilogy(range(start+1, epochs+1), train_losses[start:], label='Training',   color='royalblue', lw=1.5)
    ax.semilogy(range(start+1, epochs+1), val_losses[start:],   label='Validation', color='tomato', lw=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss (normalized)')
    ax.set_title(f'Loss Convergence (epochs {start}–{epochs})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle('Dynamic ROM – LSTM Training Results', fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(PDIR, 'training_loss.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training loss plot saved to plots/training_loss.png")


def plot_sample_predictions(model, device, df, stats, val_ids):
    """Plot full-sequence ROM vs ground truth for one validation simulation."""
    model.eval()
    sid = val_ids[0]
    sub = df[df['SimID'] == sid].sort_values('Time').reset_index(drop=True)

    t   = sub['Time'].values
    ac  = normalize(sub['AirCharge'].values,    stats['AirCharge'])
    spd = normalize(sub['Speed'].values,        stats['Speed'])
    sa  = normalize(sub['SparkAdvance'].values,  stats['SparkAdvance'])
    tq  = sub['Torque'].values

    X = torch.tensor(np.stack([ac, spd, sa], axis=1), dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_norm, _ = model(X)

    pred = denormalize(pred_norm.squeeze().cpu().numpy(), stats['Torque'])

    rmse = math.sqrt(np.mean((pred - tq)**2))

    fig, axes = plt.subplots(3, 1, figsize=(12, 9))

    ax = axes[0]
    ax.plot(t, tq,   label='Simulink',    color='steelblue',  lw=1.5)
    ax.plot(t, pred, label='LSTM ROM',    color='orangered',  lw=1.2, linestyle='--')
    ax.set_ylabel('Torque [N·m]')
    ax.set_title(f'Training Preview – Sim {sid} (SA={sub["SparkAdvance"].iloc[0]:.0f}°)  |  RMSE={rmse:.3f} N·m')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(t, denormalize(spd, stats['Speed']), color='darkgreen', lw=1.2)
    ax.set_ylabel('Speed [rad/s]')
    ax.set_title('Engine Speed (input to ROM)')
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(t, denormalize(ac, stats['AirCharge']), color='purple', lw=1.2)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Air Charge [g/s]')
    ax.set_title('Air Charge (input to ROM)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(PDIR, 'sample_predictions.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Sample prediction plot saved to plots/sample_predictions.png")


if __name__ == '__main__':
    train()
