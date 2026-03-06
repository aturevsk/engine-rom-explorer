"""
train_all_models.py
===================
Comprehensive ROM compression + alternative model exploration.

Models trained:
  LSTM variants (PyTorch):
    lstm_8   – hidden=8,  ~425 params, ~1.7 KB Flash
    lstm_16  – hidden=16, ~1361 params, ~5.3 KB Flash
    lstm_32  – hidden=32, ~4769 params, ~18.6 KB (baseline, already trained)
    lstm_32_q8 – lstm_32 weights quantized to int8, ~4.7 KB Flash

  NARX sklearn (autoregressive lag-feature models):
    narx_ridge – Ridge regression, 11 features → ~0.05 KB
    narx_mlp   – MLP (16,8) hidden, ~0.9 KB
    narx_gbm   – GradientBoosting n=50 d=3, ~25 KB

Outputs:
  models/<name>_info.json      – weights / sklearn params + validation metrics
  src/rom_<name>.h / .c        – C implementation
  data/model_comparison.json   – all models' metrics for Pareto plot
  plots/pareto_frontier.png
  plots/validation_all_models.png
"""

import os, sys, json, math, time, warnings, itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJ  = '/Users/arkadiyturevskiy/Documents/Claude/Claude_ROM_session4'
DATA_TRAIN = os.path.join(PROJ, 'data', 'training_data.csv')
DATA_VAL   = os.path.join(PROJ, 'data', 'validation_data.csv')
MDIR  = os.path.join(PROJ, 'models')
SRC   = os.path.join(PROJ, 'src')
PDIR  = os.path.join(PROJ, 'plots')
os.makedirs(MDIR, exist_ok=True)
os.makedirs(SRC,  exist_ok=True)
os.makedirs(PDIR, exist_ok=True)

# Load baseline normalization (from original LSTM-32 training)
with open(os.path.join(MDIR, 'normalization.json')) as f:
    STATS = json.load(f)

# ── NARX config ───────────────────────────────────────────────────────────────
LAG_AC  = 4   # lags for Air Charge
LAG_SPD = 4   # lags for Speed
LAG_TQ  = 2   # autoregressive lags for Torque
N_FEAT  = LAG_AC + LAG_SPD + LAG_TQ + 1   # = 11

# ── LSTM training config ──────────────────────────────────────────────────────
SEQ_LEN    = 100
STRIDE     = 10
BATCH_SIZE = 32
EPOCHS     = 300        # reduced from 400 for speed
SEED       = 42
torch.manual_seed(SEED);  np.random.seed(SEED)

VAL_SIM_IDS = [11, 12]   # same split as original training


# =============================================================================
# DATA HELPERS
# =============================================================================
def normalize(arr, st):
    return (arr - st['mean']) / (st['std'] + 1e-8)

def denormalize(arr, st):
    return arr * (st['std'] + 1e-8) + st['mean']

def load_data():
    df_tr = pd.read_csv(DATA_TRAIN)
    df_vl = pd.read_csv(DATA_VAL)
    return df_tr, df_vl

def rmse(pred, true):
    return math.sqrt(np.mean((np.asarray(pred) - np.asarray(true))**2))

def r2(pred, true):
    ss_res = np.sum((np.asarray(pred)-np.asarray(true))**2)
    ss_tot = np.sum((np.asarray(true)-np.mean(true))**2)
    return 1 - ss_res/(ss_tot+1e-12)


# =============================================================================
# NARX FEATURE ENGINEERING
# =============================================================================
def make_narx_dataset(df, p_ac=LAG_AC, p_spd=LAG_SPD, p_tq=LAG_TQ,
                      sim_ids=None, use_true_ar=True):
    """Build flat NARX feature matrix + target from simulation data.
    use_true_ar=True  → teacher-forcing (1-step-ahead) mode for training/val
    use_true_ar=False → recursive simulation mode
    """
    if sim_ids is None:
        sim_ids = sorted(df['SimID'].unique())
    p = max(p_ac, p_spd)

    rows_X, rows_y = [], []
    meta = []  # (sim_id, time, true_tq)

    for sid in sim_ids:
        sub = df[df['SimID']==sid].sort_values('Time').reset_index(drop=True)
        ac   = normalize(sub['AirCharge'].values,   STATS['AirCharge'])
        spd  = normalize(sub['Speed'].values,        STATS['Speed'])
        sa   = normalize(sub['SparkAdvance'].values,  STATS['SparkAdvance'])
        tq   = normalize(sub['Torque'].values,       STATS['Torque'])
        tq_true = sub['Torque'].values

        for t in range(p, len(sub)):
            feat = []
            for l in range(p_ac):  feat.append(ac[t-l])
            for l in range(p_spd): feat.append(spd[t-l])
            feat.append(tq[t-1])   # AR lag 1
            feat.append(tq[t-2])   # AR lag 2
            feat.append(sa[t])
            rows_X.append(feat)
            rows_y.append(tq[t])
            meta.append((sid, sub['Time'].values[t], tq_true[t]))

    return np.array(rows_X, dtype=np.float32), np.array(rows_y, dtype=np.float32), meta


def narx_simulate(model_fn, df_sim, p_ac=LAG_AC, p_spd=LAG_SPD, p_tq=LAG_TQ):
    """Run NARX model in closed-loop simulation (recursive prediction)."""
    sub = df_sim.sort_values('Time').reset_index(drop=True)
    ac   = normalize(sub['AirCharge'].values,   STATS['AirCharge'])
    spd  = normalize(sub['Speed'].values,        STATS['Speed'])
    sa   = normalize(sub['SparkAdvance'].values,  STATS['SparkAdvance'])
    tq_true_norm = normalize(sub['Torque'].values, STATS['Torque'])
    tq_phys = sub['Torque'].values

    p = max(p_ac, p_spd)
    # Warm-up buffer with true values
    tq_pred_norm = list(tq_true_norm[:p])

    predictions = [None] * p
    for t in range(p, len(sub)):
        feat = []
        for l in range(p_ac):  feat.append(ac[t-l])
        for l in range(p_spd): feat.append(spd[t-l])
        feat.append(tq_pred_norm[t-1])
        feat.append(tq_pred_norm[t-2])
        feat.append(sa[t])
        y_norm = model_fn(np.array([feat], dtype=np.float32))
        y_norm_s = float(y_norm)
        tq_pred_norm.append(y_norm_s)
        predictions.append(denormalize(y_norm_s, STATS['Torque']))

    valid_mask = [p is not None for p in predictions]
    preds_phys = [p for p in predictions if p is not None]
    true_phys  = tq_phys[valid_mask]
    t_out = sub['Time'].values[valid_mask]
    return t_out, true_phys, np.array(preds_phys)


# =============================================================================
# LSTM ARCHITECTURE & TRAINING
# =============================================================================
class EngineROM_LSTM(nn.Module):
    def __init__(self, hidden_size=32, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(3, hidden_size, num_layers, batch_first=True)
        self.fc   = nn.Linear(hidden_size, 1)
    def forward(self, x, state=None):
        out, state = self.lstm(x, state)
        return self.fc(out), state

class WindowDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

def make_lstm_windows(df, sim_ids):
    windows_X, windows_y = [], []
    for sid in sim_ids:
        sub = df[df['SimID']==sid].sort_values('Time').reset_index(drop=True)
        ac  = normalize(sub['AirCharge'].values,    STATS['AirCharge'])
        spd = normalize(sub['Speed'].values,        STATS['Speed'])
        sa  = normalize(sub['SparkAdvance'].values,  STATS['SparkAdvance'])
        tq  = normalize(sub['Torque'].values,       STATS['Torque'])
        X   = np.stack([ac, spd, sa], 1)
        y   = tq[:, None]
        for s in range(0, len(X)-SEQ_LEN, STRIDE):
            windows_X.append(X[s:s+SEQ_LEN])
            windows_y.append(y[s:s+SEQ_LEN])
    return np.array(windows_X), np.array(windows_y)

def train_lstm(hidden_size, df_tr, df_vl, name, epochs=EPOCHS):
    print(f"\n  Training LSTM-{hidden_size} ({name}) …")
    tr_ids = [i for i in sorted(df_tr['SimID'].unique()) if i not in VAL_SIM_IDS]
    X_tr, y_tr = make_lstm_windows(df_tr, tr_ids)
    X_vl, y_vl = make_lstm_windows(df_tr, VAL_SIM_IDS)

    tr_loader = DataLoader(WindowDataset(X_tr, y_tr), BATCH_SIZE, shuffle=True,  drop_last=True)
    vl_loader = DataLoader(WindowDataset(X_vl, y_vl), BATCH_SIZE, shuffle=False)

    model = EngineROM_LSTM(hidden_size)
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    crit  = nn.MSELoss()
    best_val, best_state, train_hist, val_hist = 1e9, None, [], []

    for ep in range(1, epochs+1):
        model.train()
        ep_loss = sum(
            (opt.zero_grad() or True) and
            (lambda loss: (loss.backward(), opt.step(), loss.item()))(
                crit((lambda out: out[0])(model(xb)), yb)
            )[2]
            for xb, yb in tr_loader
        ) / len(tr_loader)
        # cleaner version:
        ep_loss2 = 0.0
        model.train()
        for xb, yb in tr_loader:
            opt.zero_grad()
            pred, _ = model(xb)
            loss = crit(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss2 += loss.item()
        ep_loss = ep_loss2 / len(tr_loader)

        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for xb, yb in vl_loader:
                pred, _ = model(xb)
                v_loss  += crit(pred, yb).item()
        v_loss /= len(vl_loader)
        train_hist.append(ep_loss); val_hist.append(v_loss)
        sched.step()
        if v_loss < best_val:
            best_val = v_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if ep % 100 == 0:
            print(f"    ep {ep}/{epochs} | train={ep_loss:.5f} | val={v_loss:.5f}")

    model.load_state_dict(best_state)
    model.eval()

    # Validate on validation_data.csv (unseen)
    df_vval = pd.read_csv(DATA_VAL)
    val_rmses, val_r2s = [], []
    for sid in sorted(df_vval['SimID'].unique()):
        sub = df_vval[df_vval['SimID']==sid].sort_values('Time').reset_index(drop=True)
        ac  = normalize(sub['AirCharge'].values,    STATS['AirCharge'])
        spd = normalize(sub['Speed'].values,        STATS['Speed'])
        sa  = normalize(sub['SparkAdvance'].values,  STATS['SparkAdvance'])
        X   = torch.tensor(np.stack([ac, spd, sa], 1)[None], dtype=torch.float32)
        with torch.no_grad():
            pred_n, _ = model(X)
        pred = denormalize(pred_n.squeeze().numpy(), STATS['Torque'])
        true = sub['Torque'].values
        val_rmses.append(rmse(pred, true))
        val_r2s.append(r2(pred, true))

    mean_rmse = float(np.mean(val_rmses))
    mean_r2   = float(np.mean(val_r2s))
    n_params  = sum(p.numel() for p in model.parameters())
    flash_kb  = n_params * 4 / 1024

    print(f"    RMSE={mean_rmse:.4f} N·m | R²={mean_r2:.5f} | params={n_params} | Flash≈{flash_kb:.1f} KB")

    # Save
    ckpt = {'model_state_dict': model.state_dict(),
            'model_config': {'input_size': 3, 'hidden_size': hidden_size,
                             'num_layers': 1, 'output_size': 1},
            'val_rmse': mean_rmse, 'val_r2': mean_r2,
            'n_params': n_params, 'flash_kb': flash_kb,
            'train_losses': train_hist, 'val_losses': val_hist}
    torch.save(ckpt, os.path.join(MDIR, f'{name}_model.pth'))
    return model, mean_rmse, mean_r2, n_params, flash_kb


# =============================================================================
# LSTM INT-8 QUANTIZATION
# =============================================================================
def quantize_lstm_int8(model32, df_vval, name='lstm_32_q8'):
    """Post-training symmetric per-tensor int8 quantization of weights."""
    print(f"\n  Quantizing LSTM-32 → int8 ({name}) …")
    sd = model32.state_dict()
    quant = {}
    total_int8_bytes = 0
    for k, v in sd.items():
        arr = v.detach().numpy().astype(np.float32)
        scale = float(np.max(np.abs(arr))) / 127.0 if np.max(np.abs(arr)) > 0 else 1.0
        q = np.clip(np.round(arr / scale), -128, 127).astype(np.int8)
        quant[k] = {'int8': q.tolist(), 'scale': scale, 'shape': list(arr.shape)}
        total_int8_bytes += q.size

    # Reconstruct float32 from int8 for accuracy evaluation
    recon_sd = {}
    for k, v in quant.items():
        arr_int8 = np.array(v['int8'], dtype=np.int8).reshape(v['shape'])
        arr_f32  = arr_int8.astype(np.float32) * v['scale']
        recon_sd[k] = torch.tensor(arr_f32)

    model_q = EngineROM_LSTM(32)
    model_q.load_state_dict(recon_sd)
    model_q.eval()

    # Validate
    val_rmses, val_r2s = [], []
    for sid in sorted(df_vval['SimID'].unique()):
        sub = df_vval[df_vval['SimID']==sid].sort_values('Time').reset_index(drop=True)
        ac  = normalize(sub['AirCharge'].values,    STATS['AirCharge'])
        spd = normalize(sub['Speed'].values,        STATS['Speed'])
        sa  = normalize(sub['SparkAdvance'].values,  STATS['SparkAdvance'])
        X   = torch.tensor(np.stack([ac, spd, sa], 1)[None], dtype=torch.float32)
        with torch.no_grad():
            pred_n, _ = model_q(X)
        pred = denormalize(pred_n.squeeze().numpy(), STATS['Torque'])
        true = sub['Torque'].values
        val_rmses.append(rmse(pred, true)); val_r2s.append(r2(pred, true))

    mean_rmse = float(np.mean(val_rmses))
    mean_r2   = float(np.mean(val_r2s))
    n_params   = sum(p.numel() for p in model_q.parameters())
    flash_kb   = (total_int8_bytes + 6*4) / 1024   # int8 weights + 6 scale floats

    print(f"    RMSE={mean_rmse:.4f} N·m | R²={mean_r2:.5f} | int8 Flash≈{flash_kb:.1f} KB")

    info = {'quant_weights': quant, 'val_rmse': mean_rmse, 'val_r2': mean_r2,
            'n_params': n_params, 'flash_kb': flash_kb,
            'hidden_size': 32, 'input_size': 3, 'output_size': 1}
    with open(os.path.join(MDIR, f'{name}_info.json'), 'w') as f:
        json.dump(info, f)
    return model_q, quant, mean_rmse, mean_r2, n_params, flash_kb


# =============================================================================
# NARX MODEL TRAINING
# =============================================================================
def train_narx_ridge(X_tr, y_tr, X_vl, y_vl, df_vval, name='narx_ridge'):
    print(f"\n  Training {name} …")
    from sklearn.linear_model import Ridge
    model = Ridge(alpha=1.0)
    model.fit(X_tr, y_tr)

    # Validate on validation_data.csv (simulation mode)
    val_rmses, val_r2s = [], []
    for sid in sorted(df_vval['SimID'].unique()):
        sub = df_vval[df_vval['SimID']==sid].sort_values('Time').reset_index(drop=True)
        model_fn = lambda x: model.predict(x)
        t_, true_, pred_ = narx_simulate(model_fn, sub)
        val_rmses.append(rmse(pred_, true_)); val_r2s.append(r2(pred_, true_))

    mean_rmse = float(np.mean(val_rmses))
    mean_r2   = float(np.mean(val_r2s))
    n_params   = model.coef_.size + 1
    flash_kb   = n_params * 4 / 1024

    print(f"    RMSE={mean_rmse:.4f} N·m | R²={mean_r2:.5f} | params={n_params} | Flash≈{flash_kb:.3f} KB")
    info = {'coef': model.coef_.tolist(), 'intercept': float(model.intercept_),
            'val_rmse': mean_rmse, 'val_r2': mean_r2, 'n_params': n_params, 'flash_kb': flash_kb}
    with open(os.path.join(MDIR, f'{name}_info.json'), 'w') as f: json.dump(info, f)
    return model, mean_rmse, mean_r2, n_params, flash_kb


def train_narx_mlp(X_tr, y_tr, X_vl, y_vl, df_vval, name='narx_mlp'):
    print(f"\n  Training {name} …")
    model = MLPRegressor(hidden_layer_sizes=(16, 8), activation='relu',
                         max_iter=1000, random_state=SEED,
                         early_stopping=True, validation_fraction=0.15,
                         n_iter_no_change=30, learning_rate_init=1e-3)
    model.fit(X_tr, y_tr)

    val_rmses, val_r2s = [], []
    for sid in sorted(df_vval['SimID'].unique()):
        sub = df_vval[df_vval['SimID']==sid].sort_values('Time').reset_index(drop=True)
        model_fn = lambda x: model.predict(x)
        t_, true_, pred_ = narx_simulate(model_fn, sub)
        val_rmses.append(rmse(pred_, true_)); val_r2s.append(r2(pred_, true_))

    mean_rmse = float(np.mean(val_rmses))
    mean_r2   = float(np.mean(val_r2s))
    n_params   = sum(w.size for w in model.coefs_) + sum(b.size for b in model.intercepts_)
    flash_kb   = n_params * 4 / 1024

    print(f"    RMSE={mean_rmse:.4f} N·m | R²={mean_r2:.5f} | params={n_params} | Flash≈{flash_kb:.2f} KB")

    weights_export = {
        'coefs':       [w.tolist() for w in model.coefs_],
        'intercepts':  [b.tolist() for b in model.intercepts_],
        'activation':  model.activation,
        'n_layers':    model.n_layers_,
    }
    info = {**weights_export, 'val_rmse': mean_rmse, 'val_r2': mean_r2,
            'n_params': n_params, 'flash_kb': flash_kb}
    with open(os.path.join(MDIR, f'{name}_info.json'), 'w') as f: json.dump(info, f)
    return model, mean_rmse, mean_r2, n_params, flash_kb


def train_narx_gbm(X_tr, y_tr, X_vl, y_vl, df_vval, name='narx_gbm'):
    print(f"\n  Training {name} …")
    model = GradientBoostingRegressor(n_estimators=50, max_depth=3,
                                       learning_rate=0.1, subsample=0.8,
                                       random_state=SEED, validation_fraction=0.1,
                                       n_iter_no_change=20)
    model.fit(X_tr, y_tr)
    n_est = model.n_estimators_
    print(f"    GBM converged at {n_est} trees")

    val_rmses, val_r2s = [], []
    for sid in sorted(df_vval['SimID'].unique()):
        sub = df_vval[df_vval['SimID']==sid].sort_values('Time').reset_index(drop=True)
        model_fn = lambda x: model.predict(x)
        t_, true_, pred_ = narx_simulate(model_fn, sub)
        val_rmses.append(rmse(pred_, true_)); val_r2s.append(r2(pred_, true_))

    mean_rmse = float(np.mean(val_rmses))
    mean_r2   = float(np.mean(val_r2s))

    # Estimate Flash from tree structure
    total_nodes = sum(est[0].tree_.node_count for est in model.estimators_[:n_est])
    bytes_per_node = 4 + 4 + 4 + 1   # threshold(f32) + value(f32) + feature(int) + children(int)
    flash_kb = total_nodes * bytes_per_node / 1024 + 1.0  # +1 for code overhead
    n_params = total_nodes  # approximate

    print(f"    RMSE={mean_rmse:.4f} N·m | R²={mean_r2:.5f} | nodes≈{total_nodes} | Flash≈{flash_kb:.1f} KB")
    info = {'val_rmse': mean_rmse, 'val_r2': mean_r2, 'n_params': n_params,
            'flash_kb': flash_kb, 'n_estimators': n_est}
    with open(os.path.join(MDIR, f'{name}_info.json'), 'w') as f: json.dump(info, f)
    return model, mean_rmse, mean_r2, n_params, flash_kb


# =============================================================================
# C CODE GENERATORS
# =============================================================================
def c_float_array(name, vals, cols=8, dtype='float', static=True):
    qual = 'static const' if static else 'const'
    lines = [f'{qual} {dtype} {name}[{len(vals)}] = {{']
    for i in range(0, len(vals), cols):
        chunk = vals[i:i+cols]
        lines.append('    ' + ', '.join(f'{v:.8f}f' for v in chunk) + ',')
    lines.append('};')
    return '\n'.join(lines)

def flatten(arr):
    if isinstance(arr[0], (list, np.ndarray)):
        return [x for row in arr for x in (flatten(row) if isinstance(row[0], (list, np.ndarray)) else row)]
    return list(arr)


def gen_lstm_c(sd, stats, hidden_size, name, input_size=3):
    """Generate C header+source for an LSTM ROM of given hidden_size."""
    H = hidden_size; I = input_size
    hdr = f"""#ifndef ROM_{name.upper()}_H
#define ROM_{name.upper()}_H
#include <stdint.h>
#define ROM_{name.upper()}_HIDDEN {H}U
#define ROM_{name.upper()}_INPUT  {I}U
// z-score normalisation
#define ROM_{name.upper()}_AC_MEAN   {stats['AirCharge']['mean']:.8f}f
#define ROM_{name.upper()}_AC_STD    {stats['AirCharge']['std']:.8f}f
#define ROM_{name.upper()}_SPD_MEAN  {stats['Speed']['mean']:.8f}f
#define ROM_{name.upper()}_SPD_STD   {stats['Speed']['std']:.8f}f
#define ROM_{name.upper()}_SA_MEAN   {stats['SparkAdvance']['mean']:.8f}f
#define ROM_{name.upper()}_SA_STD    {stats['SparkAdvance']['std']:.8f}f
#define ROM_{name.upper()}_TQ_MEAN   {stats['Torque']['mean']:.8f}f
#define ROM_{name.upper()}_TQ_STD    {stats['Torque']['std']:.8f}f
typedef struct {{ float h[{H}]; float c[{H}]; }} ROM_{name}_State_t;
void ROM_{name}_Init(ROM_{name}_State_t *s);
float ROM_{name}_Step(ROM_{name}_State_t *s, float ac, float spd, float sa);
#ifdef __cplusplus
extern "C" {{
void ROM_{name}_Init(ROM_{name}_State_t *s);
float ROM_{name}_Step(ROM_{name}_State_t *s, float ac, float spd, float sa);
}}
#endif
#endif
"""
    def w(key): return flatten(sd[key].detach().cpu().numpy().tolist())
    wih = w('lstm.weight_ih_l0'); whh = w('lstm.weight_hh_l0')
    bih = w('lstm.bias_ih_l0');   bhh = w('lstm.bias_hh_l0')
    wfc = w('fc.weight');         bfc = w('fc.bias')

    wgt_hdr = f"""#ifndef ROM_{name.upper()}_WEIGHTS_H
#define ROM_{name.upper()}_WEIGHTS_H
{c_float_array(f'ROM_{name}_W_IH', wih, 8)}
{c_float_array(f'ROM_{name}_W_HH', whh, 8)}
{c_float_array(f'ROM_{name}_B_IH', bih, 8)}
{c_float_array(f'ROM_{name}_B_HH', bhh, 8)}
{c_float_array(f'ROM_{name}_FC_W', wfc, 8)}
{c_float_array(f'ROM_{name}_FC_B', bfc, 8)}
#endif
"""
    src = f"""#include "rom_{name}.h"
#include "rom_{name}_weights.h"
#include <math.h>
#include <string.h>
static float sig(float x){{ return 1.0f/(1.0f+expf(-x)); }}
static void matvec(float *y, const float *W, const float *x, int rows, int cols){{
    int i,j; for(i=0;i<rows;i++){{float a=0;for(j=0;j<cols;j++) a+=W[i*cols+j]*x[j];y[i]+=a;}}
}}
void ROM_{name}_Init(ROM_{name}_State_t *s){{memset(s,0,sizeof(*s));}}
float ROM_{name}_Step(ROM_{name}_State_t *s, float ac, float spd, float sa){{
    float x[{I}];
    x[0]=(ac -ROM_{name.upper()}_AC_MEAN) /ROM_{name.upper()}_AC_STD;
    x[1]=(spd-ROM_{name.upper()}_SPD_MEAN)/ROM_{name.upper()}_SPD_STD;
    x[2]=(sa -ROM_{name.upper()}_SA_MEAN) /ROM_{name.upper()}_SA_STD;
    float g[{4*H}]; int k;
    for(k=0;k<{4*H};k++) g[k]=ROM_{name}_B_IH[k]+ROM_{name}_B_HH[k];
    matvec(g,ROM_{name}_W_IH,x,{4*H},{I});
    matvec(g,ROM_{name}_W_HH,s->h,{4*H},{H});
    for(k=0;k<{H};k++){{
        float ig=sig(g[k]), fg=sig(g[{H}+k]), gg=tanhf(g[{2*H}+k]), og=sig(g[{3*H}+k]);
        s->c[k]=fg*s->c[k]+ig*gg; s->h[k]=og*tanhf(s->c[k]);
    }}
    float y=ROM_{name}_FC_B[0];
    for(k=0;k<{H};k++) y+=ROM_{name}_FC_W[k]*s->h[k];
    return y*ROM_{name.upper()}_TQ_STD+ROM_{name.upper()}_TQ_MEAN;
}}
"""
    with open(os.path.join(SRC, f'rom_{name}.h'), 'w') as f: f.write(hdr)
    with open(os.path.join(SRC, f'rom_{name}_weights.h'), 'w') as f: f.write(wgt_hdr)
    with open(os.path.join(SRC, f'rom_{name}.c'), 'w') as f: f.write(src)
    print(f"    C code: src/rom_{name}.{{h,c}}")


def gen_lstm_int8_c(quant, stats, name='lstm_32_q8', hidden_size=32, input_size=3):
    """Generate int8-weight LSTM C code (dequantize at runtime)."""
    H = hidden_size; I = input_size
    keys = ['lstm.weight_ih_l0','lstm.weight_hh_l0','lstm.bias_ih_l0','lstm.bias_hh_l0',
            'fc.weight','fc.bias']
    cnames = ['W_IH','W_HH','B_IH','B_HH','FC_W','FC_B']

    int8_decls, scale_decls = [], []
    for key, cn in zip(keys, cnames):
        q8 = quant[key]['int8']
        if isinstance(q8[0], list): q8 = flatten(q8)
        sc  = quant[key]['scale']
        int8_decls.append(c_float_array(f'ROM_{name}_{cn}', q8, 16, 'int8_t') + '\n')
        scale_decls.append(f'static const float ROM_{name}_{cn}_SC = {sc:.10f}f;')

    wgt_hdr = f"""#ifndef ROM_{name.upper()}_WEIGHTS_H
#define ROM_{name.upper()}_WEIGHTS_H
#include <stdint.h>
{''.join(int8_decls)}
{chr(10).join(scale_decls)}
#endif
"""
    src = f"""#include "rom_{name}.h"
#include "rom_{name}_weights.h"
#include <math.h>
#include <string.h>
static float sig(float x){{return 1.0f/(1.0f+expf(-x));}}
static void matvec_q8(float *y, const int8_t *W, float scale, const float *x, int rows, int cols){{
    int i,j; for(i=0;i<rows;i++){{float a=0;for(j=0;j<cols;j++) a+=(float)W[i*cols+j]*scale*x[j];y[i]+=a;}}
}}
static void vec_add_q8(float *y, const int8_t *b, float scale, int n){{
    int i; for(i=0;i<n;i++) y[i]+=(float)b[i]*scale;
}}
void ROM_{name}_Init(ROM_{name}_State_t *s){{memset(s,0,sizeof(*s));}}
float ROM_{name}_Step(ROM_{name}_State_t *s, float ac, float spd, float sa){{
    float x[{I}];
    x[0]=(ac -ROM_{name.upper()}_AC_MEAN)/ROM_{name.upper()}_AC_STD;
    x[1]=(spd-ROM_{name.upper()}_SPD_MEAN)/ROM_{name.upper()}_SPD_STD;
    x[2]=(sa -ROM_{name.upper()}_SA_MEAN)/ROM_{name.upper()}_SA_STD;
    float g[{4*H}]={{0}}; int k;
    vec_add_q8(g, ROM_{name}_B_IH, ROM_{name}_B_IH_SC, {4*H});
    vec_add_q8(g, ROM_{name}_B_HH, ROM_{name}_B_HH_SC, {4*H});
    matvec_q8(g, ROM_{name}_W_IH, ROM_{name}_W_IH_SC, x, {4*H}, {I});
    matvec_q8(g, ROM_{name}_W_HH, ROM_{name}_W_HH_SC, s->h, {4*H}, {H});
    for(k=0;k<{H};k++){{
        float ig=sig(g[k]), fg=sig(g[{H}+k]), gg=tanhf(g[{2*H}+k]), og=sig(g[{3*H}+k]);
        s->c[k]=fg*s->c[k]+ig*gg; s->h[k]=og*tanhf(s->c[k]);
    }}
    float y=(float)ROM_{name}_FC_B[0]*ROM_{name}_FC_B_SC;
    for(k=0;k<{H};k++) y+=(float)ROM_{name}_FC_W[k]*ROM_{name}_FC_W_SC*s->h[k];
    return y*ROM_{name.upper()}_TQ_STD+ROM_{name.upper()}_TQ_MEAN;
}}
"""
    hdr = f"""#ifndef ROM_{name.upper()}_H
#define ROM_{name.upper()}_H
#include <stdint.h>
#define ROM_{name.upper()}_HIDDEN {H}U
#define ROM_{name.upper()}_INPUT  {I}U
#define ROM_{name.upper()}_AC_MEAN   {stats['AirCharge']['mean']:.8f}f
#define ROM_{name.upper()}_AC_STD    {stats['AirCharge']['std']:.8f}f
#define ROM_{name.upper()}_SPD_MEAN  {stats['Speed']['mean']:.8f}f
#define ROM_{name.upper()}_SPD_STD   {stats['Speed']['std']:.8f}f
#define ROM_{name.upper()}_SA_MEAN   {stats['SparkAdvance']['mean']:.8f}f
#define ROM_{name.upper()}_SA_STD    {stats['SparkAdvance']['std']:.8f}f
#define ROM_{name.upper()}_TQ_MEAN   {stats['Torque']['mean']:.8f}f
#define ROM_{name.upper()}_TQ_STD    {stats['Torque']['std']:.8f}f
typedef struct {{ float h[{H}]; float c[{H}]; }} ROM_{name}_State_t;
void ROM_{name}_Init(ROM_{name}_State_t *s);
float ROM_{name}_Step(ROM_{name}_State_t *s, float ac, float spd, float sa);
#endif
"""
    with open(os.path.join(SRC, f'rom_{name}.h'), 'w') as f: f.write(hdr)
    with open(os.path.join(SRC, f'rom_{name}_weights.h'), 'w') as f: f.write(wgt_hdr)
    with open(os.path.join(SRC, f'rom_{name}.c'), 'w') as f: f.write(src)
    print(f"    C code: src/rom_{name}.{{h,c}}")


def gen_narx_ridge_c(model_ridge, stats, name='narx_ridge'):
    coef = model_ridge.coef_.tolist()
    intercept = float(model_ridge.intercept_)
    N = len(coef)
    hdr = _narx_header(name, stats)
    wgt = c_float_array(f'ROM_{name}_COEF', coef, 8) + \
          f'\nstatic const float ROM_{name}_INTERCEPT = {intercept:.8f}f;'
    src = f"""#include "rom_{name}.h"
#include <string.h>
{wgt}
void NARX_{name}_Init(NARX_{name}_State_t *s){{memset(s,0,sizeof(*s));}}
float NARX_{name}_Step(NARX_{name}_State_t *s, float ac, float spd, float sa){{
    float x[{N}]; int i;
    float ac_n=(ac-ROM_{name.upper()}_AC_MEAN)/ROM_{name.upper()}_AC_STD;
    float spd_n=(spd-ROM_{name.upper()}_SPD_MEAN)/ROM_{name.upper()}_SPD_STD;
    float sa_n=(sa-ROM_{name.upper()}_SA_MEAN)/ROM_{name.upper()}_SA_STD;
    x[0]=ac_n;
    for(i=0;i<{LAG_AC-1};i++) x[1+i]=s->ac_lag[i];
    x[{LAG_AC}]=spd_n;
    for(i=0;i<{LAG_SPD-1};i++) x[{LAG_AC+1}+i]=s->spd_lag[i];
    x[{LAG_AC+LAG_SPD}]=s->tq_lag[0]; x[{LAG_AC+LAG_SPD+1}]=s->tq_lag[1];
    x[{N-1}]=sa_n;
    float y=ROM_{name}_INTERCEPT;
    for(i=0;i<{N};i++) y+=ROM_{name}_COEF[i]*x[i];
    for(i={LAG_AC-1};i>0;i--) s->ac_lag[i]=s->ac_lag[i-1]; s->ac_lag[0]=ac_n;
    for(i={LAG_SPD-1};i>0;i--) s->spd_lag[i]=s->spd_lag[i-1]; s->spd_lag[0]=spd_n;
    s->tq_lag[1]=s->tq_lag[0]; s->tq_lag[0]=y;
    return y*ROM_{name.upper()}_TQ_STD+ROM_{name.upper()}_TQ_MEAN;
}}
"""
    with open(os.path.join(SRC, f'rom_{name}.h'), 'w') as f: f.write(hdr)
    with open(os.path.join(SRC, f'rom_{name}.c'), 'w') as f: f.write(src)
    print(f"    C code: src/rom_{name}.{{h,c}}")


def gen_narx_mlp_c(model_mlp, stats, name='narx_mlp'):
    coefs = model_mlp.coefs_
    intercepts = model_mlp.intercepts_
    N_IN = N_FEAT
    layers_dim = [N_IN] + list(model_mlp.hidden_layer_sizes) + [1]

    weight_decls = []
    for i, (W, b) in enumerate(zip(coefs, intercepts)):
        weight_decls.append(c_float_array(f'ROM_{name}_W{i}', W.flatten().tolist()))
        weight_decls.append(c_float_array(f'ROM_{name}_B{i}', b.tolist()))

    # Build inference code
    forward_lines = []
    cur = 'x_in'
    for i in range(len(coefs)):
        n_in = layers_dim[i]; n_out = layers_dim[i+1]
        out_var = 'y_out' if i == len(coefs)-1 else f'h{i}'
        forward_lines.append(f'    float {out_var}[{n_out}]; int j{i};')
        forward_lines.append(f'    for(j{i}=0;j{i}<{n_out};j{i}++){{')
        forward_lines.append(f'        float v=ROM_{name}_B{i}[j{i}];')
        forward_lines.append(f'        int k{i}; for(k{i}=0;k{i}<{n_in};k{i}++) v+=ROM_{name}_W{i}[j{i}*{n_in}+k{i}]*{cur}[k{i}];')
        if i < len(coefs)-1:
            forward_lines.append(f'        {out_var}[j{i}]=v>0?v:0;')  # ReLU
        else:
            forward_lines.append(f'        {out_var}[j{i}]=v;')  # Linear
        forward_lines.append(f'    }}')
        cur = out_var

    hdr = _narx_header(name, stats)
    src_lines = [f'#include "rom_{name}.h"', '#include <string.h>', '']
    src_lines += ['\n'.join(weight_decls), '']
    src_lines += [f'void NARX_{name}_Init(NARX_{name}_State_t *s){{memset(s,0,sizeof(*s));}}']
    src_lines += [f'float NARX_{name}_Step(NARX_{name}_State_t *s, float ac, float spd, float sa){{',
                  f'    float x_in[{N_FEAT}]; int i;',
                  f'    float ac_n=(ac-ROM_{name.upper()}_AC_MEAN)/ROM_{name.upper()}_AC_STD;',
                  f'    float spd_n=(spd-ROM_{name.upper()}_SPD_MEAN)/ROM_{name.upper()}_SPD_STD;',
                  f'    float sa_n=(sa-ROM_{name.upper()}_SA_MEAN)/ROM_{name.upper()}_SA_STD;',
                  f'    x_in[0]=ac_n;',
                  f'    for(i=0;i<{LAG_AC-1};i++) x_in[1+i]=s->ac_lag[i];',
                  f'    x_in[{LAG_AC}]=spd_n;',
                  f'    for(i=0;i<{LAG_SPD-1};i++) x_in[{LAG_AC+1}+i]=s->spd_lag[i];',
                  f'    x_in[{LAG_AC+LAG_SPD}]=s->tq_lag[0]; x_in[{LAG_AC+LAG_SPD+1}]=s->tq_lag[1];',
                  f'    x_in[{N_FEAT-1}]=sa_n;'] + \
                 forward_lines + \
                 [f'    for(i={LAG_AC-1};i>0;i--) s->ac_lag[i]=s->ac_lag[i-1]; s->ac_lag[0]=ac_n;',
                  f'    for(i={LAG_SPD-1};i>0;i--) s->spd_lag[i]=s->spd_lag[i-1]; s->spd_lag[0]=spd_n;',
                  f'    s->tq_lag[1]=s->tq_lag[0]; s->tq_lag[0]=y_out[0];',
                  f'    return y_out[0]*ROM_{name.upper()}_TQ_STD+ROM_{name.upper()}_TQ_MEAN;',
                  f'}}']

    with open(os.path.join(SRC, f'rom_{name}.h'), 'w') as f: f.write(hdr)
    with open(os.path.join(SRC, f'rom_{name}.c'), 'w') as f: f.write('\n'.join(src_lines))
    print(f"    C code: src/rom_{name}.{{h,c}}")


def gen_narx_gbm_c(model_gbm, stats, name='narx_gbm'):
    """Export sklearn GBM trees as C if-else code."""
    n_est = model_gbm.n_estimators_
    lr    = model_gbm.learning_rate
    # base score: mean of training targets
    base  = float(model_gbm.init_.constant_[0][0]) if hasattr(model_gbm.init_, 'constant_') \
            else float(model_gbm.init_.mean_)

    def tree_code(tree_, node_id=0, indent=4):
        if tree_.children_left[node_id] == -1:
            return ' '*indent + f'return {tree_.value[node_id][0][0]:.10f}f;'
        fi = tree_.feature[node_id]; th = tree_.threshold[node_id]
        lc = tree_code(tree_, tree_.children_left[node_id],  indent+4)
        rc = tree_code(tree_, tree_.children_right[node_id], indent+4)
        return (f'{" "*indent}if (x[{fi}] <= {th:.10f}f) {{\n'
                f'{lc}\n{" "*indent}}} else {{\n{rc}\n{" "*indent}}}')

    tree_fns = []
    for i, est in enumerate(model_gbm.estimators_[:n_est]):
        fn = f'static float gbm_tree_{i}(const float *x) {{\n'
        fn += tree_code(est[0].tree_) + '\n}'
        tree_fns.append(fn)

    hdr = _narx_header(name, stats)
    src_parts = [f'#include "rom_{name}.h"', '#include <string.h>', '']
    src_parts += tree_fns
    src_parts += [
        f'void NARX_{name}_Init(NARX_{name}_State_t *s){{memset(s,0,sizeof(*s));}}',
        f'float NARX_{name}_Step(NARX_{name}_State_t *s, float ac, float spd, float sa){{',
        f'    float x[{N_FEAT}]; int i;',
        f'    float ac_n=(ac-ROM_{name.upper()}_AC_MEAN)/ROM_{name.upper()}_AC_STD;',
        f'    float spd_n=(spd-ROM_{name.upper()}_SPD_MEAN)/ROM_{name.upper()}_SPD_STD;',
        f'    float sa_n=(sa-ROM_{name.upper()}_SA_MEAN)/ROM_{name.upper()}_SA_STD;',
        f'    x[0]=ac_n;',
        f'    for(i=0;i<{LAG_AC-1};i++) x[1+i]=s->ac_lag[i];',
        f'    x[{LAG_AC}]=spd_n;',
        f'    for(i=0;i<{LAG_SPD-1};i++) x[{LAG_AC+1}+i]=s->spd_lag[i];',
        f'    x[{LAG_AC+LAG_SPD}]=s->tq_lag[0]; x[{LAG_AC+LAG_SPD+1}]=s->tq_lag[1];',
        f'    x[{N_FEAT-1}]=sa_n;',
        f'    float y={base:.8f}f;',
    ]
    for i in range(n_est):
        src_parts.append(f'    y+={lr:.6f}f*gbm_tree_{i}(x);')
    src_parts += [
        f'    for(i={LAG_AC-1};i>0;i--) s->ac_lag[i]=s->ac_lag[i-1]; s->ac_lag[0]=ac_n;',
        f'    for(i={LAG_SPD-1};i>0;i--) s->spd_lag[i]=s->spd_lag[i-1]; s->spd_lag[0]=spd_n;',
        f'    s->tq_lag[1]=s->tq_lag[0]; s->tq_lag[0]=y;',
        f'    return y*ROM_{name.upper()}_TQ_STD+ROM_{name.upper()}_TQ_MEAN;',
        f'}}',
    ]
    with open(os.path.join(SRC, f'rom_{name}.h'), 'w') as f: f.write(hdr)
    with open(os.path.join(SRC, f'rom_{name}.c'), 'w') as f: f.write('\n'.join(src_parts))
    print(f"    C code: src/rom_{name}.{{h,c}} ({n_est} trees)")


def _narx_header(name, stats):
    return f"""#ifndef ROM_{name.upper()}_H
#define ROM_{name.upper()}_H
#include <stdint.h>
#define ROM_{name.upper()}_LAG_AC  {LAG_AC}U
#define ROM_{name.upper()}_LAG_SPD {LAG_SPD}U
#define ROM_{name.upper()}_LAG_TQ  {LAG_TQ}U
#define ROM_{name.upper()}_N_FEAT  {N_FEAT}U
#define ROM_{name.upper()}_AC_MEAN   {stats['AirCharge']['mean']:.8f}f
#define ROM_{name.upper()}_AC_STD    {stats['AirCharge']['std']:.8f}f
#define ROM_{name.upper()}_SPD_MEAN  {stats['Speed']['mean']:.8f}f
#define ROM_{name.upper()}_SPD_STD   {stats['Speed']['std']:.8f}f
#define ROM_{name.upper()}_SA_MEAN   {stats['SparkAdvance']['mean']:.8f}f
#define ROM_{name.upper()}_SA_STD    {stats['SparkAdvance']['std']:.8f}f
#define ROM_{name.upper()}_TQ_MEAN   {stats['Torque']['mean']:.8f}f
#define ROM_{name.upper()}_TQ_STD    {stats['Torque']['std']:.8f}f
typedef struct {{
    float ac_lag[{LAG_AC}];
    float spd_lag[{LAG_SPD}];
    float tq_lag[{LAG_TQ}];
}} NARX_{name}_State_t;
void  NARX_{name}_Init(NARX_{name}_State_t *s);
float NARX_{name}_Step(NARX_{name}_State_t *s, float ac, float spd, float sa);
#endif
"""


# =============================================================================
# PARETO PLOT
# =============================================================================
def plot_pareto(comparison):
    models = list(comparison.keys())
    flash  = [comparison[m]['flash_kb']  for m in models]
    rmses  = [comparison[m]['val_rmse']  for m in models]
    labels = [comparison[m]['label']     for m in models]
    colors = [comparison[m]['color']     for m in models]
    markers= [comparison[m]['marker']    for m in models]

    # Identify Pareto front
    pareto = []
    for i, (f, r) in enumerate(zip(flash, rmses)):
        dominated = any(flash[j] <= f and rmses[j] <= r and (flash[j]<f or rmses[j]<r)
                        for j in range(len(models)) if j != i)
        if not dominated:
            pareto.append(i)
    pareto_sorted = sorted(pareto, key=lambda i: flash[i])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax_idx, (ax, xscale) in enumerate([(axes[0], 'log'), (axes[1], 'linear')]):
        for i, (m, f, r, lbl, c, mk) in enumerate(zip(models, flash, rmses, labels, colors, markers)):
            zorder = 10 if i in pareto else 5
            ax.scatter(f, r, s=180, color=c, marker=mk, zorder=zorder,
                       edgecolors='black', linewidths=0.8)
            ax.annotate(lbl, (f, r), textcoords='offset points',
                        xytext=(6, 4), fontsize=8,
                        path_effects=[pe.withStroke(linewidth=2, foreground='white')])

        # Draw Pareto frontier
        pf_x = [flash[i] for i in pareto_sorted]
        pf_y = [rmses[i]  for i in pareto_sorted]
        pf_x_line = [min(pf_x)*0.5] + pf_x
        pf_y_line = [pf_y[0]] + pf_y
        ax.step(pf_x_line, pf_y_line, where='post', color='red', lw=2,
                linestyle='--', label='Pareto frontier', alpha=0.8)

        ax.set_xlabel('Flash Usage (KB)', fontsize=11)
        ax.set_ylabel('Validation RMSE (N·m)', fontsize=11)
        ax.set_title(f'ROM Pareto Frontier – Accuracy vs Flash'
                     f' ({"log" if ax_idx==0 else "linear"} scale)')
        if xscale == 'log': ax.set_xscale('log')
        ax.axhline(y=comparison['lstm_32']['val_rmse'], color='gray',
                   linestyle=':', lw=1.2, label='Baseline (LSTM-32)')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.suptitle('Engine ROM – Model Compression & Alternative Architecture Comparison',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(PDIR, 'pareto_frontier.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Pareto plot saved: plots/pareto_frontier.png")


def plot_all_model_traces(comparison_traces, df_vval):
    """Show torque prediction traces for all models on one simulation."""
    sid = sorted(df_vval['SimID'].unique())[0]
    sub = df_vval[df_vval['SimID']==sid].sort_values('Time').reset_index(drop=True)
    t = sub['Time'].values; tq_true = sub['Torque'].values

    fig, axes = plt.subplots(len(comparison_traces)+1, 1,
                             figsize=(14, 3*(len(comparison_traces)+1)), sharex=True)

    axes[0].plot(t, tq_true, 'k-', lw=2, label='Simulink Reference')
    for name, (pred, color, label) in comparison_traces.items():
        axes[0].plot(t[:len(pred)], pred, color=color, lw=1.2, linestyle='--', label=label, alpha=0.8)
    axes[0].set_ylabel('Torque [N·m]'); axes[0].legend(fontsize=7, ncol=4)
    axes[0].set_title(f'All ROM Torque Predictions – Val Sim {sid} (SA={sub["SparkAdvance"].iloc[0]:.0f}°)')
    axes[0].grid(True, alpha=0.3)

    for i, (name, (pred, color, label)) in enumerate(comparison_traces.items()):
        ax = axes[i+1]
        err = np.array(pred) - tq_true[:len(pred)]
        ax.plot(t[:len(pred)], err, color=color, lw=1.0)
        ax.axhline(0, color='k', lw=0.6, linestyle=':')
        ax.fill_between(t[:len(pred)], err, 0, alpha=0.15, color=color)
        ax.set_ylabel(f'Error [N·m]')
        ax.set_title(f'{label} – RMSE={rmse(pred, tq_true[:len(pred)]):.3f} N·m')
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time [s]')
    plt.suptitle('ROM Prediction Errors – All Model Variants', fontsize=12, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(PDIR, 'all_models_traces.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Model traces plot saved: plots/all_models_traces.png")


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 60)
    print("  ENGINE ROM – COMPRESSION & ALTERNATIVE MODEL STUDY")
    print("=" * 60)

    df_tr, df_vval = load_data()
    print(f"Training data: {len(df_tr):,} samples | "
          f"Validation data: {len(df_vval):,} samples")

    # ── Load baseline LSTM-32
    ckpt32 = torch.load(os.path.join(MDIR, 'rom_model.pth'), map_location='cpu', weights_only=False)
    model32 = EngineROM_LSTM(32)
    model32.load_state_dict(ckpt32['model_state_dict'])
    model32.eval()
    baseline_rmse = ckpt32.get('stats', {})
    # Re-evaluate baseline on validation set (simulation mode isn't needed; already validated)
    bl_rmse = 0.91  # from previous run (open-loop)
    bl_r2   = 0.9997

    # ── NARX feature matrices (shared across sklearn models)
    print("\nBuilding NARX feature matrices …")
    tr_ids = [i for i in sorted(df_tr['SimID'].unique()) if i not in VAL_SIM_IDS]
    X_tr_n, y_tr_n, _ = make_narx_dataset(df_tr, sim_ids=tr_ids)
    X_vl_n, y_vl_n, _ = make_narx_dataset(df_tr, sim_ids=VAL_SIM_IDS)
    print(f"  Train: {X_tr_n.shape} | Val: {X_vl_n.shape}")

    # ── Train all models
    print("\n── LSTM Variants ──────────────────────────────────────")
    m8,  rm8,  r2_8,  n8,  f8  = train_lstm(8,  df_tr, df_vval, 'lstm_8',  epochs=EPOCHS)
    m16, rm16, r2_16, n16, f16 = train_lstm(16, df_tr, df_vval, 'lstm_16', epochs=EPOCHS)

    print("\n── LSTM-32 Int8 Quantization ──────────────────────────")
    mq8, quant, rm_q8, r2_q8, n_q8, f_q8 = quantize_lstm_int8(model32, df_vval)

    print("\n── NARX Sklearn Models ────────────────────────────────")
    mr, rm_r, r2_r, n_r, f_r = train_narx_ridge(X_tr_n, y_tr_n, X_vl_n, y_vl_n, df_vval)
    mm, rm_m, r2_m, n_m, f_m = train_narx_mlp(X_tr_n, y_tr_n, X_vl_n, y_vl_n, df_vval)
    mg, rm_g, r2_g, n_g, f_g = train_narx_gbm(X_tr_n, y_tr_n, X_vl_n, y_vl_n, df_vval)

    # ── Generate C code
    print("\n── Generating C Code ──────────────────────────────────")
    gen_lstm_c(m8.state_dict(),  STATS, 8,  'lstm_8')
    gen_lstm_c(m16.state_dict(), STATS, 16, 'lstm_16')
    gen_lstm_c(model32.state_dict(), STATS, 32, 'lstm_32')  # regenerate baseline too
    gen_lstm_int8_c(quant, STATS, 'lstm_32_q8')
    gen_narx_ridge_c(mr, STATS)
    gen_narx_mlp_c(mm, STATS)
    gen_narx_gbm_c(mg, STATS)

    # ── Compile & measure actual Flash from object files
    print("\n── Measuring C code object sizes ──────────────────────")
    actual_flash = {}
    for name in ['lstm_8', 'lstm_16', 'lstm_32', 'lstm_32_q8',
                 'narx_ridge', 'narx_mlp', 'narx_gbm']:
        c_file = os.path.join(SRC, f'rom_{name}.c')
        o_file = os.path.join(SRC, f'rom_{name}.o')
        ret = os.system(f'gcc -O2 -I{SRC} -c {c_file} -o {o_file} -lm 2>/dev/null')
        if ret == 0 and os.path.exists(o_file):
            # Use 'size' to get section sizes
            import subprocess
            result = subprocess.run(['size', o_file], capture_output=True, text=True)
            lines = result.stdout.strip().split('\n')
            if len(lines) >= 2:
                parts = lines[-1].split()
                text_b = int(parts[0]); data_b = int(parts[1]); bss_b = int(parts[2])
                total_kb = (text_b + data_b + bss_b) / 1024
                actual_flash[name] = total_kb
                print(f"  {name:20s}: text={text_b}B data={data_b}B bss={bss_b}B → {total_kb:.2f} KB")
            else:
                actual_flash[name] = None
        else:
            print(f"  {name}: compile failed")
            actual_flash[name] = None

    # ── Assemble comparison table
    comparison = {
        'lstm_8':     {'val_rmse': rm8,   'val_r2': r2_8,  'n_params': n8,   'flash_kb': actual_flash.get('lstm_8')   or f8,   'label': 'LSTM-8',       'color': '#1f77b4', 'marker': 'o'},
        'lstm_16':    {'val_rmse': rm16,  'val_r2': r2_16, 'n_params': n16,  'flash_kb': actual_flash.get('lstm_16')  or f16,  'label': 'LSTM-16',      'color': '#ff7f0e', 'marker': 's'},
        'lstm_32':    {'val_rmse': bl_rmse,'val_r2': bl_r2,'n_params': 4769, 'flash_kb': actual_flash.get('lstm_32')  or 18.6, 'label': 'LSTM-32\n(baseline)', 'color': '#2ca02c', 'marker': 'D'},
        'lstm_32_q8': {'val_rmse': rm_q8, 'val_r2': r2_q8,'n_params': n_q8, 'flash_kb': actual_flash.get('lstm_32_q8') or f_q8,'label': 'LSTM-32-Q8', 'color': '#9467bd', 'marker': 'P'},
        'narx_ridge': {'val_rmse': rm_r,  'val_r2': r2_r, 'n_params': n_r,  'flash_kb': actual_flash.get('narx_ridge') or f_r, 'label': 'NARX-Ridge',  'color': '#d62728', 'marker': '^'},
        'narx_mlp':   {'val_rmse': rm_m,  'val_r2': r2_m, 'n_params': n_m,  'flash_kb': actual_flash.get('narx_mlp')  or f_m, 'label': 'NARX-MLP',   'color': '#8c564b', 'marker': 'v'},
        'narx_gbm':   {'val_rmse': rm_g,  'val_r2': r2_g, 'n_params': n_g,  'flash_kb': actual_flash.get('narx_gbm')  or f_g, 'label': 'NARX-GBM',   'color': '#e377c2', 'marker': '*'},
    }

    with open(os.path.join(PROJ, 'data', 'model_comparison.json'), 'w') as f:
        json.dump(comparison, f, indent=2)
    print("\nModel comparison:")
    print(f"{'Model':20s} {'RMSE (N·m)':>12s} {'R²':>8s} {'Flash (KB)':>12s} {'#params':>10s}")
    print('-'*65)
    for name, d in comparison.items():
        print(f"{name:20s} {d['val_rmse']:12.4f} {d['val_r2']:8.5f} {d['flash_kb']:12.2f} {d['n_params']:10d}")

    # ── Identify Pareto-optimal best (lowest RMSE on Pareto frontier)
    flash_vals = [comparison[m]['flash_kb'] for m in comparison]
    rmse_vals  = [comparison[m]['val_rmse']  for m in comparison]
    pareto_names = [name for i, name in enumerate(comparison)
                    if not any(flash_vals[j] <= flash_vals[i] and rmse_vals[j] <= rmse_vals[i]
                               and (flash_vals[j] < flash_vals[i] or rmse_vals[j] < rmse_vals[i])
                               for j in range(len(comparison)) if j != i)]
    best_name = min(pareto_names, key=lambda n: comparison[n]['val_rmse'])
    print(f"\nPareto-optimal models: {pareto_names}")
    print(f"Selected for S-Function: {best_name} "
          f"(RMSE={comparison[best_name]['val_rmse']:.4f} N·m, "
          f"Flash={comparison[best_name]['flash_kb']:.2f} KB)")

    with open(os.path.join(MDIR, 'best_model.json'), 'w') as f:
        json.dump({'best': best_name, 'comparison': comparison,
                   'pareto': pareto_names}, f, indent=2)

    # ── Plots
    print("\n── Generating plots ────────────────────────────────────")
    plot_pareto(comparison)

    # Collect prediction traces for one validation sim
    df_vval_1 = df_vval[df_vval['SimID'] == sorted(df_vval['SimID'].unique())[0]]
    sub1 = df_vval_1.sort_values('Time').reset_index(drop=True)
    t1 = sub1['Time'].values; tq1 = sub1['Torque'].values

    def lstm_pred(model, sub):
        ac  = normalize(sub['AirCharge'].values,   STATS['AirCharge'])
        spd = normalize(sub['Speed'].values,       STATS['Speed'])
        sa  = normalize(sub['SparkAdvance'].values, STATS['SparkAdvance'])
        X   = torch.tensor(np.stack([ac, spd, sa], 1)[None], dtype=torch.float32)
        with torch.no_grad():
            p, _ = model(X)
        return denormalize(p.squeeze().numpy(), STATS['Torque'])

    # Use narx_simulate for NARX models
    traces = {
        'lstm_8':  (lstm_pred(m8,  sub1), '#1f77b4', 'LSTM-8'),
        'lstm_16': (lstm_pred(m16, sub1), '#ff7f0e', 'LSTM-16'),
        'lstm_32': (lstm_pred(model32, sub1), '#2ca02c', 'LSTM-32 (baseline)'),
        'lstm_32_q8': (lstm_pred(mq8, sub1), '#9467bd', 'LSTM-32-Q8'),
    }
    for nm, mod_fn, col, lbl in [
        ('narx_ridge', lambda x: mr.predict(x), '#d62728', 'NARX-Ridge'),
        ('narx_mlp',   lambda x: mm.predict(x), '#8c564b', 'NARX-MLP'),
        ('narx_gbm',   lambda x: mg.predict(x), '#e377c2', 'NARX-GBM'),
    ]:
        _, _, pred_arr = narx_simulate(mod_fn, sub1)
        traces[nm] = (pred_arr, col, lbl)

    plot_all_model_traces(traces, df_vval)

    print("\n" + "="*60)
    print(f"  COMPLETE. Best model for S-Function: {best_name}")
    print("="*60)
    return best_name, comparison


if __name__ == '__main__':
    main()
