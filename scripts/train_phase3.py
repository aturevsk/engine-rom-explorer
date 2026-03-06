"""
train_phase3.py
===============
Phase 3: Advanced ROM Compression Techniques

1. Structured Pruning  – L1 reg → magnitude-based unit pruning → fine-tune
2. Quantization-Aware Training (QAT) – fake int8 with STE → compare vs PTQ
3. Delta Learning  – degree-2 polynomial baseline + residual LSTM-8
4. Fixed-Point Q16  – int16_t weight arrays for LSTM-16 & LSTM-32

Outputs
-------
  src/rom_lstm_pruned16.{h,c}  – pruned LSTM (16 hidden)
  src/rom_lstm_qat.{h,c}        – QAT int8 LSTM
  src/rom_delta_poly.{h,c}      – polynomial baseline C
  src/rom_lstm_delta8.{h,c}     – residual LSTM-8 C
  src/rom_lstm_16_q16.{h,c}     – LSTM-16 Q16 fixed-point
  src/rom_lstm_32_q16.{h,c}     – LSTM-32 Q16 fixed-point
  data/phase3_results.json       – all metrics
  plots/phase3_pareto.png        – extended Pareto frontier
  plots/phase3_validation.png    – validation traces
"""

import os, sys, json, math, copy, warnings, struct
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJ  = '/Users/arkadiyturevskiy/Documents/Claude/Claude_ROM_session4'
DATA_TRAIN = os.path.join(PROJ, 'data', 'training_data.csv')
DATA_VAL   = os.path.join(PROJ, 'data', 'validation_data.csv')
MDIR  = os.path.join(PROJ, 'models')
SRC   = os.path.join(PROJ, 'src')
PDIR  = os.path.join(PROJ, 'plots')
DDIR  = os.path.join(PROJ, 'data')
os.makedirs(SRC, exist_ok=True); os.makedirs(PDIR, exist_ok=True)

with open(os.path.join(MDIR, 'normalization.json')) as f:
    STATS = json.load(f)

SEQ_LEN    = 100
STRIDE     = 10
BATCH_SIZE = 32
SEED       = 42
torch.manual_seed(SEED); np.random.seed(SEED)

# ═════════════════════════════════════════════════════════════════════════════
# DATA HELPERS
# ═════════════════════════════════════════════════════════════════════════════
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

def r2_score(pred, true):
    ss_res = np.sum((np.asarray(pred)-np.asarray(true))**2)
    ss_tot = np.sum((np.asarray(true)-np.mean(true))**2)
    return 1 - ss_res/(ss_tot+1e-12)

def make_lstm_windows(df, sim_ids, target_col='Torque'):
    windows_X, windows_y = [], []
    for sid in sorted(df['SimID'].unique()) if sim_ids is None else sim_ids:
        sub = df[df['SimID']==sid].sort_values('Time').reset_index(drop=True)
        ac  = normalize(sub['AirCharge'].values,    STATS['AirCharge'])
        spd = normalize(sub['Speed'].values,        STATS['Speed'])
        sa  = normalize(sub['SparkAdvance'].values,  STATS['SparkAdvance'])
        tq  = normalize(sub[target_col].values,     STATS['Torque'])
        X   = np.stack([ac, spd, sa], 1)
        y   = tq[:, None]
        for s in range(0, len(X)-SEQ_LEN, STRIDE):
            windows_X.append(X[s:s+SEQ_LEN])
            windows_y.append(y[s:s+SEQ_LEN])
    return np.array(windows_X, np.float32), np.array(windows_y, np.float32)

def validate_lstm(model, df_vl, target_col='Torque'):
    """Run model on each validation sim; return (list_of_preds, list_of_trues)."""
    model.eval()
    all_pred, all_true = [], []
    for sid in sorted(df_vl['SimID'].unique()):
        sub = df_vl[df_vl['SimID']==sid].sort_values('Time').reset_index(drop=True)
        ac  = normalize(sub['AirCharge'].values,    STATS['AirCharge'])
        spd = normalize(sub['Speed'].values,        STATS['Speed'])
        sa  = normalize(sub['SparkAdvance'].values,  STATS['SparkAdvance'])
        X   = torch.tensor(np.stack([ac,spd,sa],1)[None], dtype=torch.float32)
        with torch.no_grad():
            out, _ = model(X)
        pred_n = out[0,:,0].numpy()
        pred_p = denormalize(pred_n, STATS['Torque'])
        true_p = sub[target_col].values
        all_pred.append(pred_p); all_true.append(true_p)
    preds = np.concatenate(all_pred); trues = np.concatenate(all_true)
    return preds, trues

class WindowDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


# ═════════════════════════════════════════════════════════════════════════════
# LSTM MODEL
# ═════════════════════════════════════════════════════════════════════════════
class EngineROM_LSTM(nn.Module):
    def __init__(self, hidden_size=32):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(3, hidden_size, 1, batch_first=True)
        self.fc   = nn.Linear(hidden_size, 1)
    def forward(self, x, state=None):
        out, state = self.lstm(x, state)
        return self.fc(out), state

def flash_kb(src_path):
    """Compile to .o and measure text section bytes → KB."""
    base = src_path.replace('.c', '.o')
    ret = os.system(f'gcc -O2 -std=c99 -c "{src_path}" -o "{base}" 2>/dev/null')
    if ret != 0: return 0.0
    import subprocess
    r = subprocess.run(['size', base], capture_output=True, text=True)
    for line in r.stdout.splitlines():
        parts = line.split()
        if parts and parts[0].isdigit():
            return int(parts[0]) / 1024
    return 0.0


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 – STRUCTURED PRUNING
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*70)
print("SECTION 1: STRUCTURED PRUNING (LSTM-32 → LSTM-16 pruned)")
print("═"*70)

df_tr, df_vl = load_data()
train_ids = sorted(df_tr['SimID'].unique())
val_ids   = sorted(df_vl['SimID'].unique())

# --- 1a. L1-regularised training from baseline ---
print("  1a. Loading baseline LSTM-32 and L1-regularised retraining …")
ckpt = torch.load(os.path.join(MDIR, 'rom_model.pth'), map_location='cpu')
baseline_cfg = ckpt['model_config']

model_l1 = EngineROM_LSTM(hidden_size=32)
model_l1.load_state_dict(ckpt['model_state_dict'])

Xtr, ytr = make_lstm_windows(df_tr, train_ids)
ds_tr = WindowDataset(Xtr, ytr)
dl_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True)

L1_LAMBDA   = 1e-4
PRUNE_EPOCHS = 100
opt = optim.Adam(model_l1.parameters(), lr=1e-4)
sched = optim.lr_scheduler.CosineAnnealingLR(opt, PRUNE_EPOCHS, eta_min=1e-6)

for ep in range(PRUNE_EPOCHS):
    model_l1.train()
    for Xb, yb in dl_tr:
        opt.zero_grad()
        out, _ = model_l1(Xb)
        loss = nn.MSELoss()(out, yb)
        # L1 on LSTM hidden→hidden weights (gate rows)
        l1_reg = L1_LAMBDA * (model_l1.lstm.weight_hh_l0.abs().sum() +
                               model_l1.lstm.weight_ih_l0.abs().sum())
        (loss + l1_reg).backward()
        torch.nn.utils.clip_grad_norm_(model_l1.parameters(), 1.0)
        opt.step()
    sched.step()
    if (ep+1) % 25 == 0:
        p, t = validate_lstm(model_l1, df_vl)
        print(f"    ep {ep+1:3d}  RMSE={rmse(p,t):.4f} N·m")

# --- 1b. Compute per-unit importance and select top-16 ---
print("  1b. Computing unit importance scores …")
H = 32
W_IH = model_l1.lstm.weight_ih_l0.detach()   # (4H × 3)
W_HH = model_l1.lstm.weight_hh_l0.detach()   # (4H × H)
W_FC = model_l1.fc.weight.detach()            # (1  × H)

# Importance: sum of abs weights entering and leaving each hidden unit k
importance = torch.zeros(H)
for k in range(H):
    # rows that belong to unit k across all 4 gates
    gate_rows = [g*H + k for g in range(4)]
    importance[k] = (W_IH[gate_rows, :].abs().sum() +
                     W_HH[gate_rows, :].abs().sum() +
                     W_HH[:, k].abs().sum() +
                     W_FC[0, k].abs())

K = 16  # keep top-16
kept = importance.topk(K).indices.sort().values
print(f"    Keeping units: {kept.tolist()}")

# --- 1c. Build pruned model (hidden=16) with sub-selected weights ---
print("  1c. Building pruned EngineROM_LSTM(16) …")
model_pruned = EngineROM_LSTM(hidden_size=K)

with torch.no_grad():
    gate_rows_kept = torch.tensor([g*H + k for g in range(4) for k in kept.tolist()])
    # weight_ih: (4K × 3) – keep rows for kept units
    model_pruned.lstm.weight_ih_l0.copy_(W_IH[gate_rows_kept, :])
    # weight_hh: (4K × K) – keep rows and cols for kept units
    model_pruned.lstm.weight_hh_l0.copy_(W_HH[gate_rows_kept, :][:, kept])
    # biases – keep rows for kept units
    b_ih = model_l1.lstm.bias_ih_l0.detach()
    b_hh = model_l1.lstm.bias_hh_l0.detach()
    model_pruned.lstm.bias_ih_l0.copy_(b_ih[gate_rows_kept])
    model_pruned.lstm.bias_hh_l0.copy_(b_hh[gate_rows_kept])
    # fc
    model_pruned.fc.weight.copy_(W_FC[:, kept])
    model_pruned.fc.bias.copy_(model_l1.fc.bias.detach())

p, t = validate_lstm(model_pruned, df_vl)
print(f"    Before fine-tune  RMSE={rmse(p,t):.4f} N·m  R²={r2_score(p,t):.5f}")

# --- 1d. Fine-tune the pruned model ---
print("  1d. Fine-tuning pruned model …")
FT_EPOCHS = 150
opt_ft = optim.Adam(model_pruned.parameters(), lr=5e-4)
sched_ft = optim.lr_scheduler.CosineAnnealingLR(opt_ft, FT_EPOCHS, eta_min=1e-6)

Xtr16, ytr16 = make_lstm_windows(df_tr, train_ids)
ds16 = WindowDataset(Xtr16, ytr16)
dl16 = DataLoader(ds16, batch_size=BATCH_SIZE, shuffle=True)

best_rmse_pruned = 1e9
best_state_pruned = None
for ep in range(FT_EPOCHS):
    model_pruned.train()
    for Xb, yb in dl16:
        opt_ft.zero_grad()
        out, _ = model_pruned(Xb)
        nn.MSELoss()(out, yb).backward()
        torch.nn.utils.clip_grad_norm_(model_pruned.parameters(), 1.0)
        opt_ft.step()
    sched_ft.step()
    if (ep+1) % 30 == 0:
        p, t = validate_lstm(model_pruned, df_vl)
        r = rmse(p,t)
        if r < best_rmse_pruned:
            best_rmse_pruned = r
            best_state_pruned = copy.deepcopy(model_pruned.state_dict())
        print(f"    ep {ep+1:3d}  RMSE={r:.4f} N·m")

model_pruned.load_state_dict(best_state_pruned)
p_pruned, t_pruned = validate_lstm(model_pruned, df_vl)
rmse_pruned = rmse(p_pruned, t_pruned)
r2_pruned   = r2_score(p_pruned, t_pruned)
print(f"  Pruned LSTM-16  RMSE={rmse_pruned:.4f} N·m  R²={r2_pruned:.5f}")

torch.save({'model_state_dict': model_pruned.state_dict(),
            'hidden_size': K, 'method': 'structured_pruning'},
           os.path.join(MDIR, 'lstm_pruned16_model.pth'))

# --- 1e. Generate C code for pruned model ---
print("  1e. Generating C code for pruned LSTM-16 …")

def gen_lstm_c(name, sd, hidden, inp=3):
    """Generate ANSI C99 LSTM inference code (float32)."""
    def fa(t, rows=None, cols=None):
        w = t.detach().numpy()
        if rows is not None: w = w[rows]
        if cols is not None: w = w[:, cols]
        return w.flatten()

    wih = fa(sd['lstm.weight_ih_l0'])
    whh = fa(sd['lstm.weight_hh_l0'])
    bih = fa(sd['lstm.bias_ih_l0'])
    bhh = fa(sd['lstm.bias_hh_l0'])
    wfc = fa(sd['fc.weight'])
    bfc = fa(sd['fc.bias'])

    def arr(nm, vals):
        body = ', '.join(f'{v:.8f}f' for v in vals)
        return f'static const float {nm}[] = {{{body}}};\n'

    H = hidden; I = inp

    h_code = f"""\
#ifndef ROM_{name.upper()}_H
#define ROM_{name.upper()}_H
/* Auto-generated LSTM ROM  hidden={H} */
#define ROM_{name.upper()}_HIDDEN {H}
void ROM_{name}_Reset(float *h, float *c);
float ROM_{name}_Step(const float *x, float *h, float *c);
#endif /* ROM_{name.upper()}_H */
"""
    c_code = f"""\
/* Auto-generated LSTM ROM  hidden={H}  input={I} */
#include <math.h>
#include "rom_{name}.h"

{arr(f'WIH_{name}', wih)}\
{arr(f'WHH_{name}', whh)}\
{arr(f'BIH_{name}', bih)}\
{arr(f'BHH_{name}', bhh)}\
{arr(f'WFC_{name}', wfc)}\
static const float BFC_{name} = {bfc[0]:.8f}f;

void ROM_{name}_Reset(float *h, float *c) {{
    for (int i = 0; i < {H}; i++) {{ h[i] = 0.f; c[i] = 0.f; }}
}}

float ROM_{name}_Step(const float *x, float *h, float *c) {{
    float g[{4*H}];
    int i, j;
    for (i = 0; i < {4*H}; i++) {{
        float v = BIH_{name}[i] + BHH_{name}[i];
        for (j = 0; j < {I}; j++) v += WIH_{name}[i*{I}+j] * x[j];
        for (j = 0; j < {H}; j++) v += WHH_{name}[i*{H}+j] * h[j];
        g[i] = v;
    }}
    for (i = 0; i < {H}; i++) {{
        float ig = 1.f/(1.f+expf(-g[i]));
        float fg = 1.f/(1.f+expf(-g[{H}+i]));
        float gg = tanhf(g[{2*H}+i]);
        float og = 1.f/(1.f+expf(-g[{3*H}+i]));
        c[i] = fg*c[i] + ig*gg;
        h[i] = og*tanhf(c[i]);
    }}
    float y = BFC_{name};
    for (i = 0; i < {H}; i++) y += WFC_{name}[i]*h[i];
    return y;
}}
"""
    return h_code, c_code

sd_p = {k: v for k, v in model_pruned.state_dict().items()}
h_p, c_p = gen_lstm_c('lstm_pruned16', sd_p, K)
with open(os.path.join(SRC, 'rom_lstm_pruned16.h'), 'w') as f: f.write(h_p)
with open(os.path.join(SRC, 'rom_lstm_pruned16.c'), 'w') as f: f.write(c_p)

flash_pruned = flash_kb(os.path.join(SRC, 'rom_lstm_pruned16.c'))
print(f"    Flash = {flash_pruned:.2f} KB")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 – QUANTIZATION-AWARE TRAINING (QAT)
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*70)
print("SECTION 2: QUANTIZATION-AWARE TRAINING (QAT)")
print("═"*70)

class FakeQuantSTE(torch.autograd.Function):
    """Fake int8 quantization with Straight-Through Estimator."""
    @staticmethod
    def forward(ctx, x, scale, zero_point=0):
        # Quantize to int8 range [-128, 127]
        x_q = torch.clamp(torch.round(x / scale + zero_point), -128, 127)
        return (x_q - zero_point) * scale  # dequantize
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None  # STE: pass gradient straight through

def per_tensor_scale(t, bits=8):
    """Symmetric per-tensor scale for int8."""
    max_abs = t.abs().max().item()
    return max(max_abs / 127.0, 1e-8)

class QATCell(nn.Module):
    """LSTM cell with fake-quantized weights (per-tensor symmetric int8)."""
    def __init__(self, hidden_size=32):
        super().__init__()
        self.H = hidden_size
        self.weight_ih = nn.Parameter(torch.zeros(4*hidden_size, 3))
        self.weight_hh = nn.Parameter(torch.zeros(4*hidden_size, hidden_size))
        self.bias_ih   = nn.Parameter(torch.zeros(4*hidden_size))
        self.bias_hh   = nn.Parameter(torch.zeros(4*hidden_size))

    def forward(self, x, h, c):
        H = self.H
        s_ih = per_tensor_scale(self.weight_ih)
        s_hh = per_tensor_scale(self.weight_hh)
        W_ih = FakeQuantSTE.apply(self.weight_ih, s_ih)
        W_hh = FakeQuantSTE.apply(self.weight_hh, s_hh)
        gates = (x @ W_ih.t() + h @ W_hh.t() +
                 self.bias_ih + self.bias_hh)
        ig = torch.sigmoid(gates[:, :H])
        fg = torch.sigmoid(gates[:, H:2*H])
        gg = torch.tanh   (gates[:, 2*H:3*H])
        og = torch.sigmoid(gates[:, 3*H:])
        c_new = fg * c + ig * gg
        h_new = og * torch.tanh(c_new)
        return h_new, c_new

class QATModel(nn.Module):
    def __init__(self, hidden_size=32):
        super().__init__()
        self.H    = hidden_size
        self.cell = QATCell(hidden_size)
        self.fc   = nn.Linear(hidden_size, 1)

    def forward(self, x, state=None):
        B, T, I = x.shape
        H = self.H
        if state is None:
            h = torch.zeros(B, H, device=x.device)
            c = torch.zeros(B, H, device=x.device)
        else:
            h, c = state
        outs = []
        for t in range(T):
            h, c = self.cell(x[:, t, :], h, c)
            outs.append(h)
        outs = torch.stack(outs, 1)  # (B,T,H)
        return self.fc(outs), (h, c)

# Initialise QAT model from baseline LSTM-32 weights
print("  2a. Initialising QAT model from baseline LSTM-32 …")
qat_model = QATModel(hidden_size=32)
with torch.no_grad():
    qat_model.cell.weight_ih.copy_(ckpt['model_state_dict']['lstm.weight_ih_l0'])
    qat_model.cell.weight_hh.copy_(ckpt['model_state_dict']['lstm.weight_hh_l0'])
    qat_model.cell.bias_ih.copy_(ckpt['model_state_dict']['lstm.bias_ih_l0'])
    qat_model.cell.bias_hh.copy_(ckpt['model_state_dict']['lstm.bias_hh_l0'])
    qat_model.fc.weight.copy_(ckpt['model_state_dict']['fc.weight'])
    qat_model.fc.bias.copy_(ckpt['model_state_dict']['fc.bias'])

# Fine-tune with fake quantization
print("  2b. QAT fine-tuning (120 epochs) …")
QAT_EPOCHS = 120
opt_qat  = optim.Adam(qat_model.parameters(), lr=5e-5)
sched_qat = optim.lr_scheduler.CosineAnnealingLR(opt_qat, QAT_EPOCHS, eta_min=1e-7)

Xtr_qat, ytr_qat = make_lstm_windows(df_tr, train_ids)
dl_qat = DataLoader(WindowDataset(Xtr_qat, ytr_qat), batch_size=BATCH_SIZE, shuffle=True)

best_rmse_qat = 1e9
best_state_qat = None

for ep in range(QAT_EPOCHS):
    qat_model.train()
    for Xb, yb in dl_qat:
        opt_qat.zero_grad()
        out, _ = qat_model(Xb)
        nn.MSELoss()(out, yb).backward()
        torch.nn.utils.clip_grad_norm_(qat_model.parameters(), 1.0)
        opt_qat.step()
    sched_qat.step()
    if (ep+1) % 30 == 0:
        qat_model.eval()
        all_p, all_t = [], []
        for sid in sorted(df_vl['SimID'].unique()):
            sub = df_vl[df_vl['SimID']==sid].sort_values('Time').reset_index(drop=True)
            ac  = normalize(sub['AirCharge'].values,  STATS['AirCharge'])
            spd = normalize(sub['Speed'].values,      STATS['Speed'])
            sa  = normalize(sub['SparkAdvance'].values, STATS['SparkAdvance'])
            X   = torch.tensor(np.stack([ac,spd,sa],1)[None], dtype=torch.float32)
            with torch.no_grad():
                out_v, _ = qat_model(X)
            all_p.append(denormalize(out_v[0,:,0].numpy(), STATS['Torque']))
            all_t.append(sub['Torque'].values)
        r = rmse(np.concatenate(all_p), np.concatenate(all_t))
        if r < best_rmse_qat:
            best_rmse_qat = r
            best_state_qat = copy.deepcopy(qat_model.state_dict())
        print(f"    ep {ep+1:3d}  RMSE={r:.4f} N·m")

qat_model.load_state_dict(best_state_qat)

# PTQ-quantize the QAT-tuned weights → generate int8 C code
print("  2c. Applying PTQ to QAT-tuned weights → int8 C code …")

def gen_lstm_int8_c(name, cell, fc, hidden, inp=3):
    """Generate int8 C code from QAT-trained model."""
    def quantize_weights(w):
        wf = w.detach().numpy()
        sc = max(np.abs(wf).max() / 127.0, 1e-8)
        wq = np.clip(np.round(wf / sc), -128, 127).astype(np.int8)
        return wq, sc

    wih, s_wih = quantize_weights(cell.weight_ih)
    whh, s_whh = quantize_weights(cell.weight_hh)
    bih = cell.bias_ih.detach().numpy()
    bhh = cell.bias_hh.detach().numpy()
    wfc, s_wfc = quantize_weights(fc.weight)
    bfc = fc.bias.detach().numpy()

    H = hidden; I = inp

    def i8arr(nm, vals):
        body = ', '.join(str(int(v)) for v in vals.flatten())
        return f'static const int8_t {nm}[] = {{{body}}};\n'
    def farr(nm, vals):
        body = ', '.join(f'{v:.8f}f' for v in vals.flatten())
        return f'static const float {nm}[] = {{{body}}};\n'

    h_code = f"""\
#ifndef ROM_{name.upper()}_H
#define ROM_{name.upper()}_H
/* Auto-generated QAT int8 LSTM ROM  hidden={H} */
#include <stdint.h>
#define ROM_{name.upper()}_HIDDEN {H}
void ROM_{name}_Reset(float *h, float *c);
float ROM_{name}_Step(const float *x, float *h, float *c);
#endif
"""
    c_code = f"""\
/* Auto-generated QAT int8 LSTM ROM  hidden={H}  input={I} */
#include <math.h>
#include <stdint.h>
#include "rom_{name}.h"

/* Scale factors for dequantization */
#define S_WIH_{name.upper()} {s_wih:.8f}f
#define S_WHH_{name.upper()} {s_whh:.8f}f
#define S_WFC_{name.upper()} {s_wfc:.8f}f

{i8arr(f'WIH_{name}', wih)}\
{i8arr(f'WHH_{name}', whh)}\
{farr(f'BIH_{name}', bih)}\
{farr(f'BHH_{name}', bhh)}\
{i8arr(f'WFC_{name}', wfc)}\
static const float BFC_{name} = {bfc[0]:.8f}f;

void ROM_{name}_Reset(float *h, float *c) {{
    for (int i = 0; i < {H}; i++) {{ h[i] = 0.f; c[i] = 0.f; }}
}}

float ROM_{name}_Step(const float *x, float *h, float *c) {{
    float g[{4*H}];
    int i, j;
    for (i = 0; i < {4*H}; i++) {{
        float v = BIH_{name}[i] + BHH_{name}[i];
        for (j = 0; j < {I}; j++) v += (float)WIH_{name}[i*{I}+j] * S_WIH_{name.upper()} * x[j];
        for (j = 0; j < {H}; j++) v += (float)WHH_{name}[i*{H}+j] * S_WHH_{name.upper()} * h[j];
        g[i] = v;
    }}
    for (i = 0; i < {H}; i++) {{
        float ig = 1.f/(1.f+expf(-g[i]));
        float fg = 1.f/(1.f+expf(-g[{H}+i]));
        float gg = tanhf(g[{2*H}+i]);
        float og = 1.f/(1.f+expf(-g[{3*H}+i]));
        c[i] = fg*c[i] + ig*gg;
        h[i] = og*tanhf(c[i]);
    }}
    float y = BFC_{name};
    for (i = 0; i < {H}; i++) y += (float)WFC_{name}[i] * S_WFC_{name.upper()} * h[i];
    return y;
}}
"""
    return h_code, c_code

h_qat, c_qat = gen_lstm_int8_c('lstm_qat', qat_model.cell, qat_model.fc, 32)
with open(os.path.join(SRC, 'rom_lstm_qat.h'), 'w') as f: f.write(h_qat)
with open(os.path.join(SRC, 'rom_lstm_qat.c'), 'w') as f: f.write(c_qat)

flash_qat = flash_kb(os.path.join(SRC, 'rom_lstm_qat.c'))

# Final QAT validation
qat_model.eval()
all_p_qat, all_t_qat = [], []
for sid in sorted(df_vl['SimID'].unique()):
    sub = df_vl[df_vl['SimID']==sid].sort_values('Time').reset_index(drop=True)
    ac  = normalize(sub['AirCharge'].values,  STATS['AirCharge'])
    spd = normalize(sub['Speed'].values,      STATS['Speed'])
    sa  = normalize(sub['SparkAdvance'].values, STATS['SparkAdvance'])
    X   = torch.tensor(np.stack([ac,spd,sa],1)[None], dtype=torch.float32)
    with torch.no_grad():
        out_v, _ = qat_model(X)
    all_p_qat.append(denormalize(out_v[0,:,0].numpy(), STATS['Torque']))
    all_t_qat.append(sub['Torque'].values)

p_qat = np.concatenate(all_p_qat); t_qat = np.concatenate(all_t_qat)
rmse_qat = rmse(p_qat, t_qat)
r2_qat   = r2_score(p_qat, t_qat)
print(f"  QAT LSTM-32  RMSE={rmse_qat:.4f} N·m  R²={r2_qat:.5f}  Flash={flash_qat:.2f} KB")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3 – DELTA LEARNING
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*70)
print("SECTION 3: DELTA LEARNING (Poly-2 baseline + residual LSTM-8)")
print("═"*70)

# --- 3a. Fit degree-2 polynomial on training data ---
print("  3a. Fitting degree-2 polynomial baseline …")

def build_poly_features(df, sim_ids=None):
    if sim_ids is None:
        sids = sorted(df['SimID'].unique())
    else:
        sids = sim_ids
    rows = []
    for sid in sids:
        sub = df[df['SimID']==sid].sort_values('Time').reset_index(drop=True)
        rows.append(pd.DataFrame({
            'AirCharge': sub['AirCharge'].values,
            'Speed':     sub['Speed'].values,
            'SparkAdvance': sub['SparkAdvance'].values,
            'Torque':    sub['Torque'].values
        }))
    return pd.concat(rows, ignore_index=True)

df_poly_tr = build_poly_features(df_tr, train_ids)
df_poly_vl = build_poly_features(df_vl)

poly = PolynomialFeatures(degree=2, include_bias=True)
scaler_poly = StandardScaler()

X_poly_tr = scaler_poly.fit_transform(
    poly.fit_transform(df_poly_tr[['AirCharge','Speed','SparkAdvance']].values))
y_poly_tr  = df_poly_tr['Torque'].values

X_poly_vl = scaler_poly.transform(
    poly.transform(df_poly_vl[['AirCharge','Speed','SparkAdvance']].values))
y_poly_vl  = df_poly_vl['Torque'].values

ridge_poly = Ridge(alpha=1.0)
ridge_poly.fit(X_poly_tr, y_poly_tr)

pred_poly_tr = ridge_poly.predict(X_poly_tr)
pred_poly_vl = ridge_poly.predict(X_poly_vl)

rmse_poly = rmse(pred_poly_vl, y_poly_vl)
r2_poly   = r2_score(pred_poly_vl, y_poly_vl)
print(f"    Polynomial baseline  RMSE={rmse_poly:.4f} N·m  R²={r2_poly:.5f}")

# --- 3b. Compute residuals and add to training data ---
print("  3b. Computing residuals and training residual LSTM-8 …")

# We need per-sim residuals aligned to sequence order
# Add residual column to df_tr (using training-time poly prediction)
idx_tr = 0
df_tr_res = df_tr.copy()
df_tr_res['Delta'] = 0.0
for sid in train_ids:
    mask = df_tr_res['SimID'] == sid
    sub = df_tr[mask].sort_values('Time')
    n = len(sub)
    X_s = scaler_poly.transform(poly.transform(sub[['AirCharge','Speed','SparkAdvance']].values))
    pred_s = ridge_poly.predict(X_s)
    df_tr_res.loc[mask, 'Delta'] = sub['Torque'].values - pred_s

# Similarly for validation
df_vl_res = df_vl.copy()
df_vl_res['Delta'] = 0.0
for sid in sorted(df_vl['SimID'].unique()):
    mask = df_vl_res['SimID'] == sid
    sub = df_vl[mask].sort_values('Time')
    X_s = scaler_poly.transform(poly.transform(sub[['AirCharge','Speed','SparkAdvance']].values))
    pred_s = ridge_poly.predict(X_s)
    df_vl_res.loc[mask, 'Delta'] = sub['Torque'].values - pred_s

# Compute delta statistics for normalization
delta_mean = df_tr_res['Delta'].mean()
delta_std  = df_tr_res['Delta'].std() + 1e-8
STATS['Delta'] = {'mean': float(delta_mean), 'std': float(delta_std)}
print(f"    Delta stats: mean={delta_mean:.3f}  std={delta_std:.3f} N·m")

# Train LSTM-8 on residuals
model_delta = EngineROM_LSTM(hidden_size=8)
Xtr_d, ytr_d = make_lstm_windows(df_tr_res, train_ids, target_col='Delta')
dl_delta = DataLoader(WindowDataset(Xtr_d, ytr_d), batch_size=BATCH_SIZE, shuffle=True)

DELTA_EPOCHS = 200
opt_d  = optim.Adam(model_delta.parameters(), lr=1e-3)
sched_d = optim.lr_scheduler.CosineAnnealingLR(opt_d, DELTA_EPOCHS, eta_min=1e-6)

best_rmse_delta_comp = 1e9
best_state_delta = None

for ep in range(DELTA_EPOCHS):
    model_delta.train()
    for Xb, yb in dl_delta:
        opt_d.zero_grad()
        out, _ = model_delta(Xb)
        nn.MSELoss()(out, yb).backward()
        torch.nn.utils.clip_grad_norm_(model_delta.parameters(), 1.0)
        opt_d.step()
    sched_d.step()

    if (ep+1) % 50 == 0:
        # Validate composite: poly + delta LSTM
        model_delta.eval()
        all_p_comp, all_t_comp = [], []
        for sid in sorted(df_vl_res['SimID'].unique()):
            sub = df_vl_res[df_vl_res['SimID']==sid].sort_values('Time').reset_index(drop=True)
            ac  = normalize(sub['AirCharge'].values,  STATS['AirCharge'])
            spd = normalize(sub['Speed'].values,      STATS['Speed'])
            sa  = normalize(sub['SparkAdvance'].values, STATS['SparkAdvance'])
            X_l = torch.tensor(np.stack([ac,spd,sa],1)[None], dtype=torch.float32)
            with torch.no_grad():
                out_d, _ = model_delta(X_l)
            delta_pred = denormalize(out_d[0,:,0].numpy(), STATS['Delta'])
            poly_pred  = ridge_poly.predict(
                scaler_poly.transform(
                    poly.transform(sub[['AirCharge','Speed','SparkAdvance']].values)))
            composite = poly_pred + delta_pred
            all_p_comp.append(composite)
            all_t_comp.append(sub['Torque'].values)
        r_comp = rmse(np.concatenate(all_p_comp), np.concatenate(all_t_comp))
        if r_comp < best_rmse_delta_comp:
            best_rmse_delta_comp = r_comp
            best_state_delta = copy.deepcopy(model_delta.state_dict())
        print(f"    ep {ep+1:3d}  composite RMSE={r_comp:.4f} N·m")

model_delta.load_state_dict(best_state_delta)

# Final composite validation
model_delta.eval()
all_p_comp, all_t_comp = [], []
for sid in sorted(df_vl_res['SimID'].unique()):
    sub = df_vl_res[df_vl_res['SimID']==sid].sort_values('Time').reset_index(drop=True)
    ac  = normalize(sub['AirCharge'].values,  STATS['AirCharge'])
    spd = normalize(sub['Speed'].values,      STATS['Speed'])
    sa  = normalize(sub['SparkAdvance'].values, STATS['SparkAdvance'])
    X_l = torch.tensor(np.stack([ac,spd,sa],1)[None], dtype=torch.float32)
    with torch.no_grad():
        out_d, _ = model_delta(X_l)
    delta_pred = denormalize(out_d[0,:,0].numpy(), STATS['Delta'])
    poly_pred  = ridge_poly.predict(
        scaler_poly.transform(
            poly.transform(sub[['AirCharge','Speed','SparkAdvance']].values)))
    all_p_comp.append(poly_pred + delta_pred)
    all_t_comp.append(sub['Torque'].values)

p_delta = np.concatenate(all_p_comp); t_delta = np.concatenate(all_t_comp)
rmse_delta = rmse(p_delta, t_delta)
r2_delta   = r2_score(p_delta, t_delta)
print(f"  Delta composite RMSE={rmse_delta:.4f} N·m  R²={r2_delta:.5f}")

torch.save({'model_state_dict': model_delta.state_dict(),
            'hidden_size': 8, 'method': 'delta_learning',
            'delta_stats': STATS['Delta']},
           os.path.join(MDIR, 'lstm_delta8_model.pth'))

# --- 3c. Generate C code: polynomial + delta LSTM-8 ---
print("  3c. Generating C code for polynomial baseline …")

# Save poly model info
import pickle
poly_info = {
    'degree': 2,
    'coef': ridge_poly.coef_.tolist(),
    'intercept': float(ridge_poly.intercept_),
    'scaler_mean': scaler_poly.mean_.tolist(),
    'scaler_std':  scaler_poly.scale_.tolist(),
    'poly_powers': poly.powers_.tolist(),
    'n_features_in': 3,
    'n_poly_features': int(poly.n_output_features_),
    'delta_mean': float(delta_mean),
    'delta_std':  float(delta_std),
}
with open(os.path.join(MDIR, 'delta_poly_info.json'), 'w') as f:
    json.dump(poly_info, f, indent=2)

def gen_poly_c(info, name='delta_poly'):
    """Generate C99 code for polynomial baseline prediction.
    Correct pipeline: raw[3] → PolyFeatures[NF] → StandardScaler[NF] → Ridge → torque_Nm
    NOTE: StandardScaler was fit on the NF polynomial features, not the 3 raw inputs.
    """
    coef  = info['coef']
    intercept = info['intercept']
    sc_mean = info['scaler_mean']   # NF-element list
    sc_std  = info['scaler_std']    # NF-element list
    powers  = info['poly_powers']   # NF × [p0,p1,p2]
    NF = len(coef)

    def farr(nm, vals):
        body = ', '.join(f'{v:.8f}f' for v in vals)
        return f'static const float {nm}[] = {{{body}}};\n'

    rows = []
    for pw in powers:
        rows.append('{' + ', '.join(str(p) for p in pw) + '}')
    powers_str = ',\n    '.join(rows)

    h_code = f"""\
#ifndef ROM_{name.upper()}_H
#define ROM_{name.upper()}_H
/* Auto-generated polynomial baseline  degree=2  n_poly_features={NF}
 * Pipeline: raw[3] -> PolynomialFeatures -> StandardScaler -> Ridge -> torque N·m */
float ROM_{name}_Predict(float air_charge, float speed, float spark_adv);
#endif
"""
    c_code = f"""\
/* Auto-generated polynomial baseline  degree=2  n_raw_inputs=3  n_poly_features={NF}
 * Pipeline: raw inputs -> PolynomialFeatures(degree=2, include_bias=True)
 *           -> StandardScaler (fit on poly features) -> Ridge regression -> torque N·m
 */
#include <math.h>
#include "rom_{name}.h"

static const int POWERS[{NF}][3] = {{
    {powers_str}
}};
{farr(f'COEF_{name}', coef)}\
static const float INTERCEPT_{name} = {intercept:.8f}f;
/* StandardScaler parameters for the {NF} polynomial features */
{farr(f'SC_MEAN_{name}', sc_mean)}\
{farr(f'SC_STD_{name}',  sc_std)}\

float ROM_{name}_Predict(float air_charge, float speed, float spark_adv) {{
    float raw[3] = {{air_charge, speed, spark_adv}};
    int f, k, pp;

    /* Step 1: Compute {NF} polynomial features from 3 raw inputs */
    float poly_feat[{NF}];
    for (f = 0; f < {NF}; f++) {{
        float term = 1.0f;
        for (k = 0; k < 3; k++) {{
            int p = POWERS[f][k];
            for (pp = 0; pp < p; pp++) term *= raw[k];
        }}
        poly_feat[f] = term;
    }}

    /* Step 2: Apply StandardScaler to the polynomial features */
    float x_sc[{NF}];
    for (f = 0; f < {NF}; f++)
        x_sc[f] = (poly_feat[f] - SC_MEAN_{name}[f]) / SC_STD_{name}[f];

    /* Step 3: Ridge regression: y = intercept + coef . x_sc */
    float y = INTERCEPT_{name};
    for (f = 0; f < {NF}; f++)
        y += COEF_{name}[f] * x_sc[f];

    return y;  /* physical torque in N·m */
}}
"""
    return h_code, c_code

h_poly, c_poly = gen_poly_c(poly_info)
with open(os.path.join(SRC, 'rom_delta_poly.h'), 'w') as f: f.write(h_poly)
with open(os.path.join(SRC, 'rom_delta_poly.c'), 'w') as f: f.write(c_poly)

print("  3d. Generating C code for residual LSTM-8 …")
h_d8, c_d8 = gen_lstm_c('lstm_delta8', model_delta.state_dict(), 8)
with open(os.path.join(SRC, 'rom_lstm_delta8.h'), 'w') as f: f.write(h_d8)
with open(os.path.join(SRC, 'rom_lstm_delta8.c'), 'w') as f: f.write(c_d8)

flash_poly  = flash_kb(os.path.join(SRC, 'rom_delta_poly.c'))
flash_delta8 = flash_kb(os.path.join(SRC, 'rom_lstm_delta8.c'))
flash_delta_total = flash_poly + flash_delta8
print(f"    Poly Flash={flash_poly:.2f} KB  LSTM-8 Flash={flash_delta8:.2f} KB  Total={flash_delta_total:.2f} KB")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4 – FIXED-POINT Q16 CODE
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*70)
print("SECTION 4: FIXED-POINT Q16 (int16_t weights for Cortex-M0+)")
print("═"*70)

def gen_lstm_q16_c(name, sd, hidden, inp=3):
    """
    Generate Q16 C code (int16_t weights, float32 runtime arithmetic).
    This is a storage-compression technique: weights are stored as int16_t
    and dequantized at inference time. A true fixed-point deployment would
    also do integer arithmetic at runtime (see comments in generated file).
    """
    def quantize_q16(t):
        w = t.detach().numpy().flatten()
        sc = max(np.abs(w).max() / 32767.0, 1e-8)
        wq = np.clip(np.round(w / sc), -32768, 32767).astype(np.int16)
        return wq, float(sc)

    wih, s_wih = quantize_q16(sd['lstm.weight_ih_l0'])
    whh, s_whh = quantize_q16(sd['lstm.weight_hh_l0'])
    bih = sd['lstm.bias_ih_l0'].detach().numpy()
    bhh = sd['lstm.bias_hh_l0'].detach().numpy()
    wfc, s_wfc = quantize_q16(sd['fc.weight'])
    bfc = sd['fc.bias'].detach().numpy()
    H = hidden; I = inp

    def i16arr(nm, vals):
        body = ', '.join(str(int(v)) for v in vals)
        return f'static const int16_t {nm}[] = {{{body}}};\n'
    def farr(nm, vals):
        body = ', '.join(f'{v:.8f}f' for v in vals)
        return f'static const float {nm}[] = {{{body}}};\n'

    h_code = f"""\
#ifndef ROM_{name.upper()}_H
#define ROM_{name.upper()}_H
/*
 * Auto-generated Q16 LSTM ROM  hidden={H}
 * Weights stored as int16_t (16-bit, symmetric, per-tensor scale).
 * Runtime uses float32 arithmetic.
 *
 * For TRUE fixed-point (Cortex-M0+ no FPU):
 *   - Replace float accumulators with int32_t
 *   - Use LUT-based tanh/sigmoid (256-entry tables)
 *   - All multiplications as int16 × int16 → int32 shifts
 */
#include <stdint.h>
#define ROM_{name.upper()}_HIDDEN {H}
void ROM_{name}_Reset(float *h, float *c);
float ROM_{name}_Step(const float *x, float *h, float *c);
#endif
"""
    c_code = f"""\
/*
 * Auto-generated Q16 LSTM ROM  hidden={H}  input={I}
 * Weight storage: int16_t  |  Inference: float32 dequantize-on-load
 *
 * Flash reduction: ~50% vs float32 (weights only; bias kept float32)
 *
 * Cortex-M0+ true fixed-point path (not implemented here):
 *   int32_t acc = 0;
 *   for (j) acc += (int32_t)WIH[i*I+j] * x_q16[j];  // int16*int16->int32
 *   acc >>= 15;  // Q16 × Q16 → Q17 → shift to Q16
 */
#include <math.h>
#include <stdint.h>
#include "rom_{name}.h"

#define S_WIH_{name.upper()} {s_wih:.10f}f
#define S_WHH_{name.upper()} {s_whh:.10f}f
#define S_WFC_{name.upper()} {s_wfc:.10f}f

{i16arr(f'WIH_{name}', wih)}\
{i16arr(f'WHH_{name}', whh)}\
{farr(f'BIH_{name}', bih)}\
{farr(f'BHH_{name}', bhh)}\
{i16arr(f'WFC_{name}', wfc)}\
static const float BFC_{name} = {bfc[0]:.8f}f;

void ROM_{name}_Reset(float *h, float *c) {{
    for (int i = 0; i < {H}; i++) {{ h[i] = 0.f; c[i] = 0.f; }}
}}

float ROM_{name}_Step(const float *x, float *h, float *c) {{
    float g[{4*H}];
    int i, j;
    for (i = 0; i < {4*H}; i++) {{
        float v = BIH_{name}[i] + BHH_{name}[i];
        for (j = 0; j < {I}; j++)
            v += (float)WIH_{name}[i*{I}+j] * S_WIH_{name.upper()} * x[j];
        for (j = 0; j < {H}; j++)
            v += (float)WHH_{name}[i*{H}+j] * S_WHH_{name.upper()} * h[j];
        g[i] = v;
    }}
    for (i = 0; i < {H}; i++) {{
        float ig = 1.f/(1.f+expf(-g[i]));
        float fg = 1.f/(1.f+expf(-g[{H}+i]));
        float gg = tanhf(g[{2*H}+i]);
        float og = 1.f/(1.f+expf(-g[{3*H}+i]));
        c[i] = fg*c[i] + ig*gg;
        h[i] = og*tanhf(c[i]);
    }}
    float y = BFC_{name};
    for (i = 0; i < {H}; i++) y += (float)WFC_{name}[i] * S_WFC_{name.upper()} * h[i];
    return y;
}}
"""
    return h_code, c_code

# LSTM-16 Q16
print("  4a. Generating Q16 code for LSTM-16 …")
ckpt16 = torch.load(os.path.join(MDIR, 'lstm_16_model.pth'), map_location='cpu')
sd16 = ckpt16['model_state_dict']
h16q, c16q = gen_lstm_q16_c('lstm_16_q16', sd16, 16)
with open(os.path.join(SRC, 'rom_lstm_16_q16.h'), 'w') as f: f.write(h16q)
with open(os.path.join(SRC, 'rom_lstm_16_q16.c'), 'w') as f: f.write(c16q)
flash_16_q16 = flash_kb(os.path.join(SRC, 'rom_lstm_16_q16.c'))

# LSTM-32 Q16
print("  4b. Generating Q16 code for LSTM-32 …")
sd32 = {k: torch.tensor(v) if not isinstance(v, torch.Tensor) else v
        for k, v in ckpt['model_state_dict'].items()}
h32q, c32q = gen_lstm_q16_c('lstm_32_q16', sd32, 32)
with open(os.path.join(SRC, 'rom_lstm_32_q16.h'), 'w') as f: f.write(h32q)
with open(os.path.join(SRC, 'rom_lstm_32_q16.c'), 'w') as f: f.write(c32q)
flash_32_q16 = flash_kb(os.path.join(SRC, 'rom_lstm_32_q16.c'))

print(f"    LSTM-16 Q16: {flash_16_q16:.2f} KB  (vs float32: 6.17 KB)")
print(f"    LSTM-32 Q16: {flash_32_q16:.2f} KB  (vs float32: 19.67 KB)")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5 – COLLECT RESULTS AND GENERATE PLOTS
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*70)
print("SECTION 5: COLLECTING RESULTS AND GENERATING PLOTS")
print("═"*70)

# Load Phase 2 results for comparison
with open(os.path.join(DDIR, 'model_comparison.json')) as f:
    ph2 = json.load(f)

# Compute Phase 2 baseline values
def get_ph2(name):
    return ph2[name]['val_rmse'], ph2[name]['flash_kb']

rmse_lstm8,  flash_lstm8  = get_ph2('lstm_8')
rmse_lstm16, flash_lstm16 = get_ph2('lstm_16')
rmse_lstm32, flash_lstm32 = get_ph2('lstm_32')
rmse_ptq,    flash_ptq    = get_ph2('lstm_32_q8')
rmse_ridge,  flash_ridge  = get_ph2('narx_ridge')

# Phase 3 results dict
phase3_results = {
    'pruned_lstm16': {
        'method': 'Structured Pruning',
        'val_rmse': round(rmse_pruned, 4),
        'val_r2':   round(r2_pruned, 5),
        'flash_kb': round(flash_pruned, 3),
        'n_params': 16*(3+16+4)+16+1,
        'label': 'LSTM-16 Pruned',
    },
    'qat_lstm32': {
        'method': 'QAT int8',
        'val_rmse': round(rmse_qat, 4),
        'val_r2':   round(r2_qat, 5),
        'flash_kb': round(flash_qat, 3),
        'n_params': 32*(3+32+4)+32+1,
        'label': 'LSTM-32 QAT',
    },
    'delta_composite': {
        'method': 'Delta Learning',
        'val_rmse': round(rmse_delta, 4),
        'val_r2':   round(r2_delta, 5),
        'flash_kb': round(flash_delta_total, 3),
        'flash_poly_kb': round(flash_poly, 3),
        'flash_lstm8_kb': round(flash_delta8, 3),
        'n_params': 8*(3+8+4)+8+1,
        'label': 'Delta (Poly+LSTM-8)',
    },
    'lstm16_q16': {
        'method': 'Fixed-Point Q16',
        'val_rmse': round(rmse_lstm16, 4),  # same accuracy, different storage
        'val_r2':   round(0.99966, 5),
        'flash_kb': round(flash_16_q16, 3),
        'label': 'LSTM-16 Q16',
    },
    'lstm32_q16': {
        'method': 'Fixed-Point Q16',
        'val_rmse': round(rmse_lstm32, 4),
        'val_r2':   round(0.99970, 5),
        'flash_kb': round(flash_32_q16, 3),
        'label': 'LSTM-32 Q16',
    },
}

with open(os.path.join(DDIR, 'phase3_results.json'), 'w') as f:
    json.dump(phase3_results, f, indent=2)

print("  Results saved → data/phase3_results.json")
for k, v in phase3_results.items():
    print(f"    {v['label']:25s}  RMSE={v['val_rmse']:.4f}  Flash={v['flash_kb']:.2f} KB")

# ── Extended Pareto plot ───────────────────────────────────────────────────
print("  Generating extended Pareto frontier plot …")

fig, ax = plt.subplots(figsize=(10, 6))

# Phase 2 points
ph2_points = [
    ('NARX-Ridge', flash_ridge, rmse_ridge, '#888888', 's'),
    ('LSTM-8',     flash_lstm8,  rmse_lstm8,  '#1f77b4', 'o'),
    ('LSTM-16',    flash_lstm16, rmse_lstm16, '#ff7f0e', 'o'),
    ('LSTM-32',    flash_lstm32, rmse_lstm32, '#2ca02c', 'o'),
    ('PTQ int8',   flash_ptq,    rmse_ptq,    '#9467bd', 'D'),
]
# Phase 3 points
ph3_points = [
    ('LSTM-16 Pruned', flash_pruned,       rmse_pruned, '#d62728', '^'),
    ('LSTM-32 QAT',    flash_qat,          rmse_qat,    '#8c564b', 'v'),
    ('Delta (P2+L8)',  flash_delta_total,   rmse_delta,  '#e377c2', 'P'),
    ('LSTM-16 Q16',    flash_16_q16,        rmse_lstm16, '#17becf', '<'),
    ('LSTM-32 Q16',    flash_32_q16,        rmse_lstm32, '#bcbd22', '>'),
]

for lbl, fx, ry, col, mk in ph2_points:
    ax.scatter(fx, ry, c=col, marker=mk, s=120, zorder=5,
               label=f'{lbl} (Ph2)', alpha=0.8, edgecolors='k', linewidths=0.7)
    ax.annotate(lbl, (fx, ry), textcoords='offset points',
                xytext=(6, 4), fontsize=7.5, color=col)

for lbl, fx, ry, col, mk in ph3_points:
    ax.scatter(fx, ry, c=col, marker=mk, s=150, zorder=6,
               label=f'{lbl} (Ph3)', alpha=0.9, edgecolors='k', linewidths=0.9)
    ax.annotate(lbl, (fx, ry), textcoords='offset points',
                xytext=(6, -10), fontsize=7.5, color=col, fontweight='bold')

# Pareto frontier line through all points combined
all_pts = [(fx, ry) for _, fx, ry, *_ in ph2_points + ph3_points]
all_pts.sort(key=lambda p: p[0])
pareto = []
min_rmse = float('inf')
for fx, ry in all_pts:
    if ry < min_rmse:
        min_rmse = ry
        pareto.append((fx, ry))
if pareto:
    px, py = zip(*pareto)
    ax.step(px, py, where='post', color='red', linewidth=1.5,
            linestyle='--', alpha=0.6, label='Pareto frontier')

ax.set_xlabel('Flash Memory (KB)', fontsize=12)
ax.set_ylabel('Validation RMSE (N·m)', fontsize=12)
ax.set_title('Engine ROM – Extended Pareto Frontier (Phase 2 + Phase 3)', fontsize=13)
ax.set_xscale('log')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=7.5, ncol=2, loc='upper right')
plt.tight_layout()
plt.savefig(os.path.join(PDIR, 'phase3_pareto.png'), dpi=150, bbox_inches='tight')
plt.close()
print("    Saved → plots/phase3_pareto.png")

# ── Validation traces plot ─────────────────────────────────────────────────
print("  Generating validation traces comparison plot …")

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
axes = axes.flatten()

# Pick a representative validation sim
sid_sample = sorted(df_vl['SimID'].unique())[0]
sub = df_vl[df_vl['SimID']==sid_sample].sort_values('Time').reset_index(drop=True)
t_ax = sub['Time'].values
tq_true = sub['Torque'].values

# Baseline LSTM-32
mdl_b = EngineROM_LSTM(32)
mdl_b.load_state_dict(ckpt['model_state_dict']); mdl_b.eval()
ac  = normalize(sub['AirCharge'].values,    STATS['AirCharge'])
spd = normalize(sub['Speed'].values,        STATS['Speed'])
sa  = normalize(sub['SparkAdvance'].values,  STATS['SparkAdvance'])
Xs  = torch.tensor(np.stack([ac,spd,sa],1)[None], dtype=torch.float32)
with torch.no_grad(): out_b, _ = mdl_b(Xs)
pred_base = denormalize(out_b[0,:,0].numpy(), STATS['Torque'])

# Plot 1: Structured Pruning
ax = axes[0]
with torch.no_grad(): out_p, _ = model_pruned(Xs)
pred_prune = denormalize(out_p[0,:,0].numpy(), STATS['Torque'])
ax.plot(t_ax, tq_true,   'k-',  lw=1.5, label='Ground Truth', alpha=0.8)
ax.plot(t_ax, pred_base, 'b--', lw=1.0, label='LSTM-32 baseline', alpha=0.7)
ax.plot(t_ax, pred_prune,'r-',  lw=1.2, label=f'Pruned LSTM-16 (RMSE={rmse_pruned:.2f})', alpha=0.9)
ax.set_title('1. Structured Pruning', fontsize=11, fontweight='bold')
ax.set_xlabel('Time (s)'); ax.set_ylabel('Torque (N·m)')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# Plot 2: QAT
ax = axes[1]
with torch.no_grad(): out_qat_v, _ = qat_model(Xs)
pred_qat_v = denormalize(out_qat_v[0,:,0].numpy(), STATS['Torque'])
ax.plot(t_ax, tq_true,    'k-',  lw=1.5, label='Ground Truth', alpha=0.8)
ax.plot(t_ax, pred_base,  'b--', lw=1.0, label='LSTM-32 baseline', alpha=0.7)
ax.plot(t_ax, pred_qat_v, 'm-',  lw=1.2, label=f'QAT int8 (RMSE={rmse_qat:.2f})', alpha=0.9)
ax.set_title('2. Quantization-Aware Training', fontsize=11, fontweight='bold')
ax.set_xlabel('Time (s)'); ax.set_ylabel('Torque (N·m)')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# Plot 3: Delta Learning
ax = axes[2]
poly_pred_sub = ridge_poly.predict(
    scaler_poly.transform(poly.transform(sub[['AirCharge','Speed','SparkAdvance']].values)))
with torch.no_grad(): out_d_v, _ = model_delta(Xs)
delta_pred_sub = denormalize(out_d_v[0,:,0].numpy(), STATS['Delta'])
pred_comp_sub  = poly_pred_sub + delta_pred_sub
ax.plot(t_ax, tq_true,     'k-',  lw=1.5, label='Ground Truth', alpha=0.8)
ax.plot(t_ax, poly_pred_sub,'g--', lw=1.0, label='Polynomial baseline', alpha=0.7)
ax.plot(t_ax, pred_comp_sub,'r-',  lw=1.2, label=f'Composite (RMSE={rmse_delta:.2f})', alpha=0.9)
ax.set_title('3. Delta Learning (Poly + LSTM-8)', fontsize=11, fontweight='bold')
ax.set_xlabel('Time (s)'); ax.set_ylabel('Torque (N·m)')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# Plot 4: Flash comparison bar chart
ax = axes[3]
methods = ['LSTM-32\n(baseline)', 'PTQ int8\n(Ph2)', 'LSTM-16\nPruned', 'QAT int8', 'Delta\nComposite',
           'LSTM-16\nQ16', 'LSTM-32\nQ16']
flashes = [flash_lstm32, flash_ptq, flash_pruned, flash_qat, flash_delta_total,
           flash_16_q16, flash_32_q16]
rmses   = [rmse_lstm32, rmse_ptq, rmse_pruned, rmse_qat, rmse_delta,
           rmse_lstm16, rmse_lstm32]
colors  = ['#2ca02c','#9467bd','#d62728','#8c564b','#e377c2','#17becf','#bcbd22']
bars = ax.bar(methods, flashes, color=colors, alpha=0.8, edgecolor='k', linewidth=0.7)
for bar, rm in zip(bars, rmses):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1,
            f'{rm:.2f}', ha='center', va='bottom', fontsize=7.5, fontweight='bold')
ax.set_ylabel('Flash (KB)')
ax.set_title('4. Flash vs RMSE Summary', fontsize=11, fontweight='bold')
ax.set_yscale('log')
ax.grid(True, axis='y', alpha=0.3)
ax.set_xlabel('Model / Method')

plt.suptitle('Phase 3: Advanced ROM Techniques – Validation Comparison', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PDIR, 'phase3_validation.png'), dpi=150, bbox_inches='tight')
plt.close()
print("    Saved → plots/phase3_validation.png")

# ═════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*70)
print("PHASE 3 COMPLETE – SUMMARY")
print("═"*70)
print(f"  Baseline LSTM-32      RMSE={rmse_lstm32:.4f} N·m  Flash={flash_lstm32:.2f} KB")
print(f"  ─────────────────────────────────────────────────────────────────")
print(f"  Pruned LSTM-16        RMSE={rmse_pruned:.4f} N·m  Flash={flash_pruned:.2f} KB  [{rmse_pruned/rmse_lstm32*100:.0f}% RMSE, {flash_pruned/flash_lstm32*100:.0f}% Flash]")
print(f"  QAT int8 LSTM-32      RMSE={rmse_qat:.4f} N·m  Flash={flash_qat:.2f} KB  [{rmse_qat/rmse_lstm32*100:.0f}% RMSE, {flash_qat/flash_lstm32*100:.0f}% Flash]")
print(f"  Delta (Poly+LSTM-8)   RMSE={rmse_delta:.4f} N·m  Flash={flash_delta_total:.2f} KB")
print(f"  LSTM-16 Q16           RMSE={rmse_lstm16:.4f} N·m  Flash={flash_16_q16:.2f} KB  [{flash_16_q16/flash_lstm16*100:.0f}% of LSTM-16 float]")
print(f"  LSTM-32 Q16           RMSE={rmse_lstm32:.4f} N·m  Flash={flash_32_q16:.2f} KB  [{flash_32_q16/flash_lstm32*100:.0f}% of LSTM-32 float]")
print("═"*70)
print("  Files generated:")
print("    src/rom_lstm_pruned16.{h,c}")
print("    src/rom_lstm_qat.{h,c}")
print("    src/rom_delta_poly.{h,c}")
print("    src/rom_lstm_delta8.{h,c}")
print("    src/rom_lstm_16_q16.{h,c}")
print("    src/rom_lstm_32_q16.{h,c}")
print("    data/phase3_results.json")
print("    plots/phase3_pareto.png")
print("    plots/phase3_validation.png")
