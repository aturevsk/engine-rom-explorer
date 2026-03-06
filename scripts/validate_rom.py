"""
validate_rom.py
===============
Validates the trained PyTorch LSTM ROM against the Simulink enginespeed model.

Strategy (open-loop validation):
1. Load validation data collected from Simulink (unseen conditions).
2. Feed Simulink-generated AirCharge and Speed signals as ROM inputs.
3. Compare ROM-predicted Torque vs Simulink reference Torque.
4. Compute RMSE, MAE, R² metrics per simulation.
5. Generate publication-quality comparison plots.
"""

import os
import json
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import torch.nn as nn

# ── Paths ────────────────────────────────────────────────────────────────────
PROJ  = '/Users/arkadiyturevskiy/Documents/Claude/Claude_ROM_session4'
VAL_DATA = os.path.join(PROJ, 'data',   'validation_data.csv')
MDIR     = os.path.join(PROJ, 'models')
PDIR     = os.path.join(PROJ, 'plots')
os.makedirs(PDIR, exist_ok=True)


# ── Model (must match train_rom.py) ─────────────────────────────────────────
class EngineROM(nn.Module):
    def __init__(self, input_size=3, hidden_size=32, num_layers=1, output_size=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc   = nn.Linear(hidden_size, output_size)

    def forward(self, x, state=None):
        out, state = self.lstm(x, state)
        return self.fc(out), state


# ── Utilities ────────────────────────────────────────────────────────────────
def normalize(arr, st):
    return (arr - st['mean']) / (st['std'] + 1e-8)

def denormalize(arr, st):
    return arr * (st['std'] + 1e-8) + st['mean']

def metrics(pred, true):
    err  = pred - true
    rmse = math.sqrt(np.mean(err**2))
    mae  = np.mean(np.abs(err))
    ss_tot = np.sum((true - np.mean(true))**2)
    ss_res = np.sum(err**2)
    r2   = 1 - ss_res / (ss_tot + 1e-12)
    return rmse, mae, r2


def load_model():
    ckpt   = torch.load(os.path.join(MDIR, 'rom_model.pth'), map_location='cpu', weights_only=False)
    cfg    = ckpt['model_config']
    model  = EngineROM(**cfg)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    stats  = ckpt['stats']
    return model, stats


def run_rom(model, df_sim, stats, device):
    """Run ROM on one simulation's Simulink input signals."""
    sub = df_sim.sort_values('Time').reset_index(drop=True)

    ac  = normalize(sub['AirCharge'].values,    stats['AirCharge'])
    spd = normalize(sub['Speed'].values,        stats['Speed'])
    sa  = normalize(sub['SparkAdvance'].values,  stats['SparkAdvance'])
    tq_ref = sub['Torque'].values

    X = torch.tensor(np.stack([ac, spd, sa], axis=1),
                     dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_norm, _ = model(X)

    tq_pred = denormalize(pred_norm.squeeze().cpu().numpy(), stats['Torque'])
    return sub['Time'].values, tq_ref, tq_pred, sub


def validate():
    print("=== ROM Validation Against Simulink ===\n")

    # ── Load
    df = pd.read_csv(VAL_DATA)
    print(f"Validation data: {len(df):,} samples | "
          f"{len(df['SimID'].unique())} simulations")

    model, stats = load_model()
    device = torch.device('cpu')
    model.to(device)

    all_metrics = []
    sim_results = []

    for sid in sorted(df['SimID'].unique()):
        df_sim = df[df['SimID'] == sid]
        sa_val = df_sim['SparkAdvance'].iloc[0]

        t, tq_ref, tq_pred, sub = run_rom(model, df_sim, stats, device)
        rmse, mae, r2 = metrics(tq_pred, tq_ref)

        all_metrics.append({'SimID': int(sid), 'SA': float(sa_val),
                             'RMSE_Nm': rmse, 'MAE_Nm': mae, 'R2': r2})
        sim_results.append((sid, sa_val, t, tq_ref, tq_pred, sub))

        print(f"  Sim {sid} | SA={sa_val:4.1f}° | "
              f"RMSE={rmse:.4f} N·m | MAE={mae:.4f} N·m | R²={r2:.5f}")

    # ── Overall metrics
    rmse_all = np.mean([m['RMSE_Nm'] for m in all_metrics])
    mae_all  = np.mean([m['MAE_Nm']  for m in all_metrics])
    r2_all   = np.mean([m['R2']      for m in all_metrics])
    print(f"\n{'─'*55}")
    print(f"Overall  |  RMSE={rmse_all:.4f} N·m  |  "
          f"MAE={mae_all:.4f} N·m  |  R²={r2_all:.5f}")

    # ── Save metrics
    metrics_path = os.path.join(PROJ, 'data', 'validation_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump({'per_sim': all_metrics,
                   'overall': {'RMSE_Nm': rmse_all,
                               'MAE_Nm':  mae_all,
                               'R2':      r2_all}}, f, indent=2)
    print(f"Metrics saved to data/validation_metrics.json")

    # ── Plots
    plot_comparison_all(sim_results)
    plot_scatter_all(sim_results, all_metrics)
    plot_error_distribution(sim_results)

    print("\nValidation complete!")
    return all_metrics


def plot_comparison_all(sim_results):
    """One figure per simulation: Simulink vs ROM torque comparison."""
    for (sid, sa, t, tq_ref, tq_pred, sub) in sim_results:
        rmse, mae, r2 = metrics(tq_pred, tq_ref)
        error = tq_pred - tq_ref

        fig = plt.figure(figsize=(14, 10))
        gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.35)

        # ── Top left: Torque comparison
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(t, tq_ref,  label='Simulink (reference)',  color='steelblue',  lw=2.0)
        ax1.plot(t, tq_pred, label='LSTM ROM (prediction)', color='orangered',  lw=1.5, linestyle='--')
        ax1.set_ylabel('Torque [N·m]', fontsize=11)
        ax1.set_title(
            f'Validation Simulation {sid}  |  SA = {sa:.0f}°  |  '
            f'RMSE = {rmse:.4f} N·m  |  R² = {r2:.5f}',
            fontsize=11, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)

        # ── Middle left: prediction error
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(t, error, color='darkgreen', lw=1.2)
        ax2.axhline(0, color='k', lw=0.8, linestyle=':')
        ax2.fill_between(t, error, 0, alpha=0.2, color='darkgreen')
        ax2.set_ylabel('Error [N·m]', fontsize=10)
        ax2.set_title('Prediction Error  (ROM − Simulink)')
        ax2.grid(True, alpha=0.3)

        # ── Middle right: Speed
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.plot(t, sub['Speed'].values, color='royalblue', lw=1.4)
        ax3.set_ylabel('Speed [rad/s]', fontsize=10)
        ax3.set_title('Engine Speed (ROM input)')
        ax3.grid(True, alpha=0.3)

        # ── Bottom left: Air Charge
        ax4 = fig.add_subplot(gs[2, 0])
        ax4.plot(t, sub['AirCharge'].values, color='purple', lw=1.4)
        ax4.set_xlabel('Time [s]')
        ax4.set_ylabel('Air Charge [g/s]', fontsize=10)
        ax4.set_title('Air Charge (ROM input)')
        ax4.grid(True, alpha=0.3)

        # ── Bottom right: Throttle
        ax5 = fig.add_subplot(gs[2, 1])
        ax5.plot(t, sub['Throttle'].values, color='darkorange', lw=1.4, drawstyle='steps-post')
        ax5.set_xlabel('Time [s]')
        ax5.set_ylabel('Throttle [deg]', fontsize=10)
        ax5.set_title('Throttle Input')
        ax5.grid(True, alpha=0.3)

        plt.suptitle('Engine ROM Validation – Dynamic Simulation Comparison',
                     fontsize=13, fontweight='bold', y=1.01)

        path = os.path.join(PDIR, f'validation_sim{sid}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Plot saved: plots/validation_sim{sid}.png")


def plot_scatter_all(sim_results, all_metrics):
    """Scatter plot: Simulink torque vs ROM torque across all validation sims."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: scatter
    ax = axes[0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(sim_results)))
    all_ref, all_pred = [], []
    for i, (sid, sa, t, tq_ref, tq_pred, sub) in enumerate(sim_results):
        ax.scatter(tq_ref, tq_pred, s=4, alpha=0.4, color=colors[i],
                   label=f'SA={sa:.0f}°')
        all_ref.extend(tq_ref)
        all_pred.extend(tq_pred)

    all_ref  = np.array(all_ref)
    all_pred = np.array(all_pred)
    lims = [all_ref.min()*0.95, all_ref.max()*1.05]
    ax.plot(lims, lims, 'k--', lw=1.5, label='Perfect fit')
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel('Simulink Torque [N·m]')
    ax.set_ylabel('ROM Torque [N·m]')
    ax.set_title('ROM vs Simulink Torque (all validation sims)')
    ax.legend(markerscale=4, fontsize=9)
    ax.grid(True, alpha=0.3)

    # Right: metrics bar chart
    ax2 = axes[1]
    sims_labels = [f"Sim {m['SimID']}\nSA={m['SA']:.0f}°" for m in all_metrics]
    rmses = [m['RMSE_Nm'] for m in all_metrics]
    r2s   = [m['R2']      for m in all_metrics]

    x = np.arange(len(sims_labels))
    bars = ax2.bar(x, rmses, color=colors, edgecolor='white', linewidth=0.5)
    ax2.set_xticks(x); ax2.set_xticklabels(sims_labels, fontsize=8)
    ax2.set_ylabel('RMSE [N·m]')
    ax2.set_title('RMSE per Validation Simulation')
    ax2.grid(True, alpha=0.3, axis='y')

    for bar, r2 in zip(bars, r2s):
        ax2.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.002,
                 f'R²={r2:.4f}', ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    path = os.path.join(PDIR, 'validation_scatter.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved: plots/validation_scatter.png")


def plot_error_distribution(sim_results):
    """Error histogram across all validation simulations."""
    all_errors = []
    for (sid, sa, t, tq_ref, tq_pred, sub) in sim_results:
        all_errors.extend((tq_pred - tq_ref).tolist())
    all_errors = np.array(all_errors)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(all_errors, bins=60, color='steelblue', edgecolor='white',
            linewidth=0.5, density=True, alpha=0.8)
    ax.axvline(np.mean(all_errors), color='red',    lw=2, linestyle='--',
               label=f'Mean = {np.mean(all_errors):.4f} N·m')
    ax.axvline(0,                   color='black',  lw=1, linestyle=':')

    # Overlay Gaussian
    from scipy.stats import norm
    mu, sigma = np.mean(all_errors), np.std(all_errors)
    x_range = np.linspace(all_errors.min(), all_errors.max(), 200)
    ax.plot(x_range, norm.pdf(x_range, mu, sigma),
            color='orangered', lw=2, label=f'N(μ={mu:.4f}, σ={sigma:.4f})')

    ax.set_xlabel('Prediction Error [N·m]')
    ax.set_ylabel('Density')
    ax.set_title('ROM Prediction Error Distribution (all validation data)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    path = os.path.join(PDIR, 'error_distribution.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved: plots/error_distribution.png")


if __name__ == '__main__':
    validate()
