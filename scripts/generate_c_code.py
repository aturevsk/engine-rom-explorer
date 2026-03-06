"""
generate_c_code.py
==================
Generates production-ready C source code for the LSTM ROM.
Reads trained weights from models/weights_export.json and generates:
  src/rom_ecm.h  - public API header
  src/rom_ecm.c  - LSTM implementation with embedded weights
  src/rom_weights.h - weight arrays (const float, ROM-friendly)

Targeting: NXP automotive MCUs (S32K, MPC5xxx, S32G)
           ARM Cortex-M / Power Architecture
           AUTOSAR-compatible coding style
"""

import os
import json
import numpy as np
import datetime

PROJ    = '/Users/arkadiyturevskiy/Documents/Claude/Claude_ROM_session4'
SRC_DIR = os.path.join(PROJ, 'src')
os.makedirs(SRC_DIR, exist_ok=True)

WEIGHTS_JSON = os.path.join(PROJ, 'models', 'weights_export.json')


def flatten(arr):
    """Flatten nested list to 1D."""
    if isinstance(arr[0], list):
        return [x for row in arr for x in row]
    return arr


def c_array(name, data_flat, dtype='float', cols=8):
    """Format a C const array with aligned columns."""
    vals = data_flat
    lines = [f'static const {dtype} {name}[{len(vals)}] = {{']
    for i in range(0, len(vals), cols):
        chunk = vals[i:i+cols]
        formatted = ', '.join(f'{v:14.8f}f' for v in chunk)
        lines.append(f'    {formatted},')
    lines.append('};')
    return '\n'.join(lines)


def generate_header(cfg, stats):
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    H   = cfg['hidden_size']
    I   = cfg['input_size']
    O   = cfg['output_size']

    ac_mean  = stats['AirCharge']['mean']
    ac_std   = stats['AirCharge']['std']
    spd_mean = stats['Speed']['mean']
    spd_std  = stats['Speed']['std']
    sa_mean  = stats['SparkAdvance']['mean']
    sa_std   = stats['SparkAdvance']['std']
    tq_mean  = stats['Torque']['mean']
    tq_std   = stats['Torque']['std']

    return f"""/******************************************************************************
 * @file    rom_ecm.h
 * @brief   Engine ROM – LSTM Reduced Order Model API
 *
 * Predicts engine torque from air charge, speed and spark advance.
 * Designed for NXP automotive MCUs (S32K, MPC5xxx, S32G).
 *
 * Model architecture:
 *   LSTM: {I} inputs → {H} hidden units → {O} output(s)
 *   Parameters: {4*(I+H+1)*H + H*O + O} total floating-point weights
 *
 * Generated: {now}
 * Source:    Claude ROM Generator (enginespeed Simulink model)
 ******************************************************************************/

#ifndef ROM_ECM_H
#define ROM_ECM_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {{
#endif

/* ── Dimensions ─────────────────────────────────────────────────────────── */
#define ROM_INPUT_SIZE    {I}U     /**< [AirCharge, Speed, SparkAdvance]     */
#define ROM_HIDDEN_SIZE   {H}U     /**< LSTM hidden state dimension           */
#define ROM_OUTPUT_SIZE   {O}U     /**< [Torque]                              */

/* ── Normalization constants (z-score: x_norm = (x - mean) / std) ─────── */
#define ROM_NORM_AC_MEAN   {ac_mean:.8f}f   /**< Air Charge mean [g/s]         */
#define ROM_NORM_AC_STD    {ac_std:.8f}f    /**< Air Charge std  [g/s]         */
#define ROM_NORM_SPD_MEAN  {spd_mean:.8f}f  /**< Speed mean [rad/s]            */
#define ROM_NORM_SPD_STD   {spd_std:.8f}f   /**< Speed std  [rad/s]            */
#define ROM_NORM_SA_MEAN   {sa_mean:.8f}f   /**< Spark Advance mean [deg]      */
#define ROM_NORM_SA_STD    {sa_std:.8f}f    /**< Spark Advance std  [deg]      */
#define ROM_NORM_TQ_MEAN   {tq_mean:.8f}f   /**< Torque mean [N·m]             */
#define ROM_NORM_TQ_STD    {tq_std:.8f}f    /**< Torque std  [N·m]             */

/* ── LSTM State (must be allocated per-instance or globally) ─────────────── */
typedef struct {{
    float h[ROM_HIDDEN_SIZE];   /**< Hidden state vector                      */
    float c[ROM_HIDDEN_SIZE];   /**< Cell state vector                         */
}} ROM_State_t;

/* ── Public API ─────────────────────────────────────────────────────────── */

/**
 * @brief  Initialize ROM state (call once at engine start or re-init).
 * @param  state  Pointer to ROM state structure.
 */
void ROM_Init(ROM_State_t * const state);

/**
 * @brief  Run one inference step of the LSTM ROM.
 *
 * Call at every control loop iteration (e.g., every 10 ms).
 * Inputs are physical-unit values; outputs are physical-unit torque.
 *
 * @param  state        Pointer to persistent ROM state.
 * @param  air_charge   Intake air charge [g/s]
 * @param  speed        Engine speed [rad/s]
 * @param  spark_adv    Spark advance [degrees]
 * @return              Predicted engine torque [N·m]
 */
float ROM_Step(ROM_State_t * const state,
               float air_charge,
               float speed,
               float spark_adv);

#ifdef __cplusplus
}}
#endif

#endif /* ROM_ECM_H */
"""


def generate_weights_header(weights, cfg):
    H = cfg['hidden_size']
    I = cfg['input_size']
    O = cfg['output_size']
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Flatten weight matrices (row-major)
    w_ih = flatten(weights['lstm']['weight_ih'])   # (4H × I)
    w_hh = flatten(weights['lstm']['weight_hh'])   # (4H × H)
    b_ih = flatten(weights['lstm']['bias_ih'])     # (4H,)
    b_hh = flatten(weights['lstm']['bias_hh'])     # (4H,)
    w_fc = flatten(weights['fc']['weight'])        # (O × H)
    b_fc = flatten(weights['fc']['bias'])          # (O,)

    lines = [f"""/******************************************************************************
 * @file    rom_weights.h
 * @brief   LSTM ROM weight arrays (auto-generated, do not edit)
 *
 * Layout (PyTorch LSTM convention):
 *   weight_ih[4H, I] – input-hidden weights  (i, f, g, o gates stacked)
 *   weight_hh[4H, H] – hidden-hidden weights
 *   bias_ih[4H]      – input-hidden biases
 *   bias_hh[4H]      – hidden-hidden biases
 *   fc_weight[O, H]  – output linear layer weight
 *   fc_bias[O]       – output linear layer bias
 *
 * Gate order: input(i) | forget(f) | cell(g) | output(o)
 * Generated: {now}
 ******************************************************************************/

#ifndef ROM_WEIGHTS_H
#define ROM_WEIGHTS_H

"""]

    lines.append(c_array('ROM_W_IH', w_ih) + '\n')
    lines.append(c_array('ROM_W_HH', w_hh) + '\n')
    lines.append(c_array('ROM_B_IH', b_ih) + '\n')
    lines.append(c_array('ROM_B_HH', b_hh) + '\n')
    lines.append(c_array('ROM_FC_W', w_fc) + '\n')
    lines.append(c_array('ROM_FC_B', b_fc) + '\n')

    lines.append('\n#endif /* ROM_WEIGHTS_H */\n')
    return '\n'.join(lines)


def generate_source(cfg):
    H   = cfg['hidden_size']
    I   = cfg['input_size']
    O   = cfg['output_size']
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    return f"""/******************************************************************************
 * @file    rom_ecm.c
 * @brief   Engine ROM – LSTM Reduced Order Model Implementation
 *
 * Implements a single-layer LSTM followed by a linear layer.
 * Optimised for embedded targets (no heap allocation, static buffers).
 * Compatible with MISRA C:2012 style guidelines for automotive software.
 *
 * Generated: {now}
 ******************************************************************************/

#include "rom_ecm.h"
#include "rom_weights.h"
#include <math.h>
#include <string.h>

/* ── Private helpers ─────────────────────────────────────────────────────── */

/* Sigmoid activation: σ(x) = 1 / (1 + e^(−x))  */
static float sigmoid_f(float x)
{{
    return 1.0f / (1.0f + expf(-x));
}}

/* Element-wise tanh  */
static float tanh_f(float x)
{{
    return tanhf(x);
}}

/**
 * @brief Dense matrix-vector multiply: y += W * x
 *        W is stored row-major: W[rows][cols]
 *        y[i] += sum_j( W[i*cols + j] * x[j] )
 */
static void matvec_add(float * const y,
                       const float * const W,
                       const float * const x,
                       uint32_t rows,
                       uint32_t cols)
{{
    uint32_t i, j;
    for (i = 0U; i < rows; i++)
    {{
        float acc = 0.0f;
        for (j = 0U; j < cols; j++)
        {{
            acc += W[i * cols + j] * x[j];
        }}
        y[i] += acc;
    }}
}}

/* ── Public functions ────────────────────────────────────────────────────── */

void ROM_Init(ROM_State_t * const state)
{{
    (void)memset(state->h, 0, sizeof(state->h));
    (void)memset(state->c, 0, sizeof(state->c));
}}

float ROM_Step(ROM_State_t * const state,
               float air_charge,
               float speed,
               float spark_adv)
{{
    /* ── Normalise inputs (z-score) ─────────────────────────────────────── */
    float x[ROM_INPUT_SIZE];
    x[0] = (air_charge - ROM_NORM_AC_MEAN)  / ROM_NORM_AC_STD;
    x[1] = (speed      - ROM_NORM_SPD_MEAN) / ROM_NORM_SPD_STD;
    x[2] = (spark_adv  - ROM_NORM_SA_MEAN)  / ROM_NORM_SA_STD;

    /* ── LSTM cell ───────────────────────────────────────────────────────── */
    /* gates[0..H-1]      = i gate (input)
     * gates[H..2H-1]     = f gate (forget)
     * gates[2H..3H-1]    = g gate (cell)
     * gates[3H..4H-1]    = o gate (output)                                  */
    float gates[4U * ROM_HIDDEN_SIZE];
    uint32_t k;

    /* Initialise gates with biases: b_ih + b_hh */
    for (k = 0U; k < (4U * ROM_HIDDEN_SIZE); k++)
    {{
        gates[k] = ROM_B_IH[k] + ROM_B_HH[k];
    }}

    /* Add input contribution: gates += W_ih * x */
    matvec_add(gates, ROM_W_IH, x, 4U * ROM_HIDDEN_SIZE, ROM_INPUT_SIZE);

    /* Add hidden contribution: gates += W_hh * h */
    matvec_add(gates, ROM_W_HH, state->h, 4U * ROM_HIDDEN_SIZE, ROM_HIDDEN_SIZE);

    /* Apply activations and update cell / hidden states */
    for (k = 0U; k < ROM_HIDDEN_SIZE; k++)
    {{
        float ig = sigmoid_f(gates[k]);                          /* input  */
        float fg = sigmoid_f(gates[ROM_HIDDEN_SIZE      + k]);   /* forget */
        float gg = tanh_f   (gates[2U * ROM_HIDDEN_SIZE + k]);   /* cell   */
        float og = sigmoid_f(gates[3U * ROM_HIDDEN_SIZE + k]);   /* output */

        state->c[k] = fg * state->c[k] + ig * gg;
        state->h[k] = og * tanh_f(state->c[k]);
    }}

    /* ── Linear output layer: torque_norm = W_fc * h + b_fc ─────────────── */
    float torque_norm = ROM_FC_B[0];
    for (k = 0U; k < ROM_HIDDEN_SIZE; k++)
    {{
        torque_norm += ROM_FC_W[k] * state->h[k];
    }}

    /* ── Denormalise output ──────────────────────────────────────────────── */
    float torque_phys = torque_norm * ROM_NORM_TQ_STD + ROM_NORM_TQ_MEAN;

    return torque_phys;
}}
"""


def main():
    print("=== Generating C Implementation ===\n")

    with open(WEIGHTS_JSON) as f:
        data = json.load(f)

    cfg    = data['config']
    stats  = data['normalization']
    weights = {'lstm': data['lstm'], 'fc': data['fc']}

    # ── rom_ecm.h
    h_path = os.path.join(SRC_DIR, 'rom_ecm.h')
    with open(h_path, 'w') as f:
        f.write(generate_header(cfg, stats))
    print(f"Generated: src/rom_ecm.h")

    # ── rom_weights.h
    w_path = os.path.join(SRC_DIR, 'rom_weights.h')
    with open(w_path, 'w') as f:
        f.write(generate_weights_header(weights, cfg))
    print(f"Generated: src/rom_weights.h  ({sum(len(flatten(v)) for v in [weights['lstm']['weight_ih'], weights['lstm']['weight_hh'], weights['lstm']['bias_ih'], weights['lstm']['bias_hh'], weights['fc']['weight'], weights['fc']['bias']])} floats)")

    # ── rom_ecm.c
    c_path = os.path.join(SRC_DIR, 'rom_ecm.c')
    with open(c_path, 'w') as f:
        f.write(generate_source(cfg))
    print(f"Generated: src/rom_ecm.c")

    # ── Summary
    H = cfg['hidden_size']
    I = cfg['input_size']
    O = cfg['output_size']
    n_params = 4*(I+H+1)*H + H*O + O
    print(f"\nModel summary:")
    print(f"  Input size:   {I}  [AirCharge, Speed, SparkAdvance]")
    print(f"  Hidden size:  {H}")
    print(f"  Output size:  {O}  [Torque]")
    print(f"  Parameters:   {n_params}")
    print(f"  Flash usage:  ~{n_params*4/1024:.1f} KB (float32)")
    print(f"  RAM usage:    ~{2*H*4} bytes (LSTM state)")
    print("\nC code generation complete!")


if __name__ == '__main__':
    main()
