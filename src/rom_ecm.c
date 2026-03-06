/******************************************************************************
 * @file    rom_ecm.c
 * @brief   Engine ROM – LSTM Reduced Order Model Implementation
 *
 * Implements a single-layer LSTM followed by a linear layer.
 * Optimised for embedded targets (no heap allocation, static buffers).
 * Compatible with MISRA C:2012 style guidelines for automotive software.
 *
 * Generated: 2026-03-04 17:57:07
 ******************************************************************************/

#include "rom_ecm.h"
#include "rom_weights.h"
#include <math.h>
#include <string.h>

/* ── Private helpers ─────────────────────────────────────────────────────── */

/* Sigmoid activation: σ(x) = 1 / (1 + e^(−x))  */
static float sigmoid_f(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

/* Element-wise tanh  */
static float tanh_f(float x)
{
    return tanhf(x);
}

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
{
    uint32_t i, j;
    for (i = 0U; i < rows; i++)
    {
        float acc = 0.0f;
        for (j = 0U; j < cols; j++)
        {
            acc += W[i * cols + j] * x[j];
        }
        y[i] += acc;
    }
}

/* ── Public functions ────────────────────────────────────────────────────── */

void ROM_Init(ROM_State_t * const state)
{
    (void)memset(state->h, 0, sizeof(state->h));
    (void)memset(state->c, 0, sizeof(state->c));
}

float ROM_Step(ROM_State_t * const state,
               float air_charge,
               float speed,
               float spark_adv)
{
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
    {
        gates[k] = ROM_B_IH[k] + ROM_B_HH[k];
    }

    /* Add input contribution: gates += W_ih * x */
    matvec_add(gates, ROM_W_IH, x, 4U * ROM_HIDDEN_SIZE, ROM_INPUT_SIZE);

    /* Add hidden contribution: gates += W_hh * h */
    matvec_add(gates, ROM_W_HH, state->h, 4U * ROM_HIDDEN_SIZE, ROM_HIDDEN_SIZE);

    /* Apply activations and update cell / hidden states */
    for (k = 0U; k < ROM_HIDDEN_SIZE; k++)
    {
        float ig = sigmoid_f(gates[k]);                          /* input  */
        float fg = sigmoid_f(gates[ROM_HIDDEN_SIZE      + k]);   /* forget */
        float gg = tanh_f   (gates[2U * ROM_HIDDEN_SIZE + k]);   /* cell   */
        float og = sigmoid_f(gates[3U * ROM_HIDDEN_SIZE + k]);   /* output */

        state->c[k] = fg * state->c[k] + ig * gg;
        state->h[k] = og * tanh_f(state->c[k]);
    }

    /* ── Linear output layer: torque_norm = W_fc * h + b_fc ─────────────── */
    float torque_norm = ROM_FC_B[0];
    for (k = 0U; k < ROM_HIDDEN_SIZE; k++)
    {
        torque_norm += ROM_FC_W[k] * state->h[k];
    }

    /* ── Denormalise output ──────────────────────────────────────────────── */
    float torque_phys = torque_norm * ROM_NORM_TQ_STD + ROM_NORM_TQ_MEAN;

    return torque_phys;
}
