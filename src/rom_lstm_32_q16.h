#ifndef ROM_LSTM_32_Q16_H
#define ROM_LSTM_32_Q16_H
/*
 * Auto-generated Q16 LSTM ROM  hidden=32
 * Weights stored as int16_t (16-bit, symmetric, per-tensor scale).
 * Runtime uses float32 arithmetic.
 *
 * For TRUE fixed-point (Cortex-M0+ no FPU):
 *   - Replace float accumulators with int32_t
 *   - Use LUT-based tanh/sigmoid (256-entry tables)
 *   - All multiplications as int16 × int16 → int32 shifts
 */
#include <stdint.h>
#define ROM_LSTM_32_Q16_HIDDEN 32
void ROM_lstm_32_q16_Reset(float *h, float *c);
float ROM_lstm_32_q16_Step(const float *x, float *h, float *c);
#endif
