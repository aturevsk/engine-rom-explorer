#ifndef ROM_LSTM_QAT_H
#define ROM_LSTM_QAT_H
/* Auto-generated QAT int8 LSTM ROM  hidden=32 */
#include <stdint.h>
#define ROM_LSTM_QAT_HIDDEN 32
void ROM_lstm_qat_Reset(float *h, float *c);
float ROM_lstm_qat_Step(const float *x, float *h, float *c);
#endif
