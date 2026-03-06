#ifndef ROM_LSTM_PRUNED16_H
#define ROM_LSTM_PRUNED16_H
/* Auto-generated LSTM ROM  hidden=16 */
#define ROM_LSTM_PRUNED16_HIDDEN 16
void ROM_lstm_pruned16_Reset(float *h, float *c);
float ROM_lstm_pruned16_Step(const float *x, float *h, float *c);
#endif /* ROM_LSTM_PRUNED16_H */
