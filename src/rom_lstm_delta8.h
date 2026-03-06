#ifndef ROM_LSTM_DELTA8_H
#define ROM_LSTM_DELTA8_H
/* Auto-generated LSTM ROM  hidden=8 */
#define ROM_LSTM_DELTA8_HIDDEN 8
void ROM_lstm_delta8_Reset(float *h, float *c);
float ROM_lstm_delta8_Step(const float *x, float *h, float *c);
#endif /* ROM_LSTM_DELTA8_H */
