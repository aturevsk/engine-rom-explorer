#ifndef ROM_LSTM_32_Q8_H
#define ROM_LSTM_32_Q8_H
#include <stdint.h>
#define ROM_LSTM_32_Q8_HIDDEN 32U
#define ROM_LSTM_32_Q8_INPUT  3U
#define ROM_LSTM_32_Q8_AC_MEAN   0.21671053f
#define ROM_LSTM_32_Q8_AC_STD    0.12929910f
#define ROM_LSTM_32_Q8_SPD_MEAN  580.25021710f
#define ROM_LSTM_32_Q8_SPD_STD   235.62429395f
#define ROM_LSTM_32_Q8_SA_MEAN   15.00000000f
#define ROM_LSTM_32_Q8_SA_STD    7.07177361f
#define ROM_LSTM_32_Q8_TQ_MEAN   25.94591248f
#define ROM_LSTM_32_Q8_TQ_STD    46.99854266f
typedef struct { float h[32]; float c[32]; } ROM_lstm_32_q8_State_t;
void ROM_lstm_32_q8_Init(ROM_lstm_32_q8_State_t *s);
float ROM_lstm_32_q8_Step(ROM_lstm_32_q8_State_t *s, float ac, float spd, float sa);
#endif
