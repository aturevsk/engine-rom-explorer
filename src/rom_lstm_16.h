#ifndef ROM_LSTM_16_H
#define ROM_LSTM_16_H
#include <stdint.h>
#define ROM_LSTM_16_HIDDEN 16U
#define ROM_LSTM_16_INPUT  3U
// z-score normalisation
#define ROM_LSTM_16_AC_MEAN   0.21671053f
#define ROM_LSTM_16_AC_STD    0.12929910f
#define ROM_LSTM_16_SPD_MEAN  580.25021710f
#define ROM_LSTM_16_SPD_STD   235.62429395f
#define ROM_LSTM_16_SA_MEAN   15.00000000f
#define ROM_LSTM_16_SA_STD    7.07177361f
#define ROM_LSTM_16_TQ_MEAN   25.94591248f
#define ROM_LSTM_16_TQ_STD    46.99854266f
typedef struct { float h[16]; float c[16]; } ROM_lstm_16_State_t;
void ROM_lstm_16_Init(ROM_lstm_16_State_t *s);
float ROM_lstm_16_Step(ROM_lstm_16_State_t *s, float ac, float spd, float sa);
#ifdef __cplusplus
extern "C" {
void ROM_lstm_16_Init(ROM_lstm_16_State_t *s);
float ROM_lstm_16_Step(ROM_lstm_16_State_t *s, float ac, float spd, float sa);
}
#endif
#endif
