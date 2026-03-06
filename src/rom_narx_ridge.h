#ifndef ROM_NARX_RIDGE_H
#define ROM_NARX_RIDGE_H
#include <stdint.h>
#define ROM_NARX_RIDGE_LAG_AC  4U
#define ROM_NARX_RIDGE_LAG_SPD 4U
#define ROM_NARX_RIDGE_LAG_TQ  2U
#define ROM_NARX_RIDGE_N_FEAT  11U
#define ROM_NARX_RIDGE_AC_MEAN   0.21671053f
#define ROM_NARX_RIDGE_AC_STD    0.12929910f
#define ROM_NARX_RIDGE_SPD_MEAN  580.25021710f
#define ROM_NARX_RIDGE_SPD_STD   235.62429395f
#define ROM_NARX_RIDGE_SA_MEAN   15.00000000f
#define ROM_NARX_RIDGE_SA_STD    7.07177361f
#define ROM_NARX_RIDGE_TQ_MEAN   25.94591248f
#define ROM_NARX_RIDGE_TQ_STD    46.99854266f
typedef struct {
    float ac_lag[4];
    float spd_lag[4];
    float tq_lag[2];
} NARX_narx_ridge_State_t;
void  NARX_narx_ridge_Init(NARX_narx_ridge_State_t *s);
float NARX_narx_ridge_Step(NARX_narx_ridge_State_t *s, float ac, float spd, float sa);
#endif
