#ifndef ROM_NARX_MLP_H
#define ROM_NARX_MLP_H
#include <stdint.h>
#define ROM_NARX_MLP_LAG_AC  4U
#define ROM_NARX_MLP_LAG_SPD 4U
#define ROM_NARX_MLP_LAG_TQ  2U
#define ROM_NARX_MLP_N_FEAT  11U
#define ROM_NARX_MLP_AC_MEAN   0.21671053f
#define ROM_NARX_MLP_AC_STD    0.12929910f
#define ROM_NARX_MLP_SPD_MEAN  580.25021710f
#define ROM_NARX_MLP_SPD_STD   235.62429395f
#define ROM_NARX_MLP_SA_MEAN   15.00000000f
#define ROM_NARX_MLP_SA_STD    7.07177361f
#define ROM_NARX_MLP_TQ_MEAN   25.94591248f
#define ROM_NARX_MLP_TQ_STD    46.99854266f
typedef struct {
    float ac_lag[4];
    float spd_lag[4];
    float tq_lag[2];
} NARX_narx_mlp_State_t;
void  NARX_narx_mlp_Init(NARX_narx_mlp_State_t *s);
float NARX_narx_mlp_Step(NARX_narx_mlp_State_t *s, float ac, float spd, float sa);
#endif
