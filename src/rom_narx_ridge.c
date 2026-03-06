#include "rom_narx_ridge.h"
#include <string.h>
static const float ROM_narx_ridge_COEF[11] = {
    1.09057403f, -0.81324351f, -0.25722757f, 0.01154179f, 0.20517664f, -0.12173340f, -0.08364743f, -0.01406897f,
    0.68009919f, 0.22202566f, 0.00317662f,
};
static const float ROM_narx_ridge_INTERCEPT = -0.00060004f;
void NARX_narx_ridge_Init(NARX_narx_ridge_State_t *s){memset(s,0,sizeof(*s));}
float NARX_narx_ridge_Step(NARX_narx_ridge_State_t *s, float ac, float spd, float sa){
    float x[11]; int i;
    float ac_n=(ac-ROM_NARX_RIDGE_AC_MEAN)/ROM_NARX_RIDGE_AC_STD;
    float spd_n=(spd-ROM_NARX_RIDGE_SPD_MEAN)/ROM_NARX_RIDGE_SPD_STD;
    float sa_n=(sa-ROM_NARX_RIDGE_SA_MEAN)/ROM_NARX_RIDGE_SA_STD;
    x[0]=ac_n;
    for(i=0;i<3;i++) x[1+i]=s->ac_lag[i];
    x[4]=spd_n;
    for(i=0;i<3;i++) x[5+i]=s->spd_lag[i];
    x[8]=s->tq_lag[0]; x[9]=s->tq_lag[1];
    x[10]=sa_n;
    float y=ROM_narx_ridge_INTERCEPT;
    for(i=0;i<11;i++) y+=ROM_narx_ridge_COEF[i]*x[i];
    for(i=3;i>0;i--) s->ac_lag[i]=s->ac_lag[i-1]; s->ac_lag[0]=ac_n;
    for(i=3;i>0;i--) s->spd_lag[i]=s->spd_lag[i-1]; s->spd_lag[0]=spd_n;
    s->tq_lag[1]=s->tq_lag[0]; s->tq_lag[0]=y;
    return y*ROM_NARX_RIDGE_TQ_STD+ROM_NARX_RIDGE_TQ_MEAN;
}
