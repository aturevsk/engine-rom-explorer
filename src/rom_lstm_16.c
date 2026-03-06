#include "rom_lstm_16.h"
#include "rom_lstm_16_weights.h"
#include <math.h>
#include <string.h>
static float sig(float x){ return 1.0f/(1.0f+expf(-x)); }
static void matvec(float *y, const float *W, const float *x, int rows, int cols){
    int i,j; for(i=0;i<rows;i++){float a=0;for(j=0;j<cols;j++) a+=W[i*cols+j]*x[j];y[i]+=a;}
}
void ROM_lstm_16_Init(ROM_lstm_16_State_t *s){memset(s,0,sizeof(*s));}
float ROM_lstm_16_Step(ROM_lstm_16_State_t *s, float ac, float spd, float sa){
    float x[3];
    x[0]=(ac -ROM_LSTM_16_AC_MEAN) /ROM_LSTM_16_AC_STD;
    x[1]=(spd-ROM_LSTM_16_SPD_MEAN)/ROM_LSTM_16_SPD_STD;
    x[2]=(sa -ROM_LSTM_16_SA_MEAN) /ROM_LSTM_16_SA_STD;
    float g[64]; int k;
    for(k=0;k<64;k++) g[k]=ROM_lstm_16_B_IH[k]+ROM_lstm_16_B_HH[k];
    matvec(g,ROM_lstm_16_W_IH,x,64,3);
    matvec(g,ROM_lstm_16_W_HH,s->h,64,16);
    for(k=0;k<16;k++){
        float ig=sig(g[k]), fg=sig(g[16+k]), gg=tanhf(g[32+k]), og=sig(g[48+k]);
        s->c[k]=fg*s->c[k]+ig*gg; s->h[k]=og*tanhf(s->c[k]);
    }
    float y=ROM_lstm_16_FC_B[0];
    for(k=0;k<16;k++) y+=ROM_lstm_16_FC_W[k]*s->h[k];
    return y*ROM_LSTM_16_TQ_STD+ROM_LSTM_16_TQ_MEAN;
}
