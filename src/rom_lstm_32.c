#include "rom_lstm_32.h"
#include "rom_lstm_32_weights.h"
#include <math.h>
#include <string.h>
static float sig(float x){ return 1.0f/(1.0f+expf(-x)); }
static void matvec(float *y, const float *W, const float *x, int rows, int cols){
    int i,j; for(i=0;i<rows;i++){float a=0;for(j=0;j<cols;j++) a+=W[i*cols+j]*x[j];y[i]+=a;}
}
void ROM_lstm_32_Init(ROM_lstm_32_State_t *s){memset(s,0,sizeof(*s));}
float ROM_lstm_32_Step(ROM_lstm_32_State_t *s, float ac, float spd, float sa){
    float x[3];
    x[0]=(ac -ROM_LSTM_32_AC_MEAN) /ROM_LSTM_32_AC_STD;
    x[1]=(spd-ROM_LSTM_32_SPD_MEAN)/ROM_LSTM_32_SPD_STD;
    x[2]=(sa -ROM_LSTM_32_SA_MEAN) /ROM_LSTM_32_SA_STD;
    float g[128]; int k;
    for(k=0;k<128;k++) g[k]=ROM_lstm_32_B_IH[k]+ROM_lstm_32_B_HH[k];
    matvec(g,ROM_lstm_32_W_IH,x,128,3);
    matvec(g,ROM_lstm_32_W_HH,s->h,128,32);
    for(k=0;k<32;k++){
        float ig=sig(g[k]), fg=sig(g[32+k]), gg=tanhf(g[64+k]), og=sig(g[96+k]);
        s->c[k]=fg*s->c[k]+ig*gg; s->h[k]=og*tanhf(s->c[k]);
    }
    float y=ROM_lstm_32_FC_B[0];
    for(k=0;k<32;k++) y+=ROM_lstm_32_FC_W[k]*s->h[k];
    return y*ROM_LSTM_32_TQ_STD+ROM_LSTM_32_TQ_MEAN;
}
