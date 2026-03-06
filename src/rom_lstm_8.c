#include "rom_lstm_8.h"
#include "rom_lstm_8_weights.h"
#include <math.h>
#include <string.h>
static float sig(float x){ return 1.0f/(1.0f+expf(-x)); }
static void matvec(float *y, const float *W, const float *x, int rows, int cols){
    int i,j; for(i=0;i<rows;i++){float a=0;for(j=0;j<cols;j++) a+=W[i*cols+j]*x[j];y[i]+=a;}
}
void ROM_lstm_8_Init(ROM_lstm_8_State_t *s){memset(s,0,sizeof(*s));}
float ROM_lstm_8_Step(ROM_lstm_8_State_t *s, float ac, float spd, float sa){
    float x[3];
    x[0]=(ac -ROM_LSTM_8_AC_MEAN) /ROM_LSTM_8_AC_STD;
    x[1]=(spd-ROM_LSTM_8_SPD_MEAN)/ROM_LSTM_8_SPD_STD;
    x[2]=(sa -ROM_LSTM_8_SA_MEAN) /ROM_LSTM_8_SA_STD;
    float g[32]; int k;
    for(k=0;k<32;k++) g[k]=ROM_lstm_8_B_IH[k]+ROM_lstm_8_B_HH[k];
    matvec(g,ROM_lstm_8_W_IH,x,32,3);
    matvec(g,ROM_lstm_8_W_HH,s->h,32,8);
    for(k=0;k<8;k++){
        float ig=sig(g[k]), fg=sig(g[8+k]), gg=tanhf(g[16+k]), og=sig(g[24+k]);
        s->c[k]=fg*s->c[k]+ig*gg; s->h[k]=og*tanhf(s->c[k]);
    }
    float y=ROM_lstm_8_FC_B[0];
    for(k=0;k<8;k++) y+=ROM_lstm_8_FC_W[k]*s->h[k];
    return y*ROM_LSTM_8_TQ_STD+ROM_LSTM_8_TQ_MEAN;
}
