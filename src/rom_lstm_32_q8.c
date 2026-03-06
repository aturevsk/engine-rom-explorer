#include "rom_lstm_32_q8.h"
#include "rom_lstm_32_q8_weights.h"
#include <math.h>
#include <string.h>
static float sig(float x){return 1.0f/(1.0f+expf(-x));}
static void matvec_q8(float *y, const int8_t *W, float scale, const float *x, int rows, int cols){
    int i,j; for(i=0;i<rows;i++){float a=0;for(j=0;j<cols;j++) a+=(float)W[i*cols+j]*scale*x[j];y[i]+=a;}
}
static void vec_add_q8(float *y, const int8_t *b, float scale, int n){
    int i; for(i=0;i<n;i++) y[i]+=(float)b[i]*scale;
}
void ROM_lstm_32_q8_Init(ROM_lstm_32_q8_State_t *s){memset(s,0,sizeof(*s));}
float ROM_lstm_32_q8_Step(ROM_lstm_32_q8_State_t *s, float ac, float spd, float sa){
    float x[3];
    x[0]=(ac -ROM_LSTM_32_Q8_AC_MEAN)/ROM_LSTM_32_Q8_AC_STD;
    x[1]=(spd-ROM_LSTM_32_Q8_SPD_MEAN)/ROM_LSTM_32_Q8_SPD_STD;
    x[2]=(sa -ROM_LSTM_32_Q8_SA_MEAN)/ROM_LSTM_32_Q8_SA_STD;
    float g[128]={0}; int k;
    vec_add_q8(g, ROM_lstm_32_q8_B_IH, ROM_lstm_32_q8_B_IH_SC, 128);
    vec_add_q8(g, ROM_lstm_32_q8_B_HH, ROM_lstm_32_q8_B_HH_SC, 128);
    matvec_q8(g, ROM_lstm_32_q8_W_IH, ROM_lstm_32_q8_W_IH_SC, x, 128, 3);
    matvec_q8(g, ROM_lstm_32_q8_W_HH, ROM_lstm_32_q8_W_HH_SC, s->h, 128, 32);
    for(k=0;k<32;k++){
        float ig=sig(g[k]), fg=sig(g[32+k]), gg=tanhf(g[64+k]), og=sig(g[96+k]);
        s->c[k]=fg*s->c[k]+ig*gg; s->h[k]=og*tanhf(s->c[k]);
    }
    float y=(float)ROM_lstm_32_q8_FC_B[0]*ROM_lstm_32_q8_FC_B_SC;
    for(k=0;k<32;k++) y+=(float)ROM_lstm_32_q8_FC_W[k]*ROM_lstm_32_q8_FC_W_SC*s->h[k];
    return y*ROM_LSTM_32_Q8_TQ_STD+ROM_LSTM_32_Q8_TQ_MEAN;
}
