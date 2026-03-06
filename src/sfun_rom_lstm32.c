/******************************************************************************
 * @file    sfun_rom_lstm32.c
 * @brief   MATLAB Level-2 MEX S-Function: Engine LSTM-32 ROM
 *
 * Wraps the LSTM-32 Reduced Order Model for use inside Simulink.
 * Compile with:
 *   mex sfun_rom_lstm32.c rom_lstm_32.c -I.
 * (run from the src/ directory, or set include path accordingly)
 *
 * Block ports:
 *   Inputs  (3): u[0]=AirCharge [g/s], u[1]=Speed [rad/s], u[2]=SparkAdv [deg]
 *   Outputs (1): y[0]=Torque [N·m]
 *
 * State: LSTM h and c vectors stored in DWork (64 floats, zero-initialised).
 ******************************************************************************/

#define S_FUNCTION_LEVEL 2
#define S_FUNCTION_NAME  sfun_rom_lstm32

/* ── MATLAB S-Function header ─────────────────────────────────────────────── */
#include "simstruc.h"

/* ── ROM implementation: include .c directly so mex sees all symbols ──────── */
#include "rom_lstm_32.c"

/* Number of LSTM state floats: h[32] + c[32] */
#define N_STATE  (2 * ROM_LSTM_32_HIDDEN)

/* =========================================================================
 * mdlInitializeSizes
 * ======================================================================= */
static void mdlInitializeSizes(SimStruct *S)
{
    ssSetNumSFcnParams(S, 0);
    if (ssGetNumSFcnParams(S) != ssGetSFcnParamsCount(S)) return;

    /* Inputs */
    if (!ssSetNumInputPorts(S, 1)) return;
    ssSetInputPortWidth(S, 0, 3);
    ssSetInputPortDirectFeedThrough(S, 0, 1);
    ssSetInputPortDataType(S, 0, SS_DOUBLE);

    /* Outputs */
    if (!ssSetNumOutputPorts(S, 1)) return;
    ssSetOutputPortWidth(S, 0, 1);
    ssSetOutputPortDataType(S, 0, SS_DOUBLE);

    /* DWork: LSTM h + c vectors as float32 */
    ssSetNumDWork(S, 1);
    ssSetDWorkWidth(S, 0, N_STATE);
    ssSetDWorkDataType(S, 0, SS_SINGLE);
    ssSetDWorkName(S, 0, "LSTMstate");

    ssSetNumSampleTimes(S, 1);
    ssSetOptions(S, SS_OPTION_EXCEPTION_FREE_CODE);
}

/* =========================================================================
 * mdlInitializeSampleTimes
 * ======================================================================= */
static void mdlInitializeSampleTimes(SimStruct *S)
{
    /* Inherited sample time – matches the driving block */
    ssSetSampleTime(S, 0, INHERITED_SAMPLE_TIME);
    ssSetOffsetTime(S, 0, 0.0);
}

/* =========================================================================
 * mdlStart  – called once before simulation; zeros LSTM state
 * ======================================================================= */
#define MDL_START
static void mdlStart(SimStruct *S)
{
    float *dw = (float *)ssGetDWork(S, 0);
    int i;
    for (i = 0; i < N_STATE; i++) dw[i] = 0.0f;
}

/* =========================================================================
 * mdlOutputs  – called every timestep
 * ======================================================================= */
static void mdlOutputs(SimStruct *S, int_T tid)
{
    const real_T *u  = ssGetInputPortRealSignal(S, 0);
    real_T       *y  = ssGetOutputPortRealSignal(S, 0);
    float        *dw = (float *)ssGetDWork(S, 0);

    /* Map DWork to ROM state struct (h first, c second) */
    ROM_lstm_32_State_t state;
    int k;
    for (k = 0; k < (int)ROM_LSTM_32_HIDDEN; k++) {
        state.h[k] = dw[k];
        state.c[k] = dw[ROM_LSTM_32_HIDDEN + k];
    }

    /* Run one LSTM step */
    float torque = ROM_lstm_32_Step(&state,
                                    (float)u[0],   /* AirCharge */
                                    (float)u[1],   /* Speed     */
                                    (float)u[2]);  /* SparkAdv  */

    /* Write output */
    y[0] = (real_T)torque;

    /* Write back updated state to DWork */
    for (k = 0; k < (int)ROM_LSTM_32_HIDDEN; k++) {
        dw[k]                       = state.h[k];
        dw[ROM_LSTM_32_HIDDEN + k]  = state.c[k];
    }
}

/* =========================================================================
 * mdlTerminate
 * ======================================================================= */
static void mdlTerminate(SimStruct *S)
{
    /* Nothing to free – all state in DWork */
    (void)S;
}

/* ── Required MEX/Simulink trailer ───────────────────────────────────────── */
#ifdef MATLAB_MEX_FILE
#include "simulink.c"
#else
#include "cg_sfun.h"
#endif
