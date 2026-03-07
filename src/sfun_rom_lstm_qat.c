/******************************************************************************
 * sfun_rom_lstm_qat.c
 * MATLAB Level-2 MEX S-Function: Engine QAT LSTM-32 ROM
 *
 * Wraps the QAT int8 LSTM-32 Reduced Order Model for use inside Simulink.
 * Inputs are z-score normalised before being passed to ROM_lstm_qat_Step().
 * The normalised output torque is de-normalised back to physical units [N.m].
 *
 * Normalisation constants (from models/normalization.json):
 *   AirCharge  : mean=0.21671053,   std=0.12929910  [g/s]
 *   Speed      : mean=580.25021710, std=235.62429395 [rad/s]
 *   SparkAdv   : mean=15.00000000,  std=7.07177361   [deg]
 *   Torque     : mean=25.94591248,  std=46.99854266  [N.m]
 *
 * Compile (from project root in terminal):
 *   SRC=src
 *   MROOT=/Applications/MATLAB_R2025b.app
 *   xcrun clang -arch arm64 -bundle -undefined dynamic_lookup \
 *     -DMATLAB_MEX_FILE -O2 \
 *     -I"$MROOT/extern/include" -I"$MROOT/simulink/include" -I"$SRC" \
 *     -o "$SRC/sfun_rom_lstm_qat.mexmaca64" "$SRC/sfun_rom_lstm_qat.c" -lm
 *
 * Block ports:
 *   Input  port 0 (width 3): [AirCharge g/s, Speed rad/s, SparkAdv deg]
 *   Output port 0 (width 1): Torque [N.m]
 *
 * Sample time: explicit discrete 0.05 s so block is not affected by
 *   Inf-sample-time sources (Constant blocks).  Adjust if the model
 *   base rate is different.
 *
 * DWork (SS_DOUBLE, width 65):
 *   dw[0..31]  = h (LSTM hidden state, 32 doubles)
 *   dw[32..63] = c (LSTM cell   state, 32 doubles)
 *   dw[64]     = reset_flag (1.0 = zeroise h,c on next real mdlOutputs call)
 *
 * STATE INIT STRATEGY (robust to compilation-phase mdlOutputs calls):
 *   mdlStart sets the reset flag (DWork[64]=1).
 *   mdlInitializeConditions also sets the reset flag.
 *   In mdlOutputs, we check ssGetSimMode: if not SS_SIMMODE_NORMAL we return
 *   a default value WITHOUT running the LSTM so the state is never corrupted.
 *   In normal simulation mode, if the reset flag is set we zero h,c first.
 ******************************************************************************/

#define S_FUNCTION_LEVEL 2
#define S_FUNCTION_NAME  sfun_rom_lstm_qat

/* MATLAB S-Function header */
#include "simstruc.h"

/* ROM implementation: include .c so mex sees all symbols in one TU */
#include "rom_lstm_qat.c"

/* LSTM hidden dimension */
#define H  ROM_LSTM_QAT_HIDDEN          /* 32 */
/* DWork layout */
#define DW_H_START   0                  /* h[0..31]   doubles */
#define DW_C_START   H                  /* c[0..31]   doubles */
#define DW_FLAG      (2*H)              /* reset flag double  */
#define N_DWORK      (2*H + 1)          /* 65 doubles total   */

/* Normalisation constants */
#define NORM_AC_MEAN     0.21671053f
#define NORM_AC_STD      0.12929910f
#define NORM_SPD_MEAN  580.25021710f
#define NORM_SPD_STD   235.62429395f
#define NORM_SA_MEAN    15.00000000f
#define NORM_SA_STD      7.07177361f
#define NORM_TQ_MEAN    25.94591248f
#define NORM_TQ_STD     46.99854266f

/* =========================================================================
 * mdlInitializeSizes
 * ======================================================================= */
static void mdlInitializeSizes(SimStruct *S)
{
    ssSetNumSFcnParams(S, 0);
    if (ssGetNumSFcnParams(S) != ssGetSFcnParamsCount(S)) return;

    /* Input port 0: width 3 */
    if (!ssSetNumInputPorts(S, 1)) return;
    ssSetInputPortWidth(S, 0, 3);
    ssSetInputPortDirectFeedThrough(S, 0, 1);
    ssSetInputPortDataType(S, 0, SS_DOUBLE);
    /* Force contiguous storage so ssGetInputPortRealSignal (vect) is valid */
    ssSetInputPortRequiredContiguous(S, 0, 1);

    /* Output port 0: width 1 */
    if (!ssSetNumOutputPorts(S, 1)) return;
    ssSetOutputPortWidth(S, 0, 1);
    ssSetOutputPortDataType(S, 0, SS_DOUBLE);

    /* DWork: 65 doubles (h[32], c[32], reset_flag) */
    ssSetNumDWork(S, 1);
    ssSetDWorkWidth(S, 0, N_DWORK);
    ssSetDWorkDataType(S, 0, SS_DOUBLE);
    ssSetDWorkName(S, 0, "LSTMstate");

    /* Explicit 0.05 s discrete sample time.
     * INHERITED_SAMPLE_TIME with Constant-block sources gives Inf sample
     * time (block runs only once).  An explicit rate is safer. */
    ssSetNumSampleTimes(S, 1);
}

/* =========================================================================
 * mdlInitializeSampleTimes
 * ======================================================================= */
static void mdlInitializeSampleTimes(SimStruct *S)
{
    ssSetSampleTime(S, 0, 0.05);
    ssSetOffsetTime(S, 0, 0.0);
    ssSetModelReferenceSampleTimeDefaultInheritance(S);
}

/* =========================================================================
 * mdlStart  - called once when simulation begins
 * ======================================================================= */
#define MDL_START
static void mdlStart(SimStruct *S)
{
    real_T *dw = (real_T *)ssGetDWork(S, 0);
    int i;
    for (i = 0; i < N_DWORK; i++) dw[i] = 0.0;
    dw[DW_FLAG] = 1.0;   /* arm the reset flag */
}

/* =========================================================================
 * mdlInitializeConditions
 * Called AFTER compilation, BEFORE first real simulation mdlOutputs.
 * This is the authoritative reset point for h and c.
 * ======================================================================= */
#define MDL_INITIALIZE_CONDITIONS
static void mdlInitializeConditions(SimStruct *S)
{
    real_T *dw = (real_T *)ssGetDWork(S, 0);
    int i;
    for (i = 0; i < N_DWORK; i++) dw[i] = 0.0;
    dw[DW_FLAG] = 1.0;   /* arm the reset flag */
}

/* =========================================================================
 * mdlOutputs  - called at every simulation time step
 * ======================================================================= */
static void mdlOutputs(SimStruct *S, int_T tid)
{
    real_T *y = ssGetOutputPortRealSignal(S, 0);
    int i;

    /* Skip LSTM during compilation/sizing queries to avoid state corruption */
    if (ssGetSimMode(S) != SS_SIMMODE_NORMAL) {
        y[0] = (real_T)NORM_TQ_MEAN;
        return;
    }

    real_T       *dw = (real_T *)ssGetDWork(S, 0);
    /* ssSetInputPortRequiredContiguous guarantees signal.vect is populated,
     * so ssGetInputPortRealSignal (which reads signal.vect) is safe here. */
    const real_T *u = ssGetInputPortRealSignal(S, 0);

    /* If reset flag set, zero LSTM state (first step of a simulation run) */
    if (dw[DW_FLAG] != 0.0) {
        for (i = 0; i < 2*H; i++) dw[i] = 0.0;
        dw[DW_FLAG] = 0.0;
    }

    /* Normalise raw physical inputs */
    float x_norm[3];
    x_norm[0] = ((float)u[0] - NORM_AC_MEAN)  / NORM_AC_STD;
    x_norm[1] = ((float)u[1] - NORM_SPD_MEAN) / NORM_SPD_STD;
    x_norm[2] = ((float)u[2] - NORM_SA_MEAN)  / NORM_SA_STD;

    /* Copy LSTM state: DWork (double) -> local float arrays */
    float h_f[H], c_f[H];
    for (i = 0; i < H; i++) {
        h_f[i] = (float)dw[DW_H_START + i];
        c_f[i] = (float)dw[DW_C_START + i];
    }

    /* Run one QAT LSTM step (updates h_f, c_f in-place) */
    float torque_norm = ROM_lstm_qat_Step(x_norm, h_f, c_f);

    /* Write updated LSTM state back: float -> DWork (double) */
    for (i = 0; i < H; i++) {
        dw[DW_H_START + i] = (real_T)h_f[i];
        dw[DW_C_START + i] = (real_T)c_f[i];
    }

    /* De-normalise and write output */
    y[0] = (real_T)(torque_norm * NORM_TQ_STD + NORM_TQ_MEAN);
}

/* =========================================================================
 * mdlTerminate
 * ======================================================================= */
static void mdlTerminate(SimStruct *S)
{
    (void)S;
}

/* Required MEX/Simulink trailer */
#ifdef MATLAB_MEX_FILE
#include "simulink.c"
#else
#include "cg_sfun.h"
#endif
