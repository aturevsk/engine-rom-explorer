/*
 * validate_roms_harness.c
 * =======================
 * Standalone C99 test harness for Pareto-optimal ROM variants.
 *
 * Reads validation_data.csv (Time,AirCharge,Speed,Torque,SparkAdvance,SimID,Throttle),
 * runs each ROM model step-by-step, accumulates RMSE and R² per simulation,
 * and writes two output streams:
 *
 *   stdout lines starting with "DATA "   – per-step predictions
 *   stdout lines starting with "METRIC " – per-sim RMSE/R²
 *   stdout line  starting with "OVERALL" – global RMSE/R²
 *
 * Models validated (Pareto-optimal set):
 *   1. NARX-Ridge       0.38 KB   autoregressive lag model
 *   2. LSTM-8           2.52 KB   recurrent, physical I/O API
 *   3. Delta composite  2.79 KB   Poly-2 baseline + LSTM-8 residual
 *   4. LSTM-16 Q16      4.05 KB   int16_t weights, normalised I/O API
 *   5. QAT LSTM-32      6.97 KB   int8_t weights, normalised I/O API
 *
 * Compile:
 *   gcc -O2 -std=c99 -I./src \
 *       src/validate_roms_harness.c \
 *       src/rom_narx_ridge.c \
 *       src/rom_lstm_8.c \
 *       src/rom_delta_poly.c \
 *       src/rom_lstm_delta8.c \
 *       src/rom_lstm_16_q16.c \
 *       src/rom_lstm_qat.c \
 *       -lm -o validate_roms
 *
 * Usage:
 *   ./validate_roms <path_to_validation_data.csv>
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ── ROM headers ──────────────────────────────────────────────────────────── */
#include "rom_narx_ridge.h"
#include "rom_lstm_8.h"
#include "rom_delta_poly.h"
#include "rom_lstm_delta8.h"
#include "rom_lstm_16_q16.h"
#include "rom_lstm_qat.h"

/* ── Normalisation constants (from models/normalization.json) ─────────────── */
#define NORM_AC_MEAN    0.21671053f
#define NORM_AC_STD     0.12929910f
#define NORM_SPD_MEAN  580.25021710f
#define NORM_SPD_STD   235.62429395f
#define NORM_SA_MEAN    15.00000000f
#define NORM_SA_STD      7.07177361f
#define NORM_TQ_MEAN    25.94591248f
#define NORM_TQ_STD     46.99854266f
/* Delta stats from models/delta_poly_info.json */
#define NORM_DELTA_MEAN  0.00000000f
#define NORM_DELTA_STD   0.87008924f

/* ── CSV parsing ──────────────────────────────────────────────────────────── */
typedef struct {
    double time;
    float  ac, spd, tq_true, sa;
    int    sim_id;
} Row;

#define MAX_ROWS 10000

static Row g_rows[MAX_ROWS];
static int g_n_rows = 0;

static int load_csv(const char *path)
{
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); return -1; }

    char line[512];
    /* skip header */
    if (!fgets(line, sizeof(line), f)) { fclose(f); return -1; }

    int n = 0;
    while (fgets(line, sizeof(line), f) && n < MAX_ROWS) {
        Row *r = &g_rows[n];
        double t, ac, spd, tq, sa, throttle;
        int sid;
        int parsed = sscanf(line, "%lf,%lf,%lf,%lf,%lf,%d,%lf",
                            &t, &ac, &spd, &tq, &sa, &sid, &throttle);
        if (parsed < 6) continue;
        r->time    = t;
        r->ac      = (float)ac;
        r->spd     = (float)spd;
        r->tq_true = (float)tq;
        r->sa      = (float)sa;
        r->sim_id  = sid;
        n++;
    }
    fclose(f);
    g_n_rows = n;
    return n;
}

/* ── Statistics accumulator ───────────────────────────────────────────────── */
typedef struct {
    double sse;          /* sum of squared errors */
    double ss_tot;       /* sum of squares for R² */
    double tq_sum;       /* for computing mean */
    double tq_sq_sum;
    int    n;
} Acc;

static void acc_reset(Acc *a) { memset(a, 0, sizeof(*a)); }

static void acc_update(Acc *a, float pred, float truth)
{
    double e = (double)pred - (double)truth;
    a->sse      += e * e;
    a->tq_sum   += truth;
    a->tq_sq_sum += (double)truth * truth;
    a->n++;
}

static double acc_rmse(const Acc *a)
{
    return (a->n > 0) ? sqrt(a->sse / a->n) : 0.0;
}

static double acc_r2(const Acc *a)
{
    if (a->n < 2) return 0.0;
    double mean = a->tq_sum / a->n;
    double ss_tot = a->tq_sq_sum - a->n * mean * mean;
    if (ss_tot < 1e-12) return 1.0;
    return 1.0 - a->sse / ss_tot;
}

/* ── Helper: normalise inputs for Phase-3 models ──────────────────────────── */
static void norm_inputs(float ac, float spd, float sa, float *x)
{
    x[0] = (ac  - NORM_AC_MEAN)  / NORM_AC_STD;
    x[1] = (spd - NORM_SPD_MEAN) / NORM_SPD_STD;
    x[2] = (sa  - NORM_SA_MEAN)  / NORM_SA_STD;
}

static float denorm_tq(float y_norm)
{
    return y_norm * NORM_TQ_STD + NORM_TQ_MEAN;
}

static float denorm_delta(float d_norm)
{
    return d_norm * NORM_DELTA_STD + NORM_DELTA_MEAN;
}

/* ════════════════════════════════════════════════════════════════════════════
 * MAIN
 * ════════════════════════════════════════════════════════════════════════════ */
int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <validation_data.csv>\n", argv[0]);
        return 1;
    }

    if (load_csv(argv[1]) < 0) return 1;
    fprintf(stderr, "Loaded %d rows from %s\n", g_n_rows, argv[1]);

    /* ── ROM state variables ──────────────────────────────────────────────── */
    NARX_narx_ridge_State_t narx_st;
    ROM_lstm_8_State_t      lstm8_st;

    /* Delta composite: poly has no state; delta LSTM-8 */
    float delta8_h[ROM_LSTM_DELTA8_HIDDEN], delta8_c[ROM_LSTM_DELTA8_HIDDEN];

    /* LSTM-16 Q16 */
    float q16_h[ROM_LSTM_16_Q16_HIDDEN], q16_c[ROM_LSTM_16_Q16_HIDDEN];

    /* QAT LSTM-32 */
    float qat_h[ROM_LSTM_QAT_HIDDEN], qat_c[ROM_LSTM_QAT_HIDDEN];

    /* ── Global accumulators ──────────────────────────────────────────────── */
    Acc g_narx, g_lstm8, g_delta, g_q16, g_qat;
    acc_reset(&g_narx); acc_reset(&g_lstm8); acc_reset(&g_delta);
    acc_reset(&g_q16);  acc_reset(&g_qat);

    /* Print header */
    printf("# DATA columns: sim_id time true_tq "
           "pred_narx pred_lstm8 pred_delta pred_q16 pred_qat\n");

    /* ── Per-simulation loop ──────────────────────────────────────────────── */
    int i = 0;
    while (i < g_n_rows) {
        int sim_id = g_rows[i].sim_id;

        /* Reset all ROM states at start of each simulation */
        NARX_narx_ridge_Init(&narx_st);
        ROM_lstm_8_Init(&lstm8_st);
        ROM_lstm_delta8_Reset(delta8_h, delta8_c);
        ROM_lstm_16_q16_Reset(q16_h, q16_c);
        ROM_lstm_qat_Reset(qat_h, qat_c);

        Acc sim_narx, sim_lstm8, sim_delta, sim_q16, sim_qat;
        acc_reset(&sim_narx); acc_reset(&sim_lstm8); acc_reset(&sim_delta);
        acc_reset(&sim_q16);  acc_reset(&sim_qat);

        /* Process all rows of this simulation */
        while (i < g_n_rows && g_rows[i].sim_id == sim_id) {
            Row *r = &g_rows[i];
            float ac  = r->ac;
            float spd = r->spd;
            float sa  = r->sa;
            float tq  = r->tq_true;

            /* 1. NARX-Ridge */
            float pred_narx  = NARX_narx_ridge_Step(&narx_st, ac, spd, sa);

            /* 2. LSTM-8 (physical I/O) */
            float pred_lstm8 = ROM_lstm_8_Step(&lstm8_st, ac, spd, sa);

            /* 3. Delta composite: poly baseline + LSTM-8 residual */
            float poly_pred  = ROM_delta_poly_Predict(ac, spd, sa);
            float x3[3]; norm_inputs(ac, spd, sa, x3);
            float delta_norm = ROM_lstm_delta8_Step(x3, delta8_h, delta8_c);
            float pred_delta = poly_pred + denorm_delta(delta_norm);

            /* 4. LSTM-16 Q16 (normalised I/O) */
            float pred_q16   = denorm_tq(ROM_lstm_16_q16_Step(x3, q16_h, q16_c));

            /* 5. QAT LSTM-32 (normalised I/O) */
            float pred_qat   = denorm_tq(ROM_lstm_qat_Step(x3, qat_h, qat_c));

            /* Accumulate */
            acc_update(&sim_narx,  pred_narx,  tq);
            acc_update(&sim_lstm8, pred_lstm8, tq);
            acc_update(&sim_delta, pred_delta, tq);
            acc_update(&sim_q16,   pred_q16,   tq);
            acc_update(&sim_qat,   pred_qat,   tq);
            acc_update(&g_narx,    pred_narx,  tq);
            acc_update(&g_lstm8,   pred_lstm8, tq);
            acc_update(&g_delta,   pred_delta, tq);
            acc_update(&g_q16,     pred_q16,   tq);
            acc_update(&g_qat,     pred_qat,   tq);

            /* Print per-step DATA line */
            printf("DATA %d %.3f %.6f %.6f %.6f %.6f %.6f %.6f\n",
                   sim_id, (float)r->time, tq,
                   pred_narx, pred_lstm8, pred_delta, pred_q16, pred_qat);

            i++;
        }

        /* Print per-sim METRIC lines */
        printf("METRIC narx_ridge  %d %d %.6f %.6f\n",
               sim_id, sim_narx.n,  acc_rmse(&sim_narx),  acc_r2(&sim_narx));
        printf("METRIC lstm_8      %d %d %.6f %.6f\n",
               sim_id, sim_lstm8.n, acc_rmse(&sim_lstm8), acc_r2(&sim_lstm8));
        printf("METRIC delta       %d %d %.6f %.6f\n",
               sim_id, sim_delta.n, acc_rmse(&sim_delta), acc_r2(&sim_delta));
        printf("METRIC lstm_16_q16 %d %d %.6f %.6f\n",
               sim_id, sim_q16.n,   acc_rmse(&sim_q16),   acc_r2(&sim_q16));
        printf("METRIC qat_lstm32  %d %d %.6f %.6f\n",
               sim_id, sim_qat.n,   acc_rmse(&sim_qat),   acc_r2(&sim_qat));
    }

    /* Print overall METRIC lines */
    printf("OVERALL narx_ridge  %d %.6f %.6f\n",
           g_narx.n,  acc_rmse(&g_narx),  acc_r2(&g_narx));
    printf("OVERALL lstm_8      %d %.6f %.6f\n",
           g_lstm8.n, acc_rmse(&g_lstm8), acc_r2(&g_lstm8));
    printf("OVERALL delta       %d %.6f %.6f\n",
           g_delta.n, acc_rmse(&g_delta), acc_r2(&g_delta));
    printf("OVERALL lstm_16_q16 %d %.6f %.6f\n",
           g_q16.n,   acc_rmse(&g_q16),   acc_r2(&g_q16));
    printf("OVERALL qat_lstm32  %d %.6f %.6f\n",
           g_qat.n,   acc_rmse(&g_qat),   acc_r2(&g_qat));

    return 0;
}
