/******************************************************************************
 * @file    rom_ecm.h
 * @brief   Engine ROM – LSTM Reduced Order Model API
 *
 * Predicts engine torque from air charge, speed and spark advance.
 * Designed for NXP automotive MCUs (S32K, MPC5xxx, S32G).
 *
 * Model architecture:
 *   LSTM: 3 inputs → 32 hidden units → 1 output(s)
 *   Parameters: 4641 total floating-point weights
 *
 * Generated: 2026-03-04 17:57:07
 * Source:    Claude ROM Generator (enginespeed Simulink model)
 ******************************************************************************/

#ifndef ROM_ECM_H
#define ROM_ECM_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Dimensions ─────────────────────────────────────────────────────────── */
#define ROM_INPUT_SIZE    3U     /**< [AirCharge, Speed, SparkAdvance]     */
#define ROM_HIDDEN_SIZE   32U     /**< LSTM hidden state dimension           */
#define ROM_OUTPUT_SIZE   1U     /**< [Torque]                              */

/* ── Normalization constants (z-score: x_norm = (x - mean) / std) ─────── */
#define ROM_NORM_AC_MEAN   0.21671053f   /**< Air Charge mean [g/s]         */
#define ROM_NORM_AC_STD    0.12929910f    /**< Air Charge std  [g/s]         */
#define ROM_NORM_SPD_MEAN  580.25021710f  /**< Speed mean [rad/s]            */
#define ROM_NORM_SPD_STD   235.62429395f   /**< Speed std  [rad/s]            */
#define ROM_NORM_SA_MEAN   15.00000000f   /**< Spark Advance mean [deg]      */
#define ROM_NORM_SA_STD    7.07177361f    /**< Spark Advance std  [deg]      */
#define ROM_NORM_TQ_MEAN   25.94591248f   /**< Torque mean [N·m]             */
#define ROM_NORM_TQ_STD    46.99854266f    /**< Torque std  [N·m]             */

/* ── LSTM State (must be allocated per-instance or globally) ─────────────── */
typedef struct {
    float h[ROM_HIDDEN_SIZE];   /**< Hidden state vector                      */
    float c[ROM_HIDDEN_SIZE];   /**< Cell state vector                         */
} ROM_State_t;

/* ── Public API ─────────────────────────────────────────────────────────── */

/**
 * @brief  Initialize ROM state (call once at engine start or re-init).
 * @param  state  Pointer to ROM state structure.
 */
void ROM_Init(ROM_State_t * const state);

/**
 * @brief  Run one inference step of the LSTM ROM.
 *
 * Call at every control loop iteration (e.g., every 10 ms).
 * Inputs are physical-unit values; outputs are physical-unit torque.
 *
 * @param  state        Pointer to persistent ROM state.
 * @param  air_charge   Intake air charge [g/s]
 * @param  speed        Engine speed [rad/s]
 * @param  spark_adv    Spark advance [degrees]
 * @return              Predicted engine torque [N·m]
 */
float ROM_Step(ROM_State_t * const state,
               float air_charge,
               float speed,
               float spark_adv);

#ifdef __cplusplus
}
#endif

#endif /* ROM_ECM_H */
