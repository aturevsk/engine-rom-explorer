/* Auto-generated polynomial baseline  degree=2  n_raw_inputs=3  n_poly_features=10
 * Pipeline: raw inputs -> PolynomialFeatures(degree=2) -> StandardScaler -> Ridge
 * Produces physical torque output in N·m.
 */
#include <math.h>
#include "rom_delta_poly.h"

static const int POWERS[10][3] = {
    {0, 0, 0},
    {1, 0, 0},
    {0, 1, 0},
    {0, 0, 1},
    {2, 0, 0},
    {1, 1, 0},
    {1, 0, 1},
    {0, 2, 0},
    {0, 1, 1},
    {0, 0, 2}
};
static const float COEF_delta_poly[] = {0.00000000f, 51.09510062f, 6.37266556f, 5.43806591f, -0.57941879f, 0.93242721f, 2.40499187f, -32.24140134f, 3.49992695f, -4.05638351f};
static const float INTERCEPT_delta_poly = 25.99088902f;
/* StandardScaler applied to the 10 polynomial features (NOT the 3 raw inputs) */
static const float SC_MEAN_delta_poly[] = {1.00000000f, 0.21610840f, 591.01675295f, 17.50000000f, 0.06378270f, 140.19982313f, 3.76342451f, 405663.21525023f, 10722.06940056f, 379.16666667f};
static const float SC_STD_delta_poly[] = {1.00000000f, 0.13068993f, 237.40769360f, 8.53912564f, 0.06253770f, 129.63364385f, 3.19152417f, 300476.80880343f, 7378.14020112f, 305.30608503f};

float ROM_delta_poly_Predict(float air_charge, float speed, float spark_adv) {
    float raw[3] = {air_charge, speed, spark_adv};
    int f, k;

    /* Step 1: Compute 10 polynomial features from 3 raw inputs */
    float poly_feat[10];
    for (f = 0; f < 10; f++) {
        float term = 1.0f;
        for (k = 0; k < 3; k++) {
            int p = POWERS[f][k];
            int pp;
            for (pp = 0; pp < p; pp++) term *= raw[k];
        }
        poly_feat[f] = term;
    }

    /* Step 2: Apply StandardScaler to polynomial features */
    float x_sc[10];
    for (f = 0; f < 10; f++)
        x_sc[f] = (poly_feat[f] - SC_MEAN_delta_poly[f]) / SC_STD_delta_poly[f];

    /* Step 3: Ridge regression predict */
    float y = INTERCEPT_delta_poly;
    for (f = 0; f < 10; f++)
        y += COEF_delta_poly[f] * x_sc[f];

    return y;  /* physical torque in N·m */
}
