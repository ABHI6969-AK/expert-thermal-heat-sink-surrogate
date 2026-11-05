# Heat-sink surrogate assessment

## Assumptions

- Ambient temperature is 40.0 °C. Die, heat-sink geometry, aluminum conductivity, TIM thickness, junction-to-case resistance, and 25 °C air properties follow the supplied assessment values.
- Air properties are held constant at the supplied values (k = 0.0262 W/m·K, ν = 1.57e-05 m²/s, Pr = 0.71); they are not recalculated at the 40 °C operating point.
- Total resistance is `R_jc + R_TIM + R_cond + R_conv`. The model includes a straight-fin efficiency approximation and corrected fin length to represent fin-tip convection.
- The source assessment's laminar developing-flow and turbulent Nusselt correlations are used with its fin-spacing Reynolds-number convention and Re transition of 2300.
- The systematic grid contains 19,375 combinations: 25 TDP values, 31 air velocities, and 25 TIM conductivities.

## Reference case and key results

At 150 W, 1 m/s, and k_TIM = 4 W/m·K, the model predicts R_total = 0.384031 °C/W and T_j = 97.6046 °C. Maximum Re in the sweep is 1101.2; the turbulent branch is not reached over these requested ranges.

## Surrogate-model comparison

The test set holds out entire air-velocity levels. Scores measure agreement with this analytic model, not measured hardware accuracy.

| model                        | target          | MAE         | RMSE       | R2       |
| ---------------------------- | --------------- | ----------- | ---------- | -------- |
| Physics-feature Ridge        | R_total_C_per_W | 8.14137e-07 | 9.4893e-07 | 1        |
| Physics-feature Ridge        | T_junction_C    | 0.000275978 | 0.00036683 | 1        |
| Distance-weighted KNN (k=12) | R_total_C_per_W | 0.00117068  | 0.00179426 | 0.970762 |
| Distance-weighted KNN (k=12) | T_junction_C    | 0.362645    | 0.564059   | 0.999172 |

## Correlation analysis

| input            | target          | pearson_r | spearman_r |
| ---------------- | --------------- | --------- | ---------- |
| TDP_W            | R_total_C_per_W | -0.0000   | 0.0000     |
| TDP_W            | T_junction_C    | 0.9743    | 0.9815     |
| air_velocity_m_s | R_total_C_per_W | -0.8090   | -0.8977    |
| air_velocity_m_s | T_junction_C    | -0.1647   | -0.1468    |
| k_tim_W_mK       | R_total_C_per_W | -0.2294   | -0.3310    |
| k_tim_W_mK       | T_junction_C    | -0.0467   | -0.0439    |

- For `R_total_C_per_W`, the strongest absolute Spearman correlation is `air_velocity_m_s` (-0.8977).
- For `T_junction_C`, the strongest absolute Spearman correlation is `TDP_W` (0.9815).

## One-factor sensitivity

Each input is varied across its full requested range while the other inputs are at their midpoints. Positive values mean the output rises as the input moves from low to high.

| target          | varied_input     | low_value | high_value | signed_output_change | absolute_output_change | rank_by_absolute_change |
| --------------- | ---------------- | --------- | ---------- | -------------------- | ---------------------- | ----------------------- |
| R_total_C_per_W | air_velocity_m_s | 0.5       | 15         | -0.133736            | 0.133736               | 1                       |
| R_total_C_per_W | k_tim_W_mK       | 1         | 12         | -0.0388007           | 0.0388007              | 2                       |
| R_total_C_per_W | TDP_W            | 30        | 250        | 0                    | 0                      | 3                       |
| T_junction_C    | TDP_W            | 30        | 250        | 66.5569              | 66.5569                | 1                       |
| T_junction_C    | air_velocity_m_s | 0.5       | 15         | -18.723              | 18.723                 | 2                       |
| T_junction_C    | k_tim_W_mK       | 1         | 12         | -5.4321              | 5.4321                 | 3                       |

## Limitations and improvements

The model assumes uniform forced airflow, constant material and air properties, and simplified fin efficiency. The Nusselt correlations and their range of validity should be reviewed by a thermal engineer. The surrogate is trained on synthetic labels from this model, so high ML scores do not prove real-device accuracy. Future work should compare against CFD and bench measurements, use temperature-dependent air properties and measured fan curves, test multiple heat-sink geometries, and use group holdouts across geometry and operating conditions.
