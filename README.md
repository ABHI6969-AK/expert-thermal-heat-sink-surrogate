# Heat Sink Parameter Sweep and Surrogate Models

This repository contains a reproducible heat sink parameter sweep and two surrogate regressors for estimating total thermal resistance and junction temperature.

Run `python3 thermal_assessment.py` to generate the dataset and refresh the model metrics, correlation analysis, sensitivity analysis, results, and summary report. The notebook provides the same workflow in Jupyter format. Required packages are listed in `requirements.txt`.

The grid contains 19,375 combinations across TDP from 30 W to 250 W, air velocity from 0.5 m/s to 15 m/s, and TIM conductivity from 1 W/m·K to 12 W/m·K. The surrogate scores measure agreement with the simplified analytic model, not accuracy against measured hardware.

The model uses the assessment geometry, aluminum conductivity, TIM thickness, and air property values. Ambient temperature is set to 40 °C. Air properties remain fixed at the supplied 25 °C values. See `summary_report.md` for assumptions, evaluation results, sensitivity findings, and limitations.

## Dataset files

- [`thermal_dataset_preview.csv`](thermal_dataset_preview.csv) is a 216-row representative sample that GitHub can render as an interactive table.
- [`thermal_dataset.csv`](thermal_dataset.csv) is the complete 19,375-row dataset. It is larger than GitHub's 512 KB interactive CSV-rendering limit, so GitHub shows it as raw text. Download it to inspect the complete data.

## Other files

- `thermal_assessment.py` contains the physics model, parameter sweep, Ridge and distance-weighted KNN surrogates, evaluation, and analysis.
- `thermal_assessment.ipynb` runs the same workflow interactively.
- `model_metrics.csv`, `correlation_analysis.csv`, and `sensitivity_analysis.csv` contain the evaluation and analysis tables.
- `results.json` stores configuration and reference case values.
- `summary_report.md` presents the technical summary.
