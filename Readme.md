<![endif]-->

**Reproducibility Package for Predictive Modelling and Visualization of Cockroach Activity**

**Overview**

This repository provides the full reproducibility package associated with the manuscript submitted to _Algorithms_. It includes the Python scripts used for (i) predictive modelling under class imbalance, (ii) statistical significance testing of model performance, and (iii) model explainability and visualization using SHAP values.

The objective of this repository is to enable transparent verification, replication, and extension of the experimental results presented in the article.

**Repository Structure**

The repository contains the following main scripts:

- **compare_imbalanced_methods.py**  
  Implements a comprehensive comparison of multiple machine learning algorithms for multi-class cockroach activity prediction under class imbalance. The script performs repeated stratified cross-validation, computes performance metrics with confidence intervals, generates precision–recall curves, and produces a consolidated performance figure.
- **statistical_significance_tests.py**  
  Performs non-parametric statistical analysis on cross-validation results using Friedman and Nemenyi tests. The script consolidates fold-level metrics across models and generates critical difference diagrams to assess statistically significant performance differences.
- **plot_figures_shap_per_class.py**  
  Generates model explainability figures using SHAP values, including global feature importance plots and class-specific SHAP visualizations (bar plots and beeswarm plots) to analyze the influence of environmental variables on predicted activity levels.

**Software Requirements**

The experiments were conducted using **Python 3.10**. The recommended setup is via **Miniconda** to ensure environment reproducibility.

Required libraries include (non-exhaustive):

- numpy, pandas, matplotlib
- scikit-learn, imbalanced-learn
- xgboost, lightgbm, catboost (optional, depending on availability)
- shap
- statds

All dependencies can be installed using:

pip install -r requirements.txt

**Environment Setup (Recommended)**

Create and activate a dedicated Conda environment:

conda create -n sadeco-ml python=3.10

conda activate sadeco-ml

pip install -r requirements.txt

**Execution Workflow**

**1. Predictive Modelling and Performance Evaluation**

Run:

python compare_imbalanced_methods.py

This script:

- Loads the dataset from a CSV file
- Trains multiple machine learning models under repeated stratified cross-validation
- Computes performance metrics (accuracy, balanced accuracy, F1 scores, AUPRC)
- Saves fold-level and out-of-fold predictions
- Generates a consolidated performance figure in PNG and SVG formats

**2. Statistical Significance Analysis**

Run:

python statistical_significance_tests.py

This script:

- Reads fold-level metrics from all evaluated models
- Applies Friedman and Nemenyi non-parametric tests
- Produces critical difference diagrams to assess statistical significance
- Saves all plots in PNG and SVG formats

**3. Model Explainability and Visualization**

Run:

python plot_figures_shap_per_class.py

This script:

- Trains a balanced XGBoost multi-class model
- Computes SHAP values for each class
- Generates global and per-class feature importance plots
- Produces SHAP beeswarm visualizations
- Saves all figures in PNG and SVG formats

**Data Availability**

The scripts expect a CSV dataset containing environmental variables (e.g., temperature, humidity, CO₂ concentration) and a categorical activity label. Due to privacy and operational constraints, the dataset used in the study is not publicly distributed.

Researchers may adapt the scripts to their own datasets by adjusting column names and configuration parameters.

**Reproducibility Notes**

- Random seeds are fixed to ensure reproducibility.
- All results are generated using out-of-fold predictions to avoid optimistic bias.
- Visualization outputs are provided in vector format (SVG) to facilitate inclusion in scientific publications.

**Citation**

If you use this code, please cite the associated article submitted to _Algorithms_.

**License**

This repository is provided for academic and research purposes. Licensing terms can be adapted according to journal or institutional requirements.

**4. Configuración XGBoost para SHAP**

La configuración específica del modelo XGBoost utilizado para generar los gráficos SHAP por clase se encuentra en `xgboost_config.py`. Esta configuración incluye:

- Path al dataset (`dataset_sadeco.csv`)
- Columna target (`Activity`)
- Hiperparámetros del modelo XGBoost
- Parámetros de split (test_size, random_state)

Para replicar los gráficos SHAP exactos, importar la configuración:

```python
from xgboost_config import DATA_CSV, TARGET_COL, XGB_PARAMS
```
