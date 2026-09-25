# Linear Regression Architecture Workshop

## Group Members and Responsibilities

This project was completed collaboratively by a team of four, following the workshop's four sequential parts. Each part depended on the completion of the previous one.

| Group Member | Role | Responsibilities |
|---|---|---|
| **Eche Oji (9078881)** | Data Sourcing & Exploratory Data Analysis | Collected California and Ontario housing price data from a CSV source, a housing/open-data API, and a relational database; built `notebooks/EDA.ipynb` demonstrating all three sources; performed initial EDA (`.head()`, `.info()`, `.describe()`, distributions, missing values) and identified the target variable and single predictor for regression. |
| **Antonio Sainz (9072844)** | Preprocessing & Linear Regression | Preprocessed the selected dataset (missing values, normalization, train/test split); implemented univariate linear regression from scratch (hypothesis, MSE cost, gradient descent) and with `scikit-learn`; evaluated both with RMSE, MAE, and R², and produced the regression visualizations in `notebooks/linear_regression.ipynb`. |
| **Sultan Atanda (9114837)** | Modularization & MLOps Configuration | Converted the working data-loading and regression code into modular, config-driven source files (`src/data_loader.py`, `preprocessing.py`, `model.py`, `evaluation.py`); built `configs/experiment_config.yaml`; established the full project structure and connected all modules into a single reproducible pipeline. |
| **John Buni (9115726)** | Experiment Tracking, Documentation & Final Integration | Implemented experiment logging to `experiments/results.csv`; wrote `requirements.txt` and `README.md`; updated `RobotPM_MLOps.ipynb` with the project's architectural changes; ran final end-to-end testing across the whole repository before submission. |

### Team Workflow

The four parts form a single sequential pipeline, each stage depending on the one before it:

**Data Sourcing & EDA → Preprocessing & Regression Modeling → Modularization & MLOps Configuration → Experiment Tracking, Documentation & Final Integration**

## Overview


This project implements a reproducible univariate linear regression workflow for housing price analysis using California and Ontario housing data.

The project demonstrates data sourcing, exploratory data analysis (EDA), preprocessing, model training, evaluation, experiment tracking, and an MLOps-style modular architecture.

The workflow supports data obtained from CSV files, a REST API, and a relational database.

## Objectives

- Load and explore housing data from CSV, API, and relational database sources.
- Identify a housing-price target and a single predictor for univariate linear regression.
- Perform exploratory data analysis and preprocessing.
- Implement linear regression from scratch using gradient descent.
- Compare the from-scratch implementation with Scikit-learn's `LinearRegression`.
- Evaluate model performance using RMSE, MAE, and R².
- Separate data loading, preprocessing, modeling, and evaluation into reusable modules.
- Use YAML configuration to control experiment settings.
- Track experiment results in CSV files for reproducibility.

## Data Sources

The project demonstrates three data-source types:

- **CSV:** Housing datasets for California and Ontario.
- **REST API:** California open-data API configured in `configs/experiment_config.yaml`.
- **Relational Database:** SQLite database containing California housing data.

The configuration file defines the locations and connection information used by the data-loading components.

## Project Structure

```text
LinearRegressionArchitecture_Group1/
│
├── configs/
│   └── experiment_config.yaml
│
├── data/
│   ├── raw/
│   └── processed/
│
├── experiments/
│   ├── results.csv
│   └── robot_results.csv
│
├── models/
│
├── notebooks/
│   ├── EDA.ipynb
│   ├── linear_regression.ipynb
│   └── ...
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── model.py
│   ├── evaluation.py
│   └── ...
│
├── Orchestrator_main.py
├── README.md
└── requirements.txt
```

### Core Housing Pipeline Modules

- `src/config.py` — loads experiment settings from the YAML configuration file.
- `src/data_loader.py` — handles CSV, REST API, and SQLite data sources.
- `src/preprocessing.py` — performs feature/target selection, missing-value handling, train/test splitting, and standardization.
- `src/model.py` — contains both the from-scratch and Scikit-learn linear regression implementations.
- `src/evaluation.py` — calculates evaluation metrics, creates regression visualizations, and saves experiment results.

## Installation and Environment Setup

Clone the repository and move into the project directory:

```powershell
git clone https://github.com/kinghighshow/LinearRegressionArchitecture_Group1.git
cd LinearRegressionArchitecture_Group1
```

Create a Python virtual environment:

```powershell
python -m venv .venv
```

Activate the environment in Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Upgrade pip and install the project dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Register the virtual environment as a Jupyter kernel:

```powershell
python -m ipykernel install --user --name linear-regression-group1 --display-name "Python (.venv - Linear Regression Group1)"
```

In VS Code, open the project notebooks and select:

```text
Python (.venv - Linear Regression Group1)
```

as the notebook kernel.

## Configuration

Experiment settings are stored in:

```text
configs/experiment_config.yaml
```

For the housing linear regression workflow, the current configuration specifies:

```yaml
modeling:
  feature: median_income
  target: median_house_value

training:
  test_size: 0.2
  random_state: 42
  learning_rate: 0.01
  iterations: 1000
```

Using a YAML configuration file keeps important experiment parameters outside the model code and makes experiments easier to reproduce and modify.

## Running the Project

### 1. Exploratory Data Analysis

Open:

```text
notebooks/EDA.ipynb
```

Select the project kernel and use **Restart Kernel and Run All Cells**.

The notebook demonstrates data loading and exploratory analysis of the project data sources.

### 2. Linear Regression

Open:

```text
notebooks/linear_regression.ipynb
```

Select the same project kernel and use **Restart Kernel and Run All Cells**.

The notebook performs preprocessing, trains the two linear regression implementations, evaluates their predictions, and displays the model results and visualizations.

## Model

The housing model uses:

- **Predictor:** `median_income`
- **Target:** `median_house_value`

Two implementations are compared.

### From-Scratch Linear Regression

`ScratchLinearRegression` implements univariate linear regression using NumPy and batch gradient descent.

The model learns an intercept and coefficient by repeatedly updating the parameters to reduce mean squared error.

### Scikit-learn Linear Regression

`SklearnLinearRegression` provides the same `fit()` and `predict()` interface while using Scikit-learn's `LinearRegression`.

Using a shared interface allows both implementations to pass through the same evaluation workflow.

## Evaluation

Both implementations are evaluated using:

- **RMSE (Root Mean Squared Error)**
- **MAE (Mean Absolute Error)**
- **R² (Coefficient of Determination)**

The verified results from the current housing experiment are:

| Model | RMSE | MAE | R² |
|---|---:|---:|---:|
| From scratch | 84,209.01 | 62,990.87 | 0.4589 |
| Scikit-learn | 84,209.01 | 62,990.87 | 0.4589 |

The nearly identical results show that the from-scratch implementation produces results consistent with the Scikit-learn implementation for the same prepared dataset and train/test split.

## Experiment Tracking

Housing regression experiment results are stored in:

```text
experiments/results.csv
```

The housing experiment records model performance and experiment parameters such as:

- model
- predictor
- learning rate
- iterations
- train/test split
- RMSE
- MAE
- R²

The robot failure-prediction workflow maintains its experiment results separately in:

```text
experiments/robot_results.csv
```

Keeping the experiment outputs separate prevents housing regression metrics and robot classification metrics from being mixed in the same results file.

Saving experiment results outside the notebook provides a persistent record of model runs and supports reproducibility.

## MLOps Architecture

The project applies several MLOps-oriented design principles.

### Separation of Concerns

Responsibilities are divided into separate modules:

```text
Data Loading
     ↓
Preprocessing
     ↓
Model Training
     ↓
Evaluation
     ↓
Experiment Tracking
```

This prevents the complete machine-learning workflow from being contained in one notebook or script.

### Configuration-Driven Experiments

Dataset locations, predictor and target variables, train/test settings, learning rate, and iteration count are stored in `configs/experiment_config.yaml`.

This reduces hard-coded experiment settings and makes configuration changes easier to reproduce.

### Reproducibility

The project supports reproducibility through:

- a dedicated Python virtual environment
- a documented `requirements.txt`
- a fixed `random_state` for the housing train/test split
- YAML-based experiment configuration
- CSV-based experiment tracking
- modular Python source files
- Jupyter notebooks that can be restarted and executed from beginning to end

## Key Design Decisions

### Shared Model Interface

The from-scratch and Scikit-learn regression implementations expose compatible `fit()` and `predict()` methods. This allows the same evaluation workflow to be used for both models.

### Train-Only Scaling

The preprocessing module fits the `StandardScaler` using the training data and then applies that fitted scaler to the test data. This avoids using information from the test set when fitting the scaler.

### Centralized Data Loading

CSV, REST API, and relational database loading are handled by `src/data_loader.py`, separating data acquisition from preprocessing and modeling.

### Persistent Experiment Results

Evaluation results can be written to CSV rather than existing only as temporary notebook output.

## Current Integration Status

The following integration tests have been completed:

- `notebooks/EDA.ipynb` — Restart Kernel + Run All completed without observed errors.
- `notebooks/linear_regression.ipynb` — Restart Kernel + Run All completed without observed errors.
- Project dependencies were tested inside a clean `.venv`.
- Missing notebook dependencies were identified and added to `requirements.txt`.
- The configuration module was repaired after duplicated source code caused an import error.
- `Orchestrator_main.py` successfully completed its robot failure-prediction training workflow after the configuration repair.

The Part 4 documentation and integration workflow has been completed and verified through clean-environment testing, notebook execution, configuration loading, module import testing, and configuration-driven experiment tracking.