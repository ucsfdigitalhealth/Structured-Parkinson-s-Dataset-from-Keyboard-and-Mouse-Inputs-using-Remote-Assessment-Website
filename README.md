# Structured Parkinson's Dataset from Keyboard and Mouse Inputs using Remote Assessment Website

This repository provides a structured dataset and analysis workflow for Parkinson's disease assessment using keyboard and mouse interaction features collected through a remote web-based assessment platform. The project includes the extracted feature dataset, model-training code, a technical-validation notebook, and generated analysis artifacts.

The project is organized around two complementary workflows:

- `main.py` provides a script-based machine-learning pipeline.
- `Dataset_Paper_Technical_Validation.ipynb` provides an interactive notebook workflow for model evaluation, exploratory analysis, and statistical testing.

## Project Structure

```text
.
|-- README.md
|-- LICENSE
|-- pyproject.toml
|-- uv.lock
|-- main.py
|-- Dataset_Paper_Technical_Validation.ipynb
|-- whitney_test.csv
|-- dataset/
|   |-- 01_raw_dataset.csv
|   `-- 03_feature_extracted_dataset.csv
|-- src/
|   |-- __init__.py
|   |-- data_loader.py
|   |-- data_preprocessor.py
|   `-- model_trainer.py
`-- analysis/
    |-- CatBoost_roc_prc_curve.png
    |-- Explainable Boosting Classifier_roc_prc_curve.png
    |-- Meta Learner_roc_prc_curve.png
    |-- comparison.png
    |-- pca.png
    |-- t-sne.png
    `-- whitney-test.png
```

## Data and Artifacts

`dataset/raw_data.csv` contains the original structured assessment data. `dataset/feature_extracted_dataset.csv` contains the extracted features used by both the script and notebook workflows.

The `analysis/` directory contains generated visual artifacts, including ROC/PR curves, model-comparison plots, PCA and t-SNE visualizations, and statistical-test plots. The file `whitney_t_test.csv` stores the tabular output of the Mann-Whitney U statistical testing workflow from the notebook.

## Requirements

This project is configured for Python 3.11.

Environment manager:

- `uv`

Main Python dependencies:

- `catboost`
- `interpret`
- `xgboost`
- `lightgbm`
- `matplotlib`
- `numpy`
- `pandas`
- `scikit-learn`
- `scipy`
- `seaborn`
- `statsmodels`
- `ipykernel`

## Installation

Clone the repository and enter the project directory:

```powershell
git clone <repository-url>
cd Structured-Parkinson-s-Dataset-from-Keyboard-and-Mouse-Inputs-using-Remote-Assessment-Website-main
```

Install `uv` if it is not already available:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

For macOS or Linux:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

For additional installation options and troubleshooting, see the official `uv` installation documentation: <https://docs.astral.sh/uv/getting-started/installation/>.

Install dependencies from the locked project environment:

```powershell
uv sync
```

## Usage

This project can be used either through the main Python script or through the notebook. The script is suitable for running the classification workflow from the command line, while the notebook is suitable for step-by-step inspection, visualization, and technical validation.

## Main Script Workflow

The `main.py` script runs the core classification pipeline using `dataset/feature_extracted_dataset.csv`.

Run the script from the repository root:

```powershell
uv run python main.py
```

The script performs the following operations:

1. Loads `dataset/feature_extracted_dataset.csv`.
2. Removes the `Session ID` column.
3. Re-labels `suspectedpd` records as `pd`.
4. One-hot encodes categorical variables.
5. Evaluates models using 5-fold stratified cross-validation.
6. Prints the mean and standard deviation for each evaluation metric.
7. Saves model-performance figures to `analysis/`.

Models evaluated by `main.py`:

- CatBoost
- Explainable Boosting Classifier
- Stacked meta learner using CatBoost, XGBoost, LightGBM, and logistic regression

Expected outputs:

```text
analysis/
|-- CatBoost_roc_prc_curve.png
|-- Explainable Boosting Classifier_roc_prc_curve.png
|-- Meta Learner_roc_prc_curve.png
`-- comparison.png
```

## Notebook Workflow

The notebook `Dataset_Paper_Technical_Validation.ipynb` provides an interactive technical-validation workflow. It includes model evaluation, inference-time reporting, PCA and t-SNE analysis, and Mann-Whitney U statistical testing with multiple-testing correction.

### Local Notebook Execution

1. Complete the installation steps above.
2. Open `Dataset_Paper_Technical_Validation.ipynb` in Jupyter, JupyterLab, or VS Code.
3. Select the project virtual environment as the notebook kernel.
4. Run all cells from top to bottom.

When executed locally, the notebook uses:

```python
project_url = "./"
```

This allows the notebook to load files from the local repository structure.

### Google Colab Execution

The first notebook cell includes optional Google Drive mounting logic. If running in Colab:

1. Upload the repository to Google Drive or mount the repository directory.
2. Update the notebook path if your project is not located at:

```python
%cd "/content/drive/MyDrive/Parkinson_data_paper/"
```

3. Run all cells sequentially.

The Colab setup cell installs the required model libraries when Colab is detected.

## Notebook Outputs

The notebook generates:

- Cross-validation metrics for the selected classifiers.
- ROC and precision-recall visualizations.
- Model-comparison plots.
- PCA visualizations.
- t-SNE visualizations across feature groups and metadata categories.
- Mann-Whitney U test results with corrected p-values.
- `whitney_t_test.csv`.

Notebook figures are displayed inline. Some figures may also be saved depending on the executed cells and environment.

## Project Notes

- Run all commands from the repository root so relative paths resolve correctly.
- The `analysis/` directory is expected to exist before running `main.py`.
- Minor numeric differences can occur across operating systems, Python builds, and package versions because some model-training routines include stochastic behavior.
- Python 3.11 is recommended for compatibility with the locked project dependencies.

## License

The code is released under the MIT License to support open-source reuse and transparency. See `LICENSE` for details.
