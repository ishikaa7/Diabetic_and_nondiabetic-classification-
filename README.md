# Diabetes Classification Using Machine Learning

## Project Overview

This project focuses on predicting diabetes using supervised machine learning techniques. The workflow includes data preprocessing, exploratory data analysis (EDA), feature scaling, model training, and performance evaluation.

The objective is to identify patterns in health-related indicators that can help classify individuals as diabetic or non-diabetic.

## Dataset

The project uses the BRFSS 2015 Diabetes Health Indicators dataset.

Files included:

* `diabetes_012_health_indicators_BRFSS2015.csv` – Original dataset
* `diabetes_processed_data_no_prediabetes.csv` – Processed dataset used for training and evaluation

## Workflow

The complete implementation is documented in `classification.ipynb`, including:

* Data Understanding
* Data Cleaning
* Exploratory Data Analysis (EDA)
* Feature Scaling
* Handling Class Distribution
* Model Training
* Model Evaluation
* Performance Comparison

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Jupyter Notebook

## Model Evaluation

The model performance is evaluated using standard classification metrics such as:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC Score

Detailed results, visualizations, and observations are available in the notebook.

## Training the Model

To train the model and generate the serialized files (`diabetes_model.pkl` and `scaler.pkl`), run all cells in:

```bash
jupyter notebook classification.ipynb
```

The notebook performs:

* Data preprocessing
* Feature scaling
* Model training
* Model evaluation
* Model serialization

Upon successful execution, the following files will be generated:

```text
diabetes_model.pkl
scaler.pkl
```

> **Note:** The trained model file (`diabetes_model.pkl`) is not included in this repository due to its size. Run `classification.ipynb` to train the model and generate the required `.pkl` files.

## How to Run

1. Clone the repository

```bash
git clone https://github.com/ishikaa7/<repository-name>.git
```

2. Install required libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

3. Launch Jupyter Notebook

```bash
jupyter notebook
```

4. Open and run

```text
classification.ipynb
```

