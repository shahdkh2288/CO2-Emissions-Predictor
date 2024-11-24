# CO2-Emissions-Predictor

This repository contains the code and resources for **Assignment 1** in the Machine Learning course. The objective of this project is to use linear and logistic regression models to analyze and predict CO₂ emissions based on various vehicle features.

## Project Overview

With climate change becoming a critical issue, understanding the impact of vehicle emissions is essential. This project involves:
- **Linear Regression:** To predict the amount of CO₂ emissions (in g/km).
- **Logistic Regression:** To classify emission levels into categories.

## Dataset

The dataset contains over 7000 records of vehicle data with 11 feature columns (such as vehicle make, model, engine size, fuel type, etc.) and 2 target columns:
- **CO₂ Emission Amount (g/km)** - Continuous target for regression.
- **Emission Class** - Categorical target for classification.

## Requirements and Key Tasks

1. **Data Analysis**:
   - Checking for missing values.
   - Ensuring numerical feature scaling consistency.
   - Pairplot visualization and correlation heatmap.

2. **Data Preprocessing**:
   - Separation of features and targets.
   - Encoding of categorical data.
   - Data shuffling and split into training and testing sets.
   - Feature scaling.

3. **Linear Regression**:
   - Implemented with gradient descent from scratch.
   - Feature selection based on correlation analysis.
   - Cost function and error visualization.

4. **Logistic Regression**:
   - Predicting emission classes using a stochastic gradient descent classifier.
   - Model accuracy assessment.

## Evaluation Metrics

- **Linear Regression**: R² Score
- **Logistic Regression**: Accuracy

## Getting Started

1. Clone this repository.
   ```bash
   git clone https://github.com/shahdkh2288/CO2-Emissions-Predictor

2. Ensure the required libraries are installed (e.g., Pandas, NumPy, Matplotlib, Seaborn, Scikit-Learn).
3. Run the preprocessing script:
   ```bash
   python DataProcessing.py
4. Build the models:
   To build the Logistic Regression model, run:
   ```bash
   python logisticModel.py
5. To build the Linear Regression model, run:
   ```bash
   python linearRegressionModel.py


## Authors
1. Shahd Khaled Ahmed
2. Alaa Albsuny
3. Nahla Hesham
4. Aliaa Adel
5. Abdelmonem Tarek

