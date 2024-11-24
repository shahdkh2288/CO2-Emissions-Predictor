import os

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_process_data(file_path, target_column_class, target_column_emission, test_size=0.2, random_state=40):

    # Load the data
    datafile = pd.read_csv(file_path)

    # 2.1 Check for missing values
    missing_values = datafile.isnull().sum()
    print("\nMissing Values in Each Column:\n", missing_values)

    # 2.2 Check whether numeric features have the same scale
    numeric_columns = datafile.select_dtypes(include=['float64', 'int64']).columns
    print("\nNumeric Columns:\n", numeric_columns)

    # Summary statistics for numeric columns
    summary_stats = datafile[numeric_columns].describe()
    print("\nSummary Statistics for Numeric Columns:\n", summary_stats)

    # Check the range of each numeric feature
    print("\nRange of Each Numeric Feature:")
    for col in numeric_columns:
        col_range = summary_stats.loc['max', col] - summary_stats.loc['min', col]
        print(f"{col}: Range = {col_range}")

    # 2.3 Visualize Histograms for Numeric Features
    sns.pairplot(datafile[numeric_columns], diag_kind='hist')
    plt.title('Pairplot of Numeric Features')
    plt.show()

    # 2.4 Visualize a Correlation Heatmap Between Numeric Columns
    correlation_matrix = datafile[numeric_columns].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Heatmap of Numeric Columns')
    plt.show()

    # Preprocessing
    # 3.1 Define Features (X)
    X = datafile.drop(columns=[target_column_class, target_column_emission])
    y_class = datafile[target_column_class]
    y_emission = datafile[target_column_emission]

    # 3.2 Handle Non-Numeric Columns
    non_numeric_cols = X.select_dtypes(exclude=['float64', 'int64']).columns
    print("\nNon-Numeric Columns:", non_numeric_cols)

    if len(non_numeric_cols) > 0:
        X = pd.get_dummies(X, drop_first=True)

    # 3.3 Split Data into Train and Test Sets
    X_train, X_test, y_train_class, y_test_class, y_train_emission, y_test_emission = train_test_split(
        X, y_class, y_emission, test_size=test_size, random_state=random_state
    )

    # 3.4 Scale Numeric Features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Check and save the processed data
    if not os.path.exists('X_train_scaled.csv'):
        pd.DataFrame(X_train_scaled, columns=X.columns).to_csv('X_train_scaled.csv', index=False)

    if not os.path.exists('X_test_scaled.csv'):
        pd.DataFrame(X_test_scaled, columns=X.columns).to_csv('X_test_scaled.csv', index=False)

    if not os.path.exists('y_train_class.csv'):
        pd.DataFrame(y_train_class).to_csv('y_train_class.csv', index=False, header=['Emission Class'])

    if not os.path.exists('y_test_class.csv'):
        pd.DataFrame(y_test_class).to_csv('y_test_class.csv', index=False, header=['Emission Class'])

    if not os.path.exists('y_train_emission.csv'):
        pd.DataFrame(y_train_emission).to_csv('y_train_emission.csv', index=False, header=['CO2 Emissions (g/km)'])

    if not os.path.exists('y_test_emission.csv'):
        pd.DataFrame(y_test_emission).to_csv('y_test_emission.csv', index=False, header=['CO2 Emissions (g/km)'])


load_and_process_data("co2_emissions_data.csv", "Emission Class", "CO2 Emissions(g/km)")

