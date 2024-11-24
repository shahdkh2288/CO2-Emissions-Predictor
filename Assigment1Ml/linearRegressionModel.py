import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt




X_train_scaled = pd.read_csv('X_train_linear_scaled.csv')
X_test_scaled = pd.read_csv('X_test_linear_scaled.csv')
y_train = pd.read_csv('y_train_emission.csv')['CO2 Emissions (g/km)']
y_test = pd.read_csv('y_test_emission.csv')['CO2 Emissions (g/km)']


X_train_selected = X_train_scaled[['Cylinders', 'Fuel Consumption Comb (L/100 km)']]
X_test_selected = X_test_scaled[['Cylinders', 'Fuel Consumption Comb (L/100 km)']]

# Linear Regression from Scratch with Gradient Descent
theta = np.array([0.0, 0.0, 0.0])  # Initializing parameters
m = len(y_train)  # Number of training samples
iterations = 5000  # Number of iterations for gradient descent
alpha = 0.1  # Learning rate

# Add a column of ones for the intercept term
X_train_selected = np.c_[np.ones(m), X_train_selected]  # Adding X0 = 1 to the features
X_test_selected = np.c_[np.ones(len(y_test)), X_test_selected]  # Adding X0 = 1 for test data


# Cost function
def calculate_cost(X, y, theta):
    return np.sum((X.dot(theta) - y) ** 2) / (2 * m)


# Initial cost
cost = calculate_cost(X_train_selected, y_train, theta)
print("Initial Cost:", cost)

# Gradient descent function
def gradient_descent(X, y, theta, alpha, iterations):
    cost_history = np.zeros(iterations)

    for iteration in range(iterations):
        hypothesis = X.dot(theta)
        loss = hypothesis - y
        gradient = X.T.dot(loss) / m
        theta -= alpha * gradient  # Update the parameters
        cost_history[iteration] = calculate_cost(X, y, theta)  # Store the cost value

    return theta, cost_history




# Perform gradient descent
theta, cost_history = gradient_descent(X_train_selected, y_train, theta, alpha, iterations)
print(theta)

# Plot cost history
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(range(iterations), cost_history, color='blue')
plt.title('Cost Function History')
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.show()

# Make predictions using the learned parameters
predictions = X_test_selected.dot(theta)

# Calculate R² score for the model
r2 = r2_score(y_test, predictions)
print(f"R² Score on Test Set: {r2}")