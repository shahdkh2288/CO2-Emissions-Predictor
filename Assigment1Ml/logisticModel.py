import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score


X_train_scaled = pd.read_csv('X_train_scaled.csv')
X_test_scaled = pd.read_csv('X_test_scaled.csv')
y_train = pd.read_csv('y_train_class.csv')['Emission Class']
y_test = pd.read_csv('y_test_class.csv')['Emission Class']


X_train_selected = X_train_scaled[['Engine Size(L)', 'Fuel Consumption Comb (mpg)']]
X_test_selected = X_test_scaled[['Engine Size(L)', 'Fuel Consumption Comb (mpg)']]


logisticModel = SGDClassifier(loss='log_loss', max_iter=1000, random_state=42)


logisticModel.fit(X_train_selected, y_train)


y_pred = logisticModel.predict(X_test_selected)


accuracy = accuracy_score(y_test, y_pred)


print(f"Accuracy of the Logistic Regression Model with Engine Size(L),Fuel Consumption Comb (mpg) : {accuracy:.2f}")


