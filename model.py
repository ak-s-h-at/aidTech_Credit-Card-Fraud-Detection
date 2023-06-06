import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, f1_score, accuracy_score, precision_score
import joblib


data = []
with open('D:/Task3- Credit Card Fraud Detection/creditcard.csv', 'r') as file:
    reader = csv.reader(file)
    next(reader)
    for row in reader:
        data.append(row)

data = np.array(data)

data = data.astype(np.float64)

data = data[~np.isnan(data).any(axis=1)]

X = data[:, 1:-2]
y = data[:, -1].astype(np.int64)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)

print("Model Performance:")
print("Recall: {:.4f}".format(recall))
print("F1 Score: {:.4f}".format(f1))
print("Accuracy: {:.4f}".format(accuracy))
print("Precision: {:.4f}".format(precision))

# Save the model and scaler
joblib.dump(model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')
