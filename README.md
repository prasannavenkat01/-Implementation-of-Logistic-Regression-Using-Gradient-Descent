# Implementation of Logistic Regression Using Gradient Descent

## Aim:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the data file and import numpy, matplotlib and scipy.
2. Visulaize the data and define the sigmoid function, cost function and gradient descent.
3. Plot the decision boundary .
4. Calculate the y-prediction.

## Program:
```
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: PRASANNA V
RegisterNumber: 212223240123
```

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset = pd.read_csv('Placement_Data.csv')
dataset
dataset = dataset.drop('sl_no',axis=1) 
dataset = dataset.drop('salary',axis=1)
dataset["gender"] = dataset["gender"].astype('category')
dataset["ssc_b"] = dataset["ssc_b"].astype('category')
dataset["hsc_b"] = dataset["hsc_b"].astype('category')
dataset["degree_t"] = dataset["degree_t"].astype('category')
dataset["workex"] = dataset["workex"].astype('category')
dataset["specialisation"] = dataset["specialisation"].astype('category')
dataset["status"] = dataset["status"].astype('category')
dataset["hsc_s"] = dataset["hsc_s"].astype('category')
dataset.dtypes
dataset["gender"] = dataset["gender"].cat.codes
dataset["ssc_b"] = dataset["ssc_b"].cat.codes
dataset["hsc_b"] = dataset["hsc_b"].cat.codes
dataset["degree_t"] = dataset["degree_t"].cat.codes
dataset["workex"] = dataset["workex"].cat.codes
dataset["specialisation"] = dataset["specialisation"].cat.codes
dataset["status"] = dataset["status"].cat.codes
dataset["hsc_s"] = dataset["hsc_s"].cat.codes
dataset
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values
Y
theta = np.random.randn(X.shape[1])
y=Y
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
def loss(theta, X, y):
    h = sigmoid(X.dot(theta))
    return -np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
def gradient_descent(theta, X, y, alpha, num_iterations):
    m = len(y)
    for i in range(num_iterations):
        h = sigmoid(X.dot(theta))
        gradient = X.T.dot(h - y) / m
        theta -= alpha * gradient
    return theta
theta = gradient_descent(theta, X, y, alpha=0.01, num_iterations=1000)
def predict(theta, X):
    h = sigmoid(X.dot(theta))
    y_pred = np.where(h >= 0.5, 1, 0)
    return y_pred
y_pred = predict(theta, X)
accuracy = np.mean(y_pred.flatten() == y)
print("Accuracy:", accuracy)
print(y_pred)
print(Y)
xnew = np.array([[0, 87, 0, 95, 0, 2, 78, 2, 0, 0, 1, 0]])
y_prednew = predict(theta, xnew)
print(y_prednew)
xnew = np.array([[0, 0, 0, 0, 0, 2, 8, 2, 0, 0, 1, 0]])
y_prednew = predict(theta, xnew)
print(y_prednew)
```
## Output:
#### Dataset
<img src="https://github.com/Jenishajustin/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119405070/ad5d1f34-ffdd-41cd-b4ae-674ea384f912">

#### Datatypes of Dataset
![Screenshot 2024-05-05 120843](https://github.com/Jenishajustin/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119405070/1731c847-b835-4b10-99b7-d5c336e4b41d)

#### Labeled Dataset
![Screenshot 2024-05-05 121003](https://github.com/Jenishajustin/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119405070/63885416-005c-4f68-a2eb-caf259cabe53)

#### Y value (dependent variable)
![Screenshot 2024-05-05 121053](https://github.com/Jenishajustin/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119405070/d8c16cfa-c167-4cac-b9dd-b7245c0a5851)

#### Accuracy
![Screenshot 2024-05-05 121135](https://github.com/Jenishajustin/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119405070/348c556c-dfb9-4d11-a098-af3f8e657d05)

#### Predicted Y value
![Screenshot 2024-05-05 121215](https://github.com/Jenishajustin/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119405070/03dd51ee-da98-4b1f-bf10-e530673b898d)

#### Y value
![Screenshot 2024-05-05 121248](https://github.com/Jenishajustin/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119405070/2ec95e50-3792-4181-bbbb-80fff7290fc8)

#### New Y predictions
![Screenshot 2024-05-05 121340](https://github.com/Jenishajustin/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119405070/9a12df40-b93c-4662-bf91-38b78e35acdd)

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.
