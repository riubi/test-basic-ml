import numpy as np
import pandas as pd


class LinearRegression:
    
    def __init__(self, learning_rate, iterations):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None
        
    def fit(self, X, Y):
        self.X = X
        self.Y = Y
        self.n_samples, self.n_features = X.shape
        
        self.weights = np.zeros(self.n_features)
        self.bias = 0
        
        for i in range(self.iterations):
            self.update_weights()
            
        return self
    
    def update_weights(self):
        Y_pred = np.dot(self.X, self.weights) + self.bias
        
        dw = (2 / self.n_samples) * np.dot(self.X.T, (Y_pred - self.Y))
        db = (2 / self.n_samples) * np.sum(Y_pred - self.Y)
        
        self.weights = self.weights - self.learning_rate * dw
        self.bias = self.bias - self.learning_rate * db
        
        return self
    
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias


df = pd.read_csv('salary_data.csv')

X = df.iloc[:, :-1].values
Y = df.iloc[:, 1].values

model = LinearRegression(iterations=1000, learning_rate=0.01)
model.fit(X, Y)

Y_pred = model.predict(X)

print(*np.round(model.weights, 2))
print(np.round(model.bias, 2))
