# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

path = 'E:\Programming\Python\MJ\Machine Learning\Andrew Ng\Machine-Learning\Ex 2\machine-learning-ex2\ex2'
os.chdir(path)
data = pd.read_csv('ex2data2.txt', header=None, names=[
                   'Test 1', 'Test 2', 'Accepted'])

data_1_o = data.loc[data['Accepted'] == 1]
data_0 = data.loc[data['Accepted'] == 0]
Y_mat = np.asanyarray(data[['Accepted']])

# Visualize data
plt.figure(figsize=(20, 15))
plt.scatter(data_0['Test 1'], data_0['Test 2'], c='Yellow')
plt.scatter(data_1_o['Test 1'], data_1_o['Test 2'], marker='+', c='Red')
plt.tick_params(color='Black')
plt.xlabel('Microchip Test 1', c='Black')
plt.ylabel('Microchip Test 2', c='Black')
plt.legend(['y = 0', 'y = 1'])
plt.show()

# Map features to polynomial
m = data.shape[0]
data_1 = data['Test 1'].to_numpy().reshape(m, 1)
data_2 = data['Test 2'].to_numpy().reshape(m, 1)
def mapfeatures(degree, x1, x2):
    out = np.ones((m,1))
    for i in range(1, degree+1):
        for j in range(i+1):
            output = (x1 ** (i-j)) * (x2 ** j)
            out = np.hstack((out, output))
    return out

X_map = mapfeatures(6, data_1, data_2)

def Sigmoid(z):
    z = 1/(1 + np.exp(-z))
    return z

# Initialize parameters
theta_size = X_map.shape[1]
initial_theta = np.zeros((theta_size,1))
Lambda = 1


def costFunctionReg(initial_theta,x,y,Lambda,m):
    J = 1/m * ((-y.T @ np.log(Sigmoid(x @ initial_theta))) - ((1 - y).T @ np.log(1 - Sigmoid(x @ initial_theta)))) + Lambda/(2*m) * np.sum(initial_theta[1:] ** 2)
    gradient_0 = 1/m * np.sum((Sigmoid(x @ initial_theta) - y))
    gradient_1 = 1/m * (x.T @ (Sigmoid(x @ initial_theta) - y))[1:] + Lambda/m * initial_theta[1:]
    gradient_tot = np.vstack((gradient_0, gradient_1))
    return J[0][0], gradient_tot

def GradientDescent(x, y, theta, alpha, num_iters, Lambda):
    m = len(y)
    J_hist = []
    for i in range(num_iters):
        cost, grad = costFunctionReg(theta, x, y, Lambda, m)
        # update theta by gradient descent
        theta = (theta - alpha*grad)
        J_hist.append(cost)
    return theta, J_hist

cost, grad = costFunctionReg(initial_theta, X_map, Y_mat, Lambda, m)
print('Cost for initial_theta is:', cost)
print('Gradient for initial_theta is:\n', grad[0:5])

# Compute and display cost and gradient with all-ones theta and lambda = 10
test_theta = np.ones((theta_size, 1))
cost_test, grad_test = costFunctionReg(test_theta, X_map, Y_mat, 10, m)
print('Cost for test_theta is:', cost_test)
print('Gradient for test_theta is:\n', grad_test[0:5])

# I am going to use GradientDescent instead of fminunc function to optimize theta
theta, J_hist = GradientDescent(X_map, Y_mat, initial_theta, 1, 800, 0.2)
print('The regularized theta using ridge regression \n', theta)

# Note that alpha, num_iters and lambda were not given so we try a few combinations and come up with the best
plt.figure(figsize=(15, 10))
plt.plot(J_hist, color = 'Red')
plt.xlabel('No of iterations')
plt.ylabel('Cost Function')
plt.show()

def mapFeaturePlot(x1, x2, degree):
    """
    take in numpy array of x1 and x2, return all polynomial terms up to the given degree
    """
    out = np.ones(1)
    for i in range(1, degree+1):
        for j in range(i+1):
            terms = (x1**(i-j) * x2**j)
            out = np.hstack((out, terms))
    return out

u_values = np.linspace(-1, 1.5, num=50)
v_values = np.linspace(-1, 1.5, num=50)
z = np.zeros((len(u_values), len(v_values)))
# We evaluate z = theta * x over the grid
for i in range(len(u_values)):
    for j in range(len(v_values)):
        z[i, j] = mapFeaturePlot(u_values[i], v_values[j], 6) @ theta

plt.figure(figsize=(20, 15))
plt.scatter(data_0['Test 1'], data_0['Test 2'], c='Yellow')
plt.scatter(data_1_o['Test 1'], data_1_o['Test 2'], marker='+', c='Red')
plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')
plt.legend(['y = 0', 'y = 1'])
plt.contour(u_values, v_values, z.T, 0)
plt.show()

def classifierPredict(theta, X):

    predictions = X @ theta

    return predictions > 0


p = classifierPredict(theta, X_map)
print("Train Accuracy:", (sum(p == Y_mat)[0]/len(Y_mat)) * 100, "%")