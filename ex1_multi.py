import numpy as np
np.set_printoptions(suppress=True)
import matplotlib.pyplot as plt

from ex1 import gradient_descent, plot_cost_history
	
# Second data to analyse, this one has three dimensions -> linear regression with multiple variables

## Import data
data = np.loadtxt("Ex1/ex1data2.txt", delimiter=",")

X = data[:, 0:2]
y = data[:, 2]

### Data normalization with bias column, set to 1
m = data.shape[0]

mu_0 = np.mean(X[:, 0])
mu_1 = np.mean(X[:, 1])

sigma_0 = np.std(X[:, 0])
sigma_1 = np.std(X[:, 1])

X_norm = np.stack((np.ones(m),
					(X[:, 0] - mu_0) / sigma_0, 
					(X[:, 1] - mu_1) / sigma_1), 
					axis=-1)

## Important variables
theta = np.zeros(3)

iterations = 400
alpha = 0.01

# Evaluation
theta, cost_history = gradient_descent(X_norm, y, theta, iterations, alpha)

plot_cost_history(cost_history)

print('Theta found by gradient descent', theta);

# Predictions
print('For Estimate the price of a 1650 sq-ft, 3 br house', np.matmul(np.array([1, (1650-mu_0)/sigma_0, (3-mu_1)/sigma_1]), theta))

#empty line for the sake of readability
print()




# Now let's rework the previous example using closed-form solution to linear regresion (normal equations) instead of error minimisation

## Import data
data = np.loadtxt("Ex1/ex1data2.txt", delimiter=",")

X = data[:, 0:2]
y = data[:, 2]

### Data with bias column, set to 1
m = data.shape[0]

X = np.stack((np.ones(m), X[:, 0], X[:, 1]), axis=-1)

## Important variables
theta = np.zeros(3)

def closed_form(X, y):
	X_T = np.matrix.transpose(X)
	
	step1 = np.matmul(X_T, X)
	step2 = np.linalg.inv(step1)
	step3 = np.matmul(step2, X_T)
	
	return np.matmul(step3, y)

# Evaluation
theta = closed_form(X, y)

print('Theta found by normal equations', theta);

# Predictions
print('For Estimate the price of a 1650 sq-ft, 3 br house', np.matmul(np.array([1, 1650, 3]), theta))
