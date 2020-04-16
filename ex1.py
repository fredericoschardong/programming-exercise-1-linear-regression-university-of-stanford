import numpy as np
np.set_printoptions(suppress=True)
import matplotlib.pyplot as plt

def gradient_descent(X, y, theta, iterations, alpha):
	m = X.shape[0]
	
	def compute_cost(X, y, theta):
		#easy way
		#error = 0
		
		#for i in range(m):
		#	hypothesis = X[i][0] * theta[0] + X[i][1] * theta[1]
		#	error += (hypothesis - y[i])**2

		#linear algebra way
		hypothesis = np.dot(X, theta) #dot product
		error = np.sum((hypothesis - y) ** 2)

		return error / (2 * m)
	
	cost_history = []
	
	for i in range(iterations):
		hypothesis = np.matmul(X, theta)
		
		for j in range(theta.shape[0]):
			theta[j] = theta[j] - alpha * np.sum((hypothesis - y) * X[:, j]) / m
			
		cost_history.append(compute_cost(X, y, theta))
		
	return theta, cost_history
	
def plot_cost_history(cost_history):
	plt.clf()
	plt.title('Cost J x iterations')
	
	plt.ylabel('Cost J')
	plt.xlabel('Number of iterations')

	plt.plot(cost_history, 'g.')
	
	plt.show()
	
if __name__ == '__main__':
	# Initialization
	## Import data
	data = np.loadtxt("ex1data1.txt", delimiter=",")

	## Initialize important variables
	m = data.shape[0]
	alpha = 0.01
	
	for iterations in [100, 1500, 3000]:
		theta = np.zeros(2)

		# Evaluation
		## Create the bias column, set to 1
		X = np.stack((np.ones(m), data[:, 0]), axis=-1)
		  
		## Second column of data
		y = data[:, 1]

		theta, cost_history = gradient_descent(X, y, theta, iterations, alpha)
		
		plt.plot(data[:, 0], np.matmul(X,theta), '-', label='linear regression using %d iterations' % iterations)

		print('Theta found by gradient descent', theta);

		# Predictions
		print('For population = 35,000, we predict a profit of', np.matmul(np.array([1, 3.5]), theta) * 10000)
		print('For population = 70,000, we predict a profit of', np.matmul(np.array([1, 7]), theta) * 10000)
		print()
		
	#plot_cost_history(cost_history)
		
	# Show Results
	plt.ylabel('Profit in $10,000s')
	plt.xlabel('Population of City in $10,000s')

	plt.plot(data[:, 0], data[:, 1], 'rx', label='Training data')
	
	plt.legend()
	plt.show()
