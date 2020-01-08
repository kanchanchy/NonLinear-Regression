import numpy as np
from scipy import spatial, linalg
from matplotlib import pyplot as plt

def kernel_function(a, b):
	sq_norm = -0.5 * spatial.distance.cdist(a, b, 'sqeuclidean')
	return np.exp(sq_norm)

def f_sign(x):
	return np.sin(0.5*x)

def GP(x1, y1, x2):
	cov11 = kernel_function(x1, x1)
	cov12 = kernel_function(x1, x2)
	solved = linalg.solve(cov11, cov12, assume_a='pos').T
	mu2 = solved @ y1
	cov22 = kernel_function(x2, x2)
	cov2 = cov22 - (solved @ cov12)
	return mu2, cov2


def plotPrior():
	x_sample = np.linspace(-4, 4, 50)
	x_matrix = np.expand_dims(x_sample, 1)
	covariance = kernel_function(x_matrix, x_matrix)
	y_samples = np.random.multivariate_normal(mean=np.zeros(50), cov=covariance, size=10)
	color_list = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
	for i in range(10):
		plt.plot(x_sample, y_samples[i], lw=1, color = color_list[i])
	plt.xlim(-4, 4)
	plt.ylim(-5, 5)
	plt.title("Sample Prior Functions", fontsize=12)
	plt.show()

def plotPosterior():
	x1_sample = [-3.8, -3.2, -3, 1, 3]
	x1 = np.expand_dims(x1_sample, 1)
	y1 = [-0.9463, -0.9996, -0.9975, 0.4794, 0.9975]
	x2_sample = np.linspace(-4, 4, 50)
	x2 = x2_sample.reshape(-1,1)

	mu2, cov2 = GP(x1, y1, x2)
	sigma2 = np.sqrt(np.diag(cov2))
	y2 = np.random.multivariate_normal(mean=mu2, cov=cov2, size=10)

	color_list = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
	for i in range(10):
		plt.plot(x2_sample, y2[i], lw=1, color = color_list[i])
	plt.title("Sample Posterior Functions", fontsize=12)
	plt.show()

	return x2_sample, mu2, sigma2


def plotMeanAndError(x2_sample, mu2, sigma2):
	x2_sine = f_sign(x2_sample)
	plt.plot(x2_sample, mu2, lw=1, color = "#FF0000", label='Mean Estimate')
	plt.plot(x2_sample, mu2 + sigma2, lw=1, color = "#13184B", label='Error Curve Above')
	plt.plot(x2_sample, mu2 - sigma2, lw=1, color = "#000000", label='Error Curve Below')
	plt.plot(x2_sample, x2_sine, lw=1, color = "#008000", label='Sign Function')
	plt.xlim(-4, 4)
	plt.ylim(-5, 5)
	plt.title("Mean and Errors", fontsize=12)
	plt.legend(loc='best', prop={'size': 12})
	plt.show()


if __name__ == "__main__":

	plotPrior()

	x2_sample, mu2, sigma2 = plotPosterior()

	plotMeanAndError(x2_sample, mu2, sigma2)


	