# NonLinear Regression
This repository implements a non-linear regression using zero mean noiseless Gaussian Processes to estimate the function values f(x) = sin(0.5 * x) for x values ranging from -4 to 4. The repository performs below specific operations:

1. It plots 10 prior functions in the interval -4 to 4.

![sample_prior](https://github.com/kanchanchy/NonLinear-Regression/blob/master/plots/prior.png)

2) It plots the mean estimate for the non-linear regression and the error curves above and below indicating confidence in the estimate.

![sample_prior](https://github.com/kanchanchy/NonLinear-Regression/blob/master/plots/mean_error.png)

3) It plots 10 posterior functions sampling the conditional distribution GP(f|D).

![sample_prior](https://github.com/kanchanchy/NonLinear-Regression/blob/master/plots/posterior.png)
