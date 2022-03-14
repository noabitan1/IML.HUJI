from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import plotly.io as pio
pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    c = UnivariateGaussian()
    true_expectation = 10
    true_var = 1
    sample = np.random.normal(true_expectation, true_var, size=1000)
    c.fit(sample)

    # Question 2 - Empirically showing sample mean is consistent
    sample_size_array = np.arange(10, 1010, 10)
    absolute_expectation_distances = []
    for sample_size in sample_size_array:
        selected_sample = sample[0:sample_size+1]
        c.fit(selected_sample)
        absolute_expectation_distances.append(abs(c.mu_-true_expectation))

    plt.scatter(sample_size_array, absolute_expectation_distances)
    plt.yticks(np.arange(0, max(absolute_expectation_distances) + 0.1, 0.1))
    plt.ylim(0, max(absolute_expectation_distances)+0.1)
    plt.title("Absolute Distance From Expectation Per Sample Size")
    plt.xlabel("Sample Size")
    plt.ylabel("Absolute Distance")
    plt.show()

    # Question 3 - Plotting Empirical PDF of fitted model
    plt.scatter(sample, c.pdf(sample))
    plt.title("Samples PDF")
    plt.xlabel("Sample Value")
    plt.ylabel("PDF")
    plt.show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    raise NotImplementedError()

    # Question 5 - Likelihood evaluation
    raise NotImplementedError()

    # Question 6 - Maximum likelihood
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()

