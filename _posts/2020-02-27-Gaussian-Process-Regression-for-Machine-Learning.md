---
layout: post
title: Machine Learning 4 - Gaussian Process Regression
tags: [Gaussian process, regression, machine learning, Bayesian inference]
comments: true
feature: https://i.imgur.com/Ds6S7lJ.png
lang: en
ref: ml4-gpr
---

Emre: How can a Gaussian process define a distribution over *infinitely many* functions, and why is that useful?

Kaan: Let’s unpack that. The classic reference is Rasmussen & Williams’ *Gaussian Processes for Machine Learning* (MIT Press, 2006). I’ll summarise the essentials here.

## Motivation

Recall ordinary linear regression. We wrote

$$
y = f(x) + \varepsilon, \qquad \varepsilon\sim\mathcal N(0,\sigma^2)
$$

and in the Bayesian version we placed a prior over the parameters $(\theta_0,\theta_1)$ of the line. Gaussian Process Regression (GPR) takes a non-parametric leap: instead of placing a prior on *parameters*, we place a prior directly on the *space of functions* $f$. Effectively, **every point on the function has its own (correlated) random variable** — infinitely many of them.

Why? Because in real problems the true function is rarely a simple line or low-degree polynomial. Think of predicting a person’s preference score for cities they may or may not know. The underlying mapping is messy and high-dimensional. GPR lets us model such relationships *while quantifying uncertainty*.

## Gaussian Process Prior

A **Gaussian process (GP)** is a collection of random variables such that *any finite subset* of them has a joint multivariate normal distribution. We write

$$
f(x) \sim \mathcal{GP}(m(x), k(x,x'))
$$

where $m(x)$ is the mean function (often 0) and $k$ is the covariance, or *kernel*, that encodes how outputs at two inputs $x$ and $x'$ co-vary.

Intuitively:

* If $k(x,x')$ is large when $x$ and $x'$ are close, the function will be **smooth**.
* If $k$ decays slowly, the function varies slowly; if it decays quickly, the function can wiggle more.

Popular kernels include the squared-exponential (RBF) and Matérn families.

### Example Prior Samples

```python
import numpy as np, matplotlib.pyplot as plt
from sklearn.gaussian_process.kernels import RBF

kernel = 1.0 * RBF(length_scale=1.0)
Xs = np.linspace(-5,5,100)[:,None]
K   = kernel(Xs)
Y   = np.random.multivariate_normal(mean=np.zeros(len(Xs)), cov=K, size=3)
for y in Y:
    plt.plot(Xs, y)
plt.title('Three samples from a GP prior')
plt.show()
```

## Conditioning on Data (Posterior)

Given observations $\mathcal D = \{(x_i, y_i)\}_{i=1}^N$, with noise variance $\sigma_n^2$, define

$$
K = K_{NN} + \sigma_n^2 I,\quad
k_* = k(X, x_*),\quad
k_{**} = k(x_*, x_*)
$$

The posterior predictive distribution at a new input $x_*$ is Gaussian:

$$
\begin{aligned}
\mu_* &= k_*^\top K^{-1} y,\\
\sigma_*^2 &= k_{**} - k_*^\top K^{-1} k_*.
\end{aligned}
$$

Thus GPR gives *both* a mean prediction and an uncertainty band — invaluable in risk-sensitive applications.

### Demo in Python

```python
import numpy as np, matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

np.random.seed(42)
X_train = np.random.uniform(-4,4,12)[:,None]
y_train = np.sin(X_train).ravel() + 0.2*np.random.randn(len(X_train))

kernel = 1.0*RBF(1.0) + WhiteKernel(0.2)
gp = GaussianProcessRegressor(kernel=kernel).fit(X_train, y_train)

X_test = np.linspace(-5,5,200)[:,None]
mu, std = gp.predict(X_test, return_std=True)

plt.plot(X_train, y_train, 'kx', label='data')
plt.plot(X_test, mu, 'b', label='mean')
plt.fill_between(X_test.ravel(), mu-2*std, mu+2*std, color='b', alpha=0.2, label='±2σ')
plt.legend(); plt.show()
```

![GPR demo](../images/gpr_demo.png)

## Kernel Engineering

The kernel is where we inject prior beliefs: periodicity, non-stationarity, separate length-scales per dimension, etc. Kernels can be **added** or **multiplied** to build rich priors. Hyper-parameters (e.g.
length-scale) are typically optimised by maximising the marginal likelihood

$$
\log p(y\mid X) = -\tfrac12 y^\top K^{-1} y - \tfrac12 \log|K| - \tfrac{N}{2}\log 2\pi.
$$

## Advantages & Limitations

Pros:

* Outputs full predictive distributions, not point estimates.
* Works with small datasets — uncertainty prevents overfitting.
* Flexible via kernels; automatically adapts model complexity.

Cons:

* $\mathcal O(N^3)$ training and $\mathcal O(N^2)$ storage — prohibitive beyond a few thousand points. Sparse or inducing-point approximations mitigate this.
* Choice of kernel crucial; poor priors yield poor performance.

## References

1. Rasmussen, C. E., & Williams, C. K. I. *Gaussian Processes for Machine Learning*. MIT Press, 2006.
2. Bishop, C. M. *Pattern Recognition and Machine Learning*. Springer, 2006 (Chapter 6).
3. Murphy, K. P. *Machine Learning: A Probabilistic Perspective*. MIT Press, 2012 (Section 16.2).
