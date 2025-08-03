---
layout: post
title: Machine Learning 2 – Essential Algorithms Every ML Practitioner Should Know
tags: [machine learning, algorithms, cheat sheet]
comments: true
feature: https://i.imgur.com/Ds6S7lJ.png
lang: en
ref: ml2-index
---

Emre: Can you give a concise overview of machine learning and highlight which algorithms are used for which types of problems?

Kaan: Absolutely.

Machine learning is usually grouped into three major categories:

1. **Unsupervised Learning**  
2. **Supervised Learning**  
3. **Reinforcement (Semi-supervised) Learning**

There is a vast literature on each, so I’ll keep things brief.  
To reach true *fluency* in ML you should be comfortable thinking about the following core topics.  Knowing where a real-world problem sits in this landscape clarifies which solution space to explore.  Future blog posts will dive into these areas one by one and steadily build a fully Turkish–English knowledge base.

---

## Fundamentals

* Shannon’s **Source Coding Theorem** – data compression limits  
* **Bayesian statistics & learning** – <https://tayyarg.github.io/dogrusal_regresyon_probleminin_bayesci_cikarimla_cozulmesi/>  
* **Cox axioms** – foundation of Bayesian probability  
* **Bayesian model comparison**  
* Information theory – **entropy estimation** ([Elements of Information Theory](http://www.cs-114.org/wp-content/uploads/2015/01/Elements_of_Information_Theory_Elements.pdf))  
* Textbook recommendation – Murphy, *Machine Learning: A Probabilistic Perspective*

## Models

* **Hidden Markov Models (HMMs)**  
* **State-Space Models (SSMs)**  
* **Boltzmann Machines**  
* **Graphical Models** – directed, undirected, factor graphs

## Algorithms

* **Expectation–Maximisation (EM)**  
* **Belief Propagation**  
* **Forward–Backward**  
* **Kalman Filtering & Extended Kalman Filtering** – <https://tayyarg.github.io/kalman-filtreleme/>  
* **Variational Methods**  
* **Laplace Approximation & BIC**  
* **Markov Chain Monte Carlo (MCMC)**  
* **Particle Filtering**  
* **Expectation Propagation**

## Unsupervised Learning

* **Factor Analysis / Dimensionality Reduction** – Principal Component Analysis (PCA)  
* **Independent Component Analysis (ICA)**  
* **Clustering with Mixture Models** – *k-means*, GMMs  
* **Singular Value Decomposition (SVD)**

## Supervised Learning

* **Linear Regression** – <https://tayyarg.github.io/dogrusal_regresyon_probleminin_bayesci_cikarimla_cozulmesi/>  
* **Gaussian Process Regression** – <http://www.gaussianprocess.org/gpml/>  
* **Logistic Regression**  
* **Decision Trees**  
* **Random Forest**  
* **Ensemble Methods**  
* **Naïve Bayes Classification**  
* **Single-Layer Perceptron**  
* **Neural Networks (Multi-layer Perceptrons) & Back-propagation**  
* **Support Vector Machines (SVMs)**

## Reinforcement Learning

* **Value Functions**  
* **Bellman Equation**  
* **Value Iteration**  
* **Policy Iteration**  
* **Q-Learning**  
* **Actor–Critic Algorithm**  
* **Temporal-Difference Learning – TD(λ)**

## Basic Learning Theory

* **Vapnik–Chervonenkis (VC) Dimension**  
* **Regularisation**

---

Plenty of excellent textbooks cover these topics in depth.  I’ve linked the ones I know best from university courses or personal study.
