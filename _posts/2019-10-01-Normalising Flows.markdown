---
layout: post
mathjax: true
title: Another Blog on Normalising Flows
date: 2019-03-01 11:12:00-0400
description: Overview of normalising flows.
---
\begin{align}
\newcommand{\x}{\mathbf{x}}
\newcommand{\z}{\mathbf{z}}
\end{align}
Here we are looking into another type of generative model called normalising flows. Many blog posts outlining normalising flows already exist, e.g, [here](https://lilianweng.github.io/lil-log/2018/10/13/flow-based-deep-generative-models.html?fbclid=IwAR0B5rOmW88VwfJAMYowU-v9xC1XJN3fXO5WKPya9K3RfhehP4O4l6jXZjc) or [here](http://akosiorek.github.io/ml/2018/04/03/norm_flows.html), but I thought I'd provide yet another one, mainly to help me consolidate what I've learnt, but also to hopefully shed light on some concepts which can be hard to grasp.

Firstly, it will be beneficial to start with some background information. For a function $F_\theta : \mathcal{X} \rightarrow \mathcal{Z}$, which maps a data sample $\x \in \mathcal{X}$ to a latent sample $z \in \mathcal{Z}$, we can obtain a relationship between the two using the change of variables formula: 

\begin{align}\label{eq:cov}
\log{p(\x)} &= \log{q(\z)} + \log{|\det{J_\x}|}.
\end{align}

Where $J_\x$ is the Jacobian of $\z$ wrt $\x$. Now, let's assume that $p(x)$ is an unknown, complex and high dimensional distribution, but we have access to a set of samples from it, let's call this dataset $\mathcal{D}$. It would be desirable if we could not only sample from $p(x)$ (as in GANs and VAES) but also perform density estimation. Doing so is not an easy task, due to the high dimensionality of the problem.

Let's assume for a minute that we are simply trying to fit a high dimensional Gaussian to the elements of $\mathcal{D}$, all we would need to do is maximise the likelihood of $\mathcal{D}$ over the mean and standard deviation. Thus would then give us a parametrised representation of the dataset $\mathcal{X}$, allowing us to sample, perform density estimation and other tasks. However, it is very unlikely that $\mathcal{X}$ will be normally distributed resulting in a very bad approximation and a useless model.

To model $\mathcal{X}$ we need to allow greater flexibility and expressiveness, something neural networks are brilliant at. Let's assume that $F_\theta$ is a neural network parametrised by $\theta$, and that $F$ is invertible, i.e. we can go from $\z$ to $\x$. Now, using the change of variables formula, we simply need to maximise $\log{q(\z)} + \log{\|\det{J_\x}\|}$ over the network parameters. This then allows us to explicitly maximise the likelihood of $p(\x)$, unlike GANs which use an adversarial approach and VAEs which maximise the evidence lowe-bound. Hence the cost function we'd use for a nerual network would be:

\begin{align}
\mathcal{L} = - \frac{1}{|\mathcal{D}|}\sum_{\x \in \mathcal{D}}\log{q(F(\z))} + \log{|\det{J_\x}|}.
\end{align}

