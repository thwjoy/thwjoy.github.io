---
layout: post
mathjax: true
title: Normalising Flows and Invertible Neural Networks
date: 2019-03-01 11:12:00-0400
description: Overview of normalising flows and inveritble neural networks.
---
\begin{align}
\newcommand{\x}{\mathbf{x}}
\newcommand{\z}{\mathbf{z}}
\end{align}
Here we are looking at two types of generative models: Normalising Flows (NF) and Invertible Neural Networks (INN). Many blog posts outlining normalising flows already exist, e.g, [here](https://lilianweng.github.io/lil-log/2018/10/13/flow-based-deep-generative-models.html?fbclid=IwAR0B5rOmW88VwfJAMYowU-v9xC1XJN3fXO5WKPya9K3RfhehP4O4l6jXZjc) or [here](http://akosiorek.github.io/ml/2018/04/03/norm_flows.html), but I thought I'd provide yet another one, mainly to help me consolidate what I've learnt, but also to hopefully shed light on some concepts which can be hard to grasp.

So why do we need NFs and INNs? Other flavours of generative models such as VAEs or GANs currently exist, and work pretty well, so why do we need another type of generative model? If we think about what we actually need from a generative model it becomes a little clearer why NFs and INNs are needed. A generative model should (ideally) be able to: offer the ability to generate real looking data-points (samples); produce a model of the true distribution (density estimation); give a score or probability that a sample is from the true distribution (likelihood evaluation). GANs address sampling very well, but they fall short when it comes to likelihood evaluation and density estimation (bar a few examples). This is understandable as GANs are trained adversarially to just produce realistic samples and not do anything else. VAEs on the other can also sample well, but focus on modelling a latent representation of the true distribution. Consequently, there is a gap in the generative for a market which is able to: sample, obtain a density estimate and also perform likelihood estimation.

Formally, what we really want to do, is to obtain a good estimate of a high-dimensional distribution $p(\x) \x \in \mathcal{X}$. Normalising flows address this issue by starting with a simple parameterised distribution $p(\z)$ and successively applying transformations $f_i(\z_i)$ until a good approximation of $p(\x)$ is obtained. Using the change of variables formula $p(y)=p(x)\vert\frac{\partial x}{\partial y}\vert$, we can obtain an expression for the approximate distribution $\tilde{p}(\x)$.

\begin{align}
\log{\tilde{p}(\x)} = \log{p_0(\z_0)} + \sum_{i=0}^{N}\log{\Bigg\vert\det{\frac{\partial f^{-1}_i(\z\_{i+1})}{\partial \z_i}}\Bigg\vert}.
\end{align}

Assuming $f_i$ is invertible, we can chain as many functions together as we like, thus creating a \textit{flow} from the complex $\tilde{p}(\x)$ to the simple $p_0(\z_0)$. This allows us to then obtain a parameterised representation of $p(\x)$ via a transformation of variables, consequently we can now perform: sampling, density estimation and likelihood evaluation.

### Learning

Let's assume that we have access to a dataset $\mathcal{D} = \lbrace\x \sim p(\x)\rbrace^M$, and we want to fit a model $\tilde{p}(\cdot)$ to it which is parameterised by $\theta$. All we do is simply maximise the likelihood of the elements of $\mathcal{D}$. This can also be seen as minimising the negative log-likelihood:

\begin{align}
\min_\theta - \frac{1}{M}\sum_{\x \in \mathcal{D}}\log{\tilde{p}(\x)}.
\end{align}

However, we need a representation for $\tilde{p}(\x)$, we can use normalising flows, but for the minute let's assume that we are simply trying to fit a high dimensional Gaussian to the elements of $\mathcal{D}$. All we would need to do is maximise the likelihood of $\mathcal{D}$ over the mean and standard deviation.
This would then give us a parametrised representation of , allowing us to not only sample, but also perform density estimation and likelihood evaluation. However,it is very unlikely that $p(x)$ will be normally distributed resulting in a very bad approximation and a useless model.

Clearly normalising flows is very applicable here, by successively applying transformations we can obtain a good approximation of $p(\x)$. Using the change of variables formula and assuming we have a latent distribution $q(\z) \z \in \mathcal{Z}$ and that the composition of $f_i \forall i \in \lbrace i,...,N\rbrace$ is represented as $F_\theta : \mathcal{Z} \rightarrow \mathcal{X}$ the fobjective can be written as:

\begin{align}\label{eq:cov}
\min_\theta - \frac{1}{M}\sum_{\x \in \mathcal{D}}\Bigg[\log{q(\z)} + \log{\Bigg\vert\det{\frac{d F^{-1}_\theta(\z)}{d\x}}\Bigg\vert}\Bigg].
\end{align}

So to learn a good representation of $p(\x)$, the only requirement is that we can evaluate the likelihood of $q(\z)$ and evaluate the log determinant of the Jacobian of $F^{-1}_\theta(\z)$ wrt $\x$ efficiently.

 
### Evaluating the log determinant

Unfortunately, evaluating the log determinant has a complexity of $\mathcal{O(d^3)}$, which is a often intractable. To address this, many methods focuss on efficent computation of the log determinant, often at the cost of expressibility or significant compute.

[NICE](https://arxiv.org/pdf/1410.8516.pdf) and [Real-NVP](https://arxiv.org/pdf/1605.08803.pdf) only applied linear transformations to a subset of the dimensions. This yielded a lower triangular Jacobian, where it's log determinant can easily be computed by summing it's diagonal. [GLOW](https://arxiv.org/pdf/1807.03039.pdf), replaced the linear layers with a $1\times1$ convolution. By representing the convolutional layer as a linear operation $\mathbf{y = W^Tx}$, we can easily see that it's inverse can be obtained and so can it's determinant at a moderate computational cost. As $\mathbf{W}$ is relatively small, bruteforcing is a satisfactory option. However $1\times1$ convolutions severely impeded the models expressibility, and still require a heavy amount of computation - restricting the authors to use a batch size of 1.

Separately, much progress has been made on invertible neural networks which are reversible by design. 


