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
\newcommand{\R}{\mathbb{R}}
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

Unfortunately, evaluating the log determinant has a complexity of $\mathcal{O(d^3)}$, which is a often intractable. To address this, many methods focuss on efficient computation of the log determinant, often at the cost of expressibility or significant compute. Below we discuss the main types of normalising flow and their corresponding advantages and disadvantages.

### Dimensions Splitting Flows

Dimensions splitting flows aim to exploit the fact that the Jacobian of a lower triangular matrix is just the product of it's diagonal. Enforcing the Jacobian to be lower triangular of a flow leads to the condition that an output dimension $i$ is only dependant on the input dimensions up to that dimension $1:i$.

#### Real-Non Volume Preserving Flows (Real-NVP)

[Real-NVP](https://arxiv.org/pdf/1605.08803.pdf) is the simplest and most intuitive autoregressive flow to understand. They implement a flow layer at index $i$ by splitting the dimensions at $k$ like so:

\begin{align}
\z_\{i+1}^{1:k} &= \z_\{i}^{1:k}\\\ \z_\{i+1}^{k:d} &= \z_\{i}^{k:d} \odot \exp{(\sigma{(\z_\{i}^{1:k})} + \mu{(\z_\{i}^{1:k})})}\label{eq:couple}.
\end{align}

Where $\sigma : \R^{1:k} \rightarrow \R^{k:d}$ and $\mu : \R^{1:k} \rightarrow \R^{k:d}$ are mappings, and $\odot$ is the Hadamard product. The log determinant of the Jacobian can be expressed as:

\begin{align}
\frac{\partial{\z_\{i+1}}}{\partial{\z_\{i}}} = \begin{bmatrix} I & 0 \\\ \frac{\partial{\z_\{i+1}^{k:d}}}{\partial{\z_\{i}^{1:k}}} & \text{diag}\[{\exp{\sigma{(\z_\{i}^{1:k})}}}\] \\ \end{bmatrix},
\end{align}

with it's determinant just being the product of it's diagonal elements. Inverting Real-NVP is easily done by performing elementwise division at the same complexity as the forward pass.

The main issue with Real-NVP is it's limited expressibility, as we can only introduce dependencies between certain pre-defined dimensions.

#### GLOW

[GLOW](https://arxiv.org/pdf/1807.03039.pdf), extended Real-NVP by first applying Actnorm - a variant of BatchNorm - and then applying an invertible 1$\times$1 convolution before finally applying an affine coupling layer similar to Real-NVP.

<p align="center">
<img src="https://thwjoy.github.io/assets/img/glow.png">
</p>

Actnorm works in a similar way to Batchnorm, but is compatible when using a batch size of 1. It applies a shift and scale parameter per channel which are initialised to 0 and 1 during an initial forward pass of the network.

To address the ordering issue of Real-NVP, an invertible 1$\times$1 convolution is introduced. This facilitates swapping of channels throughout the network, ensuring that all channels undergo a coupling transformation. By firstly representing the parameters of 1$\times$1 convolution as $W \in \R^{c \times c}$ where $c$ is the number of channels, we can reparametrise it as it's LU decomposition:

\begin{align}
W = PL(U + \text{diag}(s))
\end{align}

where $s = {\exp{\sigma{(\z_\{i}^{1:k})}}}$ from equation \ref{eq:couple}. Clearly, by keeping the representation as an LU decomposition, we are able to quickly compute the log determinant by summing the log of each $s$.

Finally GLOW performs an affine coupling layer similar to Real-NVP.

### Autoregressive Flows

Autoregressive flows aim to introduce the same feature of a lower triangular jacobian which is present in dimension splitting flows. However, rather than splitting the dimensions and introducing coupling, autoregressive flows enforce each dimension to be conditional on it's previous dimensions and not it's future ones. With this autoregressive assumption, the density of a datapoint is given as:

\begin{align}
p(\x) = \prod^{D}_{i=1}p(x_i|\x\_{<i})
\end{align}

If we consider a single conditional to be parameterised by a Gaussian, where the parameters are computing using two functions $f_{\mu_i}$ and $f_{\sigma_i}$:

\begin{align}
p(x_i|\x\_{<i}) &= \mathcal{N}(x_i | \mu_i, \sigma_i), \\\ \mu_i &= f_{\mu_i}(\x\_{<i}), \\\ \sigma_i &= \exp({f_{\sigma_i}(\x\_{<i})})^2.
\end{align}

Where a sampling can achieved by recursively computing the following:

\begin{align}\label{eq:sample}
x_i &= z_i\exp(\sigma_i) + \mu_i,
\end{align}
were $z_i \sim \mathcal{N}(0, 1)$, and it's inverse is given by:
\begin{align}\label{eq:auto}
z_i &= (x_i - \mu_i)\exp(-\sigma_i).
\end{align}
Hence, by taking the determinant of the Jacobian of \ref{eq:auto} to be $\prod_{i=1}^{D}\exp(-\sigma_i)$, we are able to construct a normalising flow:
\begin{align}\label{eq:flow_auto}
\log{p(\x)} = \log{\mathcal{N}(\z_0)} - \sum_{i=1}^{D}\sigma_i.
\end{align}
Where we can either sample from $p(\x)$ using equation \ref{eq:sample}, or compute the density of a sample $\x$ using equation \ref{eq:flow_auto}. However, naively doing so would be very slow, as for each method we have to recursively compute $\mu_i$ and $\sigma_i \forall i \in \\{1,...,D\\}$.






#### Masked Autoencoder for Distribution Estimation

[MADE](https://arxiv.org/abs/1502.03509) enforces the autoregressive assumption in an autoencoder, 


#### Masked Autoregressive Flow


#### Inverse Autoregressive Flow


