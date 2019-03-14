---
layout: post
title: Regularising GANs
date: 2019-03-01 11:12:00-0400
description: A brief overview of regularising GANs.
---

**Work in progress**

It's a well known fact that [GANs](https://arxiv.org/pdf/1406.2661.pdf) are notoriously hard to train. 
This is no surprise given the objectives of learning an approximate high-dimensional probability distribution $$p_g(x)$$, of some underlying data distribution $$q_{data}(x)$$.
This is further compounded by the fact that the training regime amounts to finding the Nash equilibrium of continuous, non-convex, high-dimensional games.
As such, complete failure in training GANs is common, resulting in phenomena such as: mode-collapse; memorization in the discriminator; exploding and vanishing gradients.
There is now a plethora of work addressing the pitfalls of GANs, but in this blog post we will be studying the stability of training through regularisation in the discriminator.

Before discussing how we regularise the discriminator, let's first discuss why we want to do so.
By quicky glancing at an optimal discriminator, we can see that it will possibly suffer from vanishing and exploding gradients, leading to instability in training:


$$
\hspace9ex D^*(x)  = \frac{q_{data}(x)}{q_{data}(x) + p_g(x)} = \sigma(\log(q_{data}(x)) - \log(p_g(x))).
$$

Clearly, when the distributions are far apart there will be minimal gradients, however, and more importantly, the gradients reach a maximum when the distributions are similar.
Unfrortunatley, this can lead to poor initial training and also to exploding gradients given large diffences between $$q_{data}(x)$$ and $$p_g(x)$$, which should be avoided at all costs.
To avoid the latter, the research community has become motivated to apply some regularity condition in an attempt to bound the output of the disriminator.




#### Control of Lipshitz





Controlling the Lipshitz constant has proven to be one of the more successful methods for regularising the discriminator.
In this setting, the goal is to obtain a discriminator from a set of $$K$$-Lipshitz continuous functions such that:

$$
\hspace9ex ||f(x) - f(x')|| \leq K||x - x'|| \thickspace \forall \thickspace x, \space x'.
$$





For a 1D case, this results in the gradient of the function $$f(x)$$, having a maximum value less than $$K$$.
This is a desirable property for the discriminator, where we would like to place a constraint on the maximum value for the gradient.
[Arjovsky et al.](link) successfully applied this idea to GANs resulting in the Wassertein GAN (WGAN), where they replaced the discriminator with a critic, which returned a score instead of returning a probability that the sample was from either $$q_{data}(x)$$ or $$p_g(x)$$.
To ensure that the critic was bounded by a 1-Lipschitz function, the weights are simply clipped to lie within a certain range.
Whilst clipping the weights is a naive way of ensuring Lipschitz continuity, it motivated the benefit of constraining the discriminator to be Lipshitz continuous.





#### Gradient Penalties

Whilst WGAN showed us Lipshitz continous functions lead to imporve stability in training, the use of weight clipping can lead to undersibale behaviour.
This issue was addressed by the addition of a [Gradient Penalty](link), where there is the addition of a parabolic regulariser on the gradient of the discriminator.

$$
\mathcal{R} = E_{x ~ p(x)}[(||\nabla_x D(x)||_2 - 1)^2]
$$




#### Weight normalisation


Whilst, GP perfomed well, simply encouraging the norm of the gradient to be close to 1 only enables us to encourage a 1-Lipshitz discriminator for regions where we have data samples.
An import caveat of this is that it fails to achieve a well generalized 1-Lipschitz function, In other words, we cannot attain stability outside of the support of $$q_{data}(x)$$ and $$p_g(x)$$.
Normalising the weights in the discriminator network is one way to solve this, showing promising results.
Whilst weight normalisation has been around for some time, the first attempt to explicitly normalise the weights of the discriminator were undertaken by [Brock et al.](link).
Similaryl to GP, they added a regulariser which encouraged the weights of each filter to be orthonormal to one another:



$$
\mathcal{R} = ||W^TW - \bold{1}||
$$



This is a desirable property in the discriminator as the norm of the weights will remain, hence ensuring a Lipschitz constant of 1.
Secondly, the orthogonal nature of the filters has been shown to help training in neural networks [Saxe et al.](link).
TODO Talk more about orthogonality, why is it useful?




The success of orthornormal regularisation drew significant attention, which naturally led to a critial investigation into whether more effective methods exist.
Spectral Normalisation was a result of such a research drive, where instead of constraining the weights to have a norm of 1, they normalised the weights by their largest singular value.
This is advantageous as it enables a tighter upper bound on the Lipschitz constant, which leads to more informative gradients in the training.



#### Closing remarks

Unfortunately, relying on regularisers for conditioning a neural network places a heavy dependance on the strength of the regularisation parameter, often leaving researchers with no other choice than to do extensive cross-validation, which is undesirable.


Clearly, optimising the generator relies firstly, on the descriminator being able to successfully distinguish between a real and fake sample, but also, and perhaps more importantly on obtaining meaningful gradients. 
%
A simple plot of the output of a discriminator leads us to the conclusion that often the discriminator provides either a vanishing or an exploding gradient.
%
To combat this, several works (cite, cite) propose to replace the probability output of the discriminator with a scalar score value.
%
Naturally, this lead on to the development of methods which replace the $[0,1]$ score of the discriminator with a scalar value, the most influential of these works was the so called Wassertein GAN (cite).
%
In their work, they propose to constrain the discriminator such that it is Lipshitz continuous. 
%
This directly addresses the exploding and vanishing gradients of the original GAN.


Naturally, constraining the Lipshitz constant can be achieved in serveral ways.
%
Initially, this involved the so called gradient penalty, but also (I'm sure there was another method)
%
However, spectral normalization addressed some serious flaws in the obove procedures.

