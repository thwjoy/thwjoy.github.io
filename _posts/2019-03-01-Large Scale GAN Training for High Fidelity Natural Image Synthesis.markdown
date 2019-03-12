---
layout: post
title: Large Scale GAN Training for High Fidelity Natural Image Synthesis
date: 2019-03-01 11:12:00-0400
description: Overview of the peper 'Large Scale GAN Training for High Fidelity Natural Image Synthesis' by Brock et al.
---

The ability of GANs to scale to high dimensions has remained a notorious hurdle in the compute vision community.
As such, training a GAN on ImageNet has often remained a single goal in itself, where until recently, the state of the art was initially set by [Zhang et al.](https://arxiv.org/pdf/1805.08318.pdf).
The seminal work by [Brock et al.](https://arxiv.org/pdf/1809.11096.pdf), showed that the state of the art can be significantly increased by making some small modifications to the architecture and to the training regime.
They also provide informative insights into the stability of GANs at large scale.
This blog post will highlight the key contributions by [Brock et al.](https://arxiv.org/pdf/1809.11096.pdf) and highlight the key findings in their extensive evaluation.


#### SA-GAN

The architecture used seems to follow a throw-the-book at it approach, where almost all components known to improve GAN performance have been used, and it certainly has worked well.
The baseline architecture used in their work was SA-GAN by [Zhang et al.](https://arxiv.org/pdf/1805.08318.pdf), which uses feature cues from long range connections.
Thus generating images which contain consisting features across all parts of the image and a high level structural representation.
This achieved using a self-attention mechanism, where spatial regions from across the image are added to local regions, resulting in information which contains high level context, but also local detail for each feature point.
[Brock et al.](https://arxiv.org/pdf/1809.11096.pdf) introduced class-conditional BatchNorm to the generator and projection to the discriminator, which projects the output of a discriminator onto the embedding of the given class.
They also make used of Orthogonal Initialisation [Saxe et al.](link), which is known to have important properties in GANs [Link](link).
Perhaps more importantly to the success was the hardware used for training, in some models they made use of 512 cores of a Google TPUv3 Pod, and enabled BatchNorm across all devices, which intuitively should improve generalisation and stability in the training. 


#### Batch Size

The first modification they made was to increase the batch size used in training, which gave a significant improvement in performance, this should be no surprise as a larger batch will cover a wider portion of the training distribution. However, whilst this is a very simple modification to make, it relies on the use of sophisticated and powerful hardware to do the training - whilst everyone can access Google TPUs, it comes at significant expense, which few universities and research centers may be willing to fund.

#### Increasing Parameters

They also experimented with increasing the dimensions of the architecture, they increased the number of filters by 50%, which gave a slight increase of 21%. They also found that increasing the height of the model didn't improve performance, one may argue that this is due to the self attention mechanism of SA-GAN already modelling the long range connections, therefore a wider receptive field would be redundant.

#### Sharing Parameters in Conditional BatchNorm

The conditional BatchNorm uses a large amount of parameters as each layer requires it's own set embedding. 
To save memory and computation, they used a shared embedding and projected this onto each layer in the network.
The introduction of this shared embedding did provide some improvement in the training and a slight decrease in FID and a slight increase in IS, however the model was trained for far fewer iterations, indicating that this lead to early failure in the training.

#### Skip-z Connections

In an effort to try and utilise the latent space at all levels in the network, they introduced skip-z connections.
These connections first perform concatenation with either a portion or all of the latent vector and the class conditional vector.
The concatenated vector is then projected onto the BatchNorm weights.
Skip-z is only reported to provide a small improvement in performance, suggesting that exposure to the latent representation at each layer is not fundamental to success of a GAN.

#### Truncation Trick

Aside from extensive insights into the performance and stability of GANs, this work also introduced the Truncation Trick, which is simply the re-sampling of the latent space if the probability is below a certain threshold.
This truncation showed improved results in terms of IS and FID scores, and intuitively this makes sense, as latent spaces with a larger probability are more likely to be sampled and hence undergo more training in that region.
However, a smaller truncated region leads to better samples but with lower variety.
They also investigated the use of other latent space distributions, but found a truncated normal to give the best performance.

However, as noted in the paper, larger models tend not to be amenable to truncation, possibly due to full utilisation of the latent space. 
Conditioning the generator to be smooth is one possible solution to this problem, and in the paper they experiment with Orthogonal Regularisation, which encourages the filters to be orthogonal to one another.
They experimented with different regularisers (although it's not clear which ones) and found the following to work the best:

$$
\mathcal{R} = ||W^TW \odot (1 - I)||^2_F
$$

This has the primary advantage of encouraging orthogonality in the filters, but unlike previous [works](link), does not constrain the norm to be 1, which is at odds to previous research directions which aim to encourage the norms of the weights to be 1.
It is also interesting that 60% of the models are amenable to truncation with Orthogonal Regularisation, suggesting that this regularisation is not completely effective for the truncation trick.

### Analysis

The analysis in this paper is excellent, and leads to some invaluable insights into the training of GANs at large scale.

#### Stability in the Generator

They report that the most informative metrics for pre-empting training collapse is to observe the three largest singular values $$\sigma_0, \sigma_1, \sigma_2$$ of the early layers in the network. They propose a solution to constrain the singular values to lie within a given range:

$$
W = W - \text{max}(0, \sigma_0 - \sigma_{clamp})v_0u_0^T,
$$
Where $$v_0$$ and $$u_0$$ are the left and right singular vectors respectively and $$\sigma_{clamp}$$ is set emperically. They observed that whilst constraining the singular values helped training, it wasn't sufficient to mitigate training collapse.
One point that eludes me is that when Spectral Normalisation was employed in the generator, the authors still claim that the above technique helped constrain the singular values, suggesting that the Spectral Normalisation in this setting was ineffective. 

#### Stability in the Discriminator

Unlike the generator, the singular values of the discriminator weights are noisy, the authors posit that this is to due to drastic periodic changes in %%p_g(x)$$, which in turn causes large gradients to the discriminator.
To mitigate this, they introduced a zero centered gradient penalty:

$$
\mathcal{R} = E_{x ~ p(x)}[(||\nabla_x D(x)||_F^2],
$$ 

which stabilised training but severely degrades performance. 


A further investigation would be to investigate if this noise is still present when training the discriminator with only real samples.
Another interesting result is that the later layers of the discriminator contain larger singular values, than the early layers.