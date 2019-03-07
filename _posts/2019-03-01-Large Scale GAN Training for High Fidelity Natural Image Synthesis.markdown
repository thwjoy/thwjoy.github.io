---
layout: post
title: Large Scale GAN Training for High Fidelity Natural Image Synthesis
date: 2019-03-01 11:12:00-0400
description: Overview of the peper 'Large Scale GAN Training for High Fidelity Natural Image Synthesis' by Brock et al.
---

The ability of GANs to scale to high dimensions has remained a notorious problem in the compute vision community.
As such, training a GAN on ImageNet has often remained a single goal in itself, where until recently, the state of the art was initially set by [Zhang et al.](https://arxiv.org/pdf/1805.08318.pdf).
The seminal work by [Brock et al.](https://arxiv.org/pdf/1809.11096.pdf), showed that the state of the art can be significantly increased by making some small modifications to the architecture and to the training regime.
They also provide informative insights into the stability of GANs at large scale.
This blog post will highlight the key contributions by [Brock et al.](https://arxiv.org/pdf/1809.11096.pdf) and highlight the key findings in their extensive evaluation.

