---
layout: post
title: Regularising GANs
date: 2019-03-01 11:12:00-0400
description: A brief overview of training GANs with spectral normalisation.
---

It's a well known fact that [GANs](https://arxiv.org/pdf/1406.2661.pdf) are notoriously hard to train. 
This is no surprise given the objectives of learning an approximate high-dimensional probability distribution $$p_g(x)$$, of some underlying data distribution $$p_{data}(x)$$.
This is further compounded by the fact that the training regime amounts to finding the Nash equilibrium of continuous, non-convex, high-dimensional games.
As such, complete failure in training GANs is common, resulting in phenomena such as: mode-collapse; memorization in the discriminator; exploding and vanishing gradients.
There is now a plethora of work addressing the pitfalls of GANs, but in this blog post we will be studying the stability of training through regularisation in the discriminator.


[\comment\]## 



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

