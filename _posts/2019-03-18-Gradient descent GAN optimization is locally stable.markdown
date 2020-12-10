---
layout: post
title: Gradient descent GAN optimization is locally stable
date: 2019-10-18 11:12:00-0400
description: Review of the paper Gradient descent GAN optimization is locally stable - NIPS 2017.
---

**Work in progress**

This paper provides three main contributions:
Firstly, they show that equilibrium points of the GAN objective are locally asymptoticcally stable; Secondly the also show that Wassertein GAN can in certain cirumstances contain limit cycles around an equilibrium; Finally, they propose a regulariser which is able to gurantee stability for the GAN and it's Wassertein variant.


#### Background

Before proceeding, it will prove useful to provide some background maths on stability.
Consider the following dynimical system:
$$
\dot{x} = f(t,x)
$$
We say that the systems is locally stable (Lyapunov), if all points that start near an equilibrium $$x^*$$ remain near that equilibirum point.
We also say that a system is asymptotically stable if it is firstly locally stable and all points that start near an equilibrium $$x^*$$, converge to the equilibrium. It is also worth nothing that exponential stability is asympototic stability which converges at en exponential rate.
A classic example is a frictionless pendulum which is only stable, whereas a pendulum with friction is asymptotically stable as all starting points will eventually converge to the point where the pendulum is motionless at the bottom.


#### The GAN objective is NOT convex-concave

Gradient descent methods will always converge if there is only one global minimum, i.e. the objective is either strictly convex or concave.
For GANs, this requires us to have a convex generator and a concave discriminator - which is often not the case, even when the G and D are linear.
The authors use a 1D example to successfully demonstrate this in the following GAN formulation:

$$
V(G,D) = E_{x ~ q_{data}}[f(a_d x + b_d))] + E_{z ~ p_{g}}[f(-(a_d (a_g z + b_g) + b_d)))]
$$


Now, since $$f$$ is concave (normally log), we can clearly see that the objective is concave for the discriminator, but it is also conave for the generator.
This is indiserable, as we are trying to minimisie over the generator and hance would like it to be convex. Intuitively, minimisaing a concave function doesn't initially make sense, as the problem should theoretically be unbounded i.e. solution of negative infinity, however this is not the case, and the authors go on to prove that even for the concave concave objective, the optimisation is locally asymptotically stable.

#### Defining 'good' equilibris of the GAN

The authors provide two assumptions for what they term a 'good' equilibirum. The first, realisable assumption is that the generated distribution matches the true data distribution with the discriminator outputing a zero over the support of the data distribution, this can be formally defined as:

Assumption I: $$p_g = q_{data}$$ and $$D(x) = 0, \forall x \in supp(q_{data})$$

This assumption is an ideal, and in reality we would not expect such a condition to hold as discriminators are not often strong enough to provide a non-zero output for even small areas where $$p_g \ne q_{data}$$.
To get over this, they introduced a relaxed (non-realisable) assumption, which states that the output of an optimal discriminator must be zero over the union of the support for $$p_g$$ and $$q_{data}$$:

Assumption I (non-realisable): $$D(x) = 0, \forall x \in supp(q_{data}) \cup supp(p_{g})$$

Whilst this assumption is little more than a relaxed version of the previous assumption, the formation of the union of the distributions enforces the condition that the discriminator is linear in its parameters.



