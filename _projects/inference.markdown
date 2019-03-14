---
layout: page
title: Inference Tutorials
description: Information on performing MLE and MAP
img: /assets/img/inference.png
---

This is a brief tutorial on MLE and MAP inference. Firstly download the script from 
<a href="https://github.com/thwjoy/B14">Github</a>, and try to run it. You'll need to ensure you have Python, Numpy, Scipy and Matplotlib installed.

If everything runs successfully, you'll see a dynamic graph appear containing three sub figures.
The top figure contains three items: The data we want to fit a model to (blue x); the model we want to fit (blue Gaussian); and the prior of the mean for the model (red Gaussian).
The second figure plots how the likelihood of the data behaves as we vary the mean of our model, the likelihood is simply the probability of the data given the parameters.
The third figure indicates how the posterior behaves, note this is not the true posterior as it is only proportional to the product of the likelihood and the prior.

<div class="img_row" style='height: 100%; width: 100%; object-fit: contain'>
    <img class="col three left" src="{{ site.baseurl }}/assets/img/inference.png" alt="" title="Main Figure" />
</div>
<div class="col three caption">
    You should be greeted with this display, where the blue curve moves from left to right.
</div>

Hopefully you can see that as we move the model from left to right, the likelihood of the data given that model increases as we approach regions where there are dense collections of points. Clearly, the model fits best when the likelihood is highest, hence, this is why we try and find the parameters which maximise the likelihood.
Similarly for the approximate posterior, which is just the likelihood weighted by the prior.

The above example gives a reasonably informative prior, however we are not always certain of our prior beliefs on parameters.
Below is an example where we have a very uninformative prior, as you can see when the prior is almost uniform the approximate posterior and the likelihood are almost identical.


<div class="img_row" style='height: 100%; width: 100%; object-fit: contain'>
    <img class="col three left" src="{{ site.baseurl }}/assets/img/uninformative_prior.png" alt="" title="Uninformative prior" />
</div>
<div class="col three caption">
    Example where the prior is not very informative.
</div>


#### So what are the pitfalls of MLE and MAP inference?

MLE and MAP give point estimates of parameters, i.e. we only get a single value and have no idea how the likelihood behaves either side of the maximum.
Hopefully you can see this in following example, where the likelihood has multiple peaks  which lead to almost similar likelihoods.

<div class="img_row" style='height: 100%; width: 100%; object-fit: contain'>
    <img class="col three left" src="{{ site.baseurl }}/assets/img/bad_distribution.png" alt="" title="Uninformative prior" />
</div>
<div class="col three caption">
    Example of informative information about the likelihood.
</div>

Whilst we can still get the value that maximises the likelihood, inspecting how the likelihood behaves can provide valuable information - which if often missed when performing MLE or MAP.
In this example, clearly the standard deviation of the model is far too small, resulting in this spikey behavior.

#### A few things for you to try

* Firstly play with the parameters of the underlying distribution and then the priors.

* Then try different distributions, maybe a uniform prior, or even a Chi-squared?

* You could also try and perform exact posterior inference (you need to compute the partition function $$p(x)$$) 

* Or even try MLE and MAP for the standard deviation.

Hopefully this has given you a bit more of an intuition into why we do inference and how the underlying dynamics work.