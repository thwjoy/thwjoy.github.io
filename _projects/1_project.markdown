---
layout: page
title: Inference Tutorials
description: Information on performing MLE and MAP
img: /assets/img/inference.png
---

Breif tutorial on MLE and MAP inference. Firstly download the following the script from 
<a href="url">Github</a>, and try and run it. You'll need to ensure you have Python, Numpy, Scipy and Matplotlib installed.

If everything runs successfully, you'll see a dyanimc graph appear containing three sub figures.
The top figure contains three items: The data we want to fit a model to (blue x); the model we want to fit (blue Gaussian); and the prior of the mean for the model (red Gaussian).
The second figure plots a trace of the likelihood of the data as we vary the mean of our model, the likelihood is simply the probability of the data given the parameters.
The third indicates the trace of the posterior, note this is not the true posterior it is only proportional to the likelihood multuplied by the prior.

<div class="img_row" style='height: 100%; width: 100%; object-fit: contain'>
    <img class="col three left" src="{{ site.baseurl }}/assets/img/inference.png" alt="" title="Main Figure" />
</div>
<div class="col three caption">
    You should be greated with this display, where the blue curve moves from left to right.
</div>

Hopefully you can see that as we move the model from left to right, the likelihood of the data given that model increases as we approach regions where there are dense points. Clearly, the model fits best when the likelihood is highest, hence why we try and find the parameters which maximise the likelihood.
Similarly for the approximate posterior, which is just the likelihood weighted by the prior.

The above example gives a resonably informative prior, however we are not always certain of our prior beliefs on parameters.
Below is an example where we have a very uninformative prior, as you can see when the prior is almost uniform the posterior is proportional to the likelihood.


<div class="img_row" style='height: 100%; width: 100%; object-fit: contain'>
    <img class="col three left" src="{{ site.baseurl }}/assets/img/uninformative_prior.png" alt="" title="Uninformative prior" />
</div>
<div class="col three caption">
    Example where prior is not very informative.
</div>


#### So why do MLE and MAP give poor estimates of parameters?

MLE and MAP give point estimates of parameters, i.e. we have no idea how the likelihood behaves either side of the maximum.
Hopefully you can see this in following example, where the likelihood has multiple peaks  which lead to almost similar likelihoods.

<div class="img_row" style='height: 100%; width: 100%; object-fit: contain'>
    <img class="col three left" src="{{ site.baseurl }}/assets/img/bad_distribution.png" alt="" title="Uninformative prior" />
</div>
<div class="col three caption">
    Example of infomative information about the likelihood.
</div>

Whilst we still get the value that maximise the likelihood, inspecting how the likelihood behaves can provide valuable information.
In this example cleary the standard deviation of the model is far too small, resulting in this spikey behaviour.

#### A few things for you to try

* Firstly play with the parameters of the underlying distribution and then the priors.

* Then try different distributions, what would you do without assuming a Gaussian model?

* You could also try and perform exact posterior inference (you need to compute the partition function $$p(x)$$) or MLE and MAP for the standard deviation.
