# Referee Report

## Reviewer

This article reanalyses the binary black hole population and uses non-parametric
models to infer the underlying properties of black holes in the universe.
I enjoyed reading this article and I applaud the authors on a very well written
and clear manuscript. I only have a few comments, which I encourage the authors
to address:

- The authors do not reference the paper by Golomb and Talbot
    (https://arxiv.org/abs/2210.12287), which also use non-parametric models
    (flexible cubic splines) to fit the binary black hole population. I encourage
    the authors to discuss this work in the introduction and include their results
    in relevant plots. In light of this work, I also think the authors should
    change the statement "We introduce the first complete non-parametric model for
    the astrophysical distribution of the binary black hole (BBH) population." in
    the abstract.

This work has now been referenced and discussed in the introduction and the new sentence was highlighted. Since both Golomb and Talbot and this paper appeared publically on arxiv on the same day and their paper had no impact on this work, I do not think it is necessary for us to include their results on our plots or discuss their results in any more detail, than mentioning their paper in the introduction. We also argue that the statement in the abstract is still fully correct in light of their paper as Golomb and Talbot only applied their flexible spline modeling to the spin distributions where we have applied our data-driven unbiased basis spline modeling to the primary mass, mass ratio, and redshift distributions **in addition** to the spin distributions. This is why we refer to our work as the first *complete* non-parametric model for the astrophysical distribution of the BBH population. To our understanding this work is the first study to do such an investigation simultaneously using data-driven models for each dimension. If this is still not clear we are open to other wordings we could rephrase this sentence as. A possible rewording could be:

"We introduce the first entirely non-parametric model for the astrophysical distribution of the binary black hole (BBH) masses, spins and redshift."


- As far as I am aware, the LVK released posterior samples for two waveform
    models, SEOBNRv4PHM and IMRPhenomXPHM. I would like the authors to explicitly
    state which posterior samples were used in their hierarchical Bayesian
    inference and discuss whether waveform model systematics could influence their
    results.

We have added a few sentences to the beginning of the results section (highlighted) that explicitly states which waveform's samples were used in this work. We do not find any signs of waveform systematics which is consistent with what was investigated and discussed in Appendix C.4 of the LVK's population analysis of GWTC-3 (https://arxiv.org/abs/2111.03634).


- I am confused why the probability distribution shown in Figure 1 has units
    of inverse solar mass. Is this a typo or am I missing something?

Thanks this has been fixed now.

- I believe there is a typo on line 411: Figure 5 shows the inferred spin
    orientation distribution not the inferred spin magnitude distribution.

Thanks this typo has been fixed.

- Since the effective spin distribution is shown in Figure 7, I would also be
    interested in seeing the inferred effective precession distribution (chi_p).
    This would allow the reader to easily interpret how the inferred tilt angle
    maps to the observed precession in the signal.

Thanks for the suggestion, we have added the corresponding chi_p distribution for the models shown in Figure 7 and briefly discussed the results in the corresponding section.

- I would be interested in seeing a study that investigates the effect of
    prior assumptions on the inferred results. For instance, how would the results
    change if a normal distribution was used for the smoothing prior scale
    hyperparameter compared to a uniform distribution?

Thanks for this suggestion and this is something we have looked into. We thoroughly investigated many different prior choices for the smoothing prior scale hyperparameter and found that the results were highly stable to these different choices. We wanted to have a prior that puts significant prior volume towards "strong smoothing" so that the model a-priori favors a smooth distribution. The two priors we decided on being the best to capture this is the uniform as presented in the paper and by placing a halfnormal prior on the inverse of the prior smoothing scale hyperparameter. While this prior does add even more a-priori preference for smooth distributions, and we do infer larger (smoother) posteriors on this parameter -- the distributions are nearly identical. We have attached some plots that show the distributions for the uniform prior (blue) and the HalfNormal inverse prior that more strongly favors smooth distributions.


## Data Editor's review:

One of our data editors has reviewed your initial manuscript submission and has the following suggestion(s) to help improve the data, software citation and/or overall content. Please treat this as you would a reviewer's comments and respond accordingly in your report to the science editor. Questions can be sent directly to the data editors at data-editors@aas.org.

- The figures all have at least one icon links associated with them. These can be a Zenodo sandbox link, a Zenodo url, and a github link. There is no text for the reader to indicate what these icon links take you to. It would be best to add some text as not everybody is familiar with the "Show our work" philosophy.

Thanks for this suggestion. I have added a short sentence that indicates where these icons/links lead to for each of the figures. If this is too much copied text into each caption, we are open to other possible suggestions that may also appropriately address this.   

- In addition, the Zenodo sandbox links are not permanent and can be removed at any time. These should be replace with true Zenodo repositories. Lastly, the authors should use the Zenodo DOI rather than the url for long term stability and discovery.

Thanks again for pointing this out, this was a mishap on our part, and we have now replaced all the zenodo links etc. to a permanent and stable zenodo repository.