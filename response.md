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

- As far as I am aware, the LVK released posterior samples for two waveform
    models, SEOBNRv4PHM and IMRPhenomXPHM. I would like the authors to explicitly
    state which posterior samples were used in their hierarchical Bayesian
    inference and discuss whether waveform model systematics could influence their
    results.

- I am confused why the probability distribution shown in Figure 1 has units
    of inverse solar mass. Is this a typo or am I missing something?

- I believe there is a typo on line 411: Figure 5 shows the inferred spin
    orientation distribution not the inferred spin magnitude distribution.

- Since the effective spin distribution is shown in Figure 7, I would also be
    interested in seeing the inferred effective precession distribution (chi_p).
    This would allow the reader to easily interpret how the inferred tilt angle
    maps to the observed precession in the signal.

- I would be interested in seeing a study that investigates the effect of
    prior assumptions on the inferred results. For instance, how would the results
    change if a normal distribution was used for the smoothing prior scale
    hyperparameter compared to a uniform distribution?

## Data Editor's review:

- One of our data editors has reviewed your initial manuscript submission and has the following suggestion(s) to help improve the data, software citation and/or overall content. Please treat this as you would a reviewer's comments and respond accordingly in your report to the science editor. Questions can be sent directly to the data editors at data-editors@aas.org.

- The figures all have at least one icon links associated with them. These can be a Zenodo sandbox link, a Zenodo url, and a github link. There is no text for the reader to indicate what these icon links take you to. It would be best to add some text as not everybody is familiar with the "Show our work" philosophy.

- In addition, the Zenodo sandbox links are not permanent and can be removed at any time. These should be replace with true Zenodo repositories. Lastly, the authors should use the Zenodo DOI rather than the url for long term stability and discovery.
