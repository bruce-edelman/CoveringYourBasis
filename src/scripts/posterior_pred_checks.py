import numpy as np
import jax.numpy as jnp
from scipy.stats import gaussian_kde
from gwpopulation.cupy_utils import trapz
from gwpopulation.conversions import convert_to_beta_parameters
from gwpopulation.models.mass import SinglePeakSmoothedMassDistribution
from gwpopulation.models.spin import iid_spin_orientation_gaussian_isotropic, iid_spin_magnitude_beta
from gwpopulation.models.redshift import PowerLawRedshift
from gwpopulation.hyperpe import HyperparameterLikelihood
from gwpopulation.vt import ResamplingVT
from bilby.core.result import read_in_result
from bilby.hyper.model import Model
from jax import jit
from utils import MSplineIndependentSpinTilts, MSplineIndependentSpinMagnitudes, MSplinePrimaryPowerlawRatio, MSplineIIDSpinMagnitudes, MSplineIIDSpinTilts
import deepdish as dd
from tqdm import trange
import paths
import warnings
warnings.filterwarnings('ignore')





def resample_posteriors(likelihood, result, injdata, save=True):
    
    original_samples = {key: to_numpy(likelihood.data[key]) for key in likelihood.data}
    original_samples["prior"] = to_numpy(likelihood.sampling_prior)
    reweighted_samples, weights = likelihood.posterior_predictive_resample(
        result.posterior, return_weights=True
    )
    original_samples["weights"] = to_numpy(weights)
    reweighted_samples = {
        key: to_numpy(reweighted_samples[key]) for key in reweighted_samples
    }
    predicted_samples = generate_posterior_predictive_from_injections(
        likelihood, result, injdata
    )
    if save:
        dd.io.save(
            os.path.join(result.outdir, f"{result.label}_samples.hdf5"),
            dict(
                original=original_samples,
                reweighted=reweighted_samples,
                predicted=predicted_samples,
                names=result.meta_data["event_ids"],
                label=result.label,
            ),
        )
    else:
        return original_samples, reweighted_samples


def generate_posterior_predictive_from_injections(likelihood, result, injdata):
    weights = xp.zeros(len(injdata["prior"]))
    for idx in trange(len(result.posterior)):
        sample = result.posterior.iloc[idx]
        likelihood.parameters.update(sample.copy())
        likelihood.parameters, added_keys = likelihood.conversion_function(
            likelihood.parameters
        )
        likelihood.selection_function.model.parameters.update(likelihood.parameters)
        new_weights = likelihood.selection_function.model.prob(injdata) / injdata["prior"]
        new_weights = (new_weights.T / xp.sum(new_weights, axis=-1)).T
        weights += new_weights
        if added_keys is not None:
            for key in added_keys:
                likelihood.parameters.pop(key)
    weights = (weights.T / xp.sum(weights, axis=-1)).T
    new_idxs = xp.asarray(
        np.random.choice(
            range(len(injdata["prior"])),
            size=(likelihood.n_posteriors, likelihood.samples_per_posterior),
            replace=True,
            p=to_numpy(weights),
        )
    )
    keys = []
    for k in injdata.keys():
        if k != "prior":
            try:
                sh = injdata[k].shape
            except (KeyError, AttributeError):
                continue
            if sh == weights.shape:
                keys.append(k)
            else:
                continue
    new_samples = {
        key: xp.vstack(
            [injdata[key][new_idxs[ii, :]] for ii in range(likelihood.n_posteriors)]
        )
        for key in keys
    }
    return new_samples
