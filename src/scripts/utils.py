import numpy as np
import jax.numpy as jnp
import paths
from bilby.core.result import read_in_result
import deepdish as dd


def save_param_cred_intervals(param_data):
    return  {'median': "{:.2f}".format(np.median(param_data)), 
             'error plus': "{:.2f}".format(np.percentile(param_data, 95)-np.mean(param_data)), 
             'error minus': "{:.2f}".format(np.median(param_data)-np.percentile(param_data, 5)),
             '5th percentile': "{:.2f}".format(np.percentile(param_data, 5)), 
             '95th percentile': "{:.2f}".format(np.percentile(param_data, 95))}


def plot_mean_and_90CI(ax, xs, ar, color, label, bounds=True, CI=90, traces=None, tracecolor='k'):

    mean = np.mean(ar, axis=0)
    ax.plot(xs, mean, color=color, label=label, lw=5, alpha=0.75)
    
    if bounds:
        low = np.percentile(ar, (100-CI)/2., axis=0)
        high = np.percentile(ar, 100-(100-CI)/2., axis=0)
        #ax.plot(xs, low, color='k', lw=0.05, alpha=0.05)
        #ax.plot(xs, high, color='k', lw=0.05, alpha=0.05)
        ax.fill_between(xs, low, high, color=color, alpha=0.08)
    
    if traces is not None:
        for _ in range(traces):
            idx = np.random.choice(ar.shape[0])
            ax.plot(xs, ar[idx], color=tracecolor, lw=0.025, alpha=0.02)  
    
    return ax


def powerlaw_pdf(xx, alpha, low, high):
    """
    $$ p(x) \propto x^{\alpha}\Theta(x-x_\mathrm{min})\Theta(x_\mathrm{max}-x) $$
    """
    prob = jnp.power(xx, alpha)
    norm = jnp.where(
        jnp.equal(alpha, -1.0),
        1.0 / jnp.log(high / low),
        (1.0 + alpha) / (jnp.power(high, 1.0 + alpha) - jnp.power(low, 1.0 + alpha)),
    )
    return jnp.where(jnp.greater(xx, high) | jnp.less(xx, low), 0.0, prob * norm)


class MSpline(object):
    def __init__(
        self, n_df, knots=None, interior_knots=None, xrange=(0, 1), k=4, proper=True
    ):
        self.order = k
        self.N = n_df
        self.xrange = xrange
        if knots is None:
            if interior_knots is None:
                interior_knots = np.linspace(*xrange, n_df - k + 2)
            if proper:
                dx = interior_knots[1] - interior_knots[0]
                knots = np.append(
                    np.append(
                        np.array([xrange[0] - 2 * dx, xrange[0] - dx]), interior_knots
                    ),
                    np.array([xrange[1] + dx, xrange[1] + 2 * dx]),
                )
            else:
                knots = np.append(
                    np.append(np.array([xrange[0]] * (k - 1)), interior_knots),
                    np.array([xrange[1]] * (k - 1)),
                )
        self.knots = knots
        self.interior_knots = knots
        assert len(self.knots) == self.N + self.order

    def _basis(self, xs, i, k):
        if self.knots[i + k] - self.knots[i] < 1e-6:
            return np.zeros_like(xs)
        elif k == 1:
            v = np.zeros_like(xs)
            v[(xs >= self.knots[i]) & (xs < self.knots[i + 1])] = 1 / (
                self.knots[i + 1] - self.knots[i]
            )
            return v
        else:
            v = (xs - self.knots[i]) * self._basis(xs, i, k - 1) + (
                self.knots[i + k] - xs
            ) * self._basis(xs, i + 1, k - 1)
            return (v * k) / ((k - 1) * (self.knots[i + k] - self.knots[i]))

    def basis(self, xs):
        return [self._basis(xs, i, k=self.order) for i in range(self.N)]

    def __call__(self, xs):
        return np.concatenate(self.basis(xs)).reshape(self.N, *xs.shape)


class Base1DMSplineModel(object):
    def __init__(
        self,
        nknots,
        xx,
        xx_inj,
        knots=None,
        interior_knots=None,
        xrange=(0, 1),
        order=3,
        prefix="c",
        domain="x",
        basis=MSpline,
    ):
        self.nknots = nknots
        self.domain = domain
        self.xmin, self.xmax = xrange
        self.order = order
        self.prefix = prefix
        self.interpolator = basis(
            nknots,
            knots=knots,
            interior_knots=interior_knots,
            xrange=xrange,
            k=order + 1,
        )
        self.variable_names = [f"{self.prefix}{i}" for i in range(self.nknots)]
        self.pe_design_matrix = jnp.array(self.interpolator(xx))
        self.inj_design_matrix = jnp.array(self.interpolator(xx_inj))
        self.funcs = [self.inj_pdf, self.pe_pdf]
        self.grid = jnp.linspace(*xrange, 2500)
        self.grid_basis = self.interpolator(self.grid)

    def norm(self, coefs):
        return 1.0 / (
            sum(
                [
                    jnp.trapz(self.grid_basis[i, :], self.grid) * coefs[i]
                    for i in range(self.nknots)
                ]
            )
        )

    def eval_spline(self, x, shape, coefs):
        coefs = coefs.reshape(shape)
        coefs /= jnp.sum(coefs)
        return jnp.sum(coefs * x, axis=0) * self.norm(coefs)

    def pe_pdf(self, coefs):
        return self.eval_spline(self.pe_design_matrix, (self.nknots, 1, 1), coefs)

    def inj_pdf(self, coefs):
        return self.eval_spline(self.inj_design_matrix, (self.nknots, 1), coefs)

    def __call__(self, ndim, coefs):
        return self.funcs[ndim - 1](coefs)


class MSplineSpinMagnitude(Base1DMSplineModel):
    def __init__(
        self,
        nknots,
        a,
        a_inj,
        knots=None,
        interior_knots=None,
        order=3,
        prefix="c",
        domain="a",
    ):
        super().__init__(
            nknots,
            a,
            a_inj,
            knots=knots,
            interior_knots=interior_knots,
            order=order,
            prefix=prefix,
            domain=domain,
        )


class MSplineSpinTilt(Base1DMSplineModel):
    def __init__(
        self,
        nknots,
        ct,
        ct_inj,
        knots=None,
        interior_knots=None,
        order=3,
        prefix="x",
        domain="cos_tilt",
    ):
        super().__init__(
            nknots,
            ct,
            ct_inj,
            knots=knots,
            interior_knots=interior_knots,
            order=order,
            prefix=prefix,
            domain=domain,
            xrange=(-1, 1),
        )


class MSplineMass(Base1DMSplineModel):
    def __init__(
        self,
        nknots,
        m,
        m_inj,
        knots=None,
        interior_knots=None,
        mmin=2,
        mmax=100,
        order=3,
        prefix="f",
        domain="mass",
    ):
        super().__init__(
            nknots,
            m,
            m_inj,
            knots=knots,
            interior_knots=interior_knots,
            xrange=(mmin, mmax),
            order=order,
            prefix=prefix,
            domain=domain,
        )
    
    
class MSplineIIDSpinMagnitudes(object):
    def __init__(
        self,
        nknots,
        a1,
        a2,
        a1_inj,
        a2_inj,
        knots=None,
        interior_knots=None,
        order=3,
        prefix="c",
    ):
        self.primary_model = MSplineSpinMagnitude(
            nknots=nknots,
            a=a1,
            a_inj=a1_inj,
            knots=knots,
            interior_knots=interior_knots,
            order=order,
            prefix=prefix,
            domain="a_1",
        )
        self.secondary_model = MSplineSpinMagnitude(
            nknots=nknots,
            a=a2,
            a_inj=a2_inj,
            knots=knots,
            interior_knots=interior_knots,
            order=order,
            prefix=prefix,
            domain="a_2",
        )

    def __call__(self, ndim, coefs):
        p_a1 = self.primary_model(ndim, coefs)
        p_a2 = self.secondary_model(ndim, coefs)
        return p_a1 * p_a2


class MSplineIndependentSpinMagnitudes(object):
    def __init__(
        self,
        nknots1,
        nknots2,
        a1,
        a2,
        a1_inj,
        a2_inj,
        knots1=None,
        interior_knots1=None,
        order1=3,
        prefix1="c",
        knots2=None,
        interior_knots2=None,
        order2=3,
        prefix2="w",
    ):
        self.primary_model = MSplineSpinMagnitude(
            nknots=nknots1,
            a=a1,
            a_inj=a1_inj,
            knots=knots1,
            interior_knots=interior_knots1,
            order=order1,
            prefix=prefix1,
            domain="a_1",
        )
        self.secondary_model = MSplineSpinMagnitude(
            nknots=nknots2,
            a=a2,
            a_inj=a2_inj,
            knots=knots2,
            interior_knots=interior_knots2,
            order=order2,
            prefix=prefix2,
            domain="a_2",
        )

    def __call__(self, ndim, pcoefs, scoefs):
        p_a1 = self.primary_model(ndim, pcoefs)
        p_a2 = self.secondary_model(ndim, scoefs)
        return p_a1 * p_a2


class MSplineIIDSpinTilts(object):
    def __init__(
        self,
        nknots,
        ct1,
        ct2,
        ct1_inj,
        ct2_inj,
        knots=None,
        interior_knots=None,
        order=3,
        prefix="x",
    ):
        self.primary_model = MSplineSpinTilt(
            nknots=nknots,
            ct=ct1,
            ct_inj=ct1_inj,
            knots=knots,
            interior_knots=interior_knots,
            order=order,
            prefix=prefix,
            domain="cos_tilt_1",
        )
        self.secondary_model = MSplineSpinTilt(
            nknots=nknots,
            ct=ct2,
            ct_inj=ct2_inj,
            knots=knots,
            interior_knots=interior_knots,
            order=order,
            prefix=prefix,
            domain="cos_tilt_2",
        )

    def __call__(self, ndim, coefs):
        p_ct1 = self.primary_model(ndim, coefs)
        p_ct2 = self.secondary_model(ndim, coefs)
        return p_ct1 * p_ct2


class MSplineIndependentSpinTilts(object):
    def __init__(
        self,
        nknots1,
        nknots2,
        ct1,
        ct2,
        ct1_inj,
        ct2_inj,
        knots1=None,
        interior_knots1=None,
        order1=3,
        prefix1="x",
        knots2=None,
        interior_knots2=None,
        order2=3,
        prefix2="z",
    ):
        self.primary_model = MSplineSpinTilt(
            nknots=nknots1,
            ct=ct1,
            ct_inj=ct1_inj,
            knots=knots1,
            interior_knots=interior_knots1,
            order=order1,
            prefix=prefix1,
            domain="cos_tilt_1",
        )
        self.secondary_model = MSplineSpinTilt(
            nknots=nknots2,
            ct=ct2,
            ct_inj=ct2_inj,
            knots=knots2,
            interior_knots=interior_knots2,
            order=order2,
            prefix=prefix2,
            domain="cos_tilt_2",
        )

    def __call__(self, ndim, pcoefs, scoefs):
        p_ct1 = self.primary_model(ndim, pcoefs)
        p_ct2 = self.secondary_model(ndim, scoefs)
        return p_ct1 * p_ct2


class MSplinePrimaryPowerlawRatio(object):
    def __init__(
        self,
        nknots,
        m1,
        m1_inj,
        mmin=2,
        mmax=100,
        knots=None,
        interior_knots=None,
        order=3,
        prefix="c",
    ):
        self.primary_model = MSplineMass(
            nknots,
            m1,
            m1_inj,
            knots=knots,
            interior_knots=interior_knots,
            mmin=mmin,
            mmax=mmax,
            order=order,
            prefix=prefix,
            domain="mass_1",
        )

    def __call__(self, m1, q, beta, mmin, coefs):
        p_m1 = self.primary_model(len(m1.shape), coefs)
        p_q = powerlaw_pdf(q, beta, mmin / m1, 1)
        return p_m1 * p_q
    
def load_iid_tilt_ppd():
    datadict = dd.io.load(paths.data / 'mspline_50m1_16iid_compspins_smoothprior_powerlaw_q_z_ppds.h5')
    xs = datadict['tilts']
    dRdct = datadict['dRdct']
    return xs, dRdct

def load_ind_tilt_ppd():
    datadict = dd.io.load(paths.data / 'mspline_50m1_16ind_compspins_smoothprior_powerlaw_q_z_ppds.h5')
    return datadict['tilts'], datadict['dRdct1'], datadict['dRdct2']

def plot_o3b_spintilt(ax, fi,ct1=False, col='tab:blue', lab='PP'):
    xs = np.linspace(-1, 1, 1000)
    _data = dd.io.load(paths.data / fi)
    lines = dict()
    for key in _data["lines"].keys():
        lines[key] = _data["lines"][key][()]
        for ii in range(len(lines[key])):
            lines[key][ii] /= np.trapz(lines[key][ii], xs)
    if ct1:
        ax = plot_mean_and_90CI(ax, xs, lines['cos_tilt_1'], color=col, label=lab, bounds=False)
    else:
        ax = plot_mean_and_90CI(ax, xs, lines['cos_tilt_2'], color=col, label=lab, bounds=False)
    return ax

def load_iid_mag_ppd():
    datadict = dd.io.load(paths.data / 'mspline_50m1_16iid_compspins_smoothprior_powerlaw_q_z_ppds.h5')
    return datadict['mags'], datadict['dRda']

def plot_o3b_spinmag(ax, fi, a1=True, col='tab:blue', lab='PP'):
    xs = np.linspace(0, 1, 1000)
    _data = dd.io.load(paths.data / fi)
    lines = dict()
    for key in _data["lines"].keys():
        lines[key] = _data["lines"][key][()]
        for ii in range(len(lines[key])):
            lines[key][ii] /= np.trapz(lines[key][ii], xs)
    if a1:
        ax = plot_mean_and_90CI(ax, xs, lines['a_1'], color=col, label=lab, bounds=False)
    else:
        ax = plot_mean_and_90CI(ax, xs, lines['a_2'], color=col, label=lab, bounds=False)
    return ax

def load_ind_mag_ppd():
    datadict = dd.io.load(paths.data / 'mspline_50m1_16ind_compspins_smoothprior_powerlaw_q_z_ppds.h5')
    return datadict['dRda1'], datadict['dRda2'], datadict['mags'], datadict['mags']


def load_ind_posterior():
    return dd.io.load(paths.data / 'mspline_50m1_16ind_compspins_smoothprior_powerlaw_q_z_ppds.h5')

def load_iid_posterior():
    return dd.io.load(paths.data / 'mspline_50m1_16iid_compspins_smoothprior_powerlaw_q_z_ppds.h5')

def load_o3b_posterior(filename):
    return read_in_result(paths.data / filename).posterior

def load_o3b_paper_run_masspdf(filename):
    """
    Generates a plot of the PPD and X% credible region for the mass distribution,
    where X=limits[1]-limits[0]
    """
    mass_1 = np.linspace(2, 100, 1000)
    mass_ratio = np.linspace(0.1, 1, 500)
        
    # load in the traces. 
    # Each entry in lines is p(m1 | Lambda_i) or p(q | Lambda_i)
    # where Lambda_i is a single draw from the hyperposterior
    # The ppd is a 2D object defined in m1 and q
    with open(filename, 'r') as _data:
        _data = dd.io.load(filename)
        marginals = _data["lines"]
    for ii in range(len(marginals['mass_1'])):
        marginals['mass_1'][ii] /= np.trapz(marginals['mass_1'][ii], mass_1)
        marginals['mass_ratio'][ii] /= np.trapz(marginals['mass_ratio'][ii], mass_ratio)
    return marginals['mass_1'], marginals['mass_ratio'], mass_1, mass_ratio

def load_mass_ppd():
    datadict = dd.io.load(paths.data / 'mspline_50m1_16iid_compspins_smoothprior_powerlaw_q_z_ppds.h5')
    return datadict['m1s'], datadict['dRdm1'], datadict['qs'], datadict['dRdq']
