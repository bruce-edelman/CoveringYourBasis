import numpy as np
import jax.numpy as jnp
import paths
from bilby.core.result import read_in_result
import deepdish as dd


def plot_mean_and_90CI(ax, xs, ar, color, label, bounds=True, CI=90, traces=None, tracecolor='k', fill_alpha=0.08):

    mean = np.mean(ar, axis=0)
    ax.plot(xs, mean, color=color, label=label, lw=5, alpha=0.75)
    
    if bounds:
        low = np.percentile(ar, (100-CI)/2., axis=0)
        high = np.percentile(ar, 100-(100-CI)/2., axis=0)
        #ax.plot(xs, low, color='k', lw=0.05, alpha=0.05)
        #ax.plot(xs, high, color='k', lw=0.05, alpha=0.05)
        ax.fill_between(xs, low, high, color=color, alpha=fill_alpha)
    
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


def load_iid_tilt_ppd():
    datadict = dd.io.load(paths.data / 'bsplines_64m1_18q_iid18mag_iid16tilt_pl16z_ppds.h5')
    xs = datadict['tilts']
    dRdct = datadict['dRdct']
    return xs, dRdct

def load_ind_tilt_ppd():
    datadict = dd.io.load(paths.data / 'bsplines_64m1_18q_ind16mag_ind12tilt_pl16z_ppds.h5')
    return datadict['tilts'], datadict['dRdct'], datadict['dRdct']

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
    datadict = dd.io.load(paths.data / 'bsplines_64m1_18q_iid18mag_iid16tilt_pl16z_ppds.h5')
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
    datadict = dd.io.load(paths.data / 'bsplines_64m1_18q_ind16mag_ind12tilt_pl16z_ppds.h5')
    return datadict['dRda'], datadict['dRda'], datadict['mags'], datadict['mags']


def load_ind_posterior():
    return dd.io.load(paths.data / 'bsplines_64m1_18q_ind16mag_ind12tilt_pl16z_posterior_samples.h5')

def load_iid_posterior():
    return dd.io.load(paths.data / 'bsplines_64m1_18q_iid18mag_iid16tilt_pl16z_posterior_samples.h5')

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
    datadict = dd.io.load(paths.data / 'bsplines_64m1_18q_iid18mag_iid16tilt_pl16z_ppds.h5')
    return datadict['m1s'], datadict['dRdm1'], datadict['qs'], datadict['dRdq']


class MSpline(object):
    def __init__(
        self,
        n_df,
        xrange=(0, 1),
        k=4,
        knots=None,
        normalize=True,
    ):
        self.order = k
        self.N = n_df
        self.xrange = xrange
        if knots is None:
            interior_knots = np.linspace(*xrange, n_df - k + 2)
            dx = interior_knots[1] - interior_knots[0]
            knots = np.concatenate(
                [
                    xrange[0] - dx * np.arange(1, k)[::-1],
                    interior_knots,
                    xrange[1] + dx * np.arange(1, k),
                ]
            )
        self.knots = knots
        self.interior_knots = knots
        assert len(self.knots) == self.N + self.order

        self.normalize = normalize
        self.basis_vols = np.ones(self.N)
        if normalize:
            grid = jnp.linspace(*xrange, 1000)
            grid_bases = jnp.array(self.bases(grid))
            self.basis_vols = jnp.array(
                [jnp.trapz(grid_bases[i, :], grid) for i in range(self.N)]
            )

    def norm(self, coefs):
        n = 1.0 / jnp.sum(self.basis_vols * coefs.flatten()) if self.normalize else 1.0
        return n

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

    def _bases(self, xs):
        return [self._basis(xs, i, k=self.order) for i in range(self.N)]

    def bases(self, xs):
        return jnp.concatenate(self._bases(xs)).reshape(self.N, *xs.shape)

    def project(self, bases, shape, coefs):
        coefs = coefs.reshape(shape)
        coefs /= jnp.sum(coefs)
        return jnp.sum(coefs * bases, axis=0) * self.norm(coefs)

    def eval(self, xs, shape, coefs):
        return self.project(self.bases(xs), shape, coefs)

    def __call__(self, xs, coefs):
        return self.eval(xs, (-1, 1), coefs)


class BSpline(MSpline):
    def __init__(
        self,
        n_df,
        xrange=(0, 1),
        k=4,
        knots=None,
        normalize=False,
    ):
        super().__init__(
            n_df=n_df,
            knots=knots,
            xrange=xrange,
            k=k,
            normalize=normalize,
        )

    def _bases(self, xs):
        return [
            (self.knots[i + self.order] - self.knots[i])
            / self.order
            * self._basis(xs, i, k=self.order)
            for i in range(self.N)
        ]

    def project(self, bases, shape, coefs):
        coefs = coefs.reshape(shape)
        return jnp.sum(coefs * bases, axis=0) * self.norm(coefs)


class LogXBSpline(BSpline):
    def __init__(self, n_df, xrange=(0.01, 1), knots=None, normalize=True, **kwargs):
        knots = None if knots is None else np.log(knots)
        xrange = np.log(xrange)
        super().__init__(n_df, knots=knots, xrange=xrange, **kwargs)

        self.normalize = normalize
        self.basis_vols = np.ones(self.N)
        if normalize:
            self.grid = jnp.linspace(*np.exp(xrange), 1000)
            self.grid_bases = jnp.array(self.bases(self.grid))

    def bases(self, xs):
        return super().bases(jnp.log(xs))


class LogYBSpline(BSpline):
    def __init__(self, n_df, xrange=(0, 1), knots=None, normalize=True, **kwargs):
        super().__init__(n_df, knots=knots, xrange=xrange, **kwargs)
        self.normalize = normalize
        if normalize:
            self.grid = jnp.linspace(*xrange, 1000)
            self.grid_bases = jnp.array(self.bases(self.grid))

    def _project(self, bases, shape, coefs):
        coefs = coefs.reshape(shape)
        return jnp.exp(jnp.sum(coefs * bases, axis=0))

    def project(self, bases, shape, coefs):
        return self._project(bases, shape, coefs) * self.norm(coefs)

    def norm(self, coefs):
        n = (
            1.0 / jnp.trapz(self._project(self.grid_bases, (-1, 1), coefs), self.grid)
            if self.normalize
            else 1.0
        )
        return n


class LogXLogYBSpline(LogYBSpline):
    def __init__(self, n_df, xrange=(0.1, 1), knots=None, normalize=True, **kwargs):
        knots = None if knots is None else np.log(knots)
        xrange = np.log(xrange)
        super().__init__(n_df, knots=knots, xrange=xrange, **kwargs)

        self.normalize = normalize
        self.basis_vols = np.ones(self.N)
        if normalize:
            self.grid = jnp.linspace(*jnp.exp(xrange), 1500)
            self.grid_bases = jnp.array(self.bases(self.grid))

    def bases(self, xs):
        return super().bases(jnp.log(xs))

    def project(self, bases, shape, coefs):
        return self._project(bases, shape, coefs) * self.norm(coefs)
    
    
class Base1DMSplineModel(object):
    def __init__(
        self,
        nknots,
        xx,
        xx_inj,
        knots=None,
        xrange=(0, 1),
        order=3,
        prefix="c",
        domain="x",
        basis=MSpline,
        **kwargs,
    ):
        self.nknots = nknots
        self.domain = domain
        self.xmin, self.xmax = xrange
        self.order = order
        self.prefix = prefix
        self.interpolator = basis(
            nknots,
            knots=knots,
            xrange=xrange,
            k=order + 1,
            **kwargs,
        )
        self.variable_names = [f"{self.prefix}{i}" for i in range(self.nknots)]
        self.pe_design_matrix = jnp.array(self.interpolator.bases(xx))
        self.inj_design_matrix = jnp.array(self.interpolator.bases(xx_inj))
        self.funcs = [self.inj_pdf, self.pe_pdf]

    def eval_spline(self, bases, shape, coefs):
        return self.interpolator.project(bases, shape, coefs)

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
        order=3,
        prefix="c",
        domain="a",
        **kwargs,
    ):
        super().__init__(
            nknots,
            a,
            a_inj,
            knots=knots,
            order=order,
            prefix=prefix,
            domain=domain,
            **kwargs,
        )


class MSplineSpinTilt(Base1DMSplineModel):
    def __init__(
        self,
        nknots,
        ct,
        ct_inj,
        knots=None,
        order=3,
        prefix="x",
        domain="cos_tilt",
        **kwargs,
    ):
        super().__init__(
            nknots,
            ct,
            ct_inj,
            knots=knots,
            order=order,
            prefix=prefix,
            domain=domain,
            xrange=(-1, 1),
            **kwargs,
        )


class MSplineChiEffective(Base1DMSplineModel):
    def __init__(
        self,
        nknots,
        chieff,
        chieff_inj,
        knots=None,
        order=3,
        prefix="x",
        domain="chi_eff",
        **kwargs,
    ):
        super().__init__(
            nknots,
            chieff,
            chieff_inj,
            knots=knots,
            order=order,
            prefix=prefix,
            domain=domain,
            xrange=(-1, 1),
            **kwargs,
        )
        
class MSplineRatio(Base1DMSplineModel):
    def __init__(
        self,
        nknots,
        q,
        q_inj,
        qmin=0,
        knots=None,
        order=3,
        prefix="u",
        **kwargs,
    ):
        super().__init__(
            nknots,
            q,
            q_inj,
            knots=knots,
            order=order,
            prefix=prefix,
            xrange=(qmin, 1),
            domain="mass_ratio",
            **kwargs,
        )


class MSplineMass(Base1DMSplineModel):
    def __init__(
        self,
        nknots,
        m,
        m_inj,
        knots=None,
        mmin=2,
        mmax=100,
        order=3,
        prefix="f",
        domain="mass",
        **kwargs,
    ):
        super().__init__(
            nknots,
            m,
            m_inj,
            knots=knots,
            xrange=(mmin, mmax),
            order=order,
            prefix=prefix,
            domain=domain,
            **kwargs,
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
        order=3,
        prefix="c",
        **kwargs,
    ):
        self.primary_model = MSplineSpinMagnitude(
            nknots=nknots,
            a=a1,
            a_inj=a1_inj,
            knots=knots,
            order=order,
            prefix=prefix,
            domain="a_1",
            **kwargs,
        )
        self.secondary_model = MSplineSpinMagnitude(
            nknots=nknots,
            a=a2,
            a_inj=a2_inj,
            knots=knots,
            order=order,
            prefix=prefix,
            domain="a_2",
            **kwargs,
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
        order1=3,
        prefix1="c",
        knots2=None,
        order2=3,
        prefix2="w",
        **kwargs,
    ):
        self.primary_model = MSplineSpinMagnitude(
            nknots=nknots1,
            a=a1,
            a_inj=a1_inj,
            knots=knots1,
            order=order1,
            prefix=prefix1,
            domain="a_1",
            **kwargs,
        )
        self.secondary_model = MSplineSpinMagnitude(
            nknots=nknots2,
            a=a2,
            a_inj=a2_inj,
            knots=knots2,
            order=order2,
            prefix=prefix2,
            domain="a_2",
            **kwargs,
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
        order=3,
        prefix="x",
        **kwargs,
    ):
        self.primary_model = MSplineSpinTilt(
            nknots=nknots,
            ct=ct1,
            ct_inj=ct1_inj,
            knots=knots,
            order=order,
            prefix=prefix,
            domain="cos_tilt_1",
            **kwargs,
        )
        self.secondary_model = MSplineSpinTilt(
            nknots=nknots,
            ct=ct2,
            ct_inj=ct2_inj,
            knots=knots,
            order=order,
            prefix=prefix,
            domain="cos_tilt_2",
            **kwargs,
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
        order1=3,
        prefix1="x",
        knots2=None,
        order2=3,
        prefix2="z",
        **kwargs,
    ):
        self.primary_model = MSplineSpinTilt(
            nknots=nknots1,
            ct=ct1,
            ct_inj=ct1_inj,
            knots=knots1,
            order=order1,
            prefix=prefix1,
            domain="cos_tilt_1",
            **kwargs,
        )
        self.secondary_model = MSplineSpinTilt(
            nknots=nknots2,
            ct=ct2,
            ct_inj=ct2_inj,
            knots=knots2,
            order=order2,
            prefix=prefix2,
            domain="cos_tilt_2",
            **kwargs,
        )

    def __call__(self, ndim, pcoefs, scoefs):
        p_ct1 = self.primary_model(ndim, pcoefs)
        p_ct2 = self.secondary_model(ndim, scoefs)
        return p_ct1 * p_ct2


class MSplinePrimaryMSplineRatio(object):
    def __init__(
        self,
        nknots_m,
        nknots_q,
        m1,
        m1_inj,
        q,
        q_inj,
        knots_m=None,
        knots_q=None,
        order_m=3,
        order_q=3,
        prefix_m="c",
        prefix_q="q",
        m1min=3.0,
        m2min=3.0,
        mmax=100.0,
        basis_m=MSpline,
        basis_q=MSpline,
        **kwargs,
    ):
        self.primary_model = MSplineMass(
            nknots_m,
            m1,
            m1_inj,
            knots=knots_m,
            mmin=m1min,
            mmax=mmax,
            order=order_m,
            prefix=prefix_m,
            domain="mass_1",
            basis=basis_m,
            **kwargs,
        )
        self.ratio_model = MSplineRatio(
            nknots_q,
            q,
            q_inj,
            qmin=m2min / mmax,
            knots=knots_q,
            order=order_q,
            prefix=prefix_q,
            basis=basis_q,
            **kwargs,
        )

    def __call__(self, ndim, mcoefs, qcoefs):
        return self.ratio_model(ndim, qcoefs) * self.primary_model(ndim, mcoefs)
