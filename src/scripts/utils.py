import numpy as np


HARDCODED_CHIEFF_KNOTS = np.array([-1.4, -1.2, -1.0, -0.8, -0.6, -0.5, -0.4, -0.32, -0.25, -0.19, -0.14, -0.1, -0.07, -0.05, -0.025, 
                   0.0, 0.025, 0.05, 0.07, 0.1, 0.14, 0.19, 0.25, 0.32, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.4])


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