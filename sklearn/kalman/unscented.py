'''
=========================================
Inference for Non-Linear Gaussian Systems
=========================================

This module contains the Unscented Kalman Filter (Wan, van der Merwe 2000)
for state estimation in systems with non-Gaussian noise and non-linear dynamics
'''
import numpy as np
from numpy import ma
from scipy import linalg

from ..base import BaseEstimator
from ..utils import array1d, array2d, check_random_state

from .standard import _last_dims


def _unscented_moments(points, weights_mu, weights_sigma):
    '''
    Calculate the weighted mean and covariance of `points`
    '''
    mu = points.T.dot(weights_mu)
    points_diff = points.T - mu[:, np.newaxis]
    sigma = points_diff.dot(np.diag(weights_sigma)).dot(points_diff.T)
    return (mu.ravel(), sigma)


def _sigma_points(mu, sigma, alpha=1e-3, beta=2.0, kappa=0.0):
    '''Calculate "sigma points" used in Unscented Kalman Filter

    Parameters
    ----------
    mu : [n_dim_state] array
        approximate mean of state at time t given observations from [0...t]
    sigma : [n_dim_state, n_dim_state] array
        approximate covariance of state at time t given observations from
        [0...t]
    alpha : float
        spread of the sigma points.  typically 1e-3.
    beta : float
        used to "incorporate prior knowledge of the distribution of the state".
        2 is optimal is the state is normally distributed.
    kappa : float
        a parameter which means ????

    Returns
    -------
    points : [2*n_dim_state+1, n_dim_state] array
        sigma points of the UKF
    weights_mean : [2*n_dim_state+1] array
        weights for calculating the empirical mean
    weights_cov : [n*n_dim_state+1] array
        weights for calculating the empirical covariance
    '''
    n_dim_state = len(mu)
    mu = array2d(mu, dtype=float)

    # compute sqrt(sigma)
    sigma2 = linalg.cholesky(sigma)

    # Calculate scaling factor for all off-center points
    lamda = (alpha * alpha) * (n_dim_state + kappa) - n_dim_state
    c = n_dim_state + lamda

    # get mu, mu + each column of sigma2 * sqrt(n_dim_state + lamda) and mu -
    # each column of sigma2 * c.  Each column of points is one of these.
    points = np.tile(mu.T, (1, 2 * n_dim_state + 1))
    points[:, 1:(n_dim_state + 1)] += sigma2.T * np.sqrt(c)
    points[:, (n_dim_state + 1):] -= sigma2.T * np.sqrt(c)

    # Calculate weights
    weights_mean = np.ones(2 * n_dim_state + 1)
    weights_mean[0] = lamda / c
    weights_mean[1:] = 0.5 / c
    weights_cov = np.copy(weights_mean)
    weights_cov[0] = lamda / c + (1 - alpha * alpha + beta)

    return (points.T, weights_mean, weights_cov)


def _unscented_transform(f, points, points_noise, weights_mean, weights_cov):
    '''
    Apply the Unscented Transform.

    Parameters
    ==========
    f : [n_dim_1, n_dim_3] -> [n_dim_2] function
        function to apply pass all points through
    points : [n_points, n_dim_1] array
        points representing state to pass through `f`
    points_noise : [n_points, n_dim_3] array
        points representing noise to pass through `f`
    weights_mean : [n_points] array
        weights used to calculate the empirical mean
    weights_cov : [n_points] array
        weights used to calculate empirical covariance

    Returns
    =======
    points_pred : [n_points, n_dim_2] array
        points passed through f
    mu_pred : [n_dim_2] array
        empirical mean
    sigma_pred : [n_dim_2, n_dim_2] array
        empirical covariance
    '''
    n_points, n_dim_state = points.shape

    # propagate points through f.  Each column is a sample point
    points_pred = np.vstack(
        [f(points[i], points_noise[i]) for i in range(n_points)]
    )

    # calculate approximate mean, covariance
    (mu_pred, sigma_pred) =   \
        _unscented_moments(points_pred, weights_mean, weights_cov)

    return (points_pred, mu_pred, sigma_pred)


def _unscented_correct(cross_sigma, mu_pred, sigma_pred, obs_mu_pred,
                       obs_sigma_pred, z):
    '''
    Correct predicted state estimates with an observation
    '''
    n_dim_state = len(mu_pred)
    n_dim_obs = len(obs_mu_pred)

    if not np.any(ma.getmask(z)):
        # calculate Kalman gain
        K = cross_sigma.dot(linalg.pinv(obs_sigma_pred))

        # correct mu, sigma
        mu_filt = mu_pred + K.dot(z - obs_mu_pred)
        sigma_filt = sigma_pred - K.dot(cross_sigma.T)
    else:
        # no Kalman gain
        K = np.zeros((n_dim_state, n_dim_obs))

        # no corrections to be made
        mu_filt = mu_pred
        sigma_filt = sigma_pred
    return (K, mu_filt, sigma_filt)


def _augmented_unscented_filter(mu_0, sigma_0, f, g, Q, R, Z):
    '''
    Apply the Unscented Kalman Filter with arbitrary noise
    '''
    T = Z.shape[0]
    n_dim_state = Q.shape[-1]
    n_dim_obs = R.shape[-1]

    mu_filt = np.zeros((T, n_dim_state))
    sigma_filt = np.zeros((T, n_dim_state, n_dim_state))

    for t in range(T):
        # Calculate sigma points for augmented state:
        #   [actual state, transition noise, observation noise]
        if t == 0:
            mu = mu_0
            sigma = sigma_0
        else:
            mu = mu_filt[t - 1]
            sigma = sigma_filt[t - 1]
        mu_aug = np.concatenate(
            [mu, np.zeros(n_dim_state), np.zeros(n_dim_obs)]
        )
        sigma_aug = np.zeros(
            (2 * n_dim_state + n_dim_obs, 2 * n_dim_state + n_dim_obs),
            dtype=float
        )
        sigma_aug[0:n_dim_state, 0:n_dim_state] = sigma
        sigma_aug[n_dim_state:2 * n_dim_state, n_dim_state:2 * n_dim_state] = Q
        sigma_aug[2 * n_dim_state:, 2 * n_dim_state:] = R

        (points_aug, weights_mu, weights_sigma) =   \
            _sigma_points(mu_aug, sigma_aug)

        # extract components of points_aug dealing with state, transition
        # noise, observation noise
        points_state = points_aug[:, 0:n_dim_state]
        points_trans = points_aug[:, n_dim_state:2 * n_dim_state]
        points_obs = points_aug[:, 2 * n_dim_state:]

        # Calculate E[x_t | z_{0:t-1}], Var(x_t | z_{0:t-1})
        if t == 0:
            points_pred = points_state
            (mu_pred, sigma_pred) =   \
                _unscented_moments(points_pred, weights_mu, weights_sigma)
        else:
            f_t1 = _last_dims(f, t - 1, ndims=1)[0]
            (points_pred, mu_pred, sigma_pred) =  \
                _unscented_transform(f_t1, points_state, points_trans,
                                     weights_mu, weights_sigma)

        # Calculate E[z_t | z_{0:t-1}], Var(z_t | z_{0:t-1})
        g_t = _last_dims(g, t, ndims=1)[0]
        (obs_points_pred, obs_mu_pred, obs_sigma_pred) =  \
            _unscented_transform(g_t, points_pred, points_obs,
                                 weights_mu, weights_sigma)

        # Calculate Cov(x_t, z_t | z_{0:t-1})
        sigma_pair = ((points_pred - mu_pred).T).   \
            dot(np.diag(weights_sigma)).  \
            dot(obs_points_pred - obs_mu_pred)

        # Calculate E[x_t | z_{0:t}], Var(x_t | z_{0:t})
        (_, mu_filt[t], sigma_filt[t]) =  \
            _unscented_correct(sigma_pair, mu_pred, sigma_pred, obs_mu_pred,
                               obs_sigma_pred, Z[t])

    return (mu_filt, sigma_filt)


def _additive_unscented_filter(mu_0, sigma_0, f, g, Q, R, Z):
    '''
    Apply the Unscented Kalman Filter with additive noise
    '''
    T = Z.shape[0]
    n_dim_state = Q.shape[0]

    mu_filt = np.zeros((T, n_dim_state))
    sigma_filt = np.zeros((T, n_dim_state, n_dim_state))

    for t in range(T):
        # Calculate sigma points for P(x_{t-1} | z_{0:t-1})
        if t == 0:
            mu = mu_0
            sigma = sigma_0
        else:
            mu = mu_filt[t - 1]
            sigma = sigma_filt[t - 1]
        (points, weights_mu, weights_sigma) =   \
            _sigma_points(mu, sigma)

        # Calculate E[x_t | z_{0:t-1}], Var(x_t | z_{0:t-1})
        if t == 0:
            points_pred = points_state
            (mu_pred, sigma_pred) =   \
                _unscented_moments(points_pred, weights_mu, weights_sigma)
        else:
            f_t1 = _last_dims(f, t - 1, ndims=1)[0]
            f_mock = lambda x, y: f_t1(x)
            (points_pred, mu_pred, sigma_pred) =  \
                _unscented_transform(f_t1, points, points,
                                     weights_mu, weights_sigma)
            sigma_pred += Q

        # Calculate E[z_t | z_{0:t-1}], Var(z_t | z_{0:t-1})
        g_t = _last_dims(g, t, ndims=1)[0]
        g_mock = lambda x, y: g_t(x)
        (obs_points_pred, obs_mu_pred, obs_sigma_pred) =  \
            _unscented_transform(g_mock, points_pred, points_pred,
                                 weights_mu, weights_sigma)
        obs_sigma_pred += R

        # Calculate Cov(x_t, z_t | z_{0:t-1})
        sigma_pair = ((points_pred - mu_pred).T).   \
            dot(np.diag(weights_sigma)).  \
            dot(obs_points_pred - obs_mu_pred)

        # Calculate E[x_t | z_{0:t}], Var(x_t | z_{0:t})
        (_, mu_filt[t], sigma_filt[t]) =  \
            _unscented_correct(sigma_pair, mu_pred, sigma_pred, obs_mu_pred,
                               obs_sigma_pred, Z[t])

    return (mu_filt, sigma_filt)


class GeneralUnscentedKalmanFilter(BaseEstimator):
    """
    Implements the General (aka Augmented) Unscented Kalman Filter governed by
    the following equations,

    .. math::

        v_t       &\sim \text{Normal}(0, Q)     \\
        w_t       &\sim \text{Normal}(0, R)     \\
        x_{t+1}   &= f_t(x_t, v_t)              \\
        z_{t}     &= g_t(x_t, w_t)

    Notice that although the input noise to the state transition equation and
    the observation equation are both normally distributed, but any non-linear
    transformation may be applied afterwards.  This allows for greater
    generality, but at the expense of computational complexity.  The complexity
    of `GeneralUnscentedKalmanFilter.filter()`is :math:`O(T(2n+m))` where :math:`T` is
    the number of time steps, :math:`n` is the size of the state space, and
    :math:`m` is the size of the observation space.

    If your noise is simply additive, consider using the
    `AdditiveUnscentedKalmanFilter`
    """
    def __init__(self, f, g, Q, R, mu_0, sigma_0, random_state=None):
        self.f = array1d(f)
        self.g = array1d(g)
        self.Q = array2d(Q)
        self.R = array2d(R)
        self.mu_0 = array1d(mu_0)
        self.sigma_0 = array2d(sigma_0)
        self.random_state = random_state

    def sample(self, T, x_0=None, random_state=None):
        n_dim_state = self.Q.shape[-1]
        n_dim_obs = self.R.shape[-1]

        # logic for instantiating rng
        if random_state is None:
            rng = check_random_state(self.random_state)
        else:
            rng = check_random_state(random_state)

        # logic for selecting initial state
        if x_0 is None:
            x_0 = rng.multivariate_normal(self.mu_0, self.sigma_0)

        # logic for generating samples
        x = np.zeros((T, n_dim_state))
        z = np.zeros((T, n_dim_obs))
        for t in range(T):
            if t == 0:
                x[0] = x_0
            else:
                f_t1 = _last_dims(self.f, t - 1, ndims=1)[0]
                Q_t1 = self.Q
                e_t1 = rng.multivariate_normal(np.zeros(n_dim_state),
                    Q_t1.newbyteorder('='))
                x[t] = f_t1(x[t - 1], e_t1)

            g_t = _last_dims(self.g, t, ndims=1)[0]
            R_t = self.R
            e_t2 = rng.multivariate_normal(np.zeros(n_dim_obs),
                R_t.newbyteorder('='))
            z[t] = g_t(x[t], e_t2)

        return (x, ma.asarray(z))

    def filter(self, Z):
        Z = ma.asarray(Z)

        (mu_filt, sigma_filt) =   \
            _augmented_unscented_filter(
                self.mu_0, self.sigma_0, self.f,
                self.g, self.Q, self.R, Z
            )
        return (mu_filt, sigma_filt)

    def predict(self, Z):
        return self.filter(Z)[0]
