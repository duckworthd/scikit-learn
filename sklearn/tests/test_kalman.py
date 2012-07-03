import numpy as np
from numpy import ma
from numpy.testing import assert_almost_equal

from nose.tools import assert_true

from ..kalman import KalmanFilter, UnscentedKalmanFilter
from sklearn.datasets import load_kalman_data

data = load_kalman_data()


def test_kalman_sampling():
    kf = KalmanFilter(
        data.transition_matrix,
        data.observation_matrix,
        data.transition_covariance,
        data.observation_covariance,
        data.transition_offsets,
        data.observation_offset,
        data.initial_state_mean,
        data.initial_state_covariance)

    (x, z) = kf.sample(100)
    assert_true(x.shape == (100, data.transition_matrix.shape[0]))
    assert_true(z.shape == (100, data.observation_matrix.shape[0]))


def test_kalman_filter_update():
    kf = KalmanFilter(
        data.transition_matrix,
        data.observation_matrix,
        data.transition_covariance,
        data.observation_covariance,
        data.transition_offsets,
        data.observation_offset,
        data.initial_state_mean,
        data.initial_state_covariance)

    # use Kalman Filter
    (x_filt, V_filt, ll) = kf.filter(X=data.data)

    # use online Kalman Filter
    T = data.data.shape[0]
    n_dim_state = data.transition_matrix.shape[0]
    x_filt2 = np.zeros((T, n_dim_state))
    V_filt2 = np.zeros((T, n_dim_state, n_dim_state))
    for t in range(T - 1):
        if t == 0:
            x_filt2[0] = data.initial_state_mean
            V_filt2[0] = data.initial_state_covariance
        (x_filt2[t + 1], V_filt2[t + 1], _) = kf.filter_update(
            x_filt2[t], V_filt2[t], data.data[t + 1], t=t
        )
    assert_true(np.all(x_filt == x_filt2))
    assert_true(np.all(V_filt == V_filt2))


def test_kalman_filter():
    kf = KalmanFilter(
        data.transition_matrix,
        data.observation_matrix,
        data.transition_covariance,
        data.observation_covariance,
        data.transition_offsets,
        data.observation_offset,
        data.initial_state_mean,
        data.initial_state_covariance)

    (x_filt, V_filt, ll) = kf.filter(X=data.data)
    for t in range(500):
        assert_true(np.linalg.norm(x_filt[t] -  \
            data.filtered_state_means[t]) < 1e-3)
        assert_true(np.linalg.norm(V_filt[t] -  \
            data.filtered_state_covariances[t]) < 1e-3)


def test_kalman_predict():
    kf = KalmanFilter(
        data.transition_matrix,
        data.observation_matrix,
        data.transition_covariance,
        data.observation_covariance,
        data.transition_offsets,
        data.observation_offset,
        data.initial_state_mean,
        data.initial_state_covariance)

    x_smooth = kf.predict(X=data.data)
    for t in reversed(range(501)):
        assert_true(np.linalg.norm(x_smooth[t] -  \
            data.smoothed_state_means[t]) < 1e-3)


def test_kalman_fit():
    # check against MATLAB dataset
    kf = KalmanFilter(
        data.transition_matrix,
        data.observation_matrix,
        data.initial_transition_covariance,
        data.initial_observation_covariance,
        data.transition_offsets,
        data.observation_offset,
        data.initial_state_mean,
        data.initial_state_covariance,
        em_vars=['transition_covariance', 'observation_covariance'])

    scores = np.zeros(5)
    for i in range(len(scores)):
        scores[i] = np.sum(kf.filter(X=data.data)[-1])
        kf.fit(X=data.data, n_iter=1)

    assert_true(np.allclose(scores, data.loglikelihoods[:5]))

    # check that EM for all parameters is working
    kf.em_vars = 'all'
    T = 30
    for i in range(len(scores)):
        kf.fit(X=data.data[0:T], n_iter=1)
        scores[i] = np.sum(kf.filter(X=data.data[0:T])[-1])
    for i in range(len(scores) - 1):
        assert_true(scores[i] < scores[i + 1])


def build_unscented_filter():
    # build transition functions
    A = np.array([[1, 1], [0, 1]])
    C = np.array([[0.5, -0.3]])
    f = lambda x, y: A.dot(x) + y
    g = lambda x, y: C.dot(x) + y

    x = np.array([1, 1])
    P = np.array([[1, 0.1], [0.1, 1]])

    Q = np.eye(2) * 2
    R = 0.5

    # build filter
    kf = UnscentedKalmanFilter(f, g, Q, R, x, P, random_state=0)

    return kf


def test_unscented_sample():
    kf = build_unscented_filter()
    (x, z) = kf.sample(100)

    assert_true(x.shape == (100, 2))
    assert_true(z.shape == (100, 1))


def test_unscented_filter():
    kf = build_unscented_filter()

    Z = ma.array([0, 1, 2, 3], mask=[True, False, False, False])
    (mu_filt, sigma_filt) = kf.filter(Z)
    mu_filt, sigma_filt = mu_filt[1:], sigma_filt[1:]

    # true unscented mean, covariance, as calculated by a MATLAB ukf_predict3
    # and ukf_update3 available from
    # http://becs.aalto.fi/en/research/bayes/ekfukf/
    mu_true = np.zeros((3, 2), dtype=float)
    mu_true[0] = [2.35637583900053, 0.92953020131845]
    mu_true[1] = [4.39153258583784, 1.15148930114305]
    mu_true[2] = [6.71906243764755, 1.52810614201467]

    sigma_true = np.zeros((3, 2, 2), dtype=float)
    sigma_true[0] = [[2.09738255033564, 1.51577181208054],
                     [1.51577181208054, 2.91778523489934]]
    sigma_true[1] = [[3.62532578216913, 3.14443733560803],
                     [3.14443733560803, 4.65898912348045]]
    sigma_true[2] = [[4.3902465859811, 3.90194406652627],
                     [3.90194406652627, 5.40957304471697]]

    for t in range(mu_true.shape[0]):
        assert_almost_equal(mu_true[t], mu_filt[t])
        assert_almost_equal(sigma_true[t], sigma_filt[t])
