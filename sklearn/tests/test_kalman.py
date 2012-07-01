import numpy as np
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
    f = lambda x: A.dot(x)
    g = lambda x: C.dot(x)

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

    Z = np.array([1, 2, 3])
    (mu_filt, sigma_filt) = kf.filter(Z)

    # true unscented mean, covariance, as calculated by a MATLAB tool
    mu_true = np.zeros((3, 2), dtype=float)
    mu_true[0] = [2.28518518514932, 1.09259259256202]
    mu_true[1] = [4.50491363401856, 1.48437587344772]
    mu_true[2] = [6.91285124118152, 1.84648139654954]

    sigma_true = np.zeros((3, 2, 2), dtype=float)
    sigma_true[0] = [[3.46802469135795, 0.862345679012293],
                     [0.862345679012293, 2.92283950617283]]
    sigma_true[1] = [[4.96013416065761, 1.99320252669258],
                     [1.99320252669258, 4.2999608698084]]
    sigma_true[2] = [[5.54236591104705, 2.48841439900969],
                     [2.48841439900969, 4.80821367463056]]

    for t in range(mu_true.shape[0]):
        assert_almost_equal(mu_true[t], mu_filt[t])
        assert_almost_equal(sigma_true[t], sigma_filt[t])
