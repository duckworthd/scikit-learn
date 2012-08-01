import numpy as np
from numpy import ma
from numpy.testing import assert_almost_equal

from nose.tools import assert_true

from sklearn.kalman import UnscentedKalmanFilter
from sklearn.datasets import load_kalman_data

data = load_kalman_data()


def build_unscented_filter():
    '''Instantiate the Unscented Kalman Filter'''
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


def check_unscented_prediction(method, mu_true, sigma_true):
    '''Check output of a method against true mean and covariances'''
    Z = ma.array([0, 1, 2, 3], mask=[True, False, False, False])
    (mu_est, sigma_est) = method(Z)
    mu_est, sigma_est = mu_est[1:], sigma_est[1:]

    for t in range(mu_true.shape[0]):
        assert_almost_equal(mu_true[t], mu_est[t])
        assert_almost_equal(sigma_true[t], sigma_est[t])


def test_unscented_sample():
    kf = build_unscented_filter()
    (x, z) = kf.sample(100)

    assert_true(x.shape == (100, 2))
    assert_true(z.shape == (100, 1))


def test_unscented_filter():
    Z = ma.array([0, 1, 2, 3], mask=[True, False, False, False])

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

    check_unscented_prediction(build_unscented_filter().filter,
                               mu_true, sigma_true)


def test_unscented_smoother():
    Z = ma.array([0, 1, 2, 3], mask=[True, False, False, False])

    # true unscented mean, covariance, as calculated by a MATLAB urts_smooth2
    # available in http://becs.aalto.fi/en/research/bayes/ekfukf/
    mu_true = np.zeros((3, 2), dtype=float)
    mu_true[0] = [2.92725011530645, 1.63582509442842]
    mu_true[1] = [4.87447429684622,  1.6467868915685]
    mu_true[2] = [6.71906243764755, 1.52810614201467]

    sigma_true = np.zeros((3, 2, 2), dtype=float)
    sigma_true[0] = [[0.993799756492982, 0.216014513083516],
                     [0.216014513083516, 1.25274857496387]]
    sigma_true[1] = [[1.57086880378025, 1.03741785934464],
                     [1.03741785934464, 2.49806235789068]]
    sigma_true[2] = [[4.3902465859811, 3.90194406652627],
                     [3.90194406652627, 5.40957304471697]]

    check_unscented_prediction(build_unscented_filter().smooth,
                               mu_true, sigma_true)
