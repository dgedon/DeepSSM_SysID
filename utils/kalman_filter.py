import numpy as np


# measurment update of KF
def KF_MU(C, R, y, P, xhat):
    """# check input
    if xhat.shape[-1] != 1:
        xhat = xhat.reshape([-1, 1])"""

    # Kalman filter coefficient
    S = C.dot(P).dot(np.transpose(C)) + R
    K = P.dot(np.transpose(C)) * np.linalg.inv(S)

    # estimated observations
    yhat_ = C.dot(xhat)

    # measurement residual error (innovation error)
    innov = y - yhat_

    # updated estimate of the current state
    xhat = xhat + K.dot(innov)

    # updated state covariane matrix
    P = P - K.dot(C).dot(P)

    # updated (filtered) output estimate y(k|k)
    yhat = C.dot(xhat)

    return yhat, xhat, P, K


# time update of KF
def KF_TU(A, B, Q, P, xhat, u):
    # update of current state
    xhat = A.dot(xhat) + B.dot(u)

    # update of covariance
    P = A.dot(P).dot(np.transpose(A)) + Q

    return xhat, P


# complete KF
def run_kalman_filter(A, B, C, Q, R, u, y):
    # get simulation time
    k_max = u.shape[-1]

    # sizes
    n_x = A.shape[-1]
    n_y = C.shape[0]

    # allocation
    xhat = np.zeros([n_x, 1])
    yhat = np.zeros([n_y, k_max])

    # initialization
    P = np.dot(B, np.transpose(B))

    for k in range(k_max):
        # measurement update
        yhat[:, k], xhat, P, K = KF_MU(C, R, y[:, :, k], P, xhat)

        # time update
        xhat, P = KF_TU(A, B, Q, P, xhat, u[:, :, k])

    return yhat
