%% complete KF
function yhat = run_kalman_filter(A, B, C, Q, R, u, y)
    % get simulation time
    k_max = size(u,2);

    % sizes
    n_x = size(A,1);
    n_y = size(C,1);

    % allocation
    xhat = zeros(n_x, 1);
    yhat = zeros(n_y, k_max);

    % initialization
    P = B*B';

    for k = 1:k_max
        % measurement update
        [yhat(:, k), xhat, P, ~] = KF_MU(C, R, y(:, k), P, xhat);

        % time update
        [xhat, P] = KF_TU(A, B, Q, P, xhat, u(:, k));

    end
end

%% measurment update of KF
function [yhat, xhat, P, K] = KF_MU(C, R, y, P, xhat)


    % Kalman filter coefficient
    S = C*P*C'+R;
    K = P*C' * inv(S);

    % estimated observations
    yhat_ = C* xhat;

    % measurement residual error (innovation error)
    innov = y - yhat_;

    % updated estimate of the current state
    xhat = xhat + K*innov;

    % updated state covariane matrix
    P = P - K*C*P;

    % updated (filtered) output estimate y(k|k)
    yhat = C*xhat;

end

%% time update of KF
function [xhat, P] = KF_TU(A, B, Q, P, xhat, u)
    % update of current state
    xhat = A*xhat + B*u;

    % update of covariance
    P = A*P*A' + Q;
end