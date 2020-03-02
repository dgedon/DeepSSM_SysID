clear
clc
close all
%% add subfolders to path

folder = fileparts(which('final_toy_lgssm_comparison.m'));
addpath(genpath(folder));


%% True states

% state space matrices
A = [0.7, 0.8;
    0, 0.1];
B = [-1; 0.1];
C = [1 0];
    
%% Init

% number of data points 
k_max_test = 5000;
k_max_train = 2000;

% Monte Carlo iterations
MC_iter = 1;

% state size of N4SID method (selected in file:
% 'final_toy_lgssm_comparison_statesize.m' with best RMSE in open loop)
nx = 2;

%% load test data

% get path
mydir = pwd;
idx = strfind(pwd, filesep);
path = mydir(1:idx(end)-1);
path = strcat(path, '/data/Toy_LGSSM/');

% load the data
file_name = 'u_test.npy';
u_test = readNPY(strcat(path, file_name));
file_name = 'y_test.npy';
y_test = readNPY(strcat(path, file_name));
y_test = y_test + randn(size(y_test));

% select correct dimensions
u_test = u_test(1:k_max_test);
y_test = y_test(1:k_max_test);  %run_toy_lgssm(u_test')';

%% MAIN

% allocation
rmse_OL = zeros(1,MC_iter);
rmse_KF = zeros(1,MC_iter);


% loop over everything
for i = 1:MC_iter
    
%     fprintf('MC_Iter=%i\n',i)

    % get new data
    u_train = (rand(1, k_max_train) - 0.5) * 5;
    y_train = run_toy_lgssm(u_train,A,B,C,0.5,1);
    % get correct sizes
    u_train = u_train';
    y_train = y_train';

    % identify model
    data = iddata(y_train, u_train);
    opt = n4sidOptions('Focus','simulation');
    init_sys = n4sid(data, nx, opt);
    sys = pem(data,init_sys);

    % test identified model in open loop
    y_test_OL = sim(sys, u_test);
    rmse_OL(i) = sqrt(mean((y_test-y_test_OL).^2));
    rmseOLidentified = sqrt(mean((y_test-y_test_OL).^2))
    
    y_test_OL_true = run_toy_lgssm(u_test',A,B,C,0,0)';
    rmse_OL_true(i) = sqrt(mean((y_test-y_test_OL_true).^2));
    rmseOLtrue = sqrt(mean((y_test-y_test_OL_true).^2)) 

    % test KF with identified model
    Q = 0.5 * eye(2);
    R = 1;
    y_test_KF_sys = run_kalman_filter(sys.A, sys.B, sys.C, Q, R, u_test', y_test');
    y_test_KF_sys = y_test_KF_sys';
    rmseKFidentified = sqrt(mean((y_test-y_test_KF_sys).^2))
    
    y_test_KF = run_kalman_filter(A, B, C, Q, R, u_test', y_test');
    y_test_KF = y_test_KF';
    %rmse_KF(i) = 
    rmseKFtrue = sqrt(mean((y_test-y_test_KF).^2))
    
    

    
    stopvar = 1;
end

% fprintf('\nmean RMSE OL: %2.4f\n',mean(rmse_OL))
% fprintf('std RMSE OL: %2.4f\n',sqrt(var(rmse_OL)))

%% function to get new data

function [y] = run_toy_lgssm(u,A,B,C,sigma_state,sigma_out)

    % get length of input
    k_max = size(u,2);

    % size of variables
    n_u = 1;
    n_y = 1;
    n_x = 2;

    % allocation
    x = zeros(n_x, k_max + 1);
    y =zeros(n_y, k_max);

    % run over all time steps
    for k = 1:k_max
        x(:, k + 1) = A * x(:, k) + B * u(:, k) + sigma_state * randn(n_x,1);
        y(:, k) = C*x(:, k) + sigma_out * randn(n_y,1);
    end

end