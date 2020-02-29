% Test which size of the state gives the highest RMSE.

clear
clc
close all
%% add subfolders to path

folder = fileparts(which('final_toy_lgssm_comparison.m'));
addpath(genpath(folder));

%% Init

% number of data points 
k_max_test = 5000;
k_max_train = 2000;
k_max_val = 2000;

% Monte Carlo iterations
MC_iter = 10;

% state size of N4SID method
nx_max = 20;

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

% select correct dimensions
u_test = u_test(1:k_max_test);
y_test = y_test(1:k_max_test);

%% MAIN

rmse = zeros(MC_iter,nx_max);

for i = 1:MC_iter
    fprintf('MC_iter=%i\n',i)

    
    % loop over everything
    for nx = 1:nx_max

        fprintf('nx=%i\n',nx)

        % get new data
        u_train = (rand(1, k_max_train) - 0.5) * 5;
        y_train = run_toy_lgssm(u_train);
        u_val = (rand(1, k_max_val) - 0.5) * 5;
        y_val = run_toy_lgssm(u_val);

        % get correct sizes
        u_train = u_train';
        y_train = y_train';
        u_val = u_val';
        y_val = y_val';

        % identify model
        data = iddata(y_train, u_train);
        opt = n4sidOptions('Focus','simulation');
        init_sys = n4sid(data, nx, opt);
        sys = pem(data,init_sys);

        % test identified model in open loop
        % x = zeros(nx,1);
        % y_val_sim = zeros(size(u_val,1),1);
        % for k = 1:size(u_val,1)
        %     y_val_sim(k,1) = sys.C * x;
        %     x = sys.A * x + sys.B * u_val(k);
        % end
        y_val_sim = sim(sys,u_val);
        
        % compute RMSE of validation output
        rmse(i,nx) = sqrt(mean((y_val-y_val_sim).^2));

    end
end

plot(mean(rmse))

%% function to get new data

function [y] = run_toy_lgssm(u)

    % define process noise
    sigma_state = sqrt(0.25);
    sigma_out = 1;

    % get length of input
    k_max = size(u,2);

    % size of variables
    n_u = 1;
    n_y = 1;
    n_x = 2;

    % state space matrices
    A = [0.7, 0.8;
         0, 0.1];
    B = [-1; 0];
    C = [1 0];

    % allocation
    x = zeros(n_x, k_max + 1);
    y =zeros(n_y, k_max);

    % run over all time steps
    for k = 1:k_max
        x(:, k + 1) = A * x(:, k) + B * u(:, k) + sigma_state * randn(n_x,1);
        y(:, k) = C*x(:, k) + sigma_out * randn(n_y,1);
    end

end