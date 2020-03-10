clear
clc
close all
%% add subfolders to path

folder = fileparts(which('final_toy_lgssm_comparison_OL.m'));
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
MC_iter = 500;

% state size of N4SID method (selected in file:
% 'final_toy_lgssm_comparison_statesize.m' with best RMSE in open loop)
nx = 2;

%% load test data

% get path
mydir = pwd;
idx = strfind(pwd, filesep);
path = mydir(1:idx(end)-1);
idx = strfind(path, filesep);
path = path(1:idx(end)-1);
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
rmse_OL_true = zeros(1,MC_iter);
rmse_OL_id = zeros(1,MC_iter);
std_id = zeros(1,MC_iter);
loglikeli = zeros(1,MC_iter);

% loop over everything
for i = 1:MC_iter
    
    fprintf('MC_Iter=%i\n',i)
    
    % get new data
    u_train = (rand(1, k_max_train) - 0.5) * 5;
    y_train = run_toy_lgssm(u_train,A,B,C,0.5,1);
    % get correct sizes
    u_train = u_train';
    y_train = y_train';
    
    % % identify model: PEM + N4SID
    % data = iddata(y_train, u_train);
    % opt = n4sidOptions('Focus','simulation');
    % init_sys = n4sid(data, nx, opt);
    % sys = pem(data,init_sys);
    % std_id(i) = sqrt(sys.NoiseVariance);
    
    % identify model: SSEST
    try
        data = iddata(y_train, u_train);
        opt = ssestOptions('Focus','simulation');
        sys_id = ssest(data,nx);
        std_id(i) = sqrt(sys_id.NoiseVariance);
    catch
        % get new data
        u_train = (rand(1, k_max_train) - 0.5) * 5;
        y_train = run_toy_lgssm(u_train,A,B,C,0.5,1);
        % get correct sizes
        u_train = u_train';
        y_train = y_train';
        % reidentify
        data = iddata(y_train, u_train);
        opt = ssestOptions('Focus','simulation');
        sys_id = ssest(data,nx);
        std_id(i) = sqrt(sys_id.NoiseVariance);
    end
    
    % test identified model in open loop
    y_test_OL_id = sim(sys_id, u_test);
    rmse_OL_id(i) = sqrt(mean((y_test-y_test_OL_id).^2));
    loglikeli(i) = get_LL(y_test,y_test_OL_id, std_id(i));
    if isnan(rmse_OL_id(i)) || isnan(loglikeli(i))
        stopvar = 1;
    end
    
    % test true model in OL
    y_test_OL_true = run_toy_lgssm(u_test',A,B,C,0,0)';
    rmse_OL_true(i) = sqrt(mean((y_test-y_test_OL_true).^2));
end

fprintf('\nmean RMSE OL identified: %2.4f\n',mean(rmse_OL_id))
fprintf('std RMSE OL identified: %2.4f\n',sqrt(var(rmse_OL_id)))
fprintf('LL OL identified: %2.4f\n',mean(loglikeli))

fprintf('\nmean RMSE OL true: %2.4f\n',mean(rmse_OL_true))
fprintf('std RMSE OL true: %2.4f\n',sqrt(var(rmse_OL_true)))

% save A,B,C and std in file
A = sys_id.A;
B = sys_id.B;
C = sys_id.C;
std = std_id(i);
yid = y_test_OL_id;
save('toy_identifiedsystem.mat','A','B','C','std','yid')

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

%%

function [LL] = get_LL(y_test,mu, std)

% number of data points
k_max = size(y_test,1);

% total LL
LL_tot = sum(-1/2 * log(2*pi*std^2) - 1/2 *1/std^2 * (y_test-mu).^2);

% LL per point
LL = LL_tot / k_max;

end