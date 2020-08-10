clc
clear

addpath 'C:/Program Files/Mosek/9.2/toolbox/R2015a'
addpath(genpath('D:/Dropbox (Sydney Uni)/Matlab/toolboxes/YALMIP-master'));

% addpath '~/mosek/9.1/toolbox/r2015a';
% addpath(genpath('~/Dropbox (Sydney Uni)/Matlab/toolboxes/yalmip_master/YALMIP-master'));

yalmip('clear')
% rng(1)

%Params
n = 100; % network size. Try 500
m = 1; % number of inputs
p = 1; % number of outputs
q = 100;

phi = @(x) tanh(x);

%parameters for W
alpha = 1.2; % try 3.0. scaling factor controlling maximum singular value of W
connectivity = 0.1; %0.1
washout = 200;

% Test on Silverbox
[u_train, y_train, u_test, y_test] = load_silverbox();



%Train an echo state network on the data.
ESN = echo_state_network_IQC_BM(n, m, p, q, connectivity, alpha, phi, washout);
ESN = ESN.ESN_Init_IEE();
ESN = ESN.train(u_train, y_train);

train_perf = ESN.test(u_train, y_train)
test_perf = ESN.test(u_test, y_test)


%Train an echo state network on the data.
ESN = echo_state_network(n, m, p, connectivity, alpha, phi, washout);
ESN = ESN.train(u_train, y_train);

train_perf = ESN.test(u_train, y_train)
test_perf = ESN.test(u_test, y_test)


%Train an echo state network with ofb .
ESN = echo_state_network_ofb(n, m, p, connectivity, alpha, phi, washout);
ESN = ESN.train(u_train, y_train);

train_perf = ESN.test(u_train, y_train)
test_perf = ESN.test(u_test, y_test)


% Plot spectra of W
plot(real(eigs(W, n)), imag(eigs(W, n)), '.')

