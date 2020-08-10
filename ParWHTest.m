clc
clear

addpath 'C:/Program Files/Mosek/9.2/toolbox/R2015a'
addpath(genpath('D:/Dropbox (Sydney Uni)/Matlab/toolboxes/yalmip_master/YALMIP-master'));

% rng(1)

%Params
n = 10; % network size. Try 500
m = 1; % number of inputs
p = 1; % number of outputs

% ESN_opts.phi = @(x) max(0, x);
% ESN_opts.phi = @(x) [tanh(x(1:end/2)); max(x(end/2+1:end), 0)];
ESN_opts.phi = @(x) tanh(x);
ESN_opts.washout = 200;

%parameters for W
alpha = 1.0; % try 3.0. scaling factor controlling maximum singular value of W
connectivity = 0.1; %0.1

% Test on Silverbox
[u_train, y_train, u_test, y_test] = load_ParWH();


%Train an echo state network on the data.
W = (sprandn(n, n, connectivity));
sigmas = svds(W);
W = alpha * W / sigmas(1);
Win = sprandn(n, m, 1);
Wofb = sprandn(n, m, 1) / sqrt(n);
bias = randn(n, 1);


% Train ESN
Wout = train_ESN(u_train, y_train, W, Win, bias, ESN_opts);

train_perf = testESN(u_train, y_train, W, Win, Wout, bias, ESN_opts)
test_perf = testESN(u_test, y_test, W, Win, Wout, bias, ESN_opts)

% Train ESN with output feedback
Wout = train_ofb_ESN(u_train, y_train, W, Win, Wofb, bias, ESN_opts);

train_perf = test_ofb_ESN(u_train, y_train, W, Win, Wofb, Wout, bias, ESN_opts)
test_perf = test_ofb_ESN(u_test, y_test, W, Win, Wofb, Wout, bias, ESN_opts)



% Plot spectra of W
plot(real(eigs(W, n)), imag(eigs(W, n)), '.')

