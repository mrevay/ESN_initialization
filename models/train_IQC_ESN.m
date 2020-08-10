function [Cy] = train_IQC_ESN(u, y, ESN_params, ESN_opts)

%Simulate ESN dynamics on training data
A = ESN_params.A;
Bw = ESN_params.Bw;
Bu = ESN_params.Bu;
Cv = ESN_params.Cv;
Dvu = ESN_params.Dvu;
b = ESN_params.b;

phi = ESN_params.phi;

n = size(W, 1);
X = zeros(n, size(y, 2));
for t = 2:size(y, 2)
   vt = Cv*X(:, t-1) + Dvu * u(:, t) + b;
   wt = phi(vt);
   X(:, t) = A * X(:, t-1) + Bw*wt + Bu * u(:, t);
end
Yhat = y(:, ESN_opts.washout:end);
Xhat = X(:, ESN_opts.washout:end);
Cy = Yhat*pinv(Xhat);

end

