classdef echo_state_network_ofb
    %ESN Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        n
        m
        p
        phi
        W
        Win
        Wout
        Wofb
        bias
        washout
    end
    
    methods
        function obj = echo_state_network_ofb(n, m, p, connectivity, alpha, phi, washout)
            obj.n = n;
            obj.m = m;
            obj.p = p;
            obj.washout = washout;

            W1 = (sprandn(n, n, connectivity));
            sigmas = svds(W1);
            obj.W = alpha * W1 / sigmas(1);
            obj.Win = sprandn(n, m, 1);
            obj.bias = randn(n, 1);
            obj.phi = phi;
            obj.Wout = zeros(p, n);
            obj.Wofb = sprandn(n, p, 1) / sqrt(n);
            
        end
        
        function obj = train(obj, u, y)

            %Simulate ESN dynamics on training data
            X = zeros(obj.n, size(y, 2));
            for tt = 2:size(y, 2)
               X(:, tt) = obj.phi(obj.W*X(:, tt-1) + obj.Win*u(:, tt) + obj.Wofb*y(:, tt-1) + obj.bias);
            end
            Yhat = y(:, obj.washout:end);
            Xhat = X(:, obj.washout:end);
            obj.Wout = Yhat*pinv(Xhat);

        end
        
        function [perf] = test(obj, u, y)

            W2 = obj.W + obj.Wofb*obj.Wout;
            %Simulate ESN dynamics on training data
            X = zeros(obj.n, size(y, 2));
            for tt = 2:size(y, 2)
                if tt < obj.washout
                    X(:, tt) = obj.phi(obj.W*X(:, tt-1) + obj.Win*u(:, tt) + obj.Wofb*y(:, tt-1) + obj.bias);
                else
                    X(:, tt) = obj.phi( W2*X(:, tt-1) + obj.Win*u(:, tt) + obj.bias);
                end
            end

            Ytilde = y(:, obj.washout:end);
            Xhat = X(:, obj.washout:end);

            Yhat = obj.Wout*Xhat;

            perf.NRMSE = norm(Ytilde - Yhat) ./ norm(Ytilde);
            perf.Yhat = Yhat;
            perf.error = obj.Wout*X - y;

            end

    end
end

