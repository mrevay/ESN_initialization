classdef echo_state_network_IQC_BM
    %ESN Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        n
        m
        p
        q
        phi
        

        A
        Bw
        Bu
        Cv
        Cy
        Dvu
        bias
        
        washout
        connectivity
        alpha
    end
    
    methods
        function obj = echo_state_network_IQC_BM(n, m, p, q, connectivity, alpha, phi, washout)
            obj.n = n;
            obj.m = m;
            obj.p = p;
            obj.q = q;

            obj.washout = washout;
            obj.connectivity = connectivity;

            obj.alpha = alpha;
            obj.phi = phi; 
        end
        
              
        function [ineq, lmi] = LMI(obj, x)
            
            n = obj.n;
            m = obj.m;
            p = obj.p;
            q = obj.q;
            
            % Initial point
            index = 0;            
            E = reshape(x(index+1:index + obj.n^2), obj.n, obj.n);
            index = index + length(E(:));
            
            F = reshape(x(index+1:index + obj.n^2), obj.n, obj.n);
            index = index + length(F(:));
            
            Bw = reshape(x(index+1:index + obj.n*obj.q), obj.n, obj.q);
            index = index + length(Bw(:));
            
            P = reshape(x(index+1:index + obj.n*obj.q), obj.n, obj.n);
            index = index + length(P(:));
            
            lambda = x(index+1:index + obj.q);
            index = index + length(lambda(:));
                                    
            T = diag(lambda);
            M = [zeros(obj.q, obj.q), T; T, -2*T];

            C2hat = [obj.Cv, zeros(obj.q, obj.q)];
            D2hat = [zeros(obj.q, obj.n), eye(obj.q)];

            Gamma = [C2hat; D2hat];
            LMI_11 = [E + E' - P, zeros(obj.n, obj.q); zeros(obj.q, obj.n), zeros(obj.q, obj.q)] - ...
                      Gamma' * M * Gamma;
                  
            LMI_21 = [F, Bw];
            
            L = reshape(x(index+1:end), 2*obj.n+obj.q, []);
            
            lmi = [LMI_11, LMI_21'; LMI_21, P] - L*L';
            ineq = 0
        end
        
        function J = objective(obj, x, Atild)
            n = obj.n;
            m = obj.m;
            p = obj.p;
            q = obj.q;
            
            % Initial point
            index = 0;            
            E = reshape(x(index+1:index + obj.n^2), obj.n, obj.n);
            index = index + length(E(:));
            
            F = reshape(x(index+1:index + obj.n^2), obj.n, obj.n);
            index = index + length(F(:));
            
            Bw = reshape(x(index+1:index + obj.n*obj.q), obj.n, obj.q);
            index = index + length(Bw(:));
            
            P = reshape(x(index+1:index + obj.n*obj.q), obj.n, obj.n);
            index = index + length(P(:));
            
            lambda = x(index+1:index + obj.q);
            index = index + length(lambda(:));
            
            J = norm(obj.Cv*Bw - E*Atild, 'fro')^2
        end
        
        function obj = ESN_Init_IEE(obj)
            
            rank = 100;
            
            % Initial point
            E0 = eye(obj.n, obj.n);
            F0 = 0*eye(obj.n);
            Bw0 = zeros(obj.n, obj.q);
            Cv0 = eye(obj.n);
            P0 = eye(obj.n, obj.n);
            lambda0 = ones(obj.q, 1);
            
            obj.Cv = eye(obj.n);
            
            %Auxilliary variable for BM method
            L0 = zeros(2*obj.n + obj.q, rank);
            
            x0 = [E0(:); F0(:); Bw0(:); P0(:); lambda0(:); L0(:)];

            
            % Target distribution for Bw
            Atild = randn(obj.q, obj.q) / sqrt(obj.q);
            eval = eigs(Atild, 1);
            Atild = obj.alpha * Atild / abs(eval);
            
            % Create opbjective and LMI constraints
            objective = @(x) obj.objective(x, Atild)
            constraints = @(x) obj.LMI(x)
            
            % Create problem
            prob.objective = objective
            prob.x0 = x0
            prob.nonlcon = constraints
            prob.solver = 'fmincon'
            prob.options = optimoptions('fmincon')
            fmincon(prob)
            
            % Make explicit and store reservoir
            Eval = value(E);
            obj.A = Eval \ value(F);
            obj.Bw = Eval \ value(B1);
            obj.Dvu = randn(obj.q, obj.m);
            
            obj.bias = randn(obj.q, 1);
        end
        
        function obj = ESN_Init_LREE(obj)
            
            
        end
        
        function obj = ESN_Init_IEE_diag(obj)
            
            E = diag(sdpvar(obj.n, 1));
            F = 0*eye(obj.n);

            B1_sparsity = sprandn(obj.n, obj.q, obj.connectivity);
            [I, J] = find(B1_sparsity);
            numel = size(I, 1);
            
            B1 = sparse(I, J, sdpvar(numel, 1), obj.n, obj.q);
%             B1 = sdpvar(obj.n, obj.q, 'full');

%             obj.Cv = sprandn(obj.q, obj.n, obj.connectivity) / sqrt(obj.n);
            obj.Cv = eye(obj.n);
            obj.Bu = randn(obj.n, obj.m);

%             P = sdpvar(obj.n, obj.n);
            P = diag(sdpvar(obj.n, 1));
            lambda = sdpvar(obj.q, 1);

            T = diag(lambda);
            M = [zeros(obj.q, obj.q), T; T, -2*T];

            C2hat = [obj.Cv, zeros(obj.q, obj.q)];
            D2hat = [zeros(obj.q, obj.n), eye(obj.q)];

            Gamma = [C2hat; D2hat];
            LMI_11 = [E + E' - P, zeros(obj.n, obj.q); zeros(obj.q, obj.n), zeros(obj.q, obj.q)] - ...
                      Gamma' * M * Gamma;
                  
            LMI_21 = [F, B1];
            LMI = [LMI_11, LMI_21'; LMI_21, P];
            
            Constraints = [LMI >= 0, P>=1E-3*eye(obj.n), lambda>=0];
            
            % Target distribution for B1
            Atild = sprandn(obj.q, obj.q, obj.connectivity);
            eval = eigs(Atild, 1);
            Atild = obj.alpha * Atild / abs(eval);
            
%           % Add hoc objective that I just invented ...
            eps = obj.Cv*B1 - E*Atild;
            J = norm(eps, 'fro')^2;
            
            optimize(Constraints, J);
            
            % Make explicit and store reservoir
            Eval = value(E);
            obj.A = Eval \ value(F);
            obj.Bw = Eval \ value(B1);
            obj.Dvu = randn(obj.q, obj.m);
            
            obj.bias = randn(obj.q, 1);
        end
        
        function obj = train(obj, u, y)

            %Simulate ESN dynamics on training data
            X = zeros(obj.n, obj.p);
            for t = 2:size(y, 2)
               vt = obj.Cv*X(:, t-1) + obj.Dvu * u(:, t) + obj.bias;
               wt = obj.phi(vt);
               X(:, t) = obj.A * X(:, t-1) + obj.Bw*wt + obj.Bu * u(:, t);
            end
            Yhat = y(:, obj.washout:end);
            Xhat = X(:, obj.washout:end);
            obj.Cy = Yhat*pinv(Xhat);

        end
        
        function [perf] = test(obj, u, y)

            %Simulate ESN dynamics on training data
            X = zeros(obj.n, obj.p);
            for t = 2:size(y, 2)
               vt = obj.Cv*X(:, t-1) + obj.Dvu * u(:, t) + obj.bias;
               wt = obj.phi(vt);
               X(:, t) = obj.A * X(:, t-1) + obj.Bw*wt + obj.Bu * u(:, t);
            end

            Ytilde = y(:, obj.washout:end);
            Xhat = X(:, obj.washout:end);

            Yhat = obj.Cy*Xhat;

            perf.NRMSE = norm(Ytilde - Yhat) ./ norm(Ytilde);
            perf.Yhat = Yhat;
            perf.error = obj.Cy*X - y;

            end

    end
end

