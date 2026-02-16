classdef DMAC_w_int_v0 < handle
    properties
        n_x             % State dimension
        n_u             % Input dimension
        n_y             % Output dimension
        Ts              % 
        C               % Output matrix for integrator term
        z_k             % Output error relative to reference vector
        q_k             % Integrator state
        rls_Pk          % RLS covariance matrix
        rls_Theta_k     % RLS optimization vector
        rls_lambda      % RLS forgetting factor
        rls_lambda_inv  % Inverse of the RLS forgetting factor
        lqr_Q           % State + integrator cost for LQR
        lqr_R           % Input cost for LQR
        randn_std       % Standard deviation of excitation signal
    end
    methods
        function obj = DMAC_w_int_v0(opts_)
            obj.n_x = opts_.n_x;
            obj.n_u = opts_.n_u;
            obj.n_y = opts_.n_y;
            obj.Ts = opts_.Ts;
            obj.C = opts_.C;
            obj.z_k = zeros(obj.n_y, 1);
            obj.q_k = zeros(obj.n_y, 1);
            obj.rls_Pk = opts_.p0*eye(obj.n_x + obj.n_u);
            obj.rls_Theta_k = zeros(obj.n_x, obj.n_x + obj.n_u);
            obj.rls_lambda = opts_.rls_lambda;
            obj.rls_lambda_inv =1/obj.rls_lambda;
            obj.lqr_Q = opts_.lqr_Q;
            obj.lqr_R = opts_.lqr_R;
            obj.randn_std = opts_.randn_std;
        end

        function [u_k, Theta_k] = oneStep(obj, xi_k, xi_km1, u_km1, r_k)
            
            Phi_km1 = [xi_km1; u_km1];
            [obj.rls_Theta_k, obj.rls_Pk, ~] = obj.rls_update(obj.rls_Theta_k, obj.rls_Pk, xi_k, Phi_km1, obj.rls_lambda, obj.rls_lambda_inv);
            
            [A_est, B_est] = obj.extract_A_B_from_Theta(obj.rls_Theta_k, obj.n_x, obj.n_u);
            [A_est_aug, B_est_aug] = obj.augment_A_B_with_integrator_state(A_est, B_est, obj.C, obj.n_x, obj.n_u, obj.n_y);
            
            % Check controllability
            [T, A_controllable, B_controllable, is_ctrl] = obj.kalman_decomposition(A_est_aug, B_est_aug);
            if (is_ctrl >= 0.5)
                %disp('System is fully controllable.');
                [~, K, ~] = idare(A_controllable, B_controllable, obj.lqr_Q, obj.lqr_R);
                K = -K;
            else
                disp('Not fully controllable! Using controllable part for control design.');

                % Use controllable part for controller design
                [~, K_controllable, ~] = idare(A_controllable, B_controllable, obj.lqr_Q, obj.lqr_R);
        
                % Transform the gain back to the original coordinates
                K = -K_controllable * T;
            end
            
            obj.z_k = r_k - obj.C*xi_k;
            obj.q_k = obj.q_k + obj.z_k*obj.Ts;
            x_aug_k = [xi_k; obj.q_k];
            u_k = K*x_aug_k + obj.randn_std*randn(1);
            Theta_k = obj.rls_Theta_k;
        end
        
        function [A, B] = extract_A_B_from_Theta(obj, Theta, n_x, n_u)
            A = Theta(:, 1:n_x);
            B = Theta(:, (n_x + 1):(n_x + n_u));
        end

        function [Aa, Ba] = augment_A_B_with_integrator_state(obj, A, B, C, n_x, n_u, n_y)
            Aa = [A, zeros(n_x, n_y); -C eye(n_y)];
            Ba = [B; zeros(n_y, n_u)];
        end

        function [Theta_km1, P_k, z_k] = rls_update(obj, Theta_km1, P_km1, y_k, Phi_km1, lambda, lambda_inv)
            % Recursive update for P_{k+1}
            gamma_k_inv = 1 / (lambda + Phi_km1.' * P_km1 * Phi_km1);
            P_k = lambda_inv*P_km1 - lambda_inv*P_km1*(Phi_km1*((gamma_k_inv*Phi_km1.') * P_km1));
            
            % Recursive update for Theta_{k+1}
            Theta_km1 = Theta_km1 + (y_k - Theta_km1 * Phi_km1) * (Phi_km1' * P_km1);
            z_k = norm(y_k - Theta_km1 * Phi_km1);
        end

        function [T, A_controllable, B_controllable, is_ctrl] = kalman_decomposition(obj, A, B)
            n = size(A, 1);
            
            % Step 1: Compute controllability matrix
            Wc = ctrb(A, B);
            
            % Step 2: Check rank of controllability matrix
            if rank(Wc) < n
                disp('Rank deficient system. Performing Kalman decomposition.');
        
                % Step 3: Perform Singular Value Decomposition (SVD) on the controllability matrix
                [U, ~, ~] = svd(Wc);
                
                % Step 4: Define the transformation matrix T (using the SVD components)
                T = U';
                
                % Step 5: Compute the decomposed system matrices
                A_controllable = T(1:n, :) * A * T(1:n, :)';
                B_controllable = T(1:n, :) * B;
                
                %A_uncontrollable = T(n+1:end, :) * A * T(n+1:end, :)';
                %B_uncontrollable = T(n+1:end, :) * B;
                is_ctrl = 0.0;
            else
                disp('System is fully controllable.');
                
                % If the system is fully controllable, the decomposition is trivial
                T = eye(n);  % Identity transformation, no need to decompose
                A_controllable = A;
                B_controllable = B;
                is_ctrl = 1.0;
                %A_uncontrollable = [];
                %B_uncontrollable = [];
            end
        end
    end
end