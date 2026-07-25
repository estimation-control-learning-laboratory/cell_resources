function [P_k, Theta_k] = MatrixRLS(phi_km1, xi_k, P_km1, Theta_km1, lambda)

% Compute Gamma_k^{-1}
gamma_k_inv = 1 / (lambda + phi_km1' * P_km1 * phi_km1);

% Update P_k (this is SP_k in your LaTeX)
P_k = (1/lambda)*P_km1 ...
        - (1/lambda)*(P_km1 * phi_km1) * gamma_k_inv * (phi_km1' * P_km1);

% Update Theta_k (USES UPDATED P_k_1)
Theta_k = Theta_km1 ...
            + (xi_k - Theta_km1 * phi_km1) * (phi_km1' * P_k);

end