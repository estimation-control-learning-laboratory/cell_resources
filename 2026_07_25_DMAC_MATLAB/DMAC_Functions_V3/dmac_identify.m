function [Theta_k, P_k] = dmac_identify(xi_km1, u_km1, xi_k, Theta_km1, P_km1, lambda)
%DMAC_IDENTIFY Online RLS identification step of the DMAC model.
%
%   [Theta_k, P_k] = DMAC_IDENTIFY(xi_km1, u_km1, xi_k, Theta_km1, P_km1, lambda)
%   builds the regressor phi_{k-1} = [xi_{k-1}; u_{k-1}] and performs one
%   causal RLS update toward the target xi_k, i.e.
%       xi_k ~ Theta_k * phi_{k-1}
%
%   Arguments use the CALLER's absolute time indices (xi_km1 is the
%   measured state one step before xi_k) — this matches how the
%   simulation loop indexes its arrays, avoiding the off-by-one
%   relabeling that existed in the previous DMAC_compute_control.m.
%
%   Inputs:
%     xi_km1, u_km1 : measured state and input at step k-1
%     xi_k          : measured state at step k (RLS target)
%     Theta_km1     : previous parameter estimate, Theta = [A_est B_est]
%     P_km1         : previous RLS covariance matrix
%     lambda        : forgetting factor
%
%   Outputs:
%     Theta_k, P_k  : updated parameter estimate and covariance
%
%   See also: MATRIXRLS, DMAC_SYNTHESIZE_GAIN

phi_km1 = [xi_km1; u_km1];
[P_k, Theta_k] = MatrixRLS(phi_km1, xi_k, P_km1, Theta_km1, lambda);
end
