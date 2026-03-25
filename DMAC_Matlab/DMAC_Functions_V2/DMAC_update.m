%% ========================================================================
%  DMAC PARAMETER AND CONTROL UPDATE (RLS + LQR)
%  ========================================================================
%
%  DESCRIPTION:
%  ------------------------------------------------------------------------
%  This file implements the core update step for Dynamic Mode Adaptive 
%  Control (DMAC). At each time step, the algorithm performs:
%
%  (1) ONLINE SYSTEM IDENTIFICATION
%      - Updates the local linear model using matrix Recursive Least Squares
%        (RLS) with forgetting factor:
%            xi_k ≈ Theta_k [xi_{k-1}; u_{k-1}]
%
%      - The update uses the causal data pair:
%            (phi_{k-1}, xi_k)
%
%  (2) MODEL EXTRACTION
%      - Extracts the estimated system matrices:
%            Theta_k = [A_est  B_est]
%
%  (3) AUGMENTED SYSTEM CONSTRUCTION
%      - Forms an augmented system with integral action for reference tracking:
%            [xi_{k+1}] = [ A_est   0 ] [xi_k] + [ B_est ] u_k
%            [ q_{k+1}]   [ -C     I ] [ q_k]   [   0   ]
%
%  (4) CONTROL SYNTHESIS
%      - Computes a full-state feedback gain via discrete-time LQR:
%            u_k = K_k [xi_k; q_k]
%
%      - If the estimated system is not controllable or the Riccati solver
%        fails, the previous gain is retained.
%
%  ------------------------------------------------------------------------
%  INPUTS:
%  ------------------------------------------------------------------------
%  phi_km1    : Regressor at step k-1, phi_{k-1} = [xi_{k-1}; u_{k-1}]
%  xi_k       : Measured state at step k
%  Theta_km1  : Previous parameter estimate
%  P_km1      : Previous RLS covariance matrix
%  K_km1      : Previous control gain
%  dmac       : Structure containing DMAC parameters:
%               - lambda : forgetting factor
%               - Q, R   : LQR weighting matrices
%               - C_xi   : output matrix for xi
%               - lxi    : state dimension
%               - lu     : input dimension
%               - ly     : output dimension
%
%  ------------------------------------------------------------------------
%  OUTPUTS:
%  ------------------------------------------------------------------------
%  Theta_k    : Updated parameter estimate
%  P_k        : Updated covariance matrix
%  K_k        : Updated control gain
%
%  ------------------------------------------------------------------------
%  KEY PROPERTIES:
%  ------------------------------------------------------------------------
%  - Fully data-driven (no prior model required)
%  - Identifies physically meaningful state-space representation
%  - Supports adaptive control under time-varying dynamics
%  - Uses causal RLS formulation suitable for real-time implementation
%
%  ------------------------------------------------------------------------
%  AUTHOR:
%  ------------------------------------------------------------------------
%  Parham Oveissi, PhD Candidate
%  Ankit Goel, Assistant Professor
%  Mechanical Engineering
%  University of Maryland, Baltimore County (UMBC)
%
%  ------------------------------------------------------------------------
%  DATE:
%  ------------------------------------------------------------------------
%  March 2026
%
%  ------------------------------------------------------------------------
%  CHANGE LOG:
%  ------------------------------------------------------------------------
%  v1.0  (Mar 2026) - Initial implementation
%        - Matrix RLS-based identification
%        - LQR-based control synthesis
%
%  v1.1  (Mar 2026) - Robustness improvements
%        - Controllability check added
%        - Safe fallback to previous gain
%        - Improved dimension handling
%
%  ------------------------------------------------------------------------
%  NOTES:
%  ------------------------------------------------------------------------
%  - Requires Control System Toolbox (idare, ctrb)
%  - Assumes Theta = [A B] structure
%  - Integral dynamics assumed unit-step unless dt is explicitly included
%
%  ========================================================================
% 
% 

function [Theta_k, P_k, K] = DMAC_update(phi_km1, xi_k, Theta_km1, P_km1, K, dmac)

%==========================================================================
% DMAC_update
%--------------------------------------------------------------------------
% Performs one Dynamic Mode Adaptive Control (DMAC) update step:
%   1) updates the identified model using matrix RLS
%   2) computes the new feedback gain from the estimated model
%
% INPUTS:
%   phi_km1    : regressor at step k-1, [xi_{k-1}; u_{k-1}]
%   xi_k       : measured state at step k
%   Theta_km1  : previous parameter estimate
%   P_km1      : previous RLS covariance matrix
%   K_km1      : previous control gain
%   dmac       : DMAC parameter structure
%
% OUTPUTS:
%   Theta_k    : updated parameter estimate
%   P_k        : updated RLS covariance matrix
%   K_k        : updated control gain
%==========================================================================

    % Estimate Ak, Bk in Theta_k
    [P_k, Theta_k]  = MatrixRLS(phi_km1, xi_k, P_km1, Theta_km1, dmac.lambda);

    % Update control gain
    K = compute_DMAC_control(Theta_k, dmac, K);

end


function K = compute_DMAC_control(Theta_km1, dmac, K)
%==========================================================================
% Computes the DMAC feedback gain using the current identified model.
% If the identified model is not suitable for control synthesis, the
% previous gain is retained.
%==========================================================================
    [A_est, B_est] = extract_A_B_from_Theta(Theta_km1, dmac);
    
    [A_est_aug, B_est_aug] = generate_augmented_A_B_DMD(A_est, B_est, dmac.C_xi, dmac);
    
    
    if rank(ctrb(A_est_aug, B_est_aug)) ~= dmac.lxi + dmac.lu
        disp('Not Controllable!!')
    else
        [~, K, ~] = idare(A_est_aug, B_est_aug, dmac.Q, dmac.R);
        K = -K;
    end

end

function [A, B] = extract_A_B_from_Theta(Theta, dmac)
%==========================================================================
% Extract A and B from Theta = [A B]
%==========================================================================
    A = Theta(:, 1:dmac.lxi);
    B = Theta(:, dmac.lxi + 1:end);
end

function [Aa, Ba] = generate_augmented_A_B_DMD(A, B, C, dmac)
%==========================================================================
% Build augmented system for integral-action tracking:
%
%   xi_{k+1} = A xi_k + B u_k
%   q_{k+1}  = q_k + (r_k - y_k) dt
%
% Ignoring the reference input in the regulator design, the augmented
% dynamics are represented as
%
%   [xi_{k+1}]   [ A   0 ] [xi_k]   [ B ] u_k
%   [ q_{k+1}] = [-C   I ] [ q_k] + [ 0 ]
%==========================================================================
Aa = [A, zeros(dmac.lxi,dmac.ly); -C eye(dmac.ly)];

Ba = [B; zeros(dmac.ly, dmac.lu)];
end
