function K = dmac_synthesize_gain(Theta_k, dmac, K_prev)
%DMAC_SYNTHESIZE_GAIN Compute the DMAC feedback gain from the identified model.
%
%   K = DMAC_SYNTHESIZE_GAIN(Theta_k, dmac, K_prev)
%   extracts [A_est, B_est] from Theta_k = [A_est B_est] and computes a
%   full-state feedback gain:
%     - dmac.integrator == 'no'  : plain discrete LQR on (A_est, B_est)
%     - dmac.integrator == 'yes': discrete LQR (via idare) on the
%       integral-augmented system built from (A_est, B_est, dmac.Cy_xi)
%
%   If the identified model is not controllable (or the augmentation is
%   not controllable, or the Riccati solver fails), the previous gain
%   K_prev is retained and a warning is printed.
%
%   Requires the Control System Toolbox (ctrb, lqr, idare).
%
%   See also: DMAC_IDENTIFY, DMAC_CONTROL_LAW

[A_est, B_est] = extract_A_B_from_Theta(Theta_k, dmac);

K = K_prev;

if strcmp(dmac.integrator, 'no')
    Q = norm(dmac.Q) * eye(size(A_est));
    R = norm(dmac.R) * eye(size(B_est, 2));

    if cond(ctrb(A_est, B_est)) > 1e10
        disp('dmac_synthesize_gain: model not controllable, retaining previous gain.');
    else
        K = -lqr(A_est, B_est, Q, R);
    end
else
    [A_aug, B_aug] = generate_augmented_A_B_DMD(A_est, B_est, dmac.Cy_xi, dmac);

    if rank(ctrb(A_aug, B_aug)) ~= dmac.lxi + dmac.lu
        disp('dmac_synthesize_gain: augmented model not controllable, retaining previous gain.');
    else
        [~, K_aug, ~] = idare(A_aug, B_aug, dmac.Q, dmac.R);
        K = -K_aug;
    end
end
end


function [A, B] = extract_A_B_from_Theta(Theta, dmac)
%EXTRACT_A_B_FROM_THETA Split Theta = [A B] into its blocks.
A = Theta(:, 1:dmac.lxi);
B = Theta(:, dmac.lxi + 1:end);
end


function [Aa, Ba] = generate_augmented_A_B_DMD(A, B, Cy_xi, dmac)
%GENERATE_AUGMENTED_A_B_DMD Build the integral-action augmented system.
%
%   Given the identified partial-state model
%       xi_{k+1} = A xi_k + B u_k
%   and the integrator state
%       q_{k+1}  = q_k + (r_k - y_k),   y_k = Cy_xi * xi_k
%   (ignoring the reference input for regulator design), the augmented
%   dynamics are
%       [xi_{k+1}]   [ A       0 ] [xi_k]   [ B ] u_k
%       [ q_{k+1}] = [-Cy_xi   I ] [ q_k] + [ 0 ]
%
%   NOTE: this uses dmac.Cy_xi (maps xi_k -> y_k), not a plant-side C
%   matrix — the two are validated to be consistent in
%   VALIDATE_DMAC_CONFIG at start-up.

Aa = [A,               zeros(dmac.lxi, dmac.ly);
      -Cy_xi,          eye(dmac.ly)];
Ba = [B; zeros(dmac.ly, dmac.lu)];
end
