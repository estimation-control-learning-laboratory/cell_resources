function [P_k_1, AB_k_1, z_k] = Recursive_Online_DMD(x, y, P_k, Theta_k, lambda)


gamma_k = 1 / (1 + x' * P_k * x);
P_k_1 = P_k - gamma_k * (P_k * x) * (x' * P_k);

% Recursive update for Theta_{k+1}
AB_k_1 = Theta_k + gamma_k * (y - Theta_k * x) * (x' * P_k);

z_k = norm(y - Theta_k * x);