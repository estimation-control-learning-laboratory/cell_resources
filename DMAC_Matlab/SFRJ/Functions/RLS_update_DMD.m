function [P_k_1, Theta_k_1, z_k] = RLS_update_DMD(x, y, P_k, Theta_k, lambda)

% Recursive update for P_{k+1}
gamma_k_inv = 1 / (lambda + x' * P_k * x);
% disp(eig(x' * P_k * x))
P_k_1 = P_k/lambda - (1/lambda)*(P_k*x)*gamma_k_inv*(x' * P_k);

% Recursive update for Theta_{k+1}
Theta_k_1 = Theta_k + (y - Theta_k * x) * (x' * P_k);
z_k = norm(y - Theta_k * x);






% % Recursive update for P_{k+1}
% gamma_k_inv = 1 / (lambda + x * P_k * x');
% % disp(eig(x' * P_k * x))
% P_k_1 = P_k/lambda - (1/lambda)*(P_k*x')*gamma_k_inv*(x * P_k);
%
% % Recursive update for Theta_{k+1}
% Theta_k_1 = Theta_k + (P_k * x')*(y - x*Theta_k);
% z_k = norm(y - x*Theta_k);
