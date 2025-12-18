function C = compute_adaptive_jacobian(x, net, mu_X, sigma_X)
% central difference
%
% Inputs:
%   x       - 2x1 input vector (unnormalized)
%   net     - trained neural network object
%   mu_X    - mean of training inputs (2x1)
%   sigma_X - std dev of training inputs (2x1)
%
% Output:
%   C - 1x2 Jacobian (C matrix) at input x

    eps = 1e-7;
    x_n = (x - mu_X) ./ sigma_X;  % Normalize input

    C = zeros(1, 2);
    for i = 1:2
        dx = zeros(2, 1);
        dx(i) = eps;

        y_plus = net(x_n + dx);
        y_minus = net(x_n - dx);

        C(1, i) = (y_plus - y_minus) / (2 * eps);
    end

    % Chain rule: account for normalization scaling
    C = C ./ sigma_X';
end
