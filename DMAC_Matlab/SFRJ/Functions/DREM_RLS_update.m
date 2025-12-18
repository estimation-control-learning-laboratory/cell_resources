function [P_k_1, AB_est_1, z_k] = DREM_RLS_update(x_bar_k, y_k, P_k, AB_est, lambda)
    % Number of channels for DREM
    n_channels = 2;
    gamma = eye(length(x_bar_k));
    M = zeros(n_channels, length(x_bar_k));
    for i = 1:n_channels
        M(i, :) = (gamma^(i - 1)) * x_bar_k;
    end

    % Error calculation for each channel
    error = zeros(n_channels, length(y_k));
    for i = 1:n_channels
        y_pred = AB_est * M(i, :)';
        error(i, :) = (y_k - y_pred)';
    end

    % Aggregate error and compute effective regressor for DREM
    z_k = sum(M .* error, 1)';   % Aggregated error as a column vector
    phi_drem = sum(M, 1)';       % Effective regressor as a column vector

    % RLS update with DREM regressor
    gain = (P_k * phi_drem) / (lambda + phi_drem' * P_k * phi_drem);
    AB_est_1 = AB_est + gain * z_k';
    P_k_1 = (P_k - gain * phi_drem' * P_k) / lambda;
end