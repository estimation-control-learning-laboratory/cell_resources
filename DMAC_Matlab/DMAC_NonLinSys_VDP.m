clc; clear; close all;
addpath DMAC_Functions\

% Simulation parameters
randn('state',2)

steps = 1000;
sys_dim = 2;
mu = 1; % Nonlinearity parameter for Van der Pol system

dt = 0.1;

% Initialization
X = zeros(sys_dim, steps);
Y = zeros(sys_dim, steps);
U = zeros(1, steps);
% S = 0;
Theta_error = zeros(1, steps);
Theta_vec = zeros(sys_dim*(sys_dim+1), steps);
K = zeros(1, sys_dim + 1);
x_int = 0;
X_Aug = zeros(sys_dim + 1, steps);  % Augmented States storage

% LQR weight matrices
Q = 1*eye(sys_dim + 1);  % Augmented state with integral state 2+1
% Q(sys_dim+1, sys_dim+1) = 1;
R = 1;
E = eye(sys_dim + 1); % Augmented for integral action 2+1
Z = zeros(sys_dim + 1, 1);

% System Dynamics

C = [1, 0];

% Estimation Parameters
R0 = 1e2*eye(sys_dim + 1); % [x; u]: sys_dim + inputdim(=1)
lambda = 0.995;

% Initial Conditions
ICs = randn(sys_dim, 1);
% ICs = [-2; -4];
x = ICs;

% Reference signal for tracking
r = 1;

% Initialize Pk and AB for RLS
AB_est = zeros(sys_dim, sys_dim + 1);
P_k = inv(R0);


for k = 1:steps
    X(:, k) = x;

    % Tracking error
    error = r - x(1);  % Error between reference and state
    x_int = x_int + error * dt;  % Integral of the tracking error

    % Augmented state vector
    x_aug = [x; x_int];
    X_Aug(:, k) = x_aug;

    u = K*x_aug + 1e-3*randn + 0*sin(4*pi/100*k);
    U(k) = u;

    x = VanDerPol_Dynamics(x, u, mu, dt);

    y = x;
    Y(:, k) = y;

    % Update Ak and Pk recursively using the new snapshot (x_k, y_k)
    x_bar_k = [X(:, k); U(k)];

    y_k = Y(:, k);   % Current state y_k = x_{k+1}

    % Recursive update for P_{k+1} and Theta_{k+1}
    [P_k_1, AB_est_1, z_k(k)] = RLS_update_DMD(x_bar_k, y_k, P_k, AB_est, lambda);
    % [P_k_1, AB_est_1, z_k(k)] = DREM_RLS_update(x_bar_k, y_k, P_k, AB_est, lambda);

    % [P_k_1, AB_est_1, z_k(k)] = Recursive_Online_DMD(x_bar_k, y_k, P_k, AB_est, lambda);

    % Update A_k and P_k for the next iteration
    AB_est = AB_est_1;
    P_k = P_k_1;

    [A_est, B_est] = extract_A_B_DMD(AB_est, sys_dim);

    [A_est_aug, B_est_aug] = generate_augmented_A_B_DMD(A_est, B_est, C, sys_dim);

    if rank(ctrb(A_est_aug, B_est_aug)) ~= sys_dim + 1
        disp('Not Controllable!!')
    else
        [~, K, ~] = idare(A_est_aug, B_est_aug, Q, R, Z, E);
        K = -K;
    end

    % Log the step
    disp(['step ' num2str(k)]);
    % Theta_error(k) = norm(AB_est - [sys_d.A sys_d.B]);
    Theta_vec(:,k) = AB_est(:);
end

%% Plot results
figure
set(gcf, 'position', [200, 100, 800, 500])
t = tiledlayout(2, 2);
t.TileSpacing = 'compact';
t.Padding = 'compact';

nexttile(1)
plot(X_Aug(1,:), 'b', 'LineWidth', 3)
hold on
ref_sig = r*ones(1, size(X_Aug,2));
plot(ref_sig, 'k--', 'LineWidth', 2)
set(gca, 'fontsize', 16);
set(gca, 'xticklabel', {[]})
xlabel('a)', 'Interpreter', 'latex', 'fontsize', 22)
ylabel('$y_k$', 'Interpreter', 'latex', 'fontsize', 22)
legend('','Reference', location = 'best')
axis tight
grid on

nexttile(2)
plot(U, 'b', 'LineWidth', 3)
set(gca, 'fontsize', 16);
set(gca, 'xticklabel', {[]})
xlabel('b)', 'Interpreter', 'latex', 'fontsize', 22)
ylabel('$u_k$', 'Interpreter', 'latex', 'fontsize', 22)
axis tight
grid on

nexttile(3)
semilogy(abs(X(1,:)-r), 'b', 'LineWidth', 3)
set(gca, 'fontsize', 16);
xlabel({'c)' 'step ($k$)'}, 'Interpreter', 'latex', 'fontsize', 22)
yticks(10.^(-5:1:5))
ylabel('$|z_k|$', 'Interpreter', 'latex', 'fontsize', 22)
axis tight
grid on

nexttile(4)
plot(Theta_vec', 'LineWidth', 3)
set(gca, 'fontsize', 16);
xlabel({'d)' 'step ($k$)'}, 'Interpreter', 'latex', 'fontsize', 22)
ylabel('$\Theta_k$', 'Interpreter', 'latex', 'fontsize', 22)
axis tight
grid on


function x_next = VanDerPol_Dynamics(x, u, mu, dt)
% Continuous Van der Pol dynamics with control input
tspan = [0 dt];
f = @(t, x) [x(2);
    mu * (1 - x(1)^2) * x(2) - x(1) + u];
[~, x_out] = ode45(f, tspan, x);
x_next = x_out(end, :)';
end