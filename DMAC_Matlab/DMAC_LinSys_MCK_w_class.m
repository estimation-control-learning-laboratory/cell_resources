clc; clear;
close all;
addpath DMAC_Functions\

% Simulation parameters
randn('state',2)

%% System parameters

n_steps = 1000;
n_x = 2;
n_u = 1;
n_y = 1;

m = 1;    % 1
ks = 2;   % 2
c = 0.5;  % 0.5
dt = 0.1; % 0.1

% System Dynamics
A_true = [0 1; -ks/m -c/m];
B_true = [0; 1/m];
C_true = [1 0];

sys_c = ss(A_true, B_true, eye(n_x), 0);
sys_d = c2d(sys_c, dt, 'tustin');

%% DMAC parameters

opts_.n_x = n_x;
opts_.n_u = n_u;
opts_.n_y = n_y;
opts_.Ts = dt;
opts_.C = [1, 0];

% RLS parameters
opts_.p0 = 1e-1;
%opts_.rls_lambda = 0.995;
opts_.rls_lambda = 1.0;

% LQR weight matrices
opts_.lqr_Q = 1*eye(n_x + n_y);  % Augmented state with integral state n_x + n_u
% opts_.lqr_Q(n_x + n_y, n_x + n_y) = 1;
opts_.lqr_R = 100*eye(n_u);

% Standard deviation of excitation signal
opts_.randn_std = 1e-2;

% DMAC controller object

DMAC_controller = DMAC_w_int_v0(opts_);

%% Vector initialization
ICs = randn(n_x, 1);
X = zeros(n_x, n_steps);
X(:,1) = ICs;
Y = zeros(n_x, n_steps);
Y(:,1) = C_true*ICs;
U = zeros(n_u, n_steps);
U(:,1) = opts_.randn_std*randn(1);

Theta_error = zeros(1, n_steps);
Theta_vec = zeros(n_x*(n_x+n_u), n_steps);

X_Aug = zeros(n_x + 1, n_steps);  % Augmented States storage

% Reference signal for tracking
r = 1;

% Initialize Pk and AB for RLS
AB_est_k = zeros(n_x, n_x + n_u);

%% Simulation Loop

for k = 2:n_steps

    X(:, k) = [sys_d.A sys_d.B]*[X(:, k-1); U(:, k-1)];
    Y(:, k) = C_true*X(:, k);
    
    [U(:, k), AB_est_k] = DMAC_controller.oneStep(X(:, k), X(:, k-1), U(:, k-1), r);

    % Log the step
    disp(['step ' num2str(k)]);
    Theta_error(k) = norm(AB_est_k - [sys_d.A sys_d.B]);
    Theta_vec(:,k) = AB_est_k(:);
end

%% Plot results
figure
set(gcf, 'position', [200, 100, 800, 500])
t = tiledlayout(2, 2);
t.TileSpacing = 'compact';
t.Padding = 'compact';

nexttile(1)
ref_sig = r*ones(1, n_steps);
plot(ref_sig, 'k--', 'LineWidth', 2)
hold on
plot(X(1,:), 'b', 'LineWidth', 3)
hold off
set(gca, 'fontsize', 16);
set(gca, 'xticklabel', {[]})
xlabel('a)', 'Interpreter', 'latex', 'fontsize', 22)
ylabel('$y_k$', 'Interpreter', 'latex', 'fontsize', 22)
legend('Reference','')
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
yticks(10.^(-5:1:5))
xlabel({'c)' 'step ($k$)'}, 'Interpreter', 'latex', 'fontsize', 22)
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

disp('True Ad:')
disp(sys_d.A)
disp('True Bd:')
disp(sys_d.B)
disp('Estimated [A B]:')
disp(AB_est_k)

