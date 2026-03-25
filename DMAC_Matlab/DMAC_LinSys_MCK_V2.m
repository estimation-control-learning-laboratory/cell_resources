clc; clear;
close all;
addpath DMAC_Functions_V2\

% Simulation parameters
randn('state',2)

N = 1000;
lx = 2;
m = 1;    % 1
ks = 2;   % 2
c = 0.5;  % 0.5
dt = 0.1; % 0.1

% Initialization
X_k = zeros(lx, N);
U = zeros(1, N);
Theta_vec = zeros(lx*(lx+1), N);
K_aug = [0, 0, 0];
q_k = 0;

% LQR weight matrices
Q = 1*eye(lx + 1);  % Augmented state with integral state 2+1
R = 1;

% System Dynamics
A_true = [0 1; -ks/m -c/m];
B_true = [0; 1/m];

sys_c = ss(A_true, B_true, eye(lx), 0);
sys_d = c2d(sys_c, dt, 'tustin');

C = [1, 0];

% Estimation Parameters
R0 = 1e2*eye(lx + 1); % [x; u]: lx + inputdim(=1)
lambda = 0.995;

% Initial Conditions
ICs = randn(lx, 1);
x_k = ICs;

% Reference signal for tracking
r = 1;

% Initialize Pk and AB for RLS
Theta_km1 = zeros(lx, lx + 1);
P_km1 = inv(R0);

for k = 1:N
    X_k(:, k) = x_k;
    Theta_vec(:,k) = Theta_km1(:);

    % Tracking error
    y_k = C*x_k;
    e_k = r - y_k;  % Error between reference and state
    q_k = q_k + e_k * dt;  % Integral of the tracking error

    % Augmented state vector
    x_aug_k = [x_k; q_k];

    % Control signal
    u_k = K_aug*x_aug_k +1e-2*randn + 0*sin(4*pi/100*k);
    U(k) = u_k;
  
    % System dynamics update
    x_kp1 = sys_d.A*x_k + sys_d.B*u_k;

    % DMAC states
    xi_k = x_k;
    xi_kp1 = x_kp1;

    % Going one step forward
    x_k = x_kp1;

    % Building the regressor
    phi_k = [xi_k; u_k];

    % Recursive update for P_{k+1} and Theta_{k+1}
    [P_k, Theta_k] = RLS_update_DMAC(phi_k, xi_kp1, P_km1, Theta_km1, lambda);

    % Update A_k and P_k for the next iteration
    Theta_km1 = Theta_k;
    P_km1 = P_k;

    % Compute control gain
    K_aug = compute_DMAC_control(Theta_km1, lx, C, Q, R, K_aug);

    % Log the step
    disp(['step ' num2str(k)]);
    
end

%% Plot results
figure
set(gcf, 'position', [200, 100, 800, 500])
t = tiledlayout(2, 2);
t.TileSpacing = 'compact';
t.Padding = 'compact';

nexttile(1)
ref_sig = r*ones(1, size(X_k,2));
plot(ref_sig, 'k--', 'LineWidth', 2)
hold on
plot(X_k(1,:), 'b', 'LineWidth', 3)
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
semilogy(abs(X_k(1,:)-r), 'b', 'LineWidth', 3)
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
disp(Theta_k)


function x_next = MCKDynamics(x, k, c, m, dt, u)
tspan = [0 dt];
A = [0 1; -k/m -c/m];
B = [0; 1/m];

f = @(t, x) A*x + B*u;
[~, x_out] = ode45(f, tspan, x);
x_next = x_out(end,:)';
end


