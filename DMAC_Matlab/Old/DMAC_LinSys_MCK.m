clc; clear;
close all;
addpath DMAC_Functions\

% Simulation parameters
randn('state',2)

N = 1000;
sys_dim = 2;
m = 1;    % 1
ks = 2;   % 2
c = 0.5;  % 0.5
dt = 0.1; % 0.1

% Initialization
X = zeros(sys_dim, N);
Y = zeros(sys_dim, N);
U = zeros(1, N);
S = 0;
Theta_error = zeros(1, N);
Theta_vec = zeros(sys_dim*(sys_dim+1), N);
K = [0, 0, 0];
x_int = 0;
X_Aug = zeros(sys_dim + 1, N);  % Augmented States storage

% LQR weight matrices
Q = 1*eye(sys_dim + 1);  % Augmented state with integral state 2+1
% Q(sys_dim+1, sys_dim+1) = 1;
R = 1;
E = eye(sys_dim + 1); % Augmented for integral action 2+1
Z = zeros(sys_dim + 1, 1);

% System Dynamics
A_true = [0 1; -ks/m -c/m];
B_true = [0; 1/m];

sys_c = ss(A_true, B_true, eye(sys_dim), 0);
sys_d = c2d(sys_c, dt, 'tustin');

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

for k = 1:N
    X(:, k) = x;

    % Tracking error
    error = r - x(1);  % Error between reference and state
    x_int = x_int + error * dt;  % Integral of the tracking error

    % Augmented state vector
    x_aug = [x; x_int];
    X_Aug(:, k) = x_aug;

    u = K*x_aug +1e-2*randn + 0*sin(4*pi/100*k);
    U(k) = u;

    % x = MCKDynamics(x, ks, c, m, dt, u);
    % x = sys_d.A * x + sys_d.B * u;
    x = [sys_d.A sys_d.B]*[x; u];

    y = x;
    Y(:, k) = y;


    x_bar_k = [X(:, k); U(k)];


    y_k = Y(:, k);   % Current state y_k = x_{k+1}

    % Recursive update for P_{k+1} and Theta_{k+1}
    [P_k_1, AB_est_1, z_k(k)] = RLS_update_DMD(x_bar_k, y_k, P_k, AB_est, lambda);

    % Update A_k and P_k for the next iteration
    AB_est = AB_est_1;
    P_k = P_k_1;

    [A_est, B_est] = extract_A_B_DMD(AB_est, sys_dim);

    [A_est_aug, B_est_aug] = generate_augmented_A_B_DMD(A_est, B_est, C, sys_dim);


    % Check controllability
    [T, A_controllable, B_controllable, A_uncontrollable, B_uncontrollable] = kalman_decomposition(A_est_aug, B_est_aug);

    if isempty(A_uncontrollable)
        disp('System is fully controllable.');
        [~, K, ~] = idare(A_controllable, B_controllable, Q, R, Z, E);
        K = -K;
    else
        disp('Not fully controllable! Using controllable part for control design.');

        % Use controllable part for controller design
        [~, K_controllable, ~] = idare(A_controllable, B_controllable, Q, R, Z, E);

        % Transform the gain back to the original coordinates
        K = -K_controllable * T;
    end

    % Log the step
    disp(['step ' num2str(k)]);
    Theta_error(k) = norm(AB_est - [sys_d.A sys_d.B]);
    Theta_vec(:,k) = AB_est(:);
end

%% Plot results
figure
set(gcf, 'position', [200, 100, 800, 500])
t = tiledlayout(2, 2);
t.TileSpacing = 'compact';
t.Padding = 'compact';

nexttile(1)
ref_sig = r*ones(1, size(X_Aug,2));
plot(ref_sig, 'k--', 'LineWidth', 2)
hold on
plot(X_Aug(1,:), 'b', 'LineWidth', 3)
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
disp(AB_est_1)


function x_next = MCKDynamics(x, k, c, m, dt, u)
tspan = [0 dt];
A = [0 1; -k/m -c/m];
B = [0; 1/m];

f = @(t, x) A*x + B*u;
[~, x_out] = ode45(f, tspan, x);
x_next = x_out(end,:)';
end



function [T, A_controllable, B_controllable, A_uncontrollable, B_uncontrollable] = kalman_decomposition(A, B)
    n = size(A, 1);
    
    % Step 1: Compute controllability matrix
    Wc = ctrb(A, B);
    
    % Step 2: Check rank of controllability matrix
    if rank(Wc) < n
        disp('Rank deficient system. Performing Kalman decomposition.');

        % Step 3: Perform Singular Value Decomposition (SVD) on the controllability matrix
        [U, ~, ~] = svd(Wc);
        
        % Step 4: Define the transformation matrix T (using the SVD components)
        T = U';
        
        % Step 5: Compute the decomposed system matrices
        A_controllable = T(1:n, :) * A * T(1:n, :)';
        B_controllable = T(1:n, :) * B;
        
        A_uncontrollable = T(n+1:end, :) * A * T(n+1:end, :)';
        B_uncontrollable = T(n+1:end, :) * B;
    else
        disp('System is fully controllable.');
        
        % If the system is fully controllable, the decomposition is trivial
        T = eye(n);  % Identity transformation, no need to decompose
        A_controllable = A;
        B_controllable = B;
        A_uncontrollable = [];
        B_uncontrollable = [];
    end
end


