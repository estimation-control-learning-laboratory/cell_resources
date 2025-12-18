clc; clear; close all;
addpath Functions\ % Ensure your custom functions are in this folder

% Works:
% (1,2,4) & (1,3,4) & (1,4,5) & (2,3,4)
% (1,3,4,5,6) & (1,3,4,5)

% Simulation parameters
randn('state', 2);
N = 2000; % Number of time steps
true_sys_dim = 6;  % True system is third order
identified_sys_dim = 3;  % Identified model is first order (position only)
m = 1; % Mass
k = 2;
dt = 0.1; % Time step

% Define the true system matrices
K = [-2*k, k, 0; k, -2*k, k; 0, k, -2*k];
A_true = [zeros(3), eye(3); inv(m)*K, zeros(3)];
B_true = [zeros(3, 1); inv(m)*[1; 0; 0]];

sys_c = ss(A_true, B_true, eye(true_sys_dim), 0);
sys_d = c2d(sys_c, dt, 'tustin');

C = [0, 0, 1, 0, 0, 0]; % Output matrix for q3 tracking

% Initialization
X_true = zeros(true_sys_dim, N); % States storage
Y = zeros(identified_sys_dim, N);
U = zeros(1, N);
S = 0;
Theta_error = zeros(1, N);
Theta_vec = zeros(identified_sys_dim*(identified_sys_dim+1), N);
K_gain = zeros(1, identified_sys_dim + 1); % Initial control gain
x_int = 0; % Integral state for tracking
X_Aug = zeros(identified_sys_dim + 1, N); % Augmented states storage

% LQR weight matrices
% here
Q = 1*eye(identified_sys_dim + 1); % Weight for augmented state
% Q(identified_sys_dim + 1, identified_sys_dim + 1) = 1; % Integral state weight
R = 1; % Control input weight
E = eye(identified_sys_dim + 1); % Identity matrix for augmentation
Z = zeros(identified_sys_dim + 1, 1); % No cross-coupling in LQR

% Estimation Parameters
R0 = 1e2* eye(identified_sys_dim + 1); % Initial covariance for RLS
lambda = 0.999; % Forgetting factor

% Initial Conditions
ICs = randn(true_sys_dim, 1); % Random initial state
x_true = ICs;
x = x_true([1,3,4]);  % Initial positions only for identified model

% Reference signal for tracking
r = 1;

% Initialize Pk and AB for RLS
AB_est = zeros(identified_sys_dim, identified_sys_dim + 1);
P_k = inv(R0);

for k = 1:N
    X_true(:, k) = x_true;

    % Tracking error
    error = r - x_true(3); % Error between reference and q3
    x_int = x_int + error * dt; % Integral of tracking error

    % Augmented state vector
    x_aug = [x; x_int];
    X_Aug(:, k) = x_aug;

    % Control input
    u = K_gain * x_aug + 1e-4 * randn + 0 * sin(4 * pi / 100 * k);
    U(k) = u;

    % Propagate system dynamics
    x_true = [sys_d.A sys_d.B] * [x_true; u];

    x = x_true([1,3,4]);
    y = x;
    Y(:, k) = y;

    % Update Ak and Pk recursively using the new snapshot (x_k, y_k)
    x_bar_k = [X_true([1,3,4], k); U(k)];
    phi = x_bar_k;
    phi_bar = kron(phi', eye(identified_sys_dim));
    S = lambda * S + phi_bar' * phi_bar;
    conds(k) = cond(S);

    y_k = Y(:, k); % Current state y_k = x_{k+1}

    % Recursive update for P_{k+1} and Theta_{k+1}
    [P_k_1, AB_est_1, z_k(k)] = RLS_update_DMD(x_bar_k, y_k, P_k, AB_est, lambda);

    % Update A_k and P_k for the next iteration
    AB_est = AB_est_1;
    P_k = P_k_1;

    [A_est, B_est] = extract_A_B_DMD(AB_est, identified_sys_dim);

    % Generate augmented matrices for FSFI control
    [A_est_aug, B_est_aug] = generate_augmented_A_B_DMD(A_est, B_est, C(1:identified_sys_dim), identified_sys_dim);

    if rank(ctrb(A_est_aug, B_est_aug)) ~= identified_sys_dim + 1
        disp('Not Controllable!!')
    else
        [~, K_gain, ~] = idare(A_est_aug, B_est_aug, Q, R, Z, E);
        K_gain = -K_gain; % Compute and store feedback gain
    end

    % Log the step
    disp(['Step ' num2str(k)]);
    Theta_vec(:,k) = AB_est(:);
    % Theta_error(k) = norm(AB_est - [sys_d.A sys_d.B]);
end


%% Plot results
figure
set(gcf, 'position', [200, 100, 800, 500])

% Get the number of rows in X
[num_states, ~] = size(X_Aug);

% Create a tiled layout with the number of rows equal to the number of states
t = tiledlayout(num_states, 1);
t.TileSpacing = 'compact';
t.Padding = 'compact';

% Loop through each state and create a plot for it
for i = 1:num_states
    nexttile
    plot(X_Aug(i, :), 'b', 'LineWidth', 3)
    set(gca, 'fontsize', 16);

    % Only remove the x-tick labels for all but the last plot
    if i < num_states
        set(gca, 'xticklabel', {[]})
    else
        xlabel('step (k)', 'Interpreter', 'latex', 'fontsize', 22)
    end

    ylabel(['$x_' num2str(i) '$'], 'Interpreter', 'latex', 'fontsize', 22)
    axis tight
    grid on
end

disp('True Ad:')
disp(sys_d.A)
disp('True Bd:')
disp(sys_d.B)
disp('Estimated [A B]:')
disp(AB_est_1)

% figure
% set(gca, 'fontsize', 16);
% title('Estimation Error')
% semilogy(Theta_error, 'LineWidth', 3)
% xlabel('step (k)', 'Interpreter', 'latex', 'fontsize', 18)
% ylabel('Estimation error', 'Interpreter', 'latex', 'fontsize', 18)
% axis tight
% grid on

% figure
% set(gca, 'fontsize', 16);
% semilogy(conds, 'LineWidth', 3)
% ylabel('Condition number', 'Interpreter', 'latex', 'fontsize', 18)
% xlabel('step (k)', 'Interpreter', 'latex', 'fontsize', 18)
% axis tight
% grid on

figure(2)
set(gcf, 'position', [200, 100, 800, 500])
t = tiledlayout(2, 2);
t.TileSpacing = 'compact';
t.Padding = 'compact';

nexttile(1)
plot(X_true(3,:), 'b', 'LineWidth', 3)
hold on
ref_sig = r*ones(1, size(X_Aug,2));
plot(ref_sig, 'k--', 'LineWidth', 2)
set(gca, 'fontsize', 16);
set(gca, 'xticklabel', {[]})
xlabel('a)', 'Interpreter', 'latex', 'fontsize', 22)
ylabel('$y_k$', 'Interpreter', 'latex', 'fontsize', 22)
axis tight
grid on
legend('','Reference', Location='best')

nexttile(2)
plot(U, 'b', 'LineWidth', 3)
set(gca, 'fontsize', 16);
set(gca, 'xticklabel', {[]})
xlabel('b)', 'Interpreter', 'latex', 'fontsize', 22)
ylabel('$u_k$', 'Interpreter', 'latex', 'fontsize', 22)
axis tight
grid on

nexttile(3)
semilogy(abs(X_true(3,:)-r), 'b', 'LineWidth', 3)
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
box on

% print(gcf,'-dpng','Figures/ThreeM/png/ThreeM_DMAC_FSFI')
% print(gcf,'-depsc','Figures/ThreeM/eps/ThreeM_DMAC_FSFI')




% figure
% set(gcf, 'position', [200, 100, 800, 350])
% semilogy(abs(X_true(3,:)-r), 'b', 'LineWidth', 3)
% set(gca, 'fontsize', 16);
% xlabel('step ($k$)', 'Interpreter', 'latex', 'fontsize', 22)
% yticks(10.^(-5:1:5))
% ylabel('$|z_k|$', 'Interpreter', 'latex', 'fontsize', 22)
% axis tight
% grid on

% print(gcf,'-dpng','Figures/ThreeM/png/ThreeM_DMAC_FSFI_zk')
% print(gcf,'-depsc','Figures/ThreeM/eps/ThreeM_DMAC_FSFI_zk')


% here
X_Aug_R_10 = X_Aug;
filename = 'ThreeM_DMAC_Sensitivity_Vars.mat';
variables = 'X_Aug_R_10';  
% save(filename,variables) 
% save(filename,variables,"-append") 