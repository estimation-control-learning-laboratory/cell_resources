clc; clear; close all;
addpath Functions\ % Ensure your custom functions are in this folder

% Simulation parameters
randn('state', 2);
N = 1500; % Number of time steps
sys_dim = 6; % Dimension of the state vector for the 3-mass system
% here
m = 1; % 1
k = 2; % 2
dt = 0.1; % Time step

% Define the true system matrices
K = [-2*k, k, 0; k, -2*k, k; 0, k, -2*k];
A_true = [zeros(3), eye(3); inv(m)*K, zeros(3)];
B_true = [zeros(3, 1); inv(m)*[1; 0; 0]];

sys_c = ss(A_true, B_true, eye(sys_dim), 0);
sys_d = c2d(sys_c, dt, 'tustin');

C = [0, 0, 1, 0, 0, 0]; % Output matrix for q3 tracking

% Initialization
X = zeros(sys_dim, N); % States storage
Y = zeros(sys_dim, N);
U = zeros(1, N);
S = 0;
Theta_error = zeros(1, N);
Theta_vec = zeros(sys_dim*(sys_dim+1), N);
K_gain = zeros(1, sys_dim + 1); % Initial control gain
x_int = 0; % Integral state for tracking
X_Aug = zeros(sys_dim + 1, N); % Augmented states storage

% LQR weight matrices
Q = 1*eye(sys_dim + 1); % Weight for augmented state
% Q(sys_dim + 1, sys_dim + 1) = 1; % Integral state weight
R = 1; % Control input weight
E = eye(sys_dim + 1); % Identity matrix for augmentation
Z = zeros(sys_dim + 1, 1); % No cross-coupling in LQR

% Estimation Parameters
R0 = 1e2 * eye(sys_dim + 1); % Initial covariance for RLS
lambda = 0.995; % Forgetting factor

% Initial Conditions
ICs = randn(sys_dim, 1); % Random initial state
x = ICs;

% Reference signal for tracking
r = 1;

% Initialize Pk and AB for RLS
AB_est = zeros(sys_dim, sys_dim + 1);
P_k = inv(R0);

for k = 1:N
    X(:, k) = x;

    % Tracking error
    error = r - x(3); % Error between reference and q3
    x_int = x_int + error * dt; % Integral of tracking error

    % Augmented state vector
    x_aug = [x; x_int];
    X_Aug(:, k) = x_aug;

    % Control input
    u = K_gain * x_aug + 1e-5 * randn + 0 * sin(4 * pi / 100 * k);
    U(k) = u;

    % Propagate system dynamics
    x = [sys_d.A sys_d.B] * [x; u];

    y = x;
    Y(:, k) = y;

    % if k == 1
    %     AB_est = zeros(sys_dim, sys_dim + 1);
    % 
    %     % Initialize Pk for RLS
    %     P_k = inv(R0);
    % else
        % Update Ak and Pk recursively using the new snapshot (x_k, y_k)
        x_bar_k = [X(:, k); U(k)];
        phi = x_bar_k;
        phi_bar = kron(phi', eye(sys_dim));
        S = lambda * S + phi_bar' * phi_bar;
        conds(k) = cond(S);

        y_k = Y(:, k); % Current state y_k = x_{k+1}

        % Recursive update for P_{k+1} and Theta_{k+1}
        [P_k_1, AB_est_1, z_k(k)] = RLS_update_DMD(x_bar_k, y_k, P_k, AB_est, lambda);

        % Update A_k and P_k for the next iteration
        AB_est = AB_est_1;
        P_k = P_k_1;

        [A_est, B_est] = extract_A_B_DMD(AB_est, sys_dim);

        % Generate augmented matrices for FSFI control
        [A_est_aug, B_est_aug] = generate_augmented_A_B_DMD(A_est, B_est, C, sys_dim);

        if rank(ctrb(A_est_aug, B_est_aug)) ~= sys_dim + 1
            disp('Not Controllable!!')
        else
            [~, K_gain, ~] = idare(A_est_aug, B_est_aug, Q, R, Z, E);
            K_gain = -K_gain; % Compute and store feedback gain
        end

        % Log the step
        disp(['Step ' num2str(k)]);
        Theta_error(k) = norm(AB_est - [sys_d.A sys_d.B]);
        Theta_vec(:,k) = AB_est(:);
    % end
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

figure(2)
set(gcf, 'position', [200, 100, 800, 1000])
t = tiledlayout(4, 2);
t.TileSpacing = 'compact';
t.Padding = 'compact';

nexttile(1)
plot(X_Aug(1,:), 'b', 'LineWidth', 3)
set(gca, 'fontsize', 16);
set(gca, 'xticklabel', {[]})
xlabel('a)', 'Interpreter', 'latex', 'fontsize', 22)
ylabel('$q_{1,k}$', 'Interpreter', 'latex', 'fontsize', 22)
axis tight
grid on

nexttile(2)
plot(X_Aug(4,:), 'b', 'LineWidth', 3)
set(gca, 'fontsize', 16);
set(gca, 'xticklabel', {[]})
xlabel('b)', 'Interpreter', 'latex', 'fontsize', 22)
ylabel('$\dot{q}_{1,k}$', 'Interpreter', 'latex', 'fontsize', 22)
axis tight
grid on

nexttile(3)
plot(X_Aug(2,:), 'b', 'LineWidth', 3)
set(gca, 'fontsize', 16);
set(gca, 'xticklabel', {[]})
xlabel('c)', 'Interpreter', 'latex', 'fontsize', 22)
ylabel('$q_{2,k}$', 'Interpreter', 'latex', 'fontsize', 22)
axis tight
grid on

nexttile(4)
plot(X_Aug(5,:), 'b', 'LineWidth', 3)
set(gca, 'fontsize', 16);
set(gca, 'xticklabel', {[]})
xlabel('d)', 'Interpreter', 'latex', 'fontsize', 22)
ylabel('$\dot{q}_{2,k}$', 'Interpreter', 'latex', 'fontsize', 22)
axis tight
grid on


nexttile(5)
plot(X_Aug(3,:), 'b', 'LineWidth', 3)
hold on
ref_sig = r*ones(1, size(X_Aug,2));
plot(ref_sig, 'k--', 'LineWidth', 2)
set(gca, 'fontsize', 16);
set(gca, 'xticklabel', {[]})
xlabel('e)', 'Interpreter', 'latex', 'fontsize', 22)
ylabel('$q_{3,k}$', 'Interpreter', 'latex', 'fontsize', 22)
axis tight
grid on
legend('','Reference')


nexttile(6)
plot(X_Aug(6,:), 'b', 'LineWidth', 3)
set(gca, 'fontsize', 16);
set(gca, 'xticklabel', {[]})
xlabel('f)', 'Interpreter', 'latex', 'fontsize', 22)
ylabel('$\dot{q}_{3,k}$', 'Interpreter', 'latex', 'fontsize', 22)
axis tight
grid on

nexttile(7)
plot(Theta_vec', 'LineWidth', 3)
set(gca, 'fontsize', 16);
xlabel({'g)' 'step ($k$)'}, 'Interpreter', 'latex', 'fontsize', 22)
ylabel('$\Theta$', 'Interpreter', 'latex', 'fontsize', 22)
axis tight
grid on
box on

nexttile(8)
plot(U, 'b', 'LineWidth', 3)
set(gca, 'fontsize', 16);
xlabel({'h)' 'step ($k$)'}, 'Interpreter', 'latex', 'fontsize', 22)
ylabel('$u$', 'Interpreter', 'latex', 'fontsize', 22)
axis tight
grid on

% print(gcf,'-dpng','Figures/ThreeM/png/ThreeM_DMAC_FSFI')
% print(gcf,'-depsc','Figures/ThreeM/eps/ThreeM_DMAC_FSFI')


% figure
% set(gca, 'fontsize', 16);
% title('Estimation Error')
% semilogy(Theta_error, 'LineWidth', 3)
% xlabel('step (k)', 'Interpreter', 'latex', 'fontsize', 18)
% ylabel('Estimation error', 'Interpreter', 'latex', 'fontsize', 18)
% axis tight
% grid on
% 
% figure
% set(gca, 'fontsize', 16);
% semilogy(conds, 'LineWidth', 3)
% ylabel('Condition number', 'Interpreter', 'latex', 'fontsize', 18)
% xlabel('step (k)', 'Interpreter', 'latex', 'fontsize', 18)
% axis tight
% grid on

figure
set(gcf, 'position', [200, 100, 800, 500])
semilogy(abs(X(3,:)-r), 'b', 'LineWidth', 3)
set(gca, 'fontsize', 16);
xlabel('step ($k$)', 'Interpreter', 'latex', 'fontsize', 22)
ylabel('$|z_k|$', 'Interpreter', 'latex', 'fontsize', 22)
axis tight
grid on

% print(gcf,'-dpng','Figures/ThreeM/png/ThreeM_DMAC_FSFI_zk')
% print(gcf,'-depsc','Figures/ThreeM/eps/ThreeM_DMAC_FSFI_zk')


% here
X_Aug_k_10 = X_Aug;
filename = 'ThreeM_System_Sensitivity_Vars.mat';
variables = 'X_Aug_k_10';  
% save(filename,variables) 
% save(filename,variables,"-append") 