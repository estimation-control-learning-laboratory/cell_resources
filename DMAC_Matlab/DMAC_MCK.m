clc; clear;
close all;
addpath Functions\

% Simulation parameters
randn('state',2)

N = 1000;
sys_dim = 2;
% here
m = 1;    % 1
ks = 2;   % 2
c = 0.5;  % 0.5
dt = 0.1; % 0.1

% Initialization
X = zeros(sys_dim, N);  % States storage
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

    u = K*x_aug +1e-3*randn + 0*sin(4*pi/100*k);
    U(k) = u;

    % x = MCKDynamics(x, ks, c, m, dt, u);
    % x = sys_d.A * x + sys_d.B * u;
    x = [sys_d.A sys_d.B]*[x; u];

    y = x;
    Y(:, k) = y;


    % if k == 1
    %     AB_est = zeros(sys_dim, sys_dim + 1);
    %
    %     % Initialize Pk for RLS
    %     P_k = inv(R0);
    %
    % else
    % Update Ak and Pk recursively using the new snapshot (x_k, y_k)
    x_bar_k = [X(:, k); U(k)];
    phi = x_bar_k;
    phi_bar = kron(phi', eye(sys_dim));
    S = lambda*S + phi_bar'*phi_bar;
    conds(k) = cond(S);

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

    % if rank(ctrb(A_est_aug, B_est_aug)) ~= sys_dim + 1
    %     disp('Not Controllable!!')
    % else
    %     [~, K, ~] = idare(A_est_aug, B_est_aug, Q, R, Z, E);
    %     % K = dlqr(A_est_aug, B_est_aug, Q, R);
    %     K = -K;
    % end

    %%

    % % Check controllability
    % if rank(ctrb(A_est_aug, B_est_aug)) < sys_dim + 1
    %     disp('Not Controllable!! -- Applying projection to controllable subspace')
    % 
    %     [T, A_est_aug_new, B_est_aug_new] = compute_controllable_form(A_est_aug, B_est_aug);
    % 
    %     try
    %         [~, K_new, ~] = idare(A_est_aug_new, B_est_aug_new, Q, R, Z, E);
    %         K = -K_new * T;  % Transform gain back to original coordinates
    %     catch
    %         disp('DARE failed even after projection. Keeping previous gain.')
    %         % keep K unchanged
    %     end
    % else
    %     [~, K, ~] = idare(A_est_aug, B_est_aug, Q, R, Z, E);
    %     K = -K;
    % end


    %%
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



    %%

    % Log the step
    disp(['step ' num2str(k)]);
    Theta_error(k) = norm(AB_est - [sys_d.A sys_d.B]);
    Theta_vec(:,k) = AB_est(:);

    % end

end

%% Plot results

figure(1)
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
    plot(X_Aug(i,:), 'b', 'LineWidth', 3)
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



figure(2)
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

% print(gcf,'-dpng','Figures/MCK/png/MCK_DMAC_FSFI')
% print(gcf,'-depsc','Figures/MCK/eps/MCK_DMAC_FSFI')






% disp('True Ac:')
% disp(A_true)
% disp('True Bc:')
% disp(B_true)
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
% ylabel('estimation error', 'Interpreter', 'latex', 'fontsize', 18)
% axis tight
% grid on
%
% figure
% set(gca, 'fontsize', 16);
% semilogy(conds, 'LineWidth', 3)
% ylabel('condition number', 'Interpreter', 'latex', 'fontsize', 18)
% xlabel('step (k)', 'Interpreter', 'latex', 'fontsize', 18)
% axis tight
% grid on
%

% figure
% set(gcf, 'position', [200, 100, 800, 350])
% semilogy(abs(X(1,:)-r), 'b', 'LineWidth', 3)
% set(gca, 'fontsize', 16);
% yticks(10.^(-5:1:5))
% xlabel('step ($k$)', 'Interpreter', 'latex', 'fontsize', 22)
% ylabel('$|z_k|$', 'Interpreter', 'latex', 'fontsize', 22)
% axis tight
% grid on

% print(gcf,'-dpng','Figures/MCK/png/MCK_DMAC_FSFI_zk')
% print(gcf,'-depsc','Figures/MCK/eps/MCK_DMAC_FSFI_zk')




% figure
% semilogy(z_k)

% here
X_Aug_k_10 = X_Aug;
filename = 'MCK_System_Sensitivity_Vars.mat';
variables = 'X_Aug_k_10';
% save(filename,variables)
% save(filename,variables,"-append")

function x_next = MCKDynamics(x, k, c, m, dt, u)
tspan = [0 dt];
A = [0 1; -k/m -c/m];
B = [0; 1/m];

f = @(t, x) A*x + B*u;
[~, x_out] = ode45(f, tspan, x);
x_next = x_out(end,:)';
end



% print(gcf,'-dpng','Figures/VDP/png/VDP_DMDc_FSFI')
% print(gcf,'-depsc','Figures/VDP/eps/VDP_DMDc_FSFI')


function [T, A_new, B_new] = compute_controllable_form(A, B)
n = size(A,1);
Wc = ctrb(A, B);

if rank(Wc) < n
    disp('Rank deficient system. Projecting onto controllable subspace.');
    [U, ~, ~] = svd(Wc);
    T = U';
else
    T = inv(Wc);  % Transformation to canonical controllable form
end

A_new = T * A / T;
B_new = T * B;
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


