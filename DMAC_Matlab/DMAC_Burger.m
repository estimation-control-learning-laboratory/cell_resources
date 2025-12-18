clc; clear; close all;
addpath Functions\

% Simulation parameters
randn('state',2)

steps = 1000;

N = 100;        % Number of grid points
control_loc = 50;
sys_dim = N;

L = 2*pi;       % Domain length
dx = L / N;     % Spatial step
dt = 0.01;      % Time step
nu = 0.1;       % Viscosity

% Initialization
X = zeros(sys_dim, steps);  % States storage
Y = zeros(sys_dim, steps);
U = zeros(1, steps);
S = 0;
Theta_error = zeros(1, steps);
Theta_vec = zeros(sys_dim*(sys_dim+1), steps);

K = zeros(1, sys_dim + 1);

x_int = 0;
X_Aug = zeros(sys_dim + 1, steps);  % Augmented States storage

% LQR weight matrices
Q = 1*eye(sys_dim + 1);  % Augmented state with integral state 2+1
R = 1;
E = eye(sys_dim + 1); % Augmented for integral action 2+1
Z = zeros(sys_dim + 1, 1);


% System Dynamics

C = zeros(1,N);
C(control_loc) = 1;

% Estimation Parameters
R0 = 1e2*eye(sys_dim + 1); % [x; u]: sys_dim + inputdim(=1)
lambda = 0.999;

% Initial Conditions
% ICs = sin(linspace(0, L, N)');
ICs = randn(N,1);

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

    u = K*x_aug + 1e-2*randn + 0*sin(4*pi/100*k);

    U(k) = u;

    uc_burg = zeros(N, 1);
    uc_burg(control_loc) = u;

    x = Burgers_Dynamics(x, nu, dx, dt, uc_burg);

    y = x;
    Y(:, k) = y;

    % Update Ak and Pk recursively using the new snapshot (x_k, y_k)
    x_bar_k = [X(:, k); U(k)];
    % phi = x_bar_k;
    % phi_bar = kron(phi', eye(sys_dim));
    % S = lambda*S + phi_bar'*phi_bar;
    % conds(k) = cond(S);

    y_k = Y(:, k);   % Current state y_k = x_{k+1}

    % Recursive update for P_{k+1} and Theta_{k+1}
    [P_k_1, AB_est_1, z_k(k)] = RLS_update_DMD(x_bar_k, y_k, P_k, AB_est, lambda);

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
    Theta_vec(:,k) = AB_est(:);


end

%% Plot results
% figure
% set(gcf, 'position', [200, 100, 800, 500])
% 
% % Get the number of rows in X
% [num_states, ~] = size(X_Aug);
% 
% % Create a tiled layout with the number of rows equal to the number of states
% t = tiledlayout(num_states, 1);
% t.TileSpacing = 'compact';
% t.Padding = 'compact';
% 
% % Loop through each state and create a plot for it
% for i = 1:num_states
%     nexttile
%     plot(X_Aug(i,:), 'b', 'LineWidth', 3)
%     set(gca, 'fontsize', 16);
% 
%     % Only remove the x-tick labels for all but the last plot
%     if i < num_states
%         set(gca, 'xticklabel', {[]})
%     else
%         xlabel('step (k)', 'Interpreter', 'latex', 'fontsize', 22)
%     end
% 
%     ylabel(['$x_' num2str(i) '$'], 'Interpreter', 'latex', 'fontsize', 22)
%     axis tight
%     grid on
% end


% figure(2)
% set(gcf, 'position', [200, 100, 800, 500])
% t = tiledlayout(2, 2);
% t.TileSpacing = 'compact';
% t.Padding = 'compact';
% 
% nexttile(1)
% plot(X_Aug(1,:), 'b', 'LineWidth', 3)
% hold on
% ref_sig = r*ones(1, size(X_Aug,2));
% plot(ref_sig, 'k--', 'LineWidth', 2)
% set(gca, 'fontsize', 16);
% set(gca, 'xticklabel', {[]})
% xlabel('a)', 'Interpreter', 'latex', 'fontsize', 22)
% ylabel('$q_k$', 'Interpreter', 'latex', 'fontsize', 22)
% legend('','Reference', location = 'best')
% axis tight
% grid on
% 
% nexttile(3)
% plot(X_Aug(2,:), 'b', 'LineWidth', 3)
% set(gca, 'fontsize', 16);
% xlabel({'c)' 'step ($k$)'}, 'Interpreter', 'latex', 'fontsize', 22)
% ylabel('$\dot{q}_k$', 'Interpreter', 'latex', 'fontsize', 22)
% axis tight
% grid on
% 
% nexttile(2)
% plot(U, 'b', 'LineWidth', 3)
% set(gca, 'fontsize', 16);
% set(gca, 'xticklabel', {[]})
% xlabel('b)', 'Interpreter', 'latex', 'fontsize', 22)
% ylabel('$u$', 'Interpreter', 'latex', 'fontsize', 22)
% axis tight
% grid on
% 
% nexttile(4)
% plot(Theta_vec', 'LineWidth', 3)
% set(gca, 'fontsize', 16);
% xlabel({'d)' 'step ($k$)'}, 'Interpreter', 'latex', 'fontsize', 22)
% ylabel('$\Theta$', 'Interpreter', 'latex', 'fontsize', 22)
% axis tight
% grid on

% print(gcf,'-dpng','Figures/VDP/png/VDP_DMAC_FSFI')
% print(gcf,'-depsc','Figures/VDP/eps/VDP_DMAC_FSFI')



figure
set(gcf, 'position', [200, 100, 800, 500])
semilogy(abs(X(1,:)-r), 'b', 'LineWidth', 3)
set(gca, 'fontsize', 16);
xlabel('step ($k$)', 'Interpreter', 'latex', 'fontsize', 22)
ylabel('$|z_k|$', 'Interpreter', 'latex', 'fontsize', 22)
axis tight
grid on


figure
plot(X_Aug(control_loc,:), 'b', 'LineWidth', 3)
hold on
ref_sig = r*ones(1, size(X_Aug,2));
plot(ref_sig, 'k--', 'LineWidth', 2)
set(gca, 'fontsize', 16);
set(gca, 'xticklabel', {[]})
xlabel('a)', 'Interpreter', 'latex', 'fontsize', 22)
ylabel('$q_k$', 'Interpreter', 'latex', 'fontsize', 22)
legend('','Reference', location = 'best')
axis tight
grid on

% print(gcf,'-dpng','Figures/VDP/png/VDP_DMAC_FSFI_zk')
% print(gcf,'-depsc','Figures/VDP/eps/VDP_DMAC_FSFI_zk')

% here
X_Aug_mu_5 = X_Aug;
filename = 'VDP_System_Sensitivity_Vars.mat';
variables = 'X_Aug_mu_5';  
% save(filename,variables) 
% save(filename,variables,"-append") 



% figure
% semilogy(z_k)

function u_next = Burgers_Dynamics(u, nu, dx, dt, u_control)
    tspan = [0 dt];
    f = @(t, u) Burgers_RHS(u, nu, dx, u_control);
    [~, u_out] = ode45(f, tspan, u);
    u_next = u_out(end, :)';
end

function du_dt = Burgers_RHS(u, nu, dx, u_control)
    N = length(u);
    du_dt = zeros(N, 1);
    
    % Periodic boundary conditions
    u_p = [u(end); u; u(1)]; 

    % Compute first derivative (convection term)
    du_dx = (u_p(3:end) - u_p(1:end-2)) / (2 * dx);

    % Compute second derivative (diffusion term)
    d2u_dx2 = (u_p(3:end) - 2*u_p(2:end-1) + u_p(1:end-2)) / (dx^2);

    % Compute time derivative of u with control input
    du_dt = -u .* du_dx + nu * d2u_dx2 + u_control; 
end



% print(gcf,'-dpng','Figures/VDP/png/VDP_DMDc_FSFI')
% print(gcf,'-depsc','Figures/VDP/eps/VDP_DMDc_FSFI')