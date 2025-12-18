clc; clear; close all;
addpath Functions\

% Simulation parameters
randn('state',2)
rand('state',2)

steps = 5000;

N = 100;        % Number of grid points
control_loc = 55;   % Among all the N grid points 
output_loc = 5;     % Among all the length(identified_locs) grid points used for identification


true_sys_dim = N;
% identified_sys_dim = 10;
% identified_locs = linspace(1, N, identified_sys_dim);
identified_locs = 1:20:100;
% 1:1(y)    2:21(n)    3:41(y)    4:61(n)    5:81(n)
identified_locs = 1:15:100;
% 1:1(y)    2:16(y)    3:31(y)    4:46(y)    5:61(y)    6:76(y)    7:91(y)

identified_sys_dim = length(identified_locs);


L = 2*pi;       % Domain length
dx = L / N;     % Spatial step
dt = 0.01;      % Time step
nu = 0.1; % nominal:0.1 others:0.05,0.5      % Viscosity

% Initialization
X_true = zeros(true_sys_dim, steps);  % States storage
Y = zeros(identified_sys_dim, steps);
U = zeros(1, steps);
S = 0;
Theta_error = zeros(1, steps);
Theta_vec = zeros(identified_sys_dim*(identified_sys_dim+1), steps);

K = zeros(1, identified_sys_dim + 1);

x_int = 0;
X_Aug = zeros(identified_sys_dim + 1, steps);  % Augmented States storage

% LQR weight matrices
Q = 10*eye(identified_sys_dim + 1); % nominal:10 others:1,100 % Augmented state with integral state 2+1
R = 0.1;                            % nominal:0.1 others:0.01,1
E = eye(identified_sys_dim + 1);  % Augmented for integral action 2+1
Z = zeros(identified_sys_dim + 1, 1);


% System Dynamics

C = zeros(1,identified_sys_dim);
C(output_loc) = 1;

% Estimation Parameters
% here
R0 = 1*1e2*eye(identified_sys_dim + 1); % nominal:1e2 others:200,1000,2000 % [x; u]: sys_dim + inputdim(=1)
lambda = 0.9995; % nominal:0.9995 others:0.999,0.9999,1

% Initial Conditions
% ICs = sin(linspace(0, L, N)');
ICs = randn(N,1);
% ICs(1) = 0;
% ICs(end) = 0;
x_true = ICs;

x = x_true(identified_locs);

% Reference signal for tracking
r = 1;

% Initialize Pk and AB for RLS
AB_est = zeros(identified_sys_dim, identified_sys_dim + 1);
P_k = inv(R0);

for k = 1:steps
    X_true(:, k) = x_true;

    % Tracking error
    error = r - x(output_loc);  % Error between reference and state
    x_int = x_int + error * dt;  % Integral of the tracking error

    % Augmented state vector
    x_aug = [x; x_int];
    X_Aug(:, k) = x_aug;

    u = K*x_aug + 1e-3*randn*1 + 0*sin(4*pi/100*k);

    U(k) = u;

    uc_burg = zeros(N, 1);
    uc_burg(control_loc) = u;


    x_true = Burgers_Dynamics(x_true, nu, dx, dt, uc_burg);
    
    x = x_true(identified_locs);
    y = x;
    Y(:, k) = y;

    % Update Ak and Pk recursively using the new snapshot (x_k, y_k)
    x_bar_k = [X_true(identified_locs, k); U(k)];
    % phi = x_bar_k;
    % phi_bar = kron(phi', eye(sys_dim));
    % S = lambda*S + phi_bar'*phi_bar;
    % conds(k) = cond(S);

    % y_k = Y(:, k);   % Current state y_k = x_{k+1}

    % Recursive update for P_{k+1} and Theta_{k+1}
    [P_k_1, AB_est_1, z_k(k)] = RLS_update_DMD(x_bar_k, y, P_k, AB_est, lambda);

    % Update A_k and P_k for the next iteration
    AB_est = AB_est_1;
    P_k = P_k_1;

    [A_est, B_est] = extract_A_B_DMD(AB_est, identified_sys_dim);

    [A_est_aug, B_est_aug] = generate_augmented_A_B_DMD(A_est, B_est, C, identified_sys_dim);

    if rank(ctrb(A_est_aug, B_est_aug)) ~= identified_sys_dim + 1
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
figure(1)
set(gcf, 'position', [200, 100, 800, 1200])
t = tiledlayout(5, 2);
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
plot(X_Aug(2,:), 'b', 'LineWidth', 3)
set(gca, 'fontsize', 16);
set(gca, 'xticklabel', {[]})
xlabel('b)', 'Interpreter', 'latex', 'fontsize', 22)
ylabel('$q_{2,k}$', 'Interpreter', 'latex', 'fontsize', 22)
axis tight
grid on

nexttile(3)
plot(X_Aug(3,:), 'b', 'LineWidth', 3)
set(gca, 'fontsize', 16);
set(gca, 'xticklabel', {[]})
xlabel('c)', 'Interpreter', 'latex', 'fontsize', 22)
ylabel('$q_{3,k}$', 'Interpreter', 'latex', 'fontsize', 22)
axis tight
grid on

nexttile(4)
plot(X_Aug(4,:), 'b', 'LineWidth', 3)
set(gca, 'fontsize', 16);
set(gca, 'xticklabel', {[]})
xlabel('d)', 'Interpreter', 'latex', 'fontsize', 22)
ylabel('$q_{4,k}$', 'Interpreter', 'latex', 'fontsize', 22)
axis tight
grid on


nexttile(5)
plot(X_Aug(5,:), 'b', 'LineWidth', 3)
hold on
ref_sig = r*ones(1, size(X_Aug,2));
plot(ref_sig, 'k--', 'LineWidth', 2)
set(gca, 'fontsize', 16);
set(gca, 'xticklabel', {[]})
xlabel('e)', 'Interpreter', 'latex', 'fontsize', 22)
ylabel('$q_{5,k}$', 'Interpreter', 'latex', 'fontsize', 22)
axis tight
grid on
legend('','Reference', Location='best')


nexttile(6)
plot(X_Aug(6,:), 'b', 'LineWidth', 3)
set(gca, 'fontsize', 16);
set(gca, 'xticklabel', {[]})
xlabel('f)', 'Interpreter', 'latex', 'fontsize', 22)
ylabel('$q_{6,k}$', 'Interpreter', 'latex', 'fontsize', 22)
axis tight
grid on

nexttile(7)
plot(X_Aug(7,:), 'b', 'LineWidth', 3)
set(gca, 'fontsize', 16);
set(gca, 'xticklabel', {[]})
xlabel('g)', 'Interpreter', 'latex', 'fontsize', 22)
ylabel('$q_{7,k}$', 'Interpreter', 'latex', 'fontsize', 22)
axis tight
grid on


nexttile(8)
plot(X_Aug(8,:), 'b', 'LineWidth', 3)
set(gca, 'fontsize', 16);
set(gca, 'xticklabel', {[]})
xlabel('h)', 'Interpreter', 'latex', 'fontsize', 22)
ylabel('$\rm{Integrator}$', 'Interpreter', 'latex', 'fontsize', 22)
axis tight
grid on

nexttile(9)
plot(Theta_vec', 'LineWidth', 3)
set(gca, 'fontsize', 16);
xlabel({'i)' 'step ($k$)'}, 'Interpreter', 'latex', 'fontsize', 22)
ylabel('$\Theta$', 'Interpreter', 'latex', 'fontsize', 22)
axis tight
grid on

nexttile(10)
plot(U, 'b', 'LineWidth', 3)
set(gca, 'fontsize', 16);
xlabel({'j)' 'step ($k$)'}, 'Interpreter', 'latex', 'fontsize', 22)
ylabel('$u$', 'Interpreter', 'latex', 'fontsize', 22)
axis tight
grid on

% print(gcf,'-dpng','Figures/Burgers/png/Burgers_DMAC_FSFI')
% print(gcf,'-depsc','Figures/Burgers/eps/Burgers_DMAC_FSFI')



figure
set(gcf, 'position', [200, 100, 800, 500])
semilogy(abs(X_Aug(output_loc,:)-r), 'b', 'LineWidth', 3)
set(gca, 'fontsize', 16);
xlabel('step ($k$)', 'Interpreter', 'latex', 'fontsize', 22)
ylabel('$|z_k|$', 'Interpreter', 'latex', 'fontsize', 22)
axis tight
grid on

% print(gcf,'-dpng','Figures/Burgers/png/Burgers_DMAC_FSFI_zk')
% print(gcf,'-depsc','Figures/Burgers/eps/Burgers_DMAC_FSFI_zk')


figure(2)
set(gcf, 'position', [200, 100, 800, 500])
t = tiledlayout(2, 2);
t.TileSpacing = 'compact';
t.Padding = 'compact';

nexttile(1)
plot(X_Aug(5,:), 'b', 'LineWidth', 3)
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
semilogy(abs(X_Aug(output_loc,:)-r), 'b', 'LineWidth', 3)
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

% print(gcf,'-dpng','Figures/Burgers/png/Burgers_DMAC_FSFI')
% print(gcf,'-depsc','Figures/Burgers/eps/Burgers_DMAC_FSFI')



% here
X_Aug_lambda_1 = X_Aug;
filename = 'Burgers_DMAC_Sensitivity_Vars.mat';
variables = 'X_Aug_lambda_1';  
% save(filename,variables) 
% save(filename,variables,"-append") 



% figure
% semilogy(z_k)



tt = 0:dt:(dt*steps);
tt = tt(1:end-1);  % x-axis

LL = linspace(0, 2*pi, 100);  % y-axis
% LL = LL(identified_locs);

% Generate example state values (Replace X_Aug with actual data)
% Ensure X_Aug has dimensions 100x8000 to match tt and LL
% zz = X_Aug(1:end-1,:);  % Example data for demonstration

zz = X_true(:,:);

% 3D Surface Plot
figure
plot_w_color_gradient_3D(tt, LL, zz)

% 2D Color Map
figure
plot_w_color_gradient_2D(tt, LL, zz)

% print(gcf,'-dpng','Figures/Burgers/png/Burgers_DMAC_states_contour')
% print(gcf,'-depsc','Figures/Burgers/eps/Burgers_DMAC_states_contour')

function plot_w_color_gradient_3D(xx, yy, zz)
    [X, Y] = meshgrid(xx, yy);
    set(gcf,'color','w');
    s = surf(X, Y, zz);
    s.LineStyle = 'none';
    grid on; box on; axis tight;
    set(gca,'TickLabelInterpreter','latex','FontSize',22)
    xlabel('$t$ (s)','Interpreter','latex','FontSize',22)
    ylabel('$x$','Interpreter','latex','FontSize',22)
    zlabel('$w_i$','Interpreter','latex','FontSize',22)
    cc = colorbar("eastoutside",...         
                    'Fontsize', 22, 'TickLabelInterpreter', 'latex');
    cc.Label.Interpreter = 'latex';
    cc.Label.String = "$w_i$";
end

function plot_w_color_gradient_2D(xx, yy, zz)
    [X, Y] = meshgrid(xx, yy);
    set(gcf,'color','w');
    s = pcolor(X, Y, zz);
    s.LineStyle = 'none';
    shading interp;
    grid on; box on; axis tight;
    set(gca,'TickLabelInterpreter','latex','FontSize',22)
    xlabel('$t$ (s)','Interpreter','latex','FontSize',22)
    yticks([0 1 2 3 4 5 6])
    ylabel('$x$','Interpreter','latex','FontSize',22)
    cc = colorbar("eastoutside",...         
                    'Fontsize', 22, 'TickLabelInterpreter', 'latex');
    cc.Label.Interpreter = 'latex';
    cc.Label.String = "$w_i$";
end

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

% function du_dt = Burgers_RHS(u, nu, dx, u_control)
%     N = length(u);
%     du_dt = zeros(N, 1);
% 
%     % Apply zero boundary conditons explicitly
%     u(1) = 0;
%     u(end) = 0;
% 
%     % Compute first derivative (convection term)
%     du_dx = zeros(N,1);
%     du_dx(2:N-1) = (u(3:N) - u(1:N-2)) / (2 * dx);
% 
%     % Compute second derivative (diffusion term)
%     d2u_dx2 = zeros(N,1);
%     d2u_dx2(2:N-1) = (u(3:N) - 2*u(2:N-1) + u(1:N-2)) / (dx^2);
% 
%     % Compute time derivative of u with control input
%     du_dt(2:N-1) = -u(2:N-1) .* du_dx(2:N-1) + nu * d2u_dx2(2:N-1) + u_control(2:N-1);
% end


