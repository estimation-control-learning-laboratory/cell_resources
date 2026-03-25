clc;
clear;
close all;
addpath DMAC_Functions_V2/

%% ========================================================================
%  RANDOM SEED
%  ========================================================================
randn('state',2)

%% ========================================================================
%  SIMULATION PARAMETERS
%  ========================================================================
sim.N  = 1000;
sim.dt = 0.1;


%% ========================================================================
%  SYSTEM PARAMETERS
%  ========================================================================

plant.lx = 2;
plant.ly = 1;
plant.lu = 1;
plant.lxi = 2;

plant.m  = 1;
plant.ks = 2;
plant.c  = 0.5;

[plant.A, plant.B ] = build_discrete_mass_spring_damper(plant, sim.dt);
plant.C     = [1 0];
plant.C_xi  = eye(2);

%% ========================================================================
%  DMAC PARAMETERS
%  ========================================================================

dmac.lambda = 0.995;
dmac.R0     = 1e2*eye(plant.lx + plant.lu);   % regressor dimension = [x;u]
dmac.Q      = 1*eye(plant.lx + plant.lu);     % augmented state = [x; q]
dmac.R      = 1*eye(plant.lu);
dmac.v_std  = 1e-2;                     % Excitation Signal
dmac.C_xi   = [1 0];
dmac.lxi    = plant.lxi;
dmac.lu     = plant.lu;
dmac.ly     = plant.ly;
%% ========================================================================
%  MEMORY ALLOCATION
%  ========================================================================
log = initialize_logs(plant.lx, sim.N);
x   = zeros(plant.lx, sim.N);
y   = zeros(plant.ly, sim.N);
u   = zeros(plant.lu, sim.N);
xi  = zeros(plant.lxi, sim.N);
q   = y;

phi = zeros(plant.lxi+plant.lu, sim.N);
%% ========================================================================
%  INITIAL CONDITIONS
%  ========================================================================
x_k = randn(plant.lx, 1);
q_k = 0;
x(:,1) = randn(plant.lx, 1);
xi(:,1) = x(:,1);
q(:,1) = 0;
r = 1;

Theta_km1 = zeros(plant.lxi, plant.lxi + plant.lu);
P_km1     = inv(dmac.R0);
K_aug     = [0 0 0];
K = zeros(plant.lu, plant.lxi+plant.ly);
%% ========================================================================
%  MAIN SIMULATION LOOP
%  ========================================================================
for k = 1:sim.N


    u(:,k) = K(:,1:plant.lxi)*xi(:,k) + K(:,plant.lxi+1:end)*q(:,k) + ...
        dmac.v_std * randn(plant.lu,1) * (k<sim.N/2) ;
    % -------------------------------------------------
    % True plant propagation
    % -------------------------------------------------
    [x(:,k+1), y(:,k), xi(:,k)] = plant_step(plant, x(:,k), u(:,k));
    
    y(:,k+1) = plant.C*x(:,k+1);
    xi(:,k+1) = plant.C_xi*x(:,k+1);

    % -------------------------------------------------
    % Integral state update
    % -------------------------------------------------
    e_k = r - y(:,k);
    q(:,k+1) = q(:,k) + e_k * sim.dt;

    % -------------------------------------------------
    % Regressor
    % -------------------------------------------------
    phi(:,k)        = [xi(:,k); u(:,k)];


    % if k>1
    %     [Theta_k, P_k, K] = DMAC_update(phi(:,k-1), xi(:,k), Theta_km1, P_km1, K, dmac);
    % else
    %     P_k = P_km1;
    %     Theta_k = Theta_km1;
    % end
    % Theta_km1       = Theta_k;
    % P_km1           = P_k;

    % [P_k, Theta_k] = RLS_update_DMAC(phi(:,k), xi(:,k+1), P_km1, Theta_km1, dmac.lambda);
    if k>1
        % this is not implementable since xi(:,k+1) is not available at
        % step k
        % [P_k, Theta_k]  = MatrixRLS(phi(:,k), xi(:,k+1), P_km1, Theta_km1, dmac.lambda);

        % the following is the correct implentation
        [P_k, Theta_k]  = MatrixRLS(phi(:,k-1), xi(:,k), P_km1, Theta_km1, dmac.lambda);
    else
        P_k = P_km1;
        Theta_k = Theta_km1;
    end
    Theta_km1       = Theta_k;
    P_km1           = P_k;

    K = compute_DMAC_control(Theta_k, plant.lx, dmac.C_xi, dmac.Q, dmac.R, K);

    % -------------------------------------------------
    % Log current data
    % -------------------------------------------------
    log.X(:,k)    = x(:,k);
    log.U(:,k)    = u(:,k);
    log.Theta_vec(:,k) = Theta_k(:);




    % -------------------------------------------------
    % Augmented state for control
    % -------------------------------------------------


    % -------------------------------------------------
    % Control input
    % -------------------------------------------------
    % u_k = compute_control_input(K_aug, phi_k, dmac);
    % 



    % % -------------------------------------------------
    % % RLS regressor and update
    % % -------------------------------------------------
    % phi_k = [x_k; u_k];


    % -------------------------------------------------
    % DMAC control update
    % -------------------------------------------------


    % -------------------------------------------------
    % Prepare for next step
    % -------------------------------------------------
    % x_k       = x_kp1;


    fprintf('step %d\n', k);
end



%% ========================================================================
%  RESULTS
%  ========================================================================
plot_results(log, r);

disp('True Ad:')
disp(plant.A)

disp('True Bd:')
disp(plant.B)

disp('Estimated [A B]:')
disp(Theta_k)

%% ========================================================================
%  LOCAL FUNCTIONS
%  ========================================================================


function [x_next, y_k, xi_k] = plant_step(plant, x_k, u_k)
% State update
x_next = plant.A * x_k + plant.B * u_k;

% Output
y_k = plant.C * x_k;

% Measured state for DMAC
xi_k = x_k;
end


function [Ad, Bd] = build_discrete_mass_spring_damper(plant, dt)
A = [0 1;
    -plant.ks/plant.m  -plant.c/plant.m];
B = [0; 1/plant.m];

sys_c = ss(A, B, eye(plant.lx), 0);
sys_d = c2d(sys_c, dt, 'tustin');

Ad = sys_d.A;
Bd = sys_d.B;
end

function log = initialize_logs(lx, N)
log.X         = zeros(lx, N);
log.U         = zeros(1, N);
log.Theta_vec = zeros(lx*(lx+1), N);
end

% function u_k = compute_control_input(K_aug, x_aug_k, dmac)
% excitation_signal  = dmac.v_std * randn;
% u_k = K_aug * x_aug_k + excitation_signal;
% end

function plot_results(log, r)
figure
set(gcf, 'position', [200, 100, 800, 500])

t = tiledlayout(2, 2);
t.TileSpacing = 'compact';
t.Padding = 'compact';

nexttile(1)
ref_sig = r * ones(1, size(log.X,2));
plot(ref_sig, 'k--', 'LineWidth', 2)
hold on
plot(log.X(1,:), 'b', 'LineWidth', 3)
hold off
set(gca, 'fontsize', 16);
set(gca, 'xticklabel', {[]})
xlabel('a)', 'Interpreter', 'latex', 'fontsize', 22)
ylabel('$y_k$', 'Interpreter', 'latex', 'fontsize', 22)
legend('Reference','')
axis tight
grid on

nexttile(2)
plot(log.U, 'b', 'LineWidth', 3)
set(gca, 'fontsize', 16);
set(gca, 'xticklabel', {[]})
xlabel('b)', 'Interpreter', 'latex', 'fontsize', 22)
ylabel('$u_k$', 'Interpreter', 'latex', 'fontsize', 22)
axis tight
grid on

nexttile(3)
semilogy(abs(log.X(1,:) - r), 'b', 'LineWidth', 3)
set(gca, 'fontsize', 16);
yticks(10.^(-5:1:5))
xlabel({'c)' 'step ($k$)'}, 'Interpreter', 'latex', 'fontsize', 22)
ylabel('$|z_k|$', 'Interpreter', 'latex', 'fontsize', 22)
axis tight
grid on

nexttile(4)
plot(log.Theta_vec', 'LineWidth', 3)
set(gca, 'fontsize', 16);
xlabel({'d)' 'step ($k$)'}, 'Interpreter', 'latex', 'fontsize', 22)
ylabel('$\Theta_k$', 'Interpreter', 'latex', 'fontsize', 22)
axis tight
grid on
end