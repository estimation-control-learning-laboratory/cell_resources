clc;
clear;
close all;
addpath DMAC_Functions_V2/

%% ========================================================================
%  RANDOM SEED
%  ========================================================================
randn('state',2)

%% ========================================================================
%  USER PARAMETERS
%  ========================================================================
sim.N  = 1000;
sim.dt = 0.1;

plant.lx = 2;
plant.ly = 1;
plant.lu = 1;
plant.lxi = 2;

plant.m  = 1;
plant.ks = 2;
plant.c  = 0.5;

ref.r = 1;

noise.u_std = 1e-2;

dmac.lambda = 0.995;
dmac.R0     = 1e2*eye(plant.lx + 1);   % regressor dimension = [x;u]
dmac.Q      = 1*eye(plant.lx + 1);     % augmented state = [x; q]
dmac.R      = 1;
dmac.v_std  = 1e-2;                     % Excitation Signal 
dmac.C_xi   = [1 0];

%% ========================================================================
%  BUILD TRUE DISCRETE-TIME MODEL
%  ========================================================================
[Ad, Bd] = build_discrete_mass_spring_damper(plant, sim.dt);

%% ========================================================================
%  MEMORY ALLOCATION
%  ========================================================================
log = initialize_logs(plant.lx, sim.N);
x   = zeros(plant.lx, sim.N);
y   = zeros(plant.ly, sim.N);
u   = zeros(plant.lu, sim.N);
xi  = zeros(plant.lxi, sim.N);

%% ========================================================================
%  INITIAL CONDITIONS
%  ========================================================================
x_k = randn(plant.lx, 1);
q_k = 0;

Theta_km1 = zeros(plant.lx, plant.lx + 1);
P_km1     = inv(dmac.R0);
K_aug     = [0 0 0];

%% ========================================================================
%  MAIN SIMULATION LOOP
%  ========================================================================
for k = 1:sim.N
    
    % -------------------------------------------------
    % True plant propagation
    % -------------------------------------------------
    [x(:,k+1), y(:,k), xi(:,k)] = plant_step(plant, x(:,k), u(:,k));
    
    % -------------------------------------------------
    % Integral state update
    % -------------------------------------------------
    e_k = ref.r - y(:,k);
    q_k = q_k + e_k * sim.dt;

    % -------------------------------------------------
    % Regressor
    % -------------------------------------------------
    phi_k = [xi(:,k); q(:,k)];


    % -------------------------------------------------
    % Log current data
    % -------------------------------------------------
    log.X(:,k)         = x_k;
    log.Theta_vec(:,k) = Theta_km1(:);

    

    
    % -------------------------------------------------
    % Augmented state for control
    % -------------------------------------------------
    

    % -------------------------------------------------
    % Control input
    % -------------------------------------------------
    u_k = compute_control_input(K_aug, phi_k, dmac);
    log.U(k) = u_k;

    

    % -------------------------------------------------
    % RLS regressor and update
    % -------------------------------------------------
    phi_k = [x_k; u_k];
    [P_k, Theta_k] = RLS_update_DMAC(phi_k, x_kp1, P_km1, Theta_km1, dmac.lambda);

    % -------------------------------------------------
    % DMAC control update
    % -------------------------------------------------
    K_aug = compute_DMAC_control(Theta_k, plant.lx, dmac.C_xi, dmac.Q, dmac.R, K_aug);

    % -------------------------------------------------
    % Prepare for next step
    % -------------------------------------------------
    x_k       = x_kp1;
    Theta_km1 = Theta_k;
    P_km1     = P_k;

    fprintf('step %d\n', k);
end

%% ========================================================================
%  RESULTS
%  ========================================================================
plot_results(log, ref.r);

disp('True Ad:')
disp(Ad)

disp('True Bd:')
disp(Bd)

disp('Estimated [A B]:')
disp(Theta_k)

%% ========================================================================
%  LOCAL FUNCTIONS
%  ========================================================================


function [x_next, y_k, xi_k] = plant_step(plant, x_k, u_k)
    % State update
    x_next = plant.Ad * x_k + plant.Bd * u_k;

    % Output
    y_k = plant.Cd * x_k;

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

function u_k = compute_control_input(K_aug, x_aug_k, dmac)
    excitation_signal  = dmac.v_std * randn;
    u_k = K_aug * x_aug_k + excitation_signal;
end

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