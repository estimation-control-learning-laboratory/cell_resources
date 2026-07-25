%% ========================================================================
%  DYNAMIC MODE ADAPTIVE CONTROL (DMAC) SIMULATION
%  ========================================================================
%
%  Runs a closed-loop discrete-time mass-spring-damper simulation under
%  Dynamic Mode Adaptive Control:
%    1) Online system identification via causal RLS         (dmac_identify)
%    2) Control synthesis via LQR/idare on the estimate      (dmac_synthesize_gain)
%    3) Control law evaluation with excitation for PE        (dmac_control_law)
%
%  See CHANGELOG.md for version history and open issues.
%
%  Requires: Control System Toolbox (c2d, lqr, idare, ctrb)
%            DMAC_Functions_V2/ in the MATLAB path (for plot_DMAC_results)
%
% ========================================================================

clc;
clear;
close all;
addpath DMAC_Functions_V3/

randn('state', 2);

%% Simulation parameters
sim.N  = 10000;
sim.dt = 0.1;

%% Plant and DMAC configuration
plant = config_plant(sim.dt);
dmac  = config_dmac(plant);
validate_dmac_config(plant, dmac);

%% Memory allocation
% x, y, xi, q, u are sized N+1: the loop writes both the current-step
% (k) and next-step (k+1) values on each iteration.
log = initialize_logs(dmac, sim.N);
x   = zeros(plant.lx,  sim.N + 1);
y   = zeros(plant.ly,  sim.N + 1);
xi  = zeros(plant.lxi, sim.N + 1);
q   = zeros(plant.ly,  sim.N + 1);
u   = zeros(plant.lu,  sim.N + 1);

Theta_k = zeros(dmac.lxi, dmac.lxi + dmac.lu);
P_k     = inv(dmac.R0);
K       = zeros(dmac.lu, dmac.lxi + dmac.ly);

%% Initial conditions
x(:, 1) = randn(plant.lx, 1);
q(:, 1) = 0;
r       = 1;                     % reference command

%% Main simulation loop
for k = 1:sim.N

    % True plant propagation
    [x(:,k+1), y(:,k), xi(:,k)] = plant_step(plant, x(:,k), u(:,k));
    y(:,k+1)  = plant.C * x(:,k+1);
    xi(:,k+1) = plant.C_xi * x(:,k+1);

    % Integral state update (dt-scaled: q approximates the continuous-time
    % integral of the tracking error, independent of the chosen sample
    % rate; see CHANGELOG.md for the tuning implication of this scaling).
    e_k = r - y(:,k);
    q(:,k+1) = q(:,k) + e_k*sim.dt;

    % Identification and gain synthesis use only the PREVIOUS transition
    % (xi(k-1),u(k-1) -> xi(k)), i.e. data that was available a full
    % sample period (dt) before it's needed below. This gives the RLS
    % update + Riccati solve a full dt of computation budget before the
    % resulting K is applied to xi(k+1)/q(k+1) — see CHANGELOG.md
    % ("computation-time buffer for K"). No update happens at k=1 (no
    % k-1 data yet); Theta/P/K retain their initial values for that step.
    if k > 1
        [Theta_k, P_k] = dmac_identify(xi(:,k-1), u(:,k-1), xi(:,k), Theta_k, P_k, dmac.lambda);
        K              = dmac_synthesize_gain(Theta_k, dmac, K);
    end

    % Full-state feedback: u(k+1) = K*xi(k+1) + (integral term). This
    % instantaneous state-to-input evaluation (a single matrix multiply)
    % is standard and assumed zero-time; K itself was computed above
    % using only data through step k, i.e. one full sample period earlier.
    u(:,k+1) = dmac_control_law(K, xi(:,k+1), q(:,k+1), dmac);

    % Log
    log.Y(:,k) = y(:,k);
    log.U(:,k) = u(:,k);
    log.Theta_vec(:,k) = Theta_k(:);

    fprintf('step %d\n', k);
end

%% Results
plot_DMAC_results(log, r);

disp('True Ad and Bd:')
disp([plant.A plant.B])

disp('Estimated [A B]:')
disp(Theta_k)