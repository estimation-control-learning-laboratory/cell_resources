%% ========================================================================
%  DYNAMIC MODE ADAPTIVE CONTROL (DMAC) SIMULATION
%  ========================================================================
%
%  DESCRIPTION:
%  ------------------------------------------------------------------------
%  This script implements Dynamic Mode Adaptive Control (DMAC) for a 
%  discrete-time mass-spring-damper system. The algorithm performs:
%
%  1) Online system identification using Recursive Least Squares (RLS)
%     to estimate the local linear model:
%           xi_{k+1} = Theta_k [xi_k; u_k]
%
%  2) Control synthesis using the estimated model via full-state feedback
%     with integral action for reference tracking.
%
%  3) Closed-loop simulation with excitation to ensure persistence of 
%     excitation (PE) for parameter convergence.
%
%  Key implementation detail:
%  - The RLS update uses a causal formulation:
%        (phi_{k-1}, xi_k)
%    instead of the non-causal (phi_k, xi_{k+1})
%
%  ------------------------------------------------------------------------
%  AUTHOR:
%  ------------------------------------------------------------------------
%  Parham Oveissi, PhD Candidate
%  Ankit Goel, Assistant Professor
%  Mechanical Engineering
%  University of Maryland, Baltimore County (UMBC)
%
%  ------------------------------------------------------------------------
%  DATE:
%  ------------------------------------------------------------------------
%  March 2026
%
%  ------------------------------------------------------------------------
%  CHANGE LOG:
%  ------------------------------------------------------------------------
%  v3  (2026_03_24) - Initial implementation
%        - DMAC with RLS-based identification
%        - Mass-spring-damper example
%        - Integral action for tracking
%
%  v4  (2026_03_24) - Implementation correction
%        - Fixed non-causal RLS update
%        - Introduced (phi_{k-1}, xi_k) formulation
%
%  v5  (2026_03_25) - Refactoring
%        - Encapsulated update in DMAC_update()
%        - Improved modularity and readability
%
%  v6  (2026_03_25) - Combined controller update in a single step
%
%  ------------------------------------------------------------------------
%  Pending Issues:
%   - (2026_03_25) the integrator state should be a discrete update
%   - (2026_03_25) timing to be checked. u_k cannot depend on xi_k
%  ------------------------------------------------------------------------
%  ------------------------------------------------------------------------
%  NOTES:
%  ------------------------------------------------------------------------
%  - Requires Control System Toolbox (for c2d)
%  - Requires DMAC_Functions_V2/ in MATLAB path
%  - Designed for research and prototyping (not optimized for real-time)
%
% ========================================================================

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
sim.N  = 10000;
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
dmac.R      = 1e0*eye(plant.lu);
dmac.v_std  = 1e-2;                     % Excitation Signal

dmac.C_xi   = [1 0];
dmac.lxi    = plant.lxi;
dmac.lu     = plant.lu;
dmac.ly     = plant.ly;
%% ========================================================================
%  MEMORY ALLOCATION
%  ========================================================================
log = initialize_logs(dmac, sim.N);
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
    
    % -------------------------------------------------
    % DMAC Update
    % -------------------------------------------------
    if k>1
        % [Theta_k, P_k, K] = DMAC_update(phi(:,k-1), xi(:,k), Theta_km1, P_km1, K, dmac);
        [u(:,k),Theta_k, P_k, K]  = DMAC_compute_control(xi(:,k),q(:,k), xi(:,k-1),u(:,k-1), dmac, Theta_km1, P_km1, K);
    else
        [u(:,k),Theta_k, P_k, K]  = DMAC_compute_control(xi(:,k),q(:,k), zeros(plant.lxi, 1),zeros(plant.lu, 1), dmac, Theta_km1, P_km1, K);
    end
    Theta_km1       = Theta_k;
    P_km1           = P_k;

   
    % -------------------------------------------------
    % True plant propagation
    % -------------------------------------------------
    [x(:,k+1), y(:,k), xi(:,k)] = plant_step(plant, x(:,k), u(:,k));
    
    % Update y(k+1) and xi(k+1) at k. 
    % THIS SHOULD NOT BE USED.
    % Not sure why this is impacting performance
    y(:,k+1)    = plant.C*x(:,k+1);
    xi(:,k+1)   = plant.C_xi*x(:,k+1);

    % -------------------------------------------------
    % Integral state update
    % -------------------------------------------------
    e_k = r - y(:,k);
    q(:,k+1) = q(:,k) + e_k * sim.dt;

    

    % -------------------------------------------------
    % Log current data
    % -------------------------------------------------
    log.Y(:,k)    = y(:,k);
    log.U(:,k)    = u(:,k);
    log.Theta_vec(:,k) = Theta_k(:);


    fprintf('step %d\n', k);
end



%% ========================================================================
%  RESULTS
%  ========================================================================
plot_DMAC_results(log, r);

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

function log = initialize_logs(dmac, N)
% log.X         = zeros(lx, N);
log.U         = zeros(dmac.lu, N);
log.Y         = zeros(dmac.ly, N);
log.Theta_vec = zeros(dmac.lxi*(dmac.lxi+dmac.lu), N);
end

% function u_k = compute_control_input(K_aug, x_aug_k, dmac)
% excitation_signal  = dmac.v_std * randn;
% u_k = K_aug * x_aug_k + excitation_signal;
% end
