clc;
clear;
close all;
addpath ../DMAC_Functions_V2/
addpath("AttitudeFunctions\")
%% ========================================================================
%  RANDOM SEED
%  ========================================================================
randn('state',2)

%% ========================================================================
%  SIMULATION PARAMETERS
%  ========================================================================
sim.N  = 6000;
sim.dt = 0.001;
sim.dt = 0.005;

%% ========================================================================
%  DMAC PARAMETERS
%  ========================================================================
dmac.integrator = 'no';
dmac.integrator = 'yes';

dmac.lx     = 3;
dmac.lxi    = 3;
dmac.lu     = 3;
dmac.ly     = 3;

dmac.lambda = 0.995;
dmac.R0     = 1e4*eye(dmac.lx + dmac.lu);   % regressor dimension = [xi;u]
dmac.Q      = 1*eye(dmac.lx + dmac.lu);     % augmented state = [xi; q]
dmac.R      = 1e0*eye(dmac.lu);
dmac.v_std  = 1e-2;                     % Excitation Signal

dmac.C_xi   = eye(3);

%% ========================================================================
%  MEMORY ALLOCATION
%  ========================================================================
log = initialize_logs(dmac, sim.N);

y   = zeros(dmac.ly, sim.N);
u   = zeros(dmac.lu, sim.N);
xi  = zeros(dmac.lxi, sim.N);
q   = y;

phi = zeros(dmac.lxi+dmac.lu, sim.N);
%% ========================================================================
%  INITIAL CONDITIONS
%  ========================================================================

Theta_k = zeros(dmac.lxi, dmac.lxi + dmac.lu);
P_k     = inv(dmac.R0);
% K_aug   = [0 0 0];
K       = zeros(dmac.lu, dmac.lxi+dmac.ly);

%% Parameters
dt = 0.01;

J = diag([0.02, 0.025, 0.03]);
J = diag([0.025, 0.025, 0.045]);

Kq = diag([2.0, 2.0, 2.0]);
Komega = 4*diag([0.3, 0.3, 0.3]);

% Linearized loop interpretation near hover:
%
% Outer loop:
%   qdot ≈ omega, omega_ref = Kq(q_ref - q)
%   e_q = q_ref - q  =>  edot_q = -Kq e_q
%   eig_outer = eig(-Kq)
%
% Inner loop:
%   omegadot ≈ J^{-1} tau, tau = Komega(omega_ref - omega)
%   e_omega = omega_ref - omega  =>  edot_omega = -J^{-1}Komega e_omega
%   eig_inner = eig(-J\Komega)
%
% For the gains below:
%   eig_outer = [-2, -2, -2]
%   eig_inner = [-60, -48, -40]
%
% Thus, the inner loop is much faster than the outer loop.


q_k     = [0.2; -0.1; 0.3];      % Euler angles [phi; theta; psi]
omega_k = [0.1; -0.2; 0.15];     % Body angular velocity [omega_1; omega_2; omega_3]
q_ref   = rand(3,1);
tau_k   = 0;

%% ========================================================================
%  MAIN SIMULATION LOOP
%  ========================================================================
gamma=zeros(3,1);
xi(:,1) = omega_k;
for k = 1:sim.N
    
    % if k==sim.N/2
    %     q_ref = rand(3,1);
    % end

    % -------------------------------------------------
    % True plant propagation
    % -------------------------------------------------
    [q_kp1, omega_kp1] = attitudeDynamicsStep(q_k, omega_k, tau_k, J, dt);
    xi(:,k+1) = omega_kp1;
    omega_ref = attitude_outer_loop(q_k,q_ref,Kq);  % Outer-loop attitude controller
    

    gamma(:,k+1) = gamma(:,k) + (omega_ref-omega_k)* sim.dt;


    [u(:,k+1),Theta_kp1, P_kp1, K]  = DMAC_compute_control(xi(:,k+1),gamma(:,k), xi(:,k),u(:,k), dmac, Theta_k, P_k, K);
    Theta_k       = Theta_kp1;
    P_k           = P_kp1;
    
    u(:,k+1) = max(min(u(:,k+1),0.2*[1;1;1]), -0.2*[1;1;1]);

    tau_k = u(:,k+1);   
    q_k = q_kp1;
    omega_k = omega_kp1;
    

    % -------------------------------------------------
    % Log current data
    % -------------------------------------------------
    log.Y(:,k)    = q_k;
    log.U(:,k)    = tau_k;
    log.Theta_vec(:,k) = Theta_kp1(:);

    q_hist(:,k)     = q_k;
    omega_hist(:,k) = omega_k;
    omega_ref_hist(:,k) = omega_ref;
    tau_hist(:,k)   = tau_k;


    fprintf('step %d\n', k);
end



%% ========================================================================
%  RESULTS
%  ========================================================================
plot_DMAC_results(log, 0);

t = (0:size(q_hist,2)-1)*dt;
figure
subplot(3,1,1);
hQ = plot(t,q_hist','LineWidth',2);
hold on;

for i = 1:3
    plot(t,q_ref(i)*ones(size(t)), ...
        '--', ...
        'Color',hQ(i).Color, ...
        'LineWidth',2.0);
end

grid on;
ylabel('q');
legend('$q_1$','$q_2$','$q_3$', ...
       '$q_{1,\rm ref}$','$q_{2,\rm ref}$','$q_{3,\rm ref}$', ...
       'Interpreter','latex','Location','best');

subplot(3,1,2);
hW = plot(t,omega_hist','LineWidth',2);
hold on;

for i = 1:3
    plot(t,omega_ref_hist(i,:), ...
        '--', ...
        'Color',hW(i).Color, ...
        'LineWidth',2);
end

grid on;
ylabel('\omega');
legend('$\omega_1$','$\omega_2$','$\omega_3$', ...
       '$\omega_{1,\rm ref}$','$\omega_{2,\rm ref}$','$\omega_{3,\rm ref}$', ...
       'Interpreter','latex','Location','best');

subplot(3,1,3);
plot(t,tau_hist','LineWidth',2);
grid on;
ylabel('\tau');
xlabel('Time [s]');





%% ========================================================================
%  LOCAL FUNCTIONS
%  ========================================================================



function log = initialize_logs(dmac, N)
% log.X         = zeros(lx, N);
log.U         = zeros(dmac.lu, N);
log.Y         = zeros(dmac.ly, N);
log.Theta_vec = zeros(dmac.lxi*(dmac.lxi+dmac.lu), N);
end
