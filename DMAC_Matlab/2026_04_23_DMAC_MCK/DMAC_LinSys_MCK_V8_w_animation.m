%% ========================================================================
%  DYNAMIC MODE ADAPTIVE CONTROL (DMAC) SIMULATION
%  ========================================================================

clc;
clear;
close all;
addpath ../DMAC_Functions_V2/

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
plant.lx  = 2;
plant.ly  = 1;
plant.lu  = 1;
plant.lxi = 2;

plant.m  = 1;
plant.ks = 2;
plant.c  = 0.5;

[plant.A, plant.B] = build_discrete_mass_spring_damper(plant, sim.dt);
plant.C    = [1 0];
plant.C_xi = eye(2);

%% ========================================================================
%  DMAC PARAMETERS
%  ========================================================================
dmac.lambda = 0.995;
dmac.R0     = 1e2 * eye(plant.lx + plant.lu);
dmac.Q      = 1 * eye(plant.lx + plant.lu);
dmac.R      = 1e4 * eye(plant.lu);
dmac.v_std  = 1e-2;

dmac.C_xi   = [1 0];
dmac.lxi    = plant.lxi;
dmac.lu     = plant.lu;
dmac.ly     = plant.ly;

%% ========================================================================
%  MEMORY ALLOCATION
%  ========================================================================
log = initialize_logs(plant, dmac, sim.N);

x   = zeros(plant.lx,  sim.N+1);
y   = zeros(plant.ly,  sim.N+1);
u   = zeros(plant.lu,  sim.N+1);
xi  = zeros(plant.lxi, sim.N+1);
q   = zeros(plant.ly,  sim.N+1);

Theta_k = zeros(plant.lxi, plant.lxi + plant.lu);
P_k     = inv(dmac.R0);
K       = zeros(plant.lu, plant.lxi + plant.ly);

%% ========================================================================
%  INITIAL CONDITIONS
%  ========================================================================
x(:,1)  = randn(plant.lx,1);
y(:,1)  = plant.C * x(:,1);
xi(:,1) = plant.C_xi * x(:,1);
q(:,1)  = 0;
r       = 1;

%% ========================================================================
%  INITIALIZE ANIMATION FIGURE
%  ========================================================================
[hfig, h] = initialize_animation(sim, plant, log, r, Theta_k, K, x(:,1));

%% ========================================================================
%  MAIN SIMULATION LOOP
%  ========================================================================
for k = 1:sim.N

    % -------------------------------------------------
    % True plant propagation
    % -------------------------------------------------
    [x(:,k+1), y(:,k), xi(:,k)] = plant_step(plant, x(:,k), u(:,k));
    y(:,k+1)  = plant.C * x(:,k+1);     % Output to be tracked
    xi(:,k+1) = plant.C_xi * x(:,k+1);  % partial state
    e_k      = r - y(:,k);              % Output error
    q(:,k+1) = q(:,k) + e_k;            % Integral state

    % -------------------------------------------------
    % DMAC Update
    % -------------------------------------------------
    [u(:,k+1), Theta_kp1, P_kp1, K] = ...
        DMAC_compute_control(xi(:,k+1), q(:,k+1), xi(:,k), u(:,k), ...
        dmac, Theta_k, P_k, K);

    Theta_k = Theta_kp1;
    P_k     = P_kp1;

    % -------------------------------------------------
    % Log current data
    % -------------------------------------------------
    log.X(:,k)         = x(:,k);
    log.Y(:,k)         = y(:,k);
    log.U(:,k)         = u(:,k);
    log.Q(:,k)         = q(:,k);
    log.Theta_vec(:,k) = Theta_k(:);
    log.K_vec(:,k)     = K(:);



    fprintf('step %d / %d\n', k, sim.N);
end

%% ========================================================================
%  FINAL LOG ENTRY
%  ========================================================================
log.X(:,sim.N+1) = x(:,sim.N+1);
log.Y(:,sim.N+1) = y(:,sim.N+1);
log.U(:,sim.N+1) = u(:,sim.N+1);
log.Q(:,sim.N+1) = q(:,sim.N+1);

%% ========================================================================
%  RESULTS
%  ========================================================================
plot_DMAC_results(log, r);

disp('True Ad and Bd:')
disp([plant.A plant.B])

disp('Estimated [A B]:')
disp(Theta_k)

%% ========================================================================
%  ANIMATION / GIF SETTINGS
%  ========================================================================
anim.enable_gif      = true;
anim.filename        = 'DMAC_mass_spring_damper.gif';
anim.playback_speed  = 2;     % 2x playback
anim.update_every    = 5;     % update figure every step

% -------------------------------------------------
% Update plots / animation
% -------------------------------------------------
for k = 1:1:sim.N
    if mod(k-1, anim.update_every) == 0
        update_animation(h, log, x(:,k), r, sim, k);

        drawnow;

        if anim.enable_gif
            frame = getframe(hfig);
            im    = frame2im(frame);
            [A, map] = rgb2ind(im, 256);
            delay = sim.dt / anim.playback_speed;

            if k == 1
                imwrite(A, map, anim.filename, 'gif', ...
                    'LoopCount', Inf, 'DelayTime', delay);
            else
                imwrite(A, map, anim.filename, 'gif', ...
                    'WriteMode', 'append', 'DelayTime', delay);
            end
        end
    end
end
%% ========================================================================
%  LOCAL FUNCTIONS
%  ========================================================================

function [x_next, y_k, xi_k] = plant_step(plant, x_k, u_k)
x_next = plant.A * x_k + plant.B * u_k;
y_k    = plant.C * x_k;
xi_k   = plant.C_xi * x_k;
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

function log = initialize_logs(plant, dmac, N)
log.X         = zeros(plant.lx, N+1);
log.U         = zeros(dmac.lu, N+1);
log.Y         = zeros(dmac.ly, N+1);
log.Q         = zeros(dmac.ly, N+1);
log.Theta_vec = zeros(dmac.lxi*(dmac.lxi+dmac.lu), N);
log.K_vec     = zeros(dmac.lu*(dmac.lxi+dmac.ly), N);
end

function [hfig, h] = initialize_animation(sim, plant, log, r, Theta_k, K, x0)

n_theta = numel(Theta_k);
n_K     = numel(K);

hfig = figure('Color','w','Position',[100 100 1200 800]);
t = tiledlayout(3,2,'TileSpacing','compact','Padding','compact');

% -------------------------------------------------
% Output tracking
% -------------------------------------------------
ax1 = nexttile(t,1);
hold(ax1,'on'); grid(ax1,'on');
h.y   = plot(ax1, nan, nan, 'LineWidth', 2);
h.ref = plot(ax1, nan, nan, '--', 'LineWidth', 2);
xlabel(ax1,'Step');
ylabel(ax1,'Output');
title(ax1,'Tracking Performance');

% -------------------------------------------------
% Control input
% -------------------------------------------------
ax2 = nexttile(t,2);
hold(ax2,'on'); grid(ax2,'on');
h.u = plot(ax2, nan, nan, 'LineWidth', 2);
xlabel(ax2,'Step');
ylabel(ax2,'u_k');
title(ax2,'Control Input');

% -------------------------------------------------
% Theta estimates
% -------------------------------------------------
ax3 = nexttile(t,3);
hold(ax3,'on'); grid(ax3,'on');
h.theta = gobjects(n_theta,1);
for i = 1:n_theta
    h.theta(i) = plot(ax3, nan, nan, 'LineWidth', 2);
end
xlabel(ax3,'Step');
ylabel(ax3,'\theta entries');
title(ax3,'Parameter Estimates');

% -------------------------------------------------
% K gains
% -------------------------------------------------
ax4 = nexttile(t,4);
hold(ax4,'on'); grid(ax4,'on');
h.K = gobjects(n_K,1);
for i = 1:n_K
    h.K(i) = plot(ax4, nan, nan, 'LineWidth', 2);
end
xlabel(ax4,'Step');
ylabel(ax4,'K entries');
title(ax4,'Feedback Gains');

% -------------------------------------------------
% Mass-spring animation
% -------------------------------------------------
ax5 = nexttile(t,[1 2]);
hold(ax5,'on'); grid(ax5,'on'); axis(ax5,'equal');
xlim(ax5,[-2 2]);
ylim(ax5,[-0.8 0.8]);
xlabel(ax5,'Position');
title(ax5,'Mass-Spring-Damper Animation');

% Wall
plot(ax5,[-1 -1],[-0.4 0.4],'k-','LineWidth',3);

pos0 = x0(1);

% Spring
h.spring = plot(ax5,[-1 pos0-0.2],[0 0],'LineWidth',2);

% Mass block
h.mass = rectangle(ax5,...
    'Position',[pos0-0.2 -0.2 0.4 0.4],...
    'Curvature',0.1,...
    'LineWidth',2);

% -------------------------------------------------
% Position markers
% -------------------------------------------------

% + at actual mass position (center of block)
h.mass_center = plot(ax5, pos0, 0, 'b+', ...
    'MarkerSize', 12, 'LineWidth', 2);

% + at desired position (initial reference)
r0 = r(1);   % assume r is available
h.ref_pos = plot(ax5, r0, 0, 'r+', ...
    'MarkerSize', 8, 'LineWidth', 2);

% Optional legend
legend(ax5, {'Wall','Spring','Actual Position','Desired Position'}, ...
    'Location','northoutside','Orientation','horizontal');


% Time text
h.text = text(ax5,-1.8,0.6,'time = 0.00 s','FontSize',12);

end

function update_animation(h, log, xk, r, sim, k)

% Output
set(h.y,   'XData',1:k,'YData',log.Y(1,1:k));
set(h.ref, 'XData',1:k,'YData',r*ones(1,k));

% Input
set(h.u, 'XData',1:k,'YData',log.U(1,1:k));

% Theta update
for i = 1:size(log.Theta_vec,1)
    set(h.theta(i), 'XData',1:k, 'YData',log.Theta_vec(i,1:k));
end

% K update
for i = 1:size(log.K_vec,1)
    set(h.K(i), 'XData',1:k, 'YData',log.K_vec(i,1:k));
end

% Animation
pos = xk(1);
set(h.mass, 'Position', [pos-0.2 -0.2 0.4 0.4]);
set(h.spring, 'XData', [-1 pos-0.2], 'YData', [0 0]);
set(h.text, 'String', sprintf('time = %.2f s', k*sim.dt));

% Update mass position
set(h.mass, 'Position', [pos-0.2 -0.2 0.4 0.4]);

% Update desired position marker
set(h.ref_pos, 'XData', r, 'YData', 0);

% Update actual position marker
set(h.mass_center, 'XData', pos, 'YData', 0);



end