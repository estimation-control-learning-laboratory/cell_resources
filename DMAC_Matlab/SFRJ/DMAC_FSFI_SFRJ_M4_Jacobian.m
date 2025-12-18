clc; clear; close all;
warning('off', 'all')
tic

load('trained_thrust_nn.mat', 'net', 'mu_X', 'sigma_X');

addpath Functions\

% Simulation Parameters
inputFilePath = 'prob_setup_test.cfg';
outputFilePath = 'prob_setup_test.cfg';

heat_flux_nominal = 2000000;
thrust_ref = 1.448e+03;
steps = 250;
sys_dim = 2;  % last: 5

% DMAC Parameters
lambda = 0.995; % Forgetting factor for online estimation
R0 = 1e2*eye(sys_dim + 1); % [x; u]: sys_dim + inputdim(=1)

Q = eye(sys_dim + 1); % Augmented state weighting
R = 1;
E = eye(sys_dim + 1);
Z = zeros(sys_dim + 1, 1);
P_k = inv(R0);
AB_est = zeros(sys_dim, sys_dim + 1);
K = zeros(1, sys_dim + 1);
% K = randn(1, sys_dim + 1);

% Memory Initialization
X = zeros(sys_dim, steps);
Y = zeros(sys_dim, steps);
U = zeros(1, steps);
X_Aug = zeros(sys_dim + 1, steps);
thrust = zeros(1, steps);

% Initialize CFD Simulation
heat_flux = heat_flux_nominal;
prev = ['MARKER_HEATFLUX= ( wall1, 0.0, wall2, ', num2str(heat_flux_nominal), ', wall3, 0.0 )'];
new = ['MARKER_HEATFLUX= ( wall1, 0.0, wall2, ', num2str(heat_flux), ', wall3, 0.0 )'];
replaceWordInCfgFile(inputFilePath, outputFilePath, prev, new);
system('del solution_flow.dat')
system('copy solution_flow_2mill.dat solution_flow.dat')
system('del restart_flow.dat')
system('del history.csv')
system('SU2_CFD prob_setup_test.cfg');

% System Dynamics
% C = [zeros(1,sys_dim-1) 1];

% Read Initial States from history.csv
Output_Data = readtable("history.csv");

avg_avg_vel_inX = Output_Data.avg_vel_inX(end);
avg_avg_vel_outX = Output_Data.avg_vel_outX(end);
avg_Pin = Output_Data.Pin(end);
avg_Pout = Output_Data.Pout(end);
avg_AvgThrust = Output_Data.AvgThrust(end) + thrust_ref;


% x(1,1) = Output_Data.avg_vel_inX(end)/avg_avg_vel_inX;
x(1,1) = Output_Data.avg_vel_outX(end)/avg_avg_vel_outX;  % last: x(2,1)
% x(3,1) = Output_Data.Pin(end)/avg_Pin;
x(2,1) = Output_Data.Pout(end)/avg_Pout;                  % last: x(4,1)
% x(3,1) = (Output_Data.AvgThrust(end) + thrust_ref)/avg_AvgThrust; % last: x(5,1)

% x(1,1) = Output_Data.avg_vel_inX(end);
% x(2,1) = Output_Data.avg_vel_outX(end);
% x(3,1) = Output_Data.Pin(end);
% x(4,1) = Output_Data.Pout(end);
% x(5,1) = Output_Data.AvgThrust(end) + thrust_ref;

x_int = 0; % Integral state initialization

% ref = 1000/avg_AvgThrust;  % last: 1500

refs = [1000/avg_AvgThrust 1000/avg_AvgThrust];
r = zeros(1,steps);


for k = 1:steps
    %% New
    if k < steps/2
        ref = refs(1);
    else
        ref = refs(2);
    end
    r(k) = ref;
    %%

    X(:,k) = x;
    
    C = compute_adaptive_jacobian([Output_Data.avg_vel_outX(end); Output_Data.Pout(end)], net, mu_X, sigma_X);

    % Tracking error and integral state
    thrust(k) = net(([Output_Data.avg_vel_outX(end); Output_Data.Pout(end)] - mu_X) ./ sigma_X);
    thrust(k) = thrust(k)/avg_AvgThrust;

    error = ref - thrust(k);
    x_int = x_int + error;

    % Augmented state vector
    x_aug = [x; x_int];
    X_Aug(:, k) = x_aug;

    % Compute control input
    u = K * x_aug + 1e-2*randn;  % last: 1e0
    U(k) = u;
    prev = ['MARKER_HEATFLUX= ( wall1, 0.0, wall2, ', num2str(heat_flux), ', wall3, 0.0 )'];

    % heat_flux = heat_flux_nominal + 1e5 * 10^u;
    heat_flux = heat_flux_nominal - 1e5 * u;
    % heat_flux = heat_flux_nominal + u;

    % Update CFD input file
    new = ['MARKER_HEATFLUX= ( wall1, 0.0, wall2, ', num2str(heat_flux), ', wall3, 0.0 )'];
    replaceWordInCfgFile(inputFilePath, outputFilePath, prev, new);
    system('del solution_flow.dat')
    system('rename restart_flow.dat solution_flow.dat')
    system('SU2_CFD prob_setup_test.cfg');


    % Read new state from history.csv
    Output_Data = readtable("history.csv");

    % x(1,1) = Output_Data.avg_vel_inX(end);
    % x(2,1) = Output_Data.avg_vel_outX(end);
    % x(3,1) = Output_Data.Pin(end);
    % x(4,1) = Output_Data.Pout(end);
    % x(5,1) = Output_Data.AvgThrust(end) + thrust_ref;

    % x(1,1) = Output_Data.avg_vel_inX(end)/avg_avg_vel_inX;
    x(1,1) = Output_Data.avg_vel_outX(end)/avg_avg_vel_outX;  % last: x(2,1)
    % x(3,1) = Output_Data.Pin(end)/avg_Pin;
    x(2,1) = Output_Data.Pout(end)/avg_Pout;                   % last: x(4,1)
    % x(3,1) = (Output_Data.AvgThrust(end) + thrust_ref)/avg_AvgThrust; % last: x(5,1)


    y = x;
    Y(:, k) = y;

    % Online Identification using DMAC
    x_bar_k = [X(:, k); U(k)];
    [P_k, AB_est, z_k(k)] = RLS_update_DMD(x_bar_k, y, P_k, AB_est, lambda);
    [A_est, B_est] = extract_A_B_DMD(AB_est, sys_dim);
    [A_est_aug, B_est_aug] = generate_augmented_A_B_DMD(A_est, B_est, C, sys_dim);

    % % Compute State-Feedback Gain
    % if rank(ctrb(A_est_aug, B_est_aug)) == sys_dim + 1
    %     [~, K, ~] = idare(A_est_aug, B_est_aug, Q, R, Z, E);
    %     K = -K;
    % else
    %     disp('System not controllable at step ' + string(k));
    % end

    %%
    % Check controllability
    if rank(ctrb(A_est_aug, B_est_aug)) < sys_dim + 1
        disp('Not Controllable!! -- Applying projection to controllable subspace')

        [T, A_est_aug_new, B_est_aug_new] = compute_controllable_form(A_est_aug, B_est_aug);

        try
            [~, K_new, ~] = idare(A_est_aug_new, B_est_aug_new, Q, R, Z, E);
            K = -K_new * T;  % Transform gain back to original coordinates
        catch
            disp('DARE failed even after projection. Keeping previous gain.')
            % keep K unchanged
        end
    else
        [~, K, ~] = idare(A_est_aug, B_est_aug, Q, R, Z, E);
        K = -K;
    end

    %%

    disp(['Step ' num2str(k) ' completed']);
end

toc

%%
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
t = tiledlayout(3, 1);
t.TileSpacing = 'compact';
t.Padding = 'compact';

nexttile(1)
plot(r, 'k--', 'LineWidth', 2)
hold on
plot(thrust, 'b', 'LineWidth', 3)
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
semilogy(abs(thrust-r), 'b', 'LineWidth', 3)
set(gca, 'fontsize', 16);
yticks(10.^(-5:1:5))
xlabel({'c)' 'step ($k$)'}, 'Interpreter', 'latex', 'fontsize', 22)
ylabel('$|z_k|$', 'Interpreter', 'latex', 'fontsize', 22)
axis tight
grid on


%%


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


