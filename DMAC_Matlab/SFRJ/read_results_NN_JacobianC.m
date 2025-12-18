clc; clear;
close all

load("DMAC_SFRJ_steps_250_lambda_0_995_R0_1e3_Q_0_1_R_1_sysdim_2_ref_1000_noise_1e_neg2_NNthrust_JacobianC_v3.mat")

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
set(gcf, 'position', [200, 100, 800, 700])
t = tiledlayout(3, 1);
t.TileSpacing = 'compact';
t.Padding = 'compact';

nexttile(1)
ref_sig = avg_AvgThrust*r;
plot(ref_sig, 'k--', 'LineWidth', 2)
hold on
plot(thrust*avg_AvgThrust, 'b', 'LineWidth', 3)
hold off
set(gca, 'fontsize', 16);
set(gca, 'xticklabel', {[]})
yticks(100:200:2000)
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

% print(gcf,'-dpng','Figures/png/DMAC_SFRJ_steps_400_lambda_0_995_R0_1e2_Q_1_R_1_sysdim_2_ref_1000_1200_noise_1e_neg2_NNthrust_JacobianC')
% print(gcf,'-depsc','Figures/eps/DMAC_SFRJ_steps_400_lambda_0_995_R0_1e2_Q_1_R_1_sysdim_2_ref_1000_1200_noise_1e_neg2_NNthrust_JacobianC')

figure(3)
set(gcf, 'position', [200, 100, 1200, 700])
t = tiledlayout(2, 1);
t.TileSpacing = 'compact';
t.Padding = 'compact';

nexttile(1)
heatflux_array = heat_flux_nominal - 1e5 * U;
plot(heatflux_array, 'bo', 'LineWidth', 1)
set(gca, 'fontsize', 16);
xlabel('step ($k$)', 'Interpreter', 'latex', 'fontsize', 22)
ylabel('Heat Flux', 'Interpreter', 'latex', 'fontsize', 22)
axis tight
grid on

nexttile(2)
plot(X(end,:)*avg_AvgThrust, 'bo', 'LineWidth', 1)
hold off
set(gca, 'fontsize', 16);
xlabel('step ($k$)', 'Interpreter', 'latex', 'fontsize', 22)
ylabel('Thrust', 'Interpreter', 'latex', 'fontsize', 22)
axis tight
grid on

% vel_outX_1 = X(1,201:end)*avg_avg_vel_outX;
% P_out_1 = X(2,201:end)*avg_Pout;
% Thrust_1 = X(3,201:end)*avg_AvgThrust;
% save('data_NN.mat', 'vel_outX_1', 'P_out_1','Thrust_1')


figure(4)
set(gcf, 'position', [200, 100, 800, 500])
t = tiledlayout(2, 2);
t.TileSpacing = 'compact';
t.Padding = 'compact';

nexttile(1)
plot(thrust*avg_AvgThrust, 'b', 'LineWidth', 3)
hold on
% ref_sig = r*ones(1, size(X_aug,2));
plot(ref_sig, 'k--', 'LineWidth', 2)
set(gca, 'fontsize', 16);
set(gca, 'xticklabel', {[]})
xlabel('a)', 'Interpreter', 'latex', 'fontsize', 22)
ylabel('$y_k$', 'Interpreter', 'latex', 'fontsize', 22)
legend('','Reference', location = 'best')
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

% print(gcf,'-dpng','Figures/png/DMAC_SFRJ_steps_250_lambda_0_995_R0_1e2_Q_1_R_1_sysdim_2_ref_1000_noise_1e_neg2_NNthrust_JacobianC_v3')
% print(gcf,'-depsc','Figures/eps/DMAC_SFRJ_steps_250_lambda_0_995_R0_1e2_Q_1_R_1_sysdim_2_ref_1000_noise_1e_neg2_NNthrust_JacobianC_v3')

%% New data

% clc; clear
% 
% load('DMAC_SFRJ_steps_300_lambda_0_995_R0_1e2_Q_1_R_1_sysdim_3_ref_1000_noise_1e_neg2.mat')
% 
% figure(4)
% set(gcf, 'position', [200, 100, 1200, 700])
% t = tiledlayout(2, 1);
% t.TileSpacing = 'compact';
% t.Padding = 'compact';
% 
% nexttile(1)
% heatflux_array = heat_flux_nominal - 1e5 * U;
% plot(heatflux_array, 'bo', 'LineWidth', 1)
% set(gca, 'fontsize', 16);
% xlabel('step ($k$)', 'Interpreter', 'latex', 'fontsize', 22)
% ylabel('Heat Flux', 'Interpreter', 'latex', 'fontsize', 22)
% axis tight
% grid on
% 
% nexttile(2)
% plot(X(end,:)*avg_AvgThrust, 'bo', 'LineWidth', 1)
% hold off
% set(gca, 'fontsize', 16);
% xlabel('step ($k$)', 'Interpreter', 'latex', 'fontsize', 22)
% ylabel('Thrust', 'Interpreter', 'latex', 'fontsize', 22)
% axis tight
% grid on
% 
% vel_outX_2 = X(1,:)*avg_avg_vel_outX;
% P_out_2 = X(2,:)*avg_Pout;
% Thrust_2 = X(3,:)*avg_AvgThrust;
% save('data_NN.mat', 'vel_outX_2', 'P_out_2','Thrust_2', '-append')





