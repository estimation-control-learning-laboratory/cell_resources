%% ========================================================================
%  PLOT_DMAC_RESULTS
%  ========================================================================
%
%  DESCRIPTION:
%  ------------------------------------------------------------------------
%  This function visualizes the closed-loop performance of the Dynamic Mode
%  Adaptive Control (DMAC) algorithm. It generates a 2x2 tiled figure with:
%
%  (a) Output tracking performance:
%      - System output y_k compared against constant reference r
%
%  (b) Control input:
%      - Control signal u_k over time
%
%  (c) Tracking error (log scale):
%      - Absolute error |y_k - r| plotted on a semilog scale to highlight
%        convergence behavior
%
%  (d) Parameter evolution:
%      - Time evolution of the estimated parameter matrix Theta_k (vectorized)
%
%  ------------------------------------------------------------------------
%  INPUTS:
%  ------------------------------------------------------------------------
%  log : structure containing simulation data
%        - log.Y          : output trajectory (y_k)
%        - log.U          : control input (u_k)
%        - log.Theta_vec  : vectorized parameter estimates over time
%
%  r   : reference value (assumed constant scalar)
%
%  ------------------------------------------------------------------------
%  OUTPUT:
%  ------------------------------------------------------------------------
%  Generates a figure with four subplots for qualitative assessment of:
%      - tracking performance
%      - control effort
%      - convergence rate
%      - parameter adaptation
%
%  ------------------------------------------------------------------------
%  NOTES:
%  ------------------------------------------------------------------------
%  - Assumes log.Y is 1 x N (SISO output). Modify plotting if multi-output.
%  - Error plot uses semilog scale for improved visibility of convergence.
%  - Theta is plotted in vectorized form; interpretation depends on model
%    structure (Theta = [A B]).
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
%  v1.0  (2026_03_25) - Initial implementation
%        - Standard 2x2 DMAC performance visualization
%
%  ========================================================================


function plot_DMAC_results(log, r)
figure
set(gcf, 'position', [200, 100, 800, 500])

t = tiledlayout(2, 2);
t.TileSpacing = 'compact';
t.Padding = 'compact';

nexttile(1)
ref_sig = r * ones(1, size(log.Y,2));
plot(ref_sig, 'k--', 'LineWidth', 2)
hold on
plot(log.Y', 'b', 'LineWidth', 3)
hold off
set(gca, 'fontsize', 16);
set(gca, 'xticklabel', {[]})
xlabel('a)', 'Interpreter', 'latex', 'fontsize', 22)
ylabel('$y_k$', 'Interpreter', 'latex', 'fontsize', 22)
legend('Reference','')
axis tight
grid on

nexttile(2)
plot(log.U', 'b', 'LineWidth', 3)
set(gca, 'fontsize', 16);
set(gca, 'xticklabel', {[]})
xlabel('b)', 'Interpreter', 'latex', 'fontsize', 22)
ylabel('$u_k$', 'Interpreter', 'latex', 'fontsize', 22)
axis tight
grid on

nexttile(3)
semilogy(abs(log.Y' - r), 'b', 'LineWidth', 3)
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