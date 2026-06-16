clc; clear; close all;
addpath("AttitudeFunctions\")
%% Parameters
dt = 0.01;

J = diag([0.02, 0.025, 0.03]);

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

for k = 1:1000

    omega_ref = attitude_outer_loop(q_k,q_ref,Kq);  % Outer-loop attitude controller
    tau_k = Komega*(omega_ref - omega_k);       % Inner-loop angular-rate controller

    % Propagate one step
    [q_k, omega_k] = attitudeDynamicsStep(q_k, omega_k, tau_k, J, dt);

    q_hist(:,k)     = q_k;
    omega_hist(:,k) = omega_k;
    omega_ref_hist(:,k) = omega_ref;
    tau_hist(:,k)   = tau_k;

end

t = (0:size(q_hist,2)-1)*dt;

subplot(3,1,1);
hQ = plot(t,q_hist','LineWidth',1.2);
hold on;

for i = 1:3
    plot(t,q_ref(i)*ones(size(t)), ...
        '--', ...
        'Color',hQ(i).Color, ...
        'LineWidth',1.0);
end

grid on;
ylabel('q');
legend('$q_1$','$q_2$','$q_3$', ...
       '$q_{1,\rm ref}$','$q_{2,\rm ref}$','$q_{3,\rm ref}$', ...
       'Interpreter','latex','Location','best');

subplot(3,1,2);
hW = plot(t,omega_hist','LineWidth',1.2);
hold on;

for i = 1:3
    plot(t,omega_ref_hist(i,:), ...
        '--', ...
        'Color',hW(i).Color, ...
        'LineWidth',1.0);
end

grid on;
ylabel('\omega');
legend('$\omega_1$','$\omega_2$','$\omega_3$', ...
       '$\omega_{1,\rm ref}$','$\omega_{2,\rm ref}$','$\omega_{3,\rm ref}$', ...
       'Interpreter','latex','Location','best');

subplot(3,1,3);
plot(t,tau_hist','LineWidth',1.2);
grid on;
ylabel('\tau');
xlabel('Time [s]');






