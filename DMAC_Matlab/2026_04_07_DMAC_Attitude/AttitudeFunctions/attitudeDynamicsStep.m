
function [q_kp1, omega_kp1] = attitudeDynamicsStep(q_k, omega_k, tau, J, dt)

x_k = [q_k; omega_k];

odefun = @(t,x) attitudeDynamicsODE(t, x, tau, J);

opts = odeset('RelTol',1e-8,'AbsTol',1e-10);

[~,x_hist] = ode45(odefun, [0 dt], x_k, opts);

x_kp1 = x_hist(end,:)';

q_kp1     = x_kp1(1:3);
omega_kp1 = x_kp1(4:6);

end