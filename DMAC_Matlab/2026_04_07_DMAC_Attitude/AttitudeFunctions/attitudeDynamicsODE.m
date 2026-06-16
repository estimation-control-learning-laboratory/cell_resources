function xdot = attitudeDynamicsODE(~, x, tau, J)

q     = x(1:3);
omega = x(4:6);

phi   = q(1);
theta = q(2);

% Euler-angle kinematics
Sinv = eulerRateMatrixInv(phi, theta);
qdot = Sinv*omega;

% Rigid-body rotational dynamics
omega_dot = J\(tau - cross(omega, J*omega));

xdot = [qdot; omega_dot];

end