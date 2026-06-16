
function omega_ref = attitude_outer_loop(q,q_ref, Kq)

% Outer-loop attitude controller
phi = q(1);
theta = q(2);
S = eulerRateMatrix(phi, theta);
omega_ref = S*Kq*(q_ref - q);

end