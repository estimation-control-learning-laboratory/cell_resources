clc
clear all
close all

% makes symbolic gains
syms("K",[3,6]);
syms("Ki",[3,3]);



syms rollRate pitchRate yawRate roll pitch yaw rollError pitchError yawError rollErrorInt pitchErrorInt yawErrorInt real
syms t_x t_y t_z real
x = [rollRate; pitchRate; yawRate; roll; pitch; yaw];
C = [zeros(3), eye(3)];
y = C*x;



u_Kx = K*x;



u_Ki = Ki*[rollErrorInt; pitchErrorInt; yawErrorInt];
% J is from https://groups.csail.mit.edu/robotics-center/public_papers/Landry15.pdf
% page 39, table 4.1
J = diag([2.3951*1e-5, 2.3951*1e-5, 3.2347*1e-5]);



% roll rate; pitch rate; yaw rate; roll; pitch; yaw % note the 'rates' are
% body rates, not euler rates.



x = [rollRate; pitchRate; yawRate; roll; pitch; yaw];
u = [t_x; t_y; t_z];



S = S_Phi_Theta_inv(pitch,roll);



xdot = x*0;



omega = [rollRate; pitchRate; yawRate];
omega_dot = inv(J)*(u - cross(omega,J*omega));
xdot(1:3,1) = omega_dot;
xdot(4:6,1) =S*omega;







% discretetize the system
A = jacobian(xdot, x);
B = jacobian(xdot, u);
Ca = [zeros(3), eye(3)];



A_lin = subs(A,{'roll','pitch','pitchRate','rollRate','yawRate'},[0,0,0,0,0])
double(B)


system_disc = c2d(ss(double(A_lin), double(B), double(Ca), []),1/1000,'zoh');
[Aa_disc, Ba_disc] = augmentMatriciesDiscrete(system_disc.A, system_disc.B, system_disc.C);



system_disc

function [Aa, Ba] = augmentMatriciesDiscrete(A, B, C)



[ny, ~] = size(C);
[~, nu] = size(B);
nx = size(A,1);



D = zeros(ny,nu);

Aa = [A, zeros(nx,ny);
    -C, eye(ny,ny)];
Ba = [B; -D];
end

function S = S_Phi_Theta_inv(Phi,Theta)
S = [1 sin(Phi)*tan(Theta) cos(Phi)*tan(Theta);
    0 cos(Phi) -sin(Phi);
    0 sin(Phi)*sec(Theta) cos(Phi)*sec(Theta)];
end