function [Ad, Bd] = build_discrete_mass_spring_damper(plant, dt)
%BUILD_DISCRETE_MASS_SPRING_DAMPER Discretize the mass-spring-damper plant.
%
%   [Ad, Bd] = BUILD_DISCRETE_MASS_SPRING_DAMPER(plant, dt) builds the
%   continuous-time mass-spring-damper state matrices from plant.m,
%   plant.ks, plant.c and discretizes them at sample time dt using the
%   Tustin (bilinear) transform.
%
%   Requires the Control System Toolbox (ss, c2d).

A = [0                 1;
     -plant.ks/plant.m -plant.c/plant.m];
B = [0; 1/plant.m];

sys_c = ss(A, B, eye(plant.lx), 0);
sys_d = c2d(sys_c, dt, 'tustin');

Ad = sys_d.A;
Bd = sys_d.B;
end
