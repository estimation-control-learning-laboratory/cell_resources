function [x_next, y_k, xi_k] = plant_step(plant, x_k, u_k)
%PLANT_STEP Propagate the true plant one discrete time step.
%
%   [x_next, y_k, xi_k] = PLANT_STEP(plant, x_k, u_k)
%     x_next : next full state,        x_{k+1} = A x_k + B u_k
%     y_k    : current tracked output, y_k     = C x_k
%     xi_k   : current measured state, xi_k    = C_xi x_k
%
%   NOTE: xi_k = plant.C_xi * x_k (previously hardcoded to xi_k = x_k,
%   which only happened to be correct when plant.lxi == plant.lx).

x_next = plant.A * x_k + plant.B * u_k;
y_k    = plant.C * x_k;
xi_k   = plant.C_xi * x_k;
end
