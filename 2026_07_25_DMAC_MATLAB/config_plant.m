function plant = config_plant(dt)
%CONFIG_PLANT Build the plant (physical system) configuration struct.
%
%   plant = CONFIG_PLANT(dt) returns a struct describing the discrete-time
%   mass-spring-damper plant used by the DMAC simulation, discretized at
%   sample time dt.
%
%   Fields:
%     lx      : full state dimension
%     ly      : output dimension (for reference tracking)
%     lu      : input dimension
%     lxi     : measured/partial state dimension. May be SMALLER than lx
%               to exercise partial-state identification.
%     m,ks,c  : mass-spring-damper physical parameters
%     A, B    : discrete-time state matrices, x_{k+1} = A x_k + B u_k
%     C       : output map,          y_k  = C * x_k          (ly  x lx)
%     C_xi    : measurement map,     xi_k = C_xi * x_k        (lxi x lx)
%
%   NOTE: C_xi must be consistent with dmac.Cy_xi such that
%       dmac.Cy_xi * plant.C_xi == plant.C
%   i.e. the tracked output y_k must be recoverable from the measured
%   state xi_k alone. This is checked in VALIDATE_DMAC_CONFIG.
%
%   See also: CONFIG_DMAC, VALIDATE_DMAC_CONFIG, PLANT_STEP

plant.lx  = 2;
plant.ly  = 1;
plant.lu  = 1;
plant.lxi = 2;      % set < plant.lx to test partial-state identification

plant.m  = 1;
plant.ks = 2;
plant.c  = 0.5;

[plant.A, plant.B] = build_discrete_mass_spring_damper(plant, dt);

plant.C    = [1 0];                          % y_k  = C * x_k
plant.C_xi = eye(plant.lxi, plant.lx);       % xi_k = C_xi * x_k
                                              % (selects the first lxi
                                              %  states; replace with a
                                              %  general selection/
                                              %  combination matrix if
                                              %  a different measured
                                              %  subspace is needed)
end
