function dmac = config_dmac(plant)
%CONFIG_DMAC Build the DMAC algorithm configuration struct.
%
%   dmac = CONFIG_DMAC(plant) returns a struct with the RLS/LQR tuning
%   parameters used by DMAC_IDENTIFY, DMAC_SYNTHESIZE_GAIN and
%   DMAC_CONTROL_LAW.
%
%   Fields:
%     lambda   : RLS forgetting factor
%     R0       : initial RLS covariance,        size (lxi+lu) x (lxi+lu)
%     Q        : LQR state penalty on the augmented [xi; q] state,
%                size (lxi+ly) x (lxi+ly). NOTE: the integrator state q
%                has dimension ly (it is built from y_k = Cy_xi*xi_k),
%                NOT lu — only build Q's block sizes from lxi/ly.
%     R        : LQR control penalty, size lu x lu
%     v_std    : std. dev. of excitation signal added to u_k (for PE)
%     Cy_xi    : maps measured state xi_k to tracked output y_k,
%                size ly x lxi. Used to build the integral-action
%                augmented system. Must satisfy
%                    Cy_xi * plant.C_xi == plant.C
%                (checked in VALIDATE_DMAC_CONFIG).
%     lxi, lu, ly : dimensions, copied from plant for convenience
%     integrator  : 'yes' | 'no' — whether to use integral action
%
%   NOTE: R0 is sized from dmac.lxi + dmac.lu (the RLS regressor
%   dimension). Q is sized from dmac.lxi + dmac.ly (the augmented
%   [xi; q] state used by the LQR/idare synthesis) — these previously
%   both coincided with sizing off plant.lx and/or plant.lu only because
%   plant.lxi == plant.lx and plant.lu == plant.ly in this example.
%
%   See also: CONFIG_PLANT, VALIDATE_DMAC_CONFIG

dmac.lxi = plant.lxi;
dmac.lu  = plant.lu;
dmac.ly  = plant.ly;

dmac.lambda = 0.995;                              % RLS forgetting factor
dmac.R0     = 1e2 * eye(dmac.lxi + dmac.lu);      % initial RLS covariance

% LQR state penalty on the augmented [xi; q] state. Built as separate
% blocks so the integrator weight can be tuned independently of the
% xi-state weight (see CHANGELOG.md for the dt-scaling discussion behind
% the 100x integrator weight below).
dmac.Q      = blkdiag(eye(dmac.lxi), 100*eye(dmac.ly));
dmac.R      = 1e4 * eye(dmac.lu);                 % LQR control penalty
dmac.v_std  = 1e-2;                               % excitation magnitude

dmac.Cy_xi  = [1 0];                              % y_k = Cy_xi * xi_k

dmac.integrator = 'yes';
end