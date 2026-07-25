function u_k = dmac_control_law(K, xi_k, q_k, dmac)
%DMAC_CONTROL_LAW Evaluate the DMAC control law given a gain and state.
%
%   u_k = DMAC_CONTROL_LAW(K, xi_k, q_k, dmac)
%     dmac.integrator == 'no'  : u_k = K * xi_k               + excitation
%     dmac.integrator == 'yes' : u_k = K(:,1:lxi)*xi_k + K(:,lxi+1:end)*q_k
%                                       + excitation
%
%   A small excitation signal (std dmac.v_std) is always added to keep
%   the closed loop persistently exciting for RLS convergence.
%
%   See also: DMAC_SYNTHESIZE_GAIN

excitation = dmac.v_std * randn(dmac.lu, 1);

if strcmp(dmac.integrator, 'no')
    u_k = K(:, 1:dmac.lxi) * xi_k + excitation;
else
    u_k = K(:, 1:dmac.lxi) * xi_k + K(:, dmac.lxi+1:end) * q_k + excitation;
end
end
