function log = initialize_logs(dmac, N)
%INITIALIZE_LOGS Preallocate the logging struct for the simulation.
%
%   log = INITIALIZE_LOGS(dmac, N) preallocates N-sample logs for the
%   tracked output, control input, and the vectorized identified model.

log.U         = zeros(dmac.lu, N);
log.Y         = zeros(dmac.ly, N);
log.Theta_vec = zeros(dmac.lxi * (dmac.lxi + dmac.lu), N);
end
