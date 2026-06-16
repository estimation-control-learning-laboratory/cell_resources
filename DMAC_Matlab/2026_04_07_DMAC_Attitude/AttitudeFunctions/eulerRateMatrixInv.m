
function Sinv = eulerRateMatrixInv(phi, theta)

cphi = cos(phi);
sphi = sin(phi);
cth  = cos(theta);

if abs(cth) < 1e-6
    warning('Euler-angle singularity approached: cos(theta) is near zero.');
end

Sinv = [ 1, sphi*tan(theta), cphi*tan(theta);
    0, cphi,           -sphi;
    0, sphi/cth,        cphi/cth ];

end