

function S = eulerRateMatrix(phi, theta)

S = [ 1, 0, -sin(theta);
    0, cos(phi), sin(phi)*cos(theta);
    0, -sin(phi), cos(phi)*cos(theta) ];

end