function [Ad, Bd] = VanLoanDiscretization(A, B, T)
% Computes discrete-time (Ad, Bd) from continuous-time (A, B)
% using block matrix exponential:
%
% exp([A B; 0 0]*T) = [Ad Bd; 0 I]

    n = size(A,1);
    m = size(B,2);

    % Construct block matrix
    M = [A, B;
         zeros(m,n), zeros(m,m)];

    % Matrix exponential
    Md = expm(M*T);

    % Extract Ad and Bd
    Ad = Md(1:n, 1:n);
    Bd = Md(1:n, n+1:n+m);
end