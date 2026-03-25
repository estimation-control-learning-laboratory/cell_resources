function K = compute_DMAC_control(Theta_km1, lx, C, Q, R, K)

[A_est, B_est] = extract_A_B_DMD(Theta_km1, lx);

[A_est_aug, B_est_aug] = generate_augmented_A_B_DMD(A_est, B_est, C, lx);


if rank(ctrb(A_est_aug, B_est_aug)) ~= lx + 1
    disp('Not Controllable!!')
else
    [~, K, ~] = idare(A_est_aug, B_est_aug, Q, R);
    K = -K;
end

end

function [A, B] = extract_A_B_DMD(AB, sys_dim)

A = AB(:, 1:sys_dim);
B = AB(:, sys_dim + 1);
end

function [Aa, Ba] = generate_augmented_A_B_DMD(A, B, C, sys_dim)

Aa = [A, zeros(sys_dim,1); -C 1];
Ba = [B; 0];
end
