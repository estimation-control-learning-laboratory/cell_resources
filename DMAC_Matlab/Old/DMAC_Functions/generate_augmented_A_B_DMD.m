function [Aa, Ba] = generate_augmented_A_B_DMD(A, B, C, sys_dim)

Aa = [A, zeros(sys_dim,1); -C 1];
Ba = [B; 0];
