function [A, B] = extract_A_B_DMD(AB, sys_dim)

A = AB(:, 1:sys_dim);
B = AB(:, sys_dim + 1);