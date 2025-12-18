function [Theta, gradient] = GD_DMD(x, y, Theta, learning_rate)

% Compute gradient
gradient = -2 * (y - Theta * x) * x';

% Update Theta using gradient descent
Theta = Theta - learning_rate * gradient;
