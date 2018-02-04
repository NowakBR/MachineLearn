function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

SigM = sigmoid(X * theta);

% This is the term after the addition: 2nd eqn on page 9.
Regularized = lambda / (2 * m) * (theta' * theta - theta(1)^2);

% Regularized cost function.  Complete 2nd eqn on page 9.
J = 1 / m * (-y' * log(SigM) - (1 - y') * log(1 - SigM)) + Regularized;

% Mask: 1st equation on page 9.
mask = ones(size(theta));
mask(1) = 0;

% Gradient: 1st eqn on page 10.
grad = 1 / m * X' * (SigM - y) + lambda / m * (theta .* mask);

% =============================================================

end
