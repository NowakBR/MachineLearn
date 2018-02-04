function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
%
% Note: grad should have the same dimensions as theta


%

% This is the first 2 eqns on page 4 using sigmoid.m
    SigM = sigmoid(X * theta); 

% Logistic cost function. The 3rd equation on page 4.
% The summation is implied by the matrix math.
    J = 1 / m * (-y' * log(SigM) - (1 - y') * log(1 - SigM));

% Gradient of the cost function.  The 1st equation on page 5.
    grad = 1 / m * ((SigM - y)' * X)';


% =============================================================

end
