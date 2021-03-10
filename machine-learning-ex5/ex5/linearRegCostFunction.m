function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

grad = zeros(size(theta));
J = 0;
% You need to return the following variables correctly 
% X = [ones(size(X)(1,1), 1) X];



% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% COST FUNCTION
J = (1/(2*m)) * sum((X * theta - y) .^ 2) + (lambda/(2*m)) * sum(theta(2:end) .^ 2);  % Scalar

% GRADIENT
sigma2 = (1/m) .* (X' * (X * theta - y));
reg2 = (lambda/m) * theta;
reg2(1) = 0;  % Not apply regularization to theta0.
grad = sigma2 + reg2;

% =========================================================================

grad = grad(:);

end
