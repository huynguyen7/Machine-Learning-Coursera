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

% Cost function
hx = sigmoid(X*theta);
sigma1 = (-y)' * log(hx) - (1-y)' * log(1-hx);
sigma2 = sum(theta(2:end) .^ 2);
J = (1/m) * sigma1 + (lambda / (2*m)) * sigma2;

% Gradient vector
thetaZero = theta;
thetaZero(1) = 0; % The first theta is not applied in regularization.. So set it to 0.

sigma = (hx - y)' * X;
grad = (1/m) * sigma + (lambda/m) * thetaZero';

% =============================================================

end
