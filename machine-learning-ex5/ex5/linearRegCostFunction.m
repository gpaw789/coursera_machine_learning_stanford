function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

%%Find J
%J(t) = (1/2m)*(sum(h(x)-y)^2)+(lambda/2m)*sum(t^2)
%h(x) is sum(theta*X)	, as per Lecture 5 slide 7
%X is 12x2; y is 12x1; theta is 2x1; lambda is 1x1
%how do I know its .^2 and not matrix multiplication? we are looking for variance for EACH entry
theta_reg = theta;
theta_reg(1,:) = [0];	%theta(0) should not be part of regularisation, set it to 0, so when the first theta(0) times lambda/2m, it will zero out
%sum twice, because the first one is across rows and the second time is across columns to give 1x1

J = (1/(2*m))*sum((X*theta-y).^2) + (lambda/(2*m))*sum(theta_reg.^2);


%%Find d/dt of J(t)%d/dt of J(t) = 1/m * sum[(h(x)-y)*x]  +   (lambda/m)*theta   for j>=1	as per ex5.pdf
%left term is 2x1 and the right term should be 2x1
grad = (1/m)*X'*(X*theta - y) + (lambda/m)*theta_reg;



% =========================================================================

grad = grad(:);

end
