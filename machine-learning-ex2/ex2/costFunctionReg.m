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

%you set the theta_regulated(1) to zero, why? so that it doesn't regulate
%also later on for the formula the first theta [theta(0)] is zero so it will cancel the 2nd term out
thetaReg = theta;
thetaReg(1) = 0;

%The instructions are not clear, you can't regulate theta(0) 
%so what you can do is to set it to zero
J = (1/m)*sum(-y.*log(sigmoid(X*theta)) - (1-y).*log(1-sigmoid(X*theta))) ...
	+ (lambda/(2*m))*sum(thetaReg'*thetaReg);

	
	
%My Solution:
%d/d0 of J(0) - for j > 0, you add lambda*theta/m at the back
%grad = ((1/m)*sum((sigmoid(X*theta)-y).*X))+ (lambda/m).*theta';	%calculate for everything first
%fprintf('Grad with lambad formula %f\n',grad(1))
%fprintf('Grad with lambad formula -minus formula %f\n',grad(1)-(lambda/m.*theta(1)))
%grad(1) = (1/m)*sum((sigmoid(X*theta)-y).*X)(1);	%overwrite grad(1) with the correct formula, probably a little slower but who cares
%fprintf('Grad with normal formula %f\n',grad(1))



%More elegant solution:
grad = ((1/m)*sum((sigmoid(X*theta)-y).*X)) + (lambda/m).*thetaReg';














% =============================================================

end
