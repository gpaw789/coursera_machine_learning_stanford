function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

J = sum(((X*theta)-y).^2)/(2*m);
%X*theta is the hypo, X is (mx2) matrix and theta is (2x1) matrix, gives a (mx1) matrix
%minus y because we need to find the error
%square the error, now it is a (mx1) matrix
%sum the matrix to give (1x) matrix
%times the per sample scaling
%this script produces J, now it can be used elsewhere to keep finding the lowest J possible
%J is cost function



% =========================================================================

end
