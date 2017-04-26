function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
	%straight from the formula:
	% 0j := 0j - a(1/m)SUM((h(x)-y)Xj)
	sigma = (X*theta-y);		%must do this as the theta doesn't change when i run the next two code
	theta(1,1) = theta(1,1) - (alpha/m)*sum(sigma.*X(:,1));		%calc "feature 1"
	theta(2,1) = theta(2,1) - (alpha/m)*sum(sigma.*X(:,2));		%calc "feature 2"
	
	%now I still don't get why times by Xj - need to study more
	%
	sigma
	theta

	computeCost(X, y, theta)
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
