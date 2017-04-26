function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
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
    %       of the cost function (computeCostMulti) and gradient here.
    %

	sigma = (X*theta)-y;		%must do this as the theta doesn't change when i run the next three code
	theta(1,1) = theta(1,1) - (alpha/m)*sum(sigma.*X(:,1));		%calc "feature 1"
	theta(2,1) = theta(2,1) - (alpha/m)*sum(sigma.*X(:,2));		%calc "feature 2" - footage
	theta(3,1) = theta(3,1) - (alpha/m)*sum(sigma.*X(:,3));		%calc "feature 3" - rooms


    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
