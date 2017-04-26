function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%%get h(x) formula, for the output layer

%get layer2 stuff
%theta1 = 25x401; theta2 = 10x26; X = 5000x400; y = 5000x1
%g(  sum(theta*X)  ), output as a 5000x25 (5000 samples and 25 layers for each sample)
X = [ones(size(X,1),1) X];		%add a ones columns to the most left (bias unit)
layer2 = sigmoid(X*Theta1');			%using vectorisation for SUM

%get layer3 stuff
%h(x) = g(    sum(layer2*X)    ), output as a 5000x10 (5000 samples and 10 K values (output) for each sample)
layer2 = [ones(size(layer2,1), 1) layer2];		%add a ones columns to the most left (bias unit)
hfunc = sigmoid(layer2*Theta2');
[max_value indices] = max(hfunc, [], 2);
size(hfunc);
%recode y
%what's the reason behind recoding y?
%y needs to be translated from 5000x1 to 5000x10, so instead of having [10; 1; 3; ... 5000th] it should be vectors of [1 0 0 0 0 0 0 0 0 0] times 5000
%so that in the cost function J, when y times log of hfunc, it will pick up the hfunc with the best value, filtering the rest to zeroes, e.g. BEFORE hfunc = [78.2 28.5 19.0 90.9], times it by y = [0 1 0 0], will give [0 28.5 0 0]
%then the cost function can be successfully calculated
vector_y = zeros(1,num_labels);
y_10 = [];
for i = 1:m
	vector_y = zeros(1,num_labels);
	vector_y(y(i)) = 1;
	y_10 = [y_10; vector_y];
end
size(y_10);

%%Find J

J = (1/m)*sum( ...
	sum( ...
	-y_10.*log(hfunc)-(1-y_10).*log(1-hfunc)
	,2)		%summing up all values in rows - across the 10 columns for SUM(K)
	);		%summing up all values in the whole column, across the 5000 samples for SUM(m)




% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
