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

%get layer2_a stuff
%theta1 = 25x401; theta2 = 10x26; X = 5000x400; y = 5000x1
%g(  sum(theta*X)  ), output as a 5000x25 (5000 samples and 25 layers for each sample)
X = [ones(size(X,1),1) X];		%add a ones columns to the most left (bias unit)
layer2_z = X*Theta1';
layer2_a = sigmoid(layer2_z);			%using vectorisation for SUM

%get layer3_a stuff
%h(x) = g(    sum(layer2_a*X)    ), output as a 5000x10 (5000 samples and 10 K values (output) for each sample)
layer2_a = [ones(size(layer2_a,1), 1) layer2_a];		%add a ones columns to the most left (bias unit)
layer3_z = layer2_a*Theta2';
layer3_a = sigmoid(layer3_z);
hfunc = layer3_a;
[max_value hfunc_indices] = max(hfunc, [], 2);

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
%%regularized cost function should not involve bias unit
%%drop the first column for Theta1 and Theta2
Theta1_reg = Theta1; Theta2_reg = Theta2;
Theta1_reg(:,[1]) = [];
Theta2_reg(:,[1]) = [];

J = (1/m)*sum( ...
	sum( ...
	-y_10.*log(hfunc)-(1-y_10).*log(1-hfunc)
	,2)		%summing up all values in rows - across the 10 columns for SUM(K)
	)+ ...		%summing up all values in the whole column, across the 5000 samples for SUM(m)
	(lambda/(2*m))* ...
	(
	sum(sum(Theta1_reg.*Theta1_reg,2)) + ... %sum the all the columns first into a vector and then row
	sum(sum(Theta2_reg.*Theta2_reg,2))		%sum all the columns first into a vector and then row
	);

	
	
	
	
%%%calculating grad using backprop

%do a feedfoward pass - done it above - got z2, a2, z3, a3

%%Finding Theta2_grad
%Finding delta_3 = ak_2 - yk  as per lecture 9 slide 7
%layer3_a is 5000x10, y_10 is 5000x10 = delta_3 = 5000x10
delta_3 = layer3_a - y_10; 

%calculate Theat2_grad as per lecture 9 slide 7
%now lecture 9 slide 8 show the for loop function, but you don't need to do that if you are using vectorisation, so effectively its just d/dT of J(T) = ak_2 * delta_3
%Theta2_grad should be 10x26, delta_3 is 5000x10, layer2_a is 5000x26
Theta2_grad = delta_3'*layer2_a;

%%Finding Theta1_grad
%Find delta_2
%Now getting delta_2 = (Theta2)*delta_3*g'(layer2_z), as per lecture 9 slide 7
%since layer2_z is 5000x25, you need to pad it with bias so its 5000x26
%delta_3 is 5000x10, Theta2 is 10x26, so the output is 5000x26, that's why layer2_z needs padding
%delta_2 will 5000x26, but we have to drop the first column bias so it lines up with X so delta_2 should be 5000x25
delta_2 = (delta_3*Theta2).*sigmoidGradient([ones(size(layer2_z),1) layer2_z]);
delta_2(:,[1])=[];	%drop first column bias

%now the formula should be delta_layerX = delta_X+1 times layerX_a
%we didn't calculate layer1_a because that's just X (input), which is 5000x401 (with bias)
%so Theta1_grad is 25x401
Theta1_grad = delta_2'*X;

%d/dT of J(T) is (1/m)*(delta), Theta1_grad is 25x401, Theta2_grad is 10x26
%regularized neural networks, as per lecture 9 slide 8
%d/dT of J(T) = (1/m)*delta 					for j = 0
%d/dT of J(T) = (1/m)*delta+(lambda/m)*(Theta) for j >= 1
Theta1_reg_0 = Theta1;	%why do I do it this way? it looks way neater, than to compute...
Theta2_reg_0 = Theta2;	%..two things and add them together
Theta1_reg_0(:,1)=[0];	%the entire first column write to zero
Theta2_reg_0(:,1)=[0];	%the entire first column write to zero

Theta1_grad = (1/m)*Theta1_grad + (lambda/m)*Theta1_reg_0;
Theta2_grad = (1/m)*Theta2_grad + (lambda/m)*Theta2_reg_0;


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
