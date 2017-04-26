function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%
%theta1 row by row is bigT(rowxcolumn)
%the 20x20 image of ith row (1x400), times bigT (theta1, 25x401)
%add the bias to make 1x401, times bigT (theta1, 25x401)
X = [ones(m, 1) X];

%what if now you times bigT (theta1, 25x401) with whole of X (5000x401)?
%transpose bigT to give X (5000x401) times (401x25), gives 5000x25

%An = g(sum(theta1.*X)) % refer to lecture 8 slide 20
%An = 5000x25 matrix
layer2 = sigmoid(X*Theta1');

%theta2 is 10x26 - make sense because there is a bias +1
%before moving to layer3, layer 2 needs to add a bias to the leftmost column as 1
%layer 2 is now 5000x26 matrix
layer2 = [ones(size(layer2,1),1) layer2];

%follow lecture 8 slide 20
%h(x) = a(3) = g(sum(theta2.*layer2))
%layer 3 is 5000x10 matrix
layer3 = sigmoid(layer2*Theta2');

%get prediction for all rows in 5000 columns - should be a 5000x10 output
%translate 5000x10 into 5000x1 vector
%get the max value for each row
%indicies will display which column did the max value came from
[max_value indices] = max(layer3, [], 2);

%indices should be 5000x1 matrix
p = indices;



% =========================================================================


end
