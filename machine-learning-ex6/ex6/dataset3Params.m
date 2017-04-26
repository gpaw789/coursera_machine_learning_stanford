function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%



%1. Poll C and sigma 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30
%2. get prediction error from each output
%2. save them into a mx3 array
%3. pick the lowest prediction error and output the C and sigma

poll = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
err = [];

for i = 1:length(poll)
	C = poll(i)
	for j = 1:length(poll)
		sigma = poll(j)
		model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 		%get model - copied from ex6
		predictions = svmPredict(model,Xval);		%verify it with CV set
		err = [err; C sigma mean(double(predictions ~=yval))];		%get the error and append
	end
end

%pick the lowest prediction error
[min_value min_index] = min(err, [], 1);	%remember: it will pick the lowest value for each column
C = err(min_index(3), 1);		%we are only interested in the error column, min_index will produce the index to where it is stored
sigma = err(min_index(3), 2);
err;


%for ex6 example the optimal values are C = 1, sigma = 0.1

% =========================================================================

end
