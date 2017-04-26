function J = costFunctionJ(X, y, theta)

%X is the design matrix containing our training examples.
%y is the class labels

m= size(X,1);	 %number of training examples: look at it by row
predictions = X*theta    %predictions of hypotehesis on all m examples

sqrErrors = (predictions-y).^2    %squared errors	-- minus works for each column if same row

J = 1/(2*m) * sum(sqrErrors)