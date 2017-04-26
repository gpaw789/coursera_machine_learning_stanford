function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returns the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);


% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.
%

%%finding new centroids using equation from ex7 slide 4
%u(k) := 1/sum(C) * sum of X(i) for a particular idx(i)
%recall idx is a vector where each element corresponds to each X, value between 1:K
centroids = [];

for k = 1:K		%poll through k to build centroids (k x n) one row at a time
	idx_k =(idx == k); 	%filter the idx to just TRUE for k
	%X*idx_k, X is mxn, idx_k is mx1
	%will filter the X with respect to idx_K, meaning only rows that idx_k has it as TRUE (1), the values will remain, otherwise will times by 0 which zeros it out
	%sum of all the values across rows only and divide by how many points matched to that particular centroid
	%think of it this way: its the average of the sum of all points positions linked to a particular centroid. This way we can determine a new centroid
	centroids = [centroids; sum(X.*idx_k,1)/sum(idx_k,1)]; 
	
end





%output should be centroids is (number of k)x(dimensions of X)


% =============================================================


end

