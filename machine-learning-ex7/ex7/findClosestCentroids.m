function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

%find closest centroids from the choice of k
%For two dimensions, X is mx2, C is kx2

%build all_distance, row of X and columns of centroids data
all_distance = zeros(size(idx, 1), size(K,1));

for row = 1:size(X, 1)
	for K_columns = 1:size(centroids, 1)
		A_vector = X(row,:)-centroids(K_columns,:);		%get the slice of entire row as a 1xy matrix
		J = A_vector*A_vector'; %using Frobenius/Eculidean Norm: ||A|| = sqrt(A*A') , its basically ||A|| = sqrt(A1^2 + A2^2). Also the sqrt cancels with squared: lecture 13 slide 19
		all_distance(row, K_columns) = J;	%build the matrix one J at a time
	end
end

%get the min value and its index from all_distance (across columns only)
[min_val, min_index] = min(all_distance, [], 2);
idx = min_index;
%idx is a vector of where each X(i) belongs to which centroids
% =============================================================

end

