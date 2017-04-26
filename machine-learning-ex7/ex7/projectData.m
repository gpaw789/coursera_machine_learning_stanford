function Z = projectData(X, U, K)
%PROJECTDATA Computes the reduced data representation when projecting only 
%on to the top k eigenvectors
%   Z = projectData(X, U, K) computes the projection of 
%   the normalized inputs X into the reduced dimensional space spanned by
%   the first K columns of U. It returns the projected examples in Z.
%

% You need to return the following variables correctly.
Z = zeros(size(X, 1), K);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the projection of the data using only the top K 
%               eigenvectors in U (first K columns). 
%               For the i-th example X(i,:), the projection on to the k-th 
%               eigenvector is given as follows:
%                    x = X(i, :)';
%                    projection_k = x' * U(:, k);
%

%%compressing dimension of X into dimension of K
%K is a single number, X is mxn, U is nxn, Ureduce ix nxk
%usually X is normalised, U is principal component
%reduce U to k dimension, and compute z using z=Ureduce'*X, as per lecture 14 slide 19
Ureduce = U(:,1:K);
Z = X*Ureduce;
%Z is mxk



% =============================================================

end