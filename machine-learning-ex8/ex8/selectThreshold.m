function [bestEpsilon bestF1] = selectThreshold(yval, pval)
%SELECTTHRESHOLD Find the best threshold (epsilon) to use for selecting
%outliers
%   [bestEpsilon bestF1] = SELECTTHRESHOLD(yval, pval) finds the best
%   threshold to use for selecting outliers based on the results from a
%   validation set (pval) and the ground truth (yval).
%

bestEpsilon = 0;
bestF1 = 0;
F1 = 0;

stepsize = (max(pval) - min(pval)) / 1000;
for epsilon = min(pval):stepsize:max(pval)
    
    % ====================== YOUR CODE HERE ======================
    % Instructions: Compute the F1 score of choosing epsilon as the
    %               threshold and place the value in F1. The code at the
    %               end of the loop will compare the F1 score for this
    %               choice of epsilon and set it to be the best epsilon if
    %               it is better than the current choice of epsilon.
    %               
    % Note: You can use predictions = (pval < epsilon) to get a binary vector
    %       of 0's and 1's of the outlier predictions
	
	
	%%calculate cvPredictions
	%the concept was not explicitly explained in lecture
	%Basically if the pval(i) is less than epsilon, it is considered to be ZERO (normal)
	%if pval(i) is more than epsilon, it is considered to be ONE (anomaly)
	%now with all the ZEROs and ONEs, this mx1 matrix will be the predicted value (cvPredictions)
	cvPredictions = (pval < epsilon);	%from Lecture 15, slide 3

	
	%%calculate true positive, from Lecture 11, slide 11
	tp = sum(cvPredictions.*yval);		%think logic: actual AND predicted

	%%calculate false positive, from Lecture 11, slide 11
	fp = sum((cvPredictions==1).*(yval==0));	%think logic: (predicted) AND NOT(actual)

	%%calculate false negative, from Lecture 11, slide 11
	fn = sum((cvPredictions==0).*(yval==1));	%think logic: NOT(predicted) AND (actual)

	%%calculate precision, from ex8, slide 6
	prec = tp/(tp+fp);
	
	%%calculate precision, from ex8, slide 6
	rec = tp/(tp+fn);
	
	%%%Finding the best F1 score means that the epsilon cut off is the most optimum
	%%calculate F1 score, from ex8, slide 6
	F1 = (2*prec*rec)/(prec+rec);







    % =============================================================

    if F1 > bestF1
       bestF1 = F1;
       bestEpsilon = epsilon;
    end
end

end
