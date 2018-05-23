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

    predictions = (pval < epsilon);
    
    % true positive if actually anormal (yval = 1) and predicted so
    tp = sum((yval == 1) & (yval == predictions));

    % false negative if actually anormal (yval = 1) but predicted not
    fn = sum((yval == 1) & (yval ~= predictions));
    
    % false positive if actually normal (yval = 0) but predicted not
    fp = sum((yval == 0) & (yval ~= predictions));

    % determine the precision and recall
    prec = tp / (tp + fp);
    rec = tp / (tp + fn);

    % determine F1 score
    F1 = (2 * prec * rec)/(prec + rec);

    % =============================================================

    if F1 > bestF1
       bestF1 = F1;
       bestEpsilon = epsilon;
    end
end

end
