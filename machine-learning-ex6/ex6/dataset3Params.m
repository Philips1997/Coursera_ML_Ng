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

potentialVal = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]; % potential values for C and sigma

l = length(potentialVal);

result = []; % store the value from 

for c = 1:l
    for sig = 1:l
        cTest = potentialVal (c);
        sigTest = potentialVal (sig);
        
        mdlTest= svmTrain(X, y, cTest, @(x1, x2) gaussianKernel(x1, x2, sigTest)); 
        prdctTest = svmPredict(mdlTest, Xval);
        
        testError = mean(double(prdctTest ~= yval));
      
        result = [result; cTest, sigTest, testError]; % store test error and parameters in the result
        
    end
end

[errMin, index] = min(result(:,3));

C = result(index,1);
sigma = result(index,2);


% =========================================================================

end
