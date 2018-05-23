function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

H = sigmoid(X * theta); %hypothesis, dimension m*1, X: m*n+1
thetaFrom1 = theta(2:end);
J = (((-1 .* y' * log(H))-((1 - y') * log(1 - H)))./m)+...
    ((lambda/(2*m))*((thetaFrom1)'*(thetaFrom1))); %regularize cost function

grad = ((H-y)' * X)' ./m; % determine grad before regularization
grad1 = grad(1); % secure grad_0 before regularization
grad = grad + ((lambda/m).* theta); % regularization
grad(1) = grad1; % restore grad_0




% =============================================================

end
