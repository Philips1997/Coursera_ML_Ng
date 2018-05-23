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

A1 = [ones(m, 1) X]; % X is 5000 x 400; A1 is 5000 x 401
Z2 = Theta1 * A1'; % Theta1 is 25 x 401, Z2 is 25 x 5000

A2 = sigmoid(Z2);
A2 = [ones(1, m); A2]; % A2 is 26 x 5000

A3 = sigmoid(Theta2 * A2); % Theta2 is 26 x 10, A3 is 10 x 5000
[x, xi] = max(A3', [], 2); 	
p = xi;

% =========================================================================


end
