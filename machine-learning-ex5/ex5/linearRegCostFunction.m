function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
% o=zeros(m-(length(theta)));
% theta=[theta;o(:,1)];
 a = X*theta;
 a1 = a-y
 b = a1.^2;
 c = sum(b);
 d = (1/(2*m))*c;
 e=(theta(2:end)).^2;
 f=sum(e);
 g=(lambda/(2*m)).*f;
 J=d+g;


 k1=X'*((X*theta)-y);
 k2=(1/m)*k1;
 k3=(theta);
 k3(1)=0;
 k4=(lambda/m)*k3;

 grad=k2+k4;



% =========================================================================

grad = grad(:);

end
