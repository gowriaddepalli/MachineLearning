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


a=-1.*y;
b=log(sigmoid(X*theta));
c=log(1-sigmoid(X*theta));
d=a'*b;
e=(1+a)'*c;
f=(1/m)*sum((d-e));
tf = [0; theta(2:length(theta));];
g=(lambda/(2*m)).*sum(tf.^2);
J=f+g;
k=(1/m)*((sigmoid(X*theta))-y)'*X;
l=(lambda/m).*tf;
grad=k'+l;



% =============================================================

end
