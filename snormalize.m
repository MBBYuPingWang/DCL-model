function f=snormalize(A)
% A is n*p matrix, n is the sample size, p is features.

% mean of A for each variable(each column)
n=size(A,1);
mu=mean(A,1); 
% A=A-mean(mu);
% mu=mean(A,1); 
A=A-repmat(mu,n,1);
% variance of each column of A
%  nu=(sum(A.^2,1)/(n)).^(0.5);    

nu=var(A).^0.5;
% nu=(n*var(A)).^0.5; DD
 f=A./repmat(nu,n,1);
f(isnan(f)) = 0;
 
% f=A;