%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Logistic Regression
% This script determines if a student will
% be admitted to the university
%
% Coded by: Brent M. Nowak, Ph.D.
% February 4, 2017
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Test the sigmoid function
a = -1000;
b = 1000;

z = (a-b) .* rand(1,100) - a;
g = sigmoid(z);

Z = sort(z);
G = sort(g);


figure(1)
plot(z);
xlabel('Iteration'); ylabel('Random Number');

figure(2)
plot(g);
xlabel('Iteration'); ylabel('Sigmoid Function');

figure(3)
plot(Z);
xlabel('Iteration'); ylabel('Sorted Random Number');

figure(3)
plot(G);
xlabel('Iteration'); ylabel('Sorted Sigmoid Function');

figure(4)
plot(z,g);
xlabel('Random Number'); ylabel('Sigmoid Function');

figure(5)
plot(z,s);
xlabel('Random Number'); ylabel('Sigmoid Function');

figure(6)
plot(Z,G);
xlabel('Sorted Random Number'); ylabel('Sorted Sigmoid Function');
