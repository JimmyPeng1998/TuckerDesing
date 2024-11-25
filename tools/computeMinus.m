function eta = computeMinus(xi)
% computeMinus Compute the minus of a tangent vector xi
%
% eta = computeMinus(xi)
% Input:
%   xi: a tangent vector represented by 
%       xi.tildecore, xi.tildeU{k}, k = 1,2,3
%
% Output:
%   eta: a tangent vector represented by 
%       eta.tildecore, eta.tildeU{k}, k = 1,2,3
%
% Reference: Desingularization of bounded-rank tensor sets,
%    Bin Gao, Renfeng Peng, Ya-xiang Yuan, https://arxiv.org/abs/2411.14093
%
% Original author: Renfeng Peng, Oct. 15, 2024.


eta.tildecore = - xi.tildecore;
eta.tildeU{1} = - xi.tildeU{1};
eta.tildeU{2} = - xi.tildeU{2};
eta.tildeU{3} = - xi.tildeU{3};