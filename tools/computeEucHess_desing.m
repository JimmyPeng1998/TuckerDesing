function eucHessian = computeEucHess_desing(X, Omega, eta)
% computeEucHess_desing Compute the Euclidean Hessian of cost function f
% along eta, i.e., \nabla^2 f(X)[eta] = \proj_\Omega (eta)
%
% eucHessian = computeEucHess_desing(X, eta)
% Input:
%   X: ttensor
%   Omega: training set (sptensor)
%   eta: a tangent vector represented by 
%       eta.tildecore, eta.tildeU{k}, k = 1,2,3
%
% Output:
%   eucHessian: the Euclidean Hessian of cost function f along eta
%
% Reference: Desingularization of bounded-rank tensor sets,
%    Bin Gao, Renfeng Peng, Ya-xiang Yuan, https://arxiv.org/abs/2411.14093
%
% Original author: Renfeng Peng, Oct. 15, 2024.


temp1 = getValsAtIndex(ttensor(eta.tildecore, {X.U{1}, X.U{2}, X.U{3}}), Omega);
temp2 = getValsAtIndex(ttensor(X.core, {eta.tildeU{1}, X.U{2}, X.U{3}}), Omega);
temp3 = getValsAtIndex(ttensor(X.core, {X.U{1}, eta.tildeU{2}, X.U{3}}), Omega);
temp4 = getValsAtIndex(ttensor(X.core, {X.U{1}, X.U{2}, eta.tildeU{3}}), Omega);


eucHessian = sptensor(Omega, temp1 + temp2 + temp3 + temp4, size(X), 0);