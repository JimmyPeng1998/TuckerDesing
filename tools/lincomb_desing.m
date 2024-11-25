function eta = lincomb_desing (eta1, a, eta2, b)
% lincomb_desing Compute linear combination of two tangent vectors
% eta = a * eta1 + b * eta2
%
% eta = lincomb_desing (eta1, a, eta2, b)
% Input:
%   eta1: a tangent vector represented by 
%       eta1.tildecore, eta1.tildeU{k}, k = 1,2,3
%   a: a constant
%   eta2: a tangent vector represented by 
%       eta2.tildecore, eta2.tildeU{k}, k = 1,2,3
%   b: a constant
%
% Output:
%   eta: a tangent vector represented by 
%       eta.tildecore, eta.tildeU{k}, k = 1,2,3
%
% Reference: Desingularization of bounded-rank tensor sets,
%    Bin Gao, Renfeng Peng, Ya-xiang Yuan, https://arxiv.org/abs/2411.14093
%
% Original author: Renfeng Peng, Sept. 28, 2024.
eta.tildecore = a * eta1.tildecore + b * eta2.tildecore;
for k = 1:3
    eta.tildeU{k} = a * eta1.tildeU{k} + b * eta2.tildeU{k};
end
end