function Xnew = retraction_desing( X, eta, s)
% retraction_desing Compute retraction onto the desingularization
%
% Xnew = retraction_desing( X, eta, s)
% Input:
%   X: ttensor
%   eta: a tangent vector represented by 
%       eta.tildecore, eta.tildeU{k}, k = 1,2,3
%   s: stepsize
%
% Output:
%   Xnew: ttensor, computed by Xnew = R_X ( s * eta )
%
% Reference: Desingularization of bounded-rank tensor sets,
%    Bin Gao, Renfeng Peng, Ya-xiang Yuan, https://arxiv.org/abs/2411.14093
%
% Original author: Renfeng Peng, Sept. 28, 2024.

r = size(X.core);

Q = cell(3,1);
% Retraction on Grassmannian
for k = 1:3
    [Q{k},~] = qr(X.U{k} + s*eta.tildeU{k}, 0);
end

% Compute (X+eta)
S = tenzeros( 2*r );

S(1:r(1), 1:r(2), 1:r(3)) = X.core + s * eta.tildecore;
S(r(1)+1:end, 1:r(2), 1:r(3)) = s * X.core;
S(1:r(1), r(2)+1:end, 1:r(3)) = s * X.core;
S(1:r(1), 1:r(2), r(3)+1:end) = s * X.core;

S = ttm( S, { Q{1}'*[X.U{1} eta.tildeU{1}], ...
              Q{2}'*[X.U{2} eta.tildeU{2}], ...
              Q{3}'*[X.U{3} eta.tildeU{3}]  });

Xnew = ttensor(S,{Q{1},Q{2},Q{3}});