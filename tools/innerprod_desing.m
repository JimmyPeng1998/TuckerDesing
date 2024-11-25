function c = innerprod_desing(varargin)
% innerprod_desing Compute the inner product of two tangent vectors eta1
% and eta2 
% 
% c = innerprod_desing (X, eta1, eta2)
% c = innerprod_desing (X, eta1, eta2, cores)
% Input:
%   X: initial guess (ttensor)
%   eta1: a tangent vector represented by 
%       eta1.tildecore, eta1.tildeU{k}, k = 1,2,3
%   eta2: a tangent vector represented by 
%       eta2.tildecore, eta2.tildeU{k}, k = 1,2,3
%   cores: pre-computed elements of core tensors, including unfolding
%     matrices and matrix multiplications
%
% Output:
%   c: inner product of two tangent vectors eta1 and eta2
%
% Reference: Desingularization of bounded-rank tensor sets,
%    Bin Gao, Renfeng Peng, Ya-xiang Yuan, https://arxiv.org/abs/2411.14093
%
% Original author: Renfeng Peng, Sept. 28, 2024.

if nargin ~= 3 && nargin ~=4
    error('Error usage: innerprod_desing!')
end

X = varargin{1};
eta1 = varargin{2};
eta2 = varargin{3};

if nargin == 4
    cores = varargin{4};
else
    r = size(X.core);
    % Pre-computing the core tensor-related terms
    cores.G1 = reshape(X.core.data, r(1), r(2)*r(3));
    cores.G2 = reshape(permute(X.core.data, [2 1 3]), r(2), r(1)*r(3));
    cores.G3 = reshape(permute(X.core.data, [3 1 2]), r(3), r(1)*r(2));
    cores.G1G1t_2I = cores.G1 * cores.G1' + 2 * speye(r(1));
    cores.G2G2t_2I = cores.G2 * cores.G2' + 2 * speye(r(2));
    cores.G3G3t_2I = cores.G3 * cores.G3' + 2 * speye(r(3));
end





% Compute the inner product of two tangent vectors
c = eta1.tildecore.data(:)' * eta2.tildecore.data(:);
temp{1} = eta2.tildeU{1} * cores.G1G1t_2I;
temp{2} = eta2.tildeU{2} * cores.G2G2t_2I;
temp{3} = eta2.tildeU{3} * cores.G3G3t_2I;
for k = 1:3
    c = c + temp{k}(:)' * eta1.tildeU{k}(:);
end
end