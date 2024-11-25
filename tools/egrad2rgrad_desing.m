function xi = egrad2rgrad_desing (varargin)
% egrad2rgrad_desing Compute the Riemannian gradient
% egrad2rgrad_desing can also project a sparse tensor to a tangent vector
% 
% xi = egrad2rgrad (X, eucgrad)
% xi = egrad2rgrad (X, eucgrad, Aneq0, Aneq1, Aneq2, Aneq3)
% xi = egrad2rgrad (X, eucgrad, Aneq0, Aneq1, Aneq2, Aneq3, cores)
% 
% Input:
%   X: initial guess (ttensor)
%   eucgrad: Euclidean gradient (sptensor)
%   Aneq0: \nabla f(X) \times_{k=1}^d U_k'
%   Aneqk: (\nabla f(X) \times_{j\neq k} U_j')_{(k)} for k=1,2,3
%   cores: pre-computed elements of core tensors, including unfolding
%     matrices and matrix multiplications
%
% Output:
%   xi: the Riemannian gradient of g
%
% Reference: Desingularization of bounded-rank tensor sets,
%    Bin Gao, Renfeng Peng, Ya-xiang Yuan, https://arxiv.org/abs/2411.14093
%
% Original author: Renfeng Peng, Sept. 28, 2024.


X = varargin{1};
eucgrad = varargin{2};

if nargin ~=2 && nargin~=6 && nargin~=7
    error("Error usage of egrad2rgrad!")
end


n = size(X);
r = size(X.core);

if nargin == 7
    % All necessary terms are pre-computed. We adopt the existing results
    Aneq0 = varargin{3};
    Aneq1 = varargin{4};
    Aneq2 = varargin{5};
    Aneq3 = varargin{6};
    cores = varargin{7};
    G1t_inv_G1G1t_2I = cores.G1t_inv_G1G1t_2I;
    G2t_inv_G2G2t_2I = cores.G2t_inv_G2G2t_2I;
    G3t_inv_G3G3t_2I = cores.G3t_inv_G3G3t_2I;
elseif nargin == 6
    % The sparse tensor times matrices are pre-computed. We adopt the
    % pre-computed results 
    Aneq0 = varargin{3};
    Aneq1 = varargin{4};
    Aneq2 = varargin{5};
    Aneq3 = varargin{6};
    
    % Core tensors are not pre-computed
    G1 = reshape(X.core.data, r(1), r(2)*r(3));
    G2 = reshape(permute(X.core.data, [2 1 3]), r(2), r(1)*r(3));
    G3 = reshape(permute(X.core.data, [3 1 2]), r(3), r(1)*r(2));
    G1G1t_2I = G1 * G1' + 2 * speye(r(1));
    G2G2t_2I = G2 * G2' + 2 * speye(r(2));
    G3G3t_2I = G3 * G3' + 2 * speye(r(3));
    G1t_inv_G1G1t_2I = G1' / G1G1t_2I;
    G2t_inv_G2G2t_2I = G2' / G2G2t_2I;
    G3t_inv_G3G3t_2I = G3' / G3G3t_2I;
else
    % All necessary terms are not pre-computed
    %
    % First, we compute the sparse tensor times matrices
    % temp   = \vec ( \nabla f(X) \times_{k=1}^d U_k' )
    % temp_k = \vec (\nabla f(X) \times_{j\neq k} U_j') for k=1,2,3
    [temp,temp1,temp2,temp3] = calcProjection_mex( eucgrad.subs', eucgrad.vals, X.U{1}', X.U{2}', X.U{3}' );
    Aneq0 = tensor(reshape(temp, [r(1), r(2), r(3)]), [r(1), r(2), r(3)]);
    Aneq1 = reshape(temp1, [n(1) r(2)*r(3)]);
    Aneq2 = reshape( permute( reshape(temp2, [r(1) n(2) r(3)]), [2 1 3]), [n(2) r(1)*r(3)]);
    Aneq3 = reshape(temp3, [r(1)*r(2) n(3)])';
    
    % Core tensors are not pre-computed
    G1 = reshape(X.core.data, r(1), r(2)*r(3));
    G2 = reshape(permute(X.core.data, [2 1 3]), r(2), r(1)*r(3));
    G3 = reshape(permute(X.core.data, [3 1 2]), r(3), r(1)*r(2));
    G1G1t_2I = G1 * G1' + 2 * speye(r(1));
    G2G2t_2I = G2 * G2' + 2 * speye(r(2));
    G3G3t_2I = G3 * G3' + 2 * speye(r(3));
    G1t_inv_G1G1t_2I = G1' / G1G1t_2I;
    G2t_inv_G2G2t_2I = G2' / G2G2t_2I;
    G3t_inv_G3G3t_2I = G3' / G3G3t_2I;
end


% Next, we compute the parameters for Riemannian gradient
xi.tildecore = Aneq0;
xi.tildeU{1} = Aneq1 * G1t_inv_G1G1t_2I;
xi.tildeU{1} = xi.tildeU{1} - X.U{1} * (X.U{1}' * xi.tildeU{1});
xi.tildeU{2} = Aneq2 * G2t_inv_G2G2t_2I;
xi.tildeU{2} = xi.tildeU{2} - X.U{2} * (X.U{2}' * xi.tildeU{2});
xi.tildeU{3} = Aneq3 * G3t_inv_G3G3t_2I;
xi.tildeU{3} = xi.tildeU{3} - X.U{3} * (X.U{3}' * xi.tildeU{3});