function rhess = ehess2rhess_desing (varargin)
% ehess2rhess_desing Compute the Riemannian Hessian
% 
% xi = ehess2rhess_desing (X, eucgrad, rgrad, eta)
% xi = ehess2rhess_desing (X, eucgrad, rgrad, eta, Aneq0, Aneq1, Aneq2, Aneq3)
% xi = ehess2rhess_desing (X, eucgrad, rgrad, eta, Aneq0, Aneq1, Aneq2, Aneq3, cores)
% Input:
%   X: initial guess (ttensor)
%   eucgrad: Euclidean gradient
%   rgrad: Riemannian gradient (stored by parameters tildecore, and tildeU{k})
%   Aneq0: \vec( \nabla f(X) \times_{k=1}^d U_k' )
%   Aneqk: \vec(\nabla f(X) \times_{j\neq k} U_j') for k=1,2,3
%   cores: pre-computed elements of core tensors, including unfolding
%     matrices and matrix multiplications
%
% Output:
%   rhess: the Riemannian Hessian of g along eta
%
% Reference: Desingularization of bounded-rank tensor sets,
%    Bin Gao, Renfeng Peng, Ya-xiang Yuan, https://arxiv.org/abs/2411.14093
%
% Original author: Renfeng Peng, Oct. 15, 2024.

X = varargin{1};
eucgrad = varargin{2};
rgrad = varargin{3};
eta = varargin{4};

if nargin ~=4 && nargin~=8 && nargin~=9
    error("Error usage of ehess2rhess_desing!")
end


n = size(X);
r = size(X.core);

%% Compute necessary components
if nargin == 9
    % All necessary terms are pre-computed. We adopt the existing results
    Aneq0 = varargin{5};
    Aneq1 = varargin{6};
    Aneq2 = varargin{7};
    Aneq3 = varargin{8};
    cores = varargin{9};
    
    G1 = cores.G1;
    G2 = cores.G2;
    G3 = cores.G3;
    G1G1t_2I = cores.G1G1t_2I;
    G2G2t_2I = cores.G2G2t_2I;
    G3G3t_2I = cores.G3G3t_2I;
    G1t_inv_G1G1t_2I = cores.G1t_inv_G1G1t_2I;
    G2t_inv_G2G2t_2I = cores.G2t_inv_G2G2t_2I;
    G3t_inv_G3G3t_2I = cores.G3t_inv_G3G3t_2I;
elseif nargin == 8
    % The sparse tensor times matrices are pre-computed. We adopt the
    % pre-computed results 
    Aneq0 = varargin{5};
    Aneq1 = varargin{6};
    Aneq2 = varargin{7};
    Aneq3 = varargin{8};
    
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





%% Compute Riemannian Hessian
% Compute \Proj_X (\nabla^2 X)
eucHessian = computeEucHess_desing(X, eucgrad.subs, eta);
% egrad2rgrad_desing projects a sparse tensor to a tangent vector
breve_eta = egrad2rgrad_desing(X, eucHessian);


% Compute \Proj_X (DP (\nabla X))
tildeU1t_Aneq1 = eta.tildeU{1}'*Aneq1;
tildeU2t_Aneq2 = eta.tildeU{2}'*Aneq2;
tildeU3t_Aneq3 = eta.tildeU{3}'*Aneq3;

hat_eta.tildecore = reshape(tildeU1t_Aneq1, [r(1) r(2) r(3)]) + ...
    permute(reshape(tildeU2t_Aneq2, [r(2) r(1) r(3)]), [2,1,3]) + ...
    reshape((tildeU3t_Aneq3)', [r(1) r(2) r(3)]);
hat_eta.tildecore = tensor(hat_eta.tildecore);

hat_eta.tildecore = hat_eta.tildecore + ...
    ttm(X.core, tildeU1t_Aneq1 * G1t_inv_G1G1t_2I, 1) + ...
    ttm(X.core, tildeU2t_Aneq2 * G2t_inv_G2G2t_2I, 2) + ...
    ttm(X.core, tildeU3t_Aneq3 * G3t_inv_G3G3t_2I, 3);

hat_eta.tildecore = -hat_eta.tildecore;



[   A_tildeU2_U3, A_U2_tildeU3, ...
    A_tildeU1_U3, A_U1_tildeU3, ...
    A_tildeU1_U2, A_U1_tildeU2] = computeAdotneqk(uint32(eucgrad.subs)', eucgrad.vals, ...
    X.U{1}', X.U{2}', X.U{3}', ...
    eta.tildeU{1}', eta.tildeU{2}', eta.tildeU{3}');


A_tildeU2_U3 = reshape(A_tildeU2_U3, [n(1) r(2)*r(3)]);
A_U2_tildeU3 = reshape(A_U2_tildeU3, [n(1) r(2)*r(3)]);
A_tildeU1_U3 = reshape( permute( reshape(A_tildeU1_U3, [r(1) n(2) r(3)]), [2 1 3]), [n(2) r(1)*r(3)]);
A_U1_tildeU3 = reshape( permute( reshape(A_U1_tildeU3, [r(1) n(2) r(3)]), [2 1 3]), [n(2) r(1)*r(3)]);
A_tildeU1_U2 = reshape(A_tildeU1_U2, [r(1)*r(2) n(3)])';
A_U1_tildeU2 = reshape(A_U1_tildeU2, [r(1)*r(2) n(3)])';

A_dotV1_G1t_inv_G1G1t_2I = (A_tildeU2_U3 + A_U2_tildeU3) * G1t_inv_G1G1t_2I;
A_dotV2_G2t_inv_G2G2t_2I = (A_tildeU1_U3 + A_U1_tildeU3) * G2t_inv_G2G2t_2I;
A_dotV3_G3t_inv_G3G3t_2I = (A_tildeU1_U2 + A_U1_tildeU2) * G3t_inv_G3G3t_2I;

dotG1 = reshape(eta.tildecore.data, r(1), r(2)*r(3));
dotG2 = reshape(permute(eta.tildecore.data, [2 1 3]), r(2), r(1)*r(3));
dotG3 = reshape(permute(eta.tildecore.data, [3 1 2]), r(3), r(1)*r(2));



hat_eta.tildeU{1} = A_dotV1_G1t_inv_G1G1t_2I + (Aneq1 - rgrad.tildeU{1} * G1) * (dotG1' / G1G1t_2I);
hat_eta.tildeU{2} = A_dotV2_G2t_inv_G2G2t_2I + (Aneq2 - rgrad.tildeU{2} * G2) * (dotG2' / G2G2t_2I);
hat_eta.tildeU{3} = A_dotV3_G3t_inv_G3G3t_2I + (Aneq3 - rgrad.tildeU{3} * G3) * (dotG3' / G3G3t_2I);

hat_eta.tildeU{1} = hat_eta.tildeU{1} - X.U{1} * (X.U{1}' * hat_eta.tildeU{1});
hat_eta.tildeU{2} = hat_eta.tildeU{2} - X.U{2} * (X.U{2}' * hat_eta.tildeU{2});
hat_eta.tildeU{3} = hat_eta.tildeU{3} - X.U{3} * (X.U{3}' * hat_eta.tildeU{3});



%% Summing up the results
rhess.tildecore = breve_eta.tildecore + hat_eta.tildecore;
rhess.tildeU{1} = breve_eta.tildeU{1} + hat_eta.tildeU{1};
rhess.tildeU{2} = breve_eta.tildeU{2} + hat_eta.tildeU{2};
rhess.tildeU{3} = breve_eta.tildeU{3} + hat_eta.tildeU{3};






