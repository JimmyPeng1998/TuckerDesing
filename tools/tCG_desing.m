function [eta, info] = tCG_desing(X, Delta, opts, eucgrad, rgrad, Aneq0, Aneq1, Aneq2, Aneq3, cores)
% tCG_desing truncated conjugate gradients on a tangent space
% 
% [eta, info] = tCG_desing(X, Delta, opts, eucgrad, rgrad, Aneq0, Aneq1, Aneq2, Aneq3, cores)
% Input:
%   X: initial guess (ttensor)
%   Delta: trust-region radius
%   Opts: user-defined options
%       - maxiter: maximum iteration (dimension of manifold by default)
%       - kappa: convergence tolerance (0.1 by default)
%       - theta: convergence tolerance (1 by default)
%   eucgrad: Euclidean gradient
%   rgrad: Riemannian gradient (stored by parameters tildecore, and tildeU{k})
%   Aneq0: pre-computed \vec( \nabla f(X) \times_{k=1}^d U_k' )
%   Aneqk: pre-computed \vec(\nabla f(X) \times_{j\neq k} U_j') for k=1,2,3
%   cores: pre-computed elements of core tensors, including unfolding
%     matrices and matrix multiplications
% 
% Output:
%   eta: a tangent vector represented by 
%       eta.tildecore, eta.tildeU{k}, k = 1,2,3
%   info: stop reason of tCG
%
% Reference: Desingularization of bounded-rank tensor sets,
%    Bin Gao, Renfeng Peng, Ya-xiang Yuan, https://arxiv.org/abs/2411.14093
%
% Original author: Renfeng Peng, Oct. 15, 2024.
n = size(X);
r = size(X.core);

if ~isfield(opts, 'kappa') opts.kappa = 0.1;     end
if ~isfield(opts, 'theta') opts.theta = 1;       end
if ~isfield(opts, 'maxiter') opts.maxiter = prod(r) + n * r'; end


% Initialization
tildeU = cell(3,1);
for k = 1:3
    tildeU{k} = zeros(n(k),r(k));
end

v.tildecore = tenzeros(r);
v.tildeU = tildeU;
res = computeMinus(rgrad);
p = res;

norm_init = innerprod_desing(X, res, res, cores);
norm_res = norm_init;


for k = 1:opts.maxiter
    Hp = ehess2rhess_desing (X, eucgrad, rgrad, p, Aneq0, Aneq1, Aneq2, Aneq3, cores);
    ptHp = innerprod_desing (X, p, Hp, cores);
    alpha = norm_res / ptHp;
    
    v_plus = lincomb_desing (v, 1, p, alpha);
    
    
    if ptHp <= 0 || innerprod_desing(X, v_plus, v_plus, cores) >= Delta^2
        if ptHp <= 0
            info.stopping = 'negative curvature';
        else 
            info.stopping = 'exceeded trust region';
        end
        a = innerprod_desing(X, p, p, cores);
        b = 2*innerprod_desing(X, v, p, cores);
        c = innerprod_desing(X, v, v, cores) - Delta^2;
        t = (sqrt(b^2-4*a*c)-b) / (2*a);
        vnew = lincomb_desing (v, 1, p, t);
        break
    end
    
    
    vnew = v_plus;
    resnew = lincomb_desing (res, 1, Hp, -alpha);
    norm_resnew = innerprod_desing(X, resnew, resnew, cores);
    
    if norm_resnew <= norm_init * min(norm_init^(opts.theta), opts.kappa^2)
        if norm_resnew <= norm_init^(1+opts.theta)
            info.stopping = 'reached targeted theta';
        else
            info.stopping = 'reached targeted kappa';
        end
        break
    end
    beta = norm_resnew / norm_res;
    pnew = lincomb_desing (resnew, 1, p, beta);

    
    
    
    v = vnew;
    res = resnew;
    p = pnew;
    norm_res = norm_resnew;
end


info.iters = k;
if k == opts.maxiter && ~isfield(info,'stopping')
    info.stopping = 'reached maximum iteration';
end

eta = vnew;












