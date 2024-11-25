function stats = RTR_desing ( A_Omega, X, A_Gamma, opts )
% RTR_desing Riemannian gradient descent on desingularization of the set of
% bounded Tucker rank tensors
%
%
% stats = RTR_desing ( A_Omega, X, A_Gamma, opts )
% Input:
%   A_Omega: training set (sptensor)
%   X: initial guess (ttensor)
%   A_Gamma: test set to evaluate the recovery performance (sptensor)
%   r: rank parameter
%   Opts: user-defined options
%       - maxiter: maximum iteration (1000)
%       - verbose: verbosity (1)
%       - tol: tolerance of training error (1e-6)
%       - difftol: relative difference of training error (1e-8)
%       - lastit: the last iterate (false)
%       - rho_prime: accept/reject ratio (0.1)
%       - kappa: tCG parameter (0.1)
%       - Delta_bar: maximun trust-region radius (sqrt(dim))
%       - Delta: initial trust-region radius (sqrt(dim)/16)
%
% Output:
%   stats.errorOmega: training error ||P_Omega X-P_Omega A|| / ||P_Omega A||
%   stats.errorGamma: test error ||P_Gamma X-P_Gamma A|| / ||P_Gamma A||
%   stats.duration: time elapsed (second)
%   stats.conv: convergence (true or false)
%   stats.X_RTR_desing: the last iterate
%   stats.sigmak: singular values of mode-k unfolding matrices for k=1,2,3
%
% Reference: Desingularization of bounded-rank tensor sets,
%    Bin Gao, Renfeng Peng, Ya-xiang Yuan, https://arxiv.org/abs/2411.14093
%
% Original author: Renfeng Peng, Oct. 15, 2024.

%% Preparation
if ~isfield( opts, 'maxiter');          opts.maxiter = 1000;            end
if ~isfield( opts, 'verbose');          opts.verbose = 1;               end
if ~isfield( opts, 'tol');              opts.tol = 1e-6;                end
if ~isfield( opts, 'difftol');          opts.difftol = 1e-8;            end
if ~isfield( opts, 'lastit');           opts.lastit = false;            end
if ~isfield( opts, 'singularvalues');   opts.singularvalues = false;    end
if ~isfield( opts, 'rho_prime');        opts.rho_prime = 0.1;           end
if ~isfield( opts, 'kappa');            opts.kappa = 0.1;               end
if ~isfield( opts, 'theta');            opts.theta = 1;                 end

conv=false;
n = size(X);
r = size(X.core);

dim = prod(r) + n * r';



if ~isfield( opts, 'Delta_bar');        opts.Delta_bar = sqrt(dim);     end
if ~isfield( opts, 'Delta');            opts.Delta = opts.Delta_bar/16; end
Delta_bar = opts.Delta_bar;
Delta = opts.Delta;


fprintf('RTR_desing starts.\n')
%% Initial errors
PAOmega=norm(A_Omega);
PAGamma=norm(A_Gamma);
tic
errorOmega(1)=sqrt(2*calcFunction(A_Omega,X))/PAOmega;
errorGamma(1)=sqrt(2*calcFunction(A_Gamma,X))/PAGamma;
duration(1)=toc;

if opts.verbose == 1
    fprintf("Iter 0:        training error %.4e, test error %.4e\n",...
        errorOmega(1), errorGamma(1))
end

if opts.singularvalues == true
    stats_sigma = computeSingularValues(X);
    stats.sigma1 = stats_sigma.sigma1;
    stats.sigma2 = stats_sigma.sigma2;
    stats.sigma3 = stats_sigma.sigma3;
end

stats.acc(1) = true;


for t=1:opts.maxiter
    tic
    
    %% Pre-computations
    if stats.acc(t) == true
        % Euclidean gradient
        eucGrad = calcGradient(A_Omega, X);
        
        % Compute sparse tensor times matrices
        % temp   = \vec ( \nabla f(X) \times_{k=1}^d U_k' )
        % temp_k = \vec (\nabla f(X) \times_{j\neq k} U_j') for k=1,2,3
        [temp,temp1,temp2,temp3] = calcProjection_mex( eucGrad.subs', eucGrad.vals, X.U{1}', X.U{2}', X.U{3}' );
        Aneq0 = tensor(reshape(temp, [r(1), r(2), r(3)]), [r(1), r(2), r(3)]);
        Aneq1 = reshape(temp1, [n(1) r(2)*r(3)]);
        Aneq2 = reshape( permute( reshape(temp2, [r(1) n(2) r(3)]), [2 1 3]), [n(2) r(1)*r(3)]);
        Aneq3 = reshape(temp3, [r(1)*r(2) n(3)])';
        
        
        % Pre-computing the core tensor-related terms
        cores.G1 = reshape(X.core.data, r(1), r(2)*r(3));
        cores.G2 = reshape(permute(X.core.data, [2 1 3]), r(2), r(1)*r(3));
        cores.G3 = reshape(permute(X.core.data, [3 1 2]), r(3), r(1)*r(2));
        cores.G1G1t_2I = cores.G1 * cores.G1' + 2 * speye(r(1));
        cores.G2G2t_2I = cores.G2 * cores.G2' + 2 * speye(r(2));
        cores.G3G3t_2I = cores.G3 * cores.G3' + 2 * speye(r(3));
        cores.G1t_inv_G1G1t_2I = cores.G1' / cores.G1G1t_2I;
        cores.G2t_inv_G2G2t_2I = cores.G2' / cores.G2G2t_2I;
        cores.G3t_inv_G3G3t_2I = cores.G3' / cores.G3G3t_2I;
        
        
        
        % Compute Riemannian gradient
        rgrad = egrad2rgrad_desing (X, eucGrad, Aneq0, Aneq1, Aneq2, Aneq3, cores);
    end
    
    
    
    %% Riemannian trust-region method
    % Solving subproblem
    opts_CG.kappa = opts.kappa;
    opts_CG.theta = opts.theta;
    opts_CG.maxiter = prod(r) + n * r';
    [eta, info] = tCG_desing(X, Delta, opts_CG, eucGrad, rgrad, Aneq0, Aneq1, Aneq2, Aneq3, cores);
    rHess_eta = ehess2rhess_desing (X, eucGrad, rgrad, eta, Aneq0, Aneq1, Aneq2, Aneq3, cores);
    
    % Retraction
    X_plus = retraction_desing( X, eta, 1);
    f_plus = calcFunction(A_Omega,X_plus);
    
    
    % Accept or reject the tentative next iterate
    Ared = ((errorOmega(t)*PAOmega)^2/2 - f_plus);
    Pred = Ared - innerprod_desing(X, eta, rgrad, cores) - innerprod_desing(X, eta, rHess_eta, cores)/2;
    rho = Ared / Pred;
    
    if rho > opts.rho_prime
        stats.acc(t+1) = true;
        point_str = 'Acc';
        Xnew = X_plus;
    else
        stats.acc(t+1) = false;
        point_str = 'REJ';
        Xnew = X;
    end
    
    
    % Update the trust-region radius
    if rho < 1/4
        Delta = Delta/4;
        radius_str = 'TR-';
    elseif rho > 3/4 && abs(innerprod_desing(X, eta, eta, cores) - Delta^2) < 1e-13
        Delta = min(2*Delta, Delta_bar);
        radius_str = 'TR+';
    else
        radius_str = 'TR ';
    end
    
    duration(t+1)=toc;
    
    
    
    
    % Evaluate training and test errors
    if stats.acc(t+1) == true
        errorOmega(t+1)=sqrt(2*f_plus)/PAOmega;
        errorGamma(t+1)=sqrt(2*calcFunction(A_Gamma,Xnew))/PAGamma;
    else
        errorOmega(t+1)=errorOmega(t);
        errorGamma(t+1)=errorGamma(t);
    end
    
    
    
    if opts.singularvalues == true
        stats_sigma = computeSingularValues(X);
        stats.sigma1 = [stats.sigma1 stats_sigma.sigma1];
        stats.sigma2 = [stats.sigma2 stats_sigma.sigma2];
        stats.sigma3 = [stats.sigma3 stats_sigma.sigma3];
    end
    
    
    
    
    if opts.verbose == 1
        fprintf("Iter %d: %s%s training error %.4e, test error %.4e, inner iter=%d, %s\n",...
            t, point_str, radius_str, errorOmega(t+1), errorGamma(t+1), info.iters, info.stopping)
    end
    
    
    
    
    if abs(errorOmega(t+1)-errorOmega(t))/errorOmega(t)  < opts.difftol && stats.acc(t+1) == true
        fprintf('RTR_desing stagnates after %d steps.\n',t)
        break
    end
    
    if errorOmega(t+1)  < opts.tol
        fprintf('RTR_desing converges after %d steps.\n',t)
        conv=true;
        break
    end
    
    
    
    X=Xnew;
    
    
end


% stats = struct('errorOmega',errorOmega,'errorGamma',errorGamma,'duration',cumsum(duration),'conv',conv);

stats.errorOmega = errorOmega;
stats.errorGamma = errorGamma;
stats.duration = cumsum(duration);
stats.conv = conv;


if opts.lastit==true
    stats.X_RTR_desing=Xnew;
end
end


function stats = computeSingularValues(X)
C = X.core.data;
r = size(C);

[~,S,~]=svd(reshape(C,[r(1) r(2)*r(3)]),'econ');
stats.sigma1 = diag(S)';
% stats.sigma1 = stats.sigma1(1:r(1));


[~,S,~]=svd(reshape(permute(C,[2 1 3]),[r(2) r(1)*r(3)]),'econ');
stats.sigma2 = diag(S)';
% stats.sigma2 = stats.sigma2(1:r(2));


[~,S,~]=svd(reshape(C,[r(1)*r(2) r(3)]),'econ');
stats.sigma3 = diag(S)';
end



