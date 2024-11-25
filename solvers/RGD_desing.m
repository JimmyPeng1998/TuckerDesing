function stats = RGD_desing ( A_Omega, X, A_Gamma, opts )
% RGD_desing Riemannian gradient descent on desingularization of the set of
% bounded Tucker rank tensors 
%
% stats = RGD_desing ( A_Omega, X, A_Gamma, opts )
% Input:
%   A_Omega: training set (sptensor)
%   X: initial guess (ttensor)
%   A_Gamma: test set to evaluate the recovery performance (sptensor)
%   Opts: user-defined options
%       - maxiter: maximum iteration (1000)
%       - verbose: verbosity (1)
%       - tol: tolerance of training error (1e-6)
%       - difftol: relative difference of training error (1e-8)
%       - lastit: the last iterate (false)
%       - singularvalues: history of singular values (false)
%
% Output: 
%   stats.errorOmega: training error ||P_Omega X-P_Omega A|| / ||P_Omega A||
%   stats.errorGamma: test error ||P_Gamma X-P_Gamma A|| / ||P_Gamma A||
%   stats.duration: time elapsed (second)
%   stats.conv: convergence (true or false)
%   stats.X_RGD_desing: the last iterate
%   stats.sigmak: singular values of mode-k unfolding matrices for k=1,2,3
%
% Reference: Desingularization of bounded-rank tensor sets,
%    Bin Gao, Renfeng Peng, Ya-xiang Yuan, https://arxiv.org/abs/2411.14093
%
% Original author: Renfeng Peng, Sept. 28, 2024.


%% Preparation
if ~isfield( opts, 'maxiter');          opts.maxiter = 1000;            end
if ~isfield( opts, 'verbose');          opts.verbose = 1;               end
if ~isfield( opts, 'tol');              opts.tol = 1e-6;                end
if ~isfield( opts, 'difftol');          opts.difftol = 1e-8;            end
if ~isfield( opts, 'lastit');           opts.lastit = false;            end
if ~isfield( opts, 'singularvalues');   opts.singularvalues = false;    end


conv=false;


fprintf('RGD_desing starts.\n')
%% Initial errors
PAOmega=norm(A_Omega);
PAGamma=norm(A_Gamma);
tic
errorOmega(1)=sqrt(2*calcFunction(A_Omega,X))/PAOmega;
errorGamma(1)=sqrt(2*calcFunction(A_Gamma,X))/PAGamma;
duration(1)=toc;

if opts.verbose == 1
    fprintf("Iter 0: training error %.4e, test error %.4e\n",errorOmega(1),errorGamma(1))
end

if opts.singularvalues == true
    stats_sigma = computeSingularValues(X);
    stats.sigma1 = stats_sigma.sigma1;
    stats.sigma2 = stats_sigma.sigma2;
    stats.sigma3 = stats_sigma.sigma3;
end


for t=1:opts.maxiter
    tic
    
    % Euclidean gradient
    eucGrad = calcGradient(A_Omega,X);
    % Compute Riemannian gradient
    xi = egrad2rgrad_desing (X, eucGrad);
    eta = computeMinus(xi);
    % Compute initial stepsize
    s = calcInitial_mex( eucGrad.subs', eucGrad.vals, ...
                          X.core.data, X.U{1}', X.U{2}', X.U{3}',...
                          xi.tildecore.data, xi.tildeU{1}', xi.tildeU{2}', xi.tildeU{3}');
    % Retraction
    Xnew = retraction_desing( X, eta, s);
    duration(t+1)=toc;
    
    
    % Evaluate training and test errors
    errorOmega(t+1)=sqrt(2*calcFunction(A_Omega,Xnew))/PAOmega;
    errorGamma(t+1)=sqrt(2*calcFunction(A_Gamma,Xnew))/PAGamma;
    
    
    
    if opts.singularvalues == true
        stats_sigma = computeSingularValues(X);
        stats.sigma1 = [stats.sigma1 stats_sigma.sigma1];
        stats.sigma2 = [stats.sigma2 stats_sigma.sigma2];
        stats.sigma3 = [stats.sigma3 stats_sigma.sigma3];
    end
    
    
    if opts.verbose == 1
        fprintf("Iter %d: training error %.4e, test error %.4e\n",...
            t,errorOmega(t+1),errorGamma(t+1))
    end
    
    
    
    
    if abs(errorOmega(t+1)-errorOmega(t))/errorOmega(t)  < opts.difftol
        fprintf('RGD_desing stagnates after %d steps.\n',t)
        break
    end
    
    if errorOmega(t+1)  < opts.tol
        fprintf('RGD_desing converges after %d steps.\n',t)
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
    stats.X_RGD_desing=Xnew;
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


