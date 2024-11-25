clear
clc


rng(19)


%% Default settings
maxIter=1000;
n = [100 150 200];
r = [2 3 4];


workingr = r + [3 3 3]; % over-estimated rank parameter
p = 0.05;



A = makeRandnTensor( n, r );


% create the sampling set
subs = makeOmegaSet( n, round(p*prod(n)) );
vals = getValsAtIndex(A, subs);
A_Omega = sptensor( subs, vals, n, 0);
normPAOmega=norm(A_Omega);

% create the test set to compare:
subs_Test = makeOmegaSet( n, round(p*prod(n)) );
vals_Test = getValsAtIndex(A, subs_Test);
A_Gamma = sptensor( subs_Test, vals_Test, n, 0);
normPAGamma=norm(A_Gamma);
        

% initial guess
X_init = makeRandnTensor( n, workingr );





%% Test on desingularization methods
opts = struct( 'maxiter', maxIter, 'gradtol', eps, 'tol', 1e-12, ...
    'verbose', 1, 'lastit', false );
stats_RGD = RGD_desing ( A_Omega, X_init, A_Gamma, opts );
stats_RCG = RCG_desing ( A_Omega, X_init, A_Gamma, opts );
stats_RTR = RTR_desing ( A_Omega, X_init, A_Gamma, opts );



