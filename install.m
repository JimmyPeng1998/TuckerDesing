clear
clc



disp('TuckerTC_desing: Low-rank tensor completion in Tucker decomposition via desingularization ...')


disp('Adding paths ...')
addpath tensor_toolbox-v3.4/
addpath tools/
addpath solvers/


fprintf("Compiling mex files.\n")
cd tools/
mex calcFunction_mex.c
mex calcGradient_mex.c
mex calcInitial_mex.c
mex calcProjection_mex.c
mex computeAdotneqk.c
mex getValsAtIndex_mex.c


disp('Install successful!')
cd ..