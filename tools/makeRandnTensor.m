function X = makeRandnTensor( n, k )
%   makeRandnTensor Create a random Tucker tensor
%   X = makeRandnTensor( n, k ) creates a random Tucker tensor X stored as a 
%   ttensor object. The entries of the core tensor the basis factors are chosen
%   independently from the standard normal distribution. Finally, the basis
%   factors are orthogonalized using a QR procedure. 

%   GeomCG Tensor Completion. Copyright 2013 by
%   Michael Steinlechner
%   Questions and contact: michael.steinlechner@epfl.ch
%   BSD 2-clause license, see LICENSE.txt
%
%   Revised: Renfeng Peng, Sept. 28, 2024.

    [U1,~] = qr( randn( n(1), k(1) ), 0);
    [U2,~] = qr( randn( n(2), k(2) ), 0);
    [U3,~] = qr( randn( n(3), k(3) ), 0);

    C = tensor(randn(k),[k(1) k(2) k(3)]);
    

    X = ttensor( C, {U1, U2, U3} );
end
