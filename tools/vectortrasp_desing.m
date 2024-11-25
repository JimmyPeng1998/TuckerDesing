function xi = vectortrasp_desing(X,Y,eta)
% vectortrasp_desing Projecting a tangent vector eta in T_X M to T_Y M
% xi = vectortrasp_desing(X,Y,eta)
% Input:
%   X: ttensor
%   Y: ttensor
%   eta: a tangent vector represented by 
%       eta.tildecore, eta.tildeU{k}, k = 1,2,3
%
% Output:
%   xi: a tangent vector represented by 
%       xi.tildecore, xi.tildeU{k}, k = 1,2,3
%
% Reference: Desingularization of bounded-rank tensor sets,
%    Bin Gao, Renfeng Peng, Ya-xiang Yuan, https://arxiv.org/abs/2411.14093
%
% Original author: Renfeng Peng, Oct. 15, 2024.

n = size(X);
r = size(X.core);

% Pre-computations
tildeG = cell(3,1);
tildeG{1} = reshape(Y.core.data, [r(1) r(2)*r(3)]);
tildeG{2} = reshape(permute(Y.core.data, [2,1,3]), [r(2) r(1)*r(3)]);
tildeG{3} = reshape(Y.core.data, [r(1)*r(2) r(3)])';


% Compute the vector transport
tildeUtU = cell(3,1);
tildeUtdotU = cell(3,1);
temp = cell(3,1);
for k = 1:3
    tildeUtU{k} = Y.U{k}' * X.U{k};
    tildeUtdotU{k} = Y.U{k}' * eta.tildeU{k};
    temp{k} = X.U{k} * tildeUtdotU{k}' + eta.tildeU{k} * tildeUtU{k}';
end


xi.tildecore = ttm(eta.tildecore, tildeUtU) + ...
               ttm(X.core, {tildeUtdotU{1}, tildeUtU{2}, tildeUtU{3}}) + ...
               ttm(X.core, {tildeUtU{1}, tildeUtdotU{2}, tildeUtU{3}}) + ...
               ttm(X.core, {tildeUtU{1}, tildeUtU{2}, tildeUtdotU{3}});


comp1 = ttm(eta.tildecore, {X.U{1}, tildeUtU{2}, tildeUtU{3}}) + ...
        ttm(X.core, {eta.tildeU{1}, tildeUtU{2}, tildeUtU{3}}) + ...   
        ttm(X.core, {X.U{1}, tildeUtdotU{2}, tildeUtU{3}}) + ...   
        ttm(X.core, {X.U{1}, tildeUtU{2}, tildeUtdotU{3}});
comp1 = double(comp1);
comp1 = reshape(comp1,[n(1) r(2)*r(3)]) * tildeG{1}' + 2 * temp{1};
xi.tildeU{1} = (comp1 - Y.U{1} * (Y.U{1}' * comp1)) / (2*speye(r(1)) + tildeG{1} * tildeG{1}');

comp2 = ttm(eta.tildecore, {tildeUtU{1}, X.U{2}, tildeUtU{3}}) + ...
        ttm(X.core, {tildeUtdotU{1}, X.U{2}, tildeUtU{3}}) + ...   
        ttm(X.core, {tildeUtU{1}, eta.tildeU{2}, tildeUtU{3}}) + ...   
        ttm(X.core, {tildeUtU{1}, X.U{2}, tildeUtdotU{3}});
comp2 = double(comp2);
comp2 = reshape(permute(comp2,[2,1,3]),[n(2) r(1)*r(3)]) * tildeG{2}' + 2 * temp{2};
xi.tildeU{2} = (comp2 - Y.U{2} * (Y.U{2}' * comp2)) / (2*speye(r(2)) + tildeG{2} * tildeG{2}');

comp3 = ttm(eta.tildecore, {tildeUtU{1}, tildeUtU{2}, X.U{3}}) + ...
        ttm(X.core, {tildeUtdotU{1}, tildeUtU{2}, X.U{3}}) + ...   
        ttm(X.core, {tildeUtU{1}, tildeUtdotU{2}, X.U{3}}) + ...   
        ttm(X.core, {tildeUtU{1}, tildeUtU{2}, eta.tildeU{3}});
comp3 = double(comp3);
comp3 = reshape(comp3, [r(1)*r(2) n(3)])' * tildeG{3}' + 2 * temp{3};
xi.tildeU{3} = (comp3 - Y.U{3} * (Y.U{3}' * comp3)) / (2*speye(r(3)) + tildeG{3} * tildeG{3}');





           
           
           


