function [W,V] = trainSIGCCAdecoders(X,Y,Q,rho,covMethod)
% TRAINSIGCCADECODERS Solve the SI-GCCA problem to train SI-GCCA decoders
% based on K EEG signals of subjects attending the same given stimulus.
%
%   Input parameters:
%       X [DOUBLE]: EEG tensor (time x channel/lag x subject)
%       Y [DOUBLE]: stimulus matrix (time x channel/lag)
%       Q [INTEGER]: subspace dimension/number of components to
%                               extract
%       rho [DOUBLE]: stimulus hyperparameter
%       covMethod [STRING]: 'cov' (classical sample covariance) or 'lwcov'
%                           (ledoit-wolf estimation)
%
%   Output:
%       W [DOUBLE]: EEG decoders (channel/lag x subject x component)
%       V [DOUBLE]: stimulus encoder (channel/lag x component)

% Author: Simon Geirnaert, KU Leuven, ESAT & Dept. of Neurosciences
% Correspondence: simon.geirnaert@esat.kuleuven.be

%% Construct correlation matrices
[~,M,K] = size(X);
P = size(Y,2);

Xfl = tens2mat(X,1,2:3);
XflEnh = [Xfl,Y];
switch covMethod
    case 'none'
        Rxx = cov(XflEnh);
    case 'lwcov'
        Rxx = lwcov(XflEnh);
end
Rxx = (size(XflEnh,1)-1)*Rxx;
Rdxx = zeros(K*M+P,K*M+P);
for k = 1:K
    Rdxx((k-1)*M+1:k*M,(k-1)*M+1:k*M) = Rxx((k-1)*M+1:k*M,(k-1)*M+1:k*M);
end
Rdxx(K*M+1:end,K*M+1:end) = Rxx(K*M+1:end,K*M+1:end);

Rxx(1:end,K*M+1:end) = rho*Rxx(1:end,K*M+1:end);

%% Compute GEVD with reduced number of components for speedup
[W,Lambda] = eigs(Rdxx,Rxx,Q,'smallestabs');

%% Correct scaling
W = W./vecnorm(W);
RxxRho = Rxx;
RxxRho(K*M+1:end,:) = rho*RxxRho(K*M+1:end,:);
W = W*sqrtm(inv(Lambda*W'*RxxRho*W*Lambda));

%% Extract filters
V = W(K*M+1:end,:);
W = reshape(W(1:K*M,:),[M,K,Q]);
end