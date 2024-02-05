function [W,V] = trainSIcorrCAdecoders(X,Y,Q,gamma,regularization)
% TRAINSICORRCADECODERS Solve the SI-corrCA problem to train SI-corrCA decoders
% based on K EEG signals of subjects attending the same given stimulus.
%
%   Input parameters:
%       X [DOUBLE]: EEG tensor (time x channel/lag x subject)
%       Y [DOUBLE]: stimulus matrix (time x channel/lag)
%       Q [INTEGER]: subspace dimension/number of components to
%                               extract
%       gamma [DOUBLE]: stimulus hyperparameter
%       regularization [STRUCT]: fields
%           'method': 'none' (classical sample covariance), 'lwcov'
%                           (ledoit-wolf estimation), or 'l2' (diagonal
%                           loading)
%           'mu': hyperparameter in case of 'l2'
%
%   Output:
%       W [DOUBLE]: EEG decoders (channel/lag x component)
%       V [DOUBLE]: stimulus encoders (channel/lag x component)

% Author: Simon Geirnaert, KU Leuven, ESAT & Dept. of Neurosciences
% Correspondence: simon.geirnaert@esat.kuleuven.be

%% Construct correlation matrices
[~,M,~] = size(X);
P = size(Y,2);

Xapp = tens2mat(permute(X,[2,1,3]),1,2:3)'; Xsum = sum(X,3);
sumRkl = cov(Xsum);
RyyNoReg = cov(Y);

switch regularization.method
    case 'none'
        avgRxx = cov(Xapp);
        RyyReg = cov(Y);
    case 'lwcov'
        avgRxx = lwcov(Xapp);
        RyyReg = lwcov(Y);
    case 'l2'
        avgRxx = cov(Xapp); avgRxx = avgRxx + regularization.mu*trace(avgRxx)./M*eye(M);
        RyyReg = cov(Y); RyyReg = RyyReg + regularization.mu*trace(RyyReg)./P*eye(P);
end

avgRxx = (size(Xapp,1)-1)*avgRxx;
sumRkl = (size(Xsum,1)-1)*sumRkl;
RyyNoReg = (size(Y,1)-1)*RyyNoReg;
RyyReg = (size(Y,1)-1)*RyyReg;

Rl = [avgRxx,zeros(M,P);zeros(P,M),RyyReg];
Rr = [sumRkl,gamma*Xsum'*Y;Y'*Xsum,gamma*RyyNoReg];

%% Compute GEVD with reduced number of components for speedup
[W,Lambda] = eigs(Rl,Rr,Q,'smallestabs');

%% Correct scaling
W = W./vecnorm(W);
Xenh = [Xsum,gamma*Y];
W = (1./diag(Lambda).*sqrt(1./diag((Xenh*W)'*Xenh*W)))'.*W;

%% Extract filters
V = W(M+1:end,:);
W = W(1:M,:);
end