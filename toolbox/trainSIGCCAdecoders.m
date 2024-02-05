function [W,V] = trainSIGCCAdecoders(X,Y,Q,gamma,regularization)
% TRAINSIGCCADECODERS Solve the SI-GCCA problem to train SI-GCCA decoders
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
%       W [DOUBLE]: EEG decoders (channel/lag x subject x component)
%       V [DOUBLE]: stimulus encoders (channel/lag x component)

% Author: Simon Geirnaert, KU Leuven, ESAT & Dept. of Neurosciences
% Correspondence: simon.geirnaert@esat.kuleuven.be

%% Construct correlation matrices
[~,M,K] = size(X);
P = size(Y,2);

Xfl = tens2mat(X,1,2:3);
XflEnh = [Xfl,Y];
Rxx = cov(XflEnh);

Rdxx = zeros(K*M+P,K*M+P);
switch regularization.method
    case 'none'
        for k = 1:K
            Rdxx((k-1)*M+1:k*M,(k-1)*M+1:k*M) = Rxx((k-1)*M+1:k*M,(k-1)*M+1:k*M);
        end
        Rdxx(K*M+1:end,K*M+1:end) = Rxx(K*M+1:end,K*M+1:end);
    case 'l2'
        for k = 1:K
            Rdxx((k-1)*M+1:k*M,(k-1)*M+1:k*M) = Rxx((k-1)*M+1:k*M,(k-1)*M+1:k*M) + regularization.mu*trace(Rdxx((k-1)*M+1:k*M,(k-1)*M+1:k*M))./M*eye(M);
        end
        Rdxx(K*M+1:end,K*M+1:end) = lwcov(Y);
    case 'lwcov'
        for k = 1:K
            Rdxx((k-1)*M+1:k*M,(k-1)*M+1:k*M) = lwcov(X(:,:,k));
        end
        Rdxx(K*M+1:end,K*M+1:end) = lwcov(Y);
end

Rxx(1:end,K*M+1:end) = gamma*Rxx(1:end,K*M+1:end);

%% Compute GEVD with reduced number of components for speedup
[W,Lambda] = eigs(Rdxx,Rxx,Q,'smallestabs');

%% Correct scaling
W = W./vecnorm(W);
RxxRho = Rxx;
RxxRho(K*M+1:end,:) = gamma*RxxRho(K*M+1:end,:);
W = (1./diag(Lambda).*sqrt(1./diag(W'*RxxRho*W)))'.*W;

%% Extract filters
V = W(K*M+1:end,:);
W = reshape(W(1:K*M,:),[M,K,Q]);
end