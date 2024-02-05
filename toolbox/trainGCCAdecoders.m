function W = trainGCCAdecoders(X,Q,regularization)
% TRAINGCCADECODERS Solve the MAXVAR-GCCA problem to train GCCA decoders
% based on K EEG signals of subjects attending the same stimulus.
%
%   Input parameters:
%       X [DOUBLE]: EEG tensor (time x channel/lag x subject)
%       Q [INTEGER]: subspace dimension/number of components to
%                               extract
%       regularization [STRUCT]: fields
%           'method': 'none' (classical sample covariance), 'lwcov'
%                           (ledoit-wolf estimation), or 'l2' (diagonal
%                           loading)
%           'mu': hyperparameter in case of 'l2'
%
%   Output:
%       W [DOUBLE]: EEG decoders (channel/lag x subject x component)

% Author: Simon Geirnaert, KU Leuven, ESAT & Dept. of Neurosciences
% Correspondence: simon.geirnaert@esat.kuleuven.be

%% Construct correlation matrices
[~,M,K] = size(X);

Xfl = tens2mat(X,1,2:3);
switch regularization.method
    case {'none','l2'}
        Rxx = cov(Xfl);
    case 'lwcov'
        Rxx = lwcov(Xfl);
end
Rxx = (size(Xfl,1)-1)*Rxx;
Rdxx = zeros(K*M,K*M);
for k = 1:K
    Rdxx((k-1)*M+1:k*M,(k-1)*M+1:k*M) = Rxx((k-1)*M+1:k*M,(k-1)*M+1:k*M);
end

if strcmp(regularization.method,'l2')
    for k = 1:K
        Rdxx((k-1)*M+1:k*M,(k-1)*M+1:k*M) = Rdxx((k-1)*M+1:k*M,(k-1)*M+1:k*M) + regularization.mu*trace(Rdxx((k-1)*M+1:k*M,(k-1)*M+1:k*M))./M*eye(M);
    end
end

%% Compute GEVD with reduced number of components for speedup
Rdxx = (Rdxx+Rdxx')/2; Rxx = (Rxx+Rxx')/2; % ensure symmetry
[W,Lambda] = eigs(Rdxx,Rxx,Q,'smallestabs');

%% Correct scaling
W = W./vecnorm(W);
W = sqrt(1./(diag(Lambda).^2.*diag((Xfl*W)'*Xfl*W)))'.*W;

%% Extract filters
W = reshape(W,[M,K,Q]);
end