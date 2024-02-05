function W = trainCorrCAdecoders(X,Q,regularization)
% TRAINCORRCADECODERS Solve the MAXVAR-corrCA problem to train corrCA decoders
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
%       W [DOUBLE]: EEG decoders (channel/lag x component)

% Author: Simon Geirnaert, KU Leuven, ESAT & Dept. of Neurosciences
% Correspondence: simon.geirnaert@esat.kuleuven.be

%% Construct correlation matrices
[~,M,~] = size(X);
Xapp = tens2mat(permute(X,[2,1,3]),1,2:3)'; Xsum = sum(X,3);
sumRkl = cov(Xsum);

switch regularization.method
    case 'none'
        avgRxx = cov(Xapp);
    case 'lwcov'
        avgRxx = lwcov(Xapp);
    case 'l2'
        avgRxx = cov(Xapp); avgRxx = avgRxx + regularization.mu*trace(avgRxx)./M*eye(M);
end

%% Compute GEVD with reduced number of components for speedup
[W,Lambda] = eigs(avgRxx,sumRkl,Q,'smallestabs');

%% Correct scaling
W = W./vecnorm(W);
W = sqrt(1./(diag(Lambda).^2.*diag((Xsum*W)'*Xsum*W)))'.*W;

end