function W = trainGCCAdecoders(X,Q,covMethod)
% TRAINGCCADECODERS Solve the MAXVAR-GCCA problem to train GCCA decoders
% based on K EEG signals of subjects attending the same stimulus.
%
%   Input parameters:
%       X [DOUBLE]: EEG tensor (time x channel/lag x subject)
%       Q [INTEGER]: subspace dimension/number of components to
%                               extract
%       covMethod [STRING]: 'cov' (classical sample covariance) or 'lwcov'
%                           (ledoit-wolf estimation)
%
%   Output:
%       W [DOUBLE]: EEG decoders (channel/lag x subject x component)

% Author: Simon Geirnaert, KU Leuven, ESAT & Dept. of Neurosciences
% Correspondence: simon.geirnaert@esat.kuleuven.be

%% Construct correlation matrices
[~,M,K] = size(X);

Xfl = tens2mat(X,1,2:3);
switch covMethod
    case 'none'
        Rxx = cov(Xfl);
    case 'lwcov'
        Rxx = lwcov(Xfl);
end
Rxx = (size(Xfl,1)-1)*Rxx;
Rdxx = zeros(K*M,K*M);
for k = 1:K
    Rdxx((k-1)*M+1:k*M,(k-1)*M+1:k*M) = Rxx((k-1)*M+1:k*M,(k-1)*M+1:k*M);
end

%% Compute GEVD with reduced number of components for speedup
[W,Lambda] = eigs(Rdxx,Rxx,Q,'smallestabs');

%% Correct scaling
W = W./vecnorm(W);
W = W*sqrtm(inv(W'*Rdxx*W*Lambda));

%% Extract filters
W = reshape(W,[M,K,Q]);
end