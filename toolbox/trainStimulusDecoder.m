function d = trainStimulusDecoder(X,Y,regularization)
% TRAINSTIMULUSDECODER Train a stimulus decoder.
%
%   Input parameters:
%       X [DOUBLE]: EEG matrix (time x channel/lag)
%       Y [DOUBLE]: stimulus matrix (time x channel/lag)
%       regularization [STRUCT]: fields
%           'method': 'none' (classical sample covariance), 'lwcov'
%                           (ledoit-wolf estimation), or 'l2' (diagonal
%                           loading)
%
%   Output:
%       d [DOUBLE]: EEG stimulus decoder (channel/lag)

% Author: Simon Geirnaert, KU Leuven, ESAT & Dept. of Neurosciences
% Correspondence: simon.geirnaert@esat.kuleuven.be

switch regularization.method
    case 'none'
        d = (X'*X)\(X'*Y);
    case 'lwcov'
        Rxx = lwcov(X);
        d = Rxx\(X'*Y);
end

end