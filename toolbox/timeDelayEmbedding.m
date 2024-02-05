function Y = timeDelayEmbedding(X,L,fs)
% TIMEDELAYEMBEDDING Time-delay embedding by hankelizing the EEG.
%
%   Input parameters:
%       X [DOUBLE]: EEG tensor (time x channel x trial)
%       L [DOUBLE]: range of time lags (in seconds)
%       fs [DOUBLE]: sampling frequency

% Author: Simon Geirnaert, KU Leuven, ESAT & Dept. of Neurosciences
% Correspondence: simon.geirnaert@esat.kuleuven.be

zerosBefore = abs(min(round(L(1)*fs),0));
zerosAfter = abs(max(round(L(2)*fs),0));
T = size(X,1);
nbLags = zerosBefore+zerosAfter+1;

Y = zeros(size(X,1),nbLags*size(X,2),size(X,3));
for tr = 1:size(X,3)
    for ch = 1:size(X,2)
        fC = [X(:,ch,tr);zeros(zerosAfter,1)];
        fC = fC(end-T+1:end);
        fR = [X(zerosAfter+1:-1:1,ch,tr)',zeros(1,zerosBefore)];
        Y(:,(ch-1)*nbLags+1:ch*nbLags,tr) = toeplitz(fC,fR);
    end
end


end