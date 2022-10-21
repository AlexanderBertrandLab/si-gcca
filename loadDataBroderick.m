function [eeg,envelope,fs,trialLengthSamp] = loadDataBroderick(subject,preprocessing,trialLength,datapath)
% LOADDATA Load the EEG and envelope of a given subject. The trials are 
% preprocessed with the given parameters in preprocessing (normalization, 
% rereferencing and channel selection).
%
%   Input parameters:
%       subject [INTEGER]: subject number to load the data from
%       preprocessing [STRUCT]: preprocessing structure with normalization
%                               field (binary) and potential channel 
%                               selection (vector of channels).
%       trialLength [DOUBLE]: trial length to cut
%       datapath [STRING]: path to where your datafolder is

% Author: Simon Geirnaert, KU Leuven, ESAT & Dept. of Neurosciences
% Correspondence: simon.geirnaert@esat.kuleuven.be

eeg = []; envelope = [];
for run = 1:20
    %% Load data
    data = load([datapath,'/Natural Speech/EEG/Subject',num2str(subject),'/Subject',num2str(subject),'_Run',num2str(run),'.mat']);

    eegTemp = double(data.eegData);
    fs = data.fs;
    
    stim = load([datapath,'/Natural Speech/Stimuli/Envelopes/audio',num2str(run),'_128Hz.mat']);
    envelopeTemp = stim.env;

    %% rereference to mastoids
    eegTemp = eegTemp-mean(double(data.mastoids),2);

    %% filtering
    d = designfilt('bandpassiir','FilterOrder',4, ...
        'HalfPowerFrequency1',preprocessing.filtering.low,'HalfPowerFrequency2',preprocessing.filtering.high, ...
        'SampleRate',fs);
    eegTemp = filtfilt(d,eegTemp);
    envelopeTemp = filtfilt(d,envelopeTemp);

    %% resampling
    fsPos = 2*preprocessing.filtering.high:fs;
    fsNew = fsPos(find(mod(fs,2*preprocessing.filtering.high:fs)==0,1,'first'));

    envelopeTemp = resample(envelopeTemp,fsNew,fs);
    eegTemp = resample(eegTemp,fsNew,fs);

    fs = fsNew;

    %% Cut data
    trialLengthSamp = trialLength*fs;

    llen = floor(min(size(eegTemp,1),length(envelopeTemp))/trialLengthSamp)*trialLengthSamp;
    eegTemp = eegTemp(1:llen,:); envelopeTemp = envelopeTemp(1:llen);
    eegTemp = reshape(eegTemp,[trialLengthSamp,llen/trialLengthSamp,size(eegTemp,2)]);
    eegTemp = permute(eegTemp,[1 3 2]);
    envelopeTemp = reshape(envelopeTemp,[trialLengthSamp,llen/trialLengthSamp]);

    eeg = cat(3,eeg,eegTemp);
    envelope = [envelope,envelopeTemp];

end


%% preprocessing

% channel selection
if ~isempty(preprocessing.eegChanSel)
    eeg = eeg(:,preprocessing.eegChanSel,:);
end

% normalization
ntrials = size(eeg,3);
if preprocessing.normalization
    for tr = 1:ntrials
        eeg(:,:,tr) = eeg(:,:,tr) - mean(eeg(:,:,tr),1);
        eeg(:,:,tr) = eeg(:,:,tr)./norm(eeg(:,:,tr),'fro')*size(eeg,2);
    end
end


