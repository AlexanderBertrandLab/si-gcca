%% TEST VARIOUS AMOUNTS OF TRAINING DATA FOR MAXVAR-GCCA

% Author: Simon Geirnaert, KU Leuven, ESAT & Dept. of Neurosciences
% Correspondence: simon.geirnaert@esat.kuleuven.be

clear; close all; clc;

%% Setup: parameters
params.subjects = 1:10; % subjects to include in experiments
params.datapath = 'C:\Users\sgeirnae\Documents\data\broderick'; % path where your data is
params.saving.save = false; params.saving.name = 'test'; % saving parameters
params.windowLength = 60; % correlation window length to use (in seconds)

params.amountOfTrainingData = [1:10,12:2:20,25:5:45]; % amount of training data for GEVD computation
params.nbReps = 50; % amount of repetitions of sampling the data

% preprocessing
params.preprocessing.normalization = true; % 1: with normalization of regression matrices (column-wise), 0: without normalization
params.preprocessing.eegChanSel = [];
% params.preprocessing.eegChanSel = [1,3,7,15,19,21,28,36,43,54,58,59,68,71,80,83,85,93,100,103,115,119,127]; % smarting mbraintrain for biosemi 128-channel
params.preprocessing.filtering.low = 1; % lower band for filtering
params.preprocessing.filtering.high = 4; % upper band for filtering

% decoder and result estimation
params.decoder.Leeg = [-0.25,0.25]; % integration window EEG
params.decoder.Q = 1; % subspace dimension/number of components to use
params.regularization = 'lwcov'; % 'none': no regularization, 'lwcov': ridge regression with Ledoit-Wolf

%% Initialization
pairwiseCorrs = cell(length(params.amountOfTrainingData),1); % pairwise Pearson correlation coefficients
pairCoding = nchoosek(1:length(params.subjects),2); % to index corrs and indicate the pairing of subjects

%% Load all data and construct data matrices
fprintf('Loading all data\n');
for s = 1:length(params.subjects)
    fprintf('   Subject %i/%i\n',s,length(params.subjects))
    %% Load all data (EEG and speech envelopes, already preprocessed) and build data matrices
    % load all data of subject s
    sb = params.subjects(s);
    [eegS,~,fs,trialLength] = loadDataBroderick(sb,params.preprocessing,params.windowLength,params.datapath);

    % time-delay embedding
    eegS = timeDelayEmbedding(eegS,params.decoder.Leeg,fs);

    % initialize eeg variable
    if s == 1
        eeg = zeros(size(eegS,1),size(eegS,2),size(eegS,3),length(params.subjects));
        for trS = 1:length(params.amountOfTrainingData)
            pairwiseCorrs{trS} = zeros(size(pairCoding,1),size(eeg,3)-params.amountOfTrainingData(trS),params.nbReps,params.decoder.Q);
        end
    end
    eeg(:,:,:,s) = eegS;
end

%% Loop over amounts of training data
nbTrials = size(eeg,3);

for trS = 1:length(params.amountOfTrainingData)
    for rep = 1:params.nbReps
        fprintf('Training/testing training size %i/%i, rep %i/%i\n',trS,length(params.amountOfTrainingData),rep,params.nbReps);

        % determine training and test data
        ri = randperm(nbTrials,params.amountOfTrainingData(trS));
        idxTrain = false(nbTrials,1);
        idxTrain(ri) = 1;
        idxTest = logical(1-idxTrain);

        X = struct;
        X.test = eeg(:,:,idxTest,:);
        X.train = eeg(:,:,idxTrain,:);

        %% train GCCA decoders
        W = trainGCCAdecoders(reshape(permute(X.train,[1,3,2,4]),[size(X.train,1)*size(X.train,3),size(X.train,2),size(X.train,4)]),params.decoder.Q,params.regularization);

        %% apply decoders
        yTest = zeros(size(X.test,1),size(X.test,3),size(X.test,4),params.decoder.Q);
        for s = 1:length(params.subjects)
            yTest(:,:,s,:) = permute(squeeze(tmprod(X.test(:,:,:,s),squeeze(W(:,s,:))',2)),[1,3,2]);
        end

        %% cut in windows and compute ISC
        % segmentation
        yWindowedTest = segmentize(yTest,'Segsize',params.windowLength*fs);
        yWindowedTest = reshape(yWindowedTest,[size(yWindowedTest,1),size(yWindowedTest,2)*size(yWindowedTest,3),size(yWindowedTest,4),size(yWindowedTest,5)]);

        % compute pairwise correlations
        for cmb = 1:size(pairCoding,1)
            for q = 1:params.decoder.Q
                pairwiseCorrs{trS}(cmb,:,rep,q) = diag(corr(yWindowedTest(:,:,pairCoding(cmb,1),q),yWindowedTest(:,:,pairCoding(cmb,2),q),'Type','Pearson'));
            end
        end
    end
    ISC = cellfun(@(x) squeeze(mean(x,1)), pairwiseCorrs, 'UniformOutput',false);
    disp(cell2mat(cellfun(@(x) mean(mean(x,2),1), ISC, 'UniformOutput', false)));
end
ISC = cellfun(@(x) squeeze(mean(x,1)), pairwiseCorrs, 'UniformOutput',false);

% display results
fprintf('\n Average ISC per component/amount of training data:\n');
disp(cell2mat(cellfun(@(x) mean(mean(x,2),1), ISC, 'UniformOutput', false)));

%% save results
if params.saving.save
    save(['results-',params.saving.name],'pairwiseCorrs','ISC','params');
end