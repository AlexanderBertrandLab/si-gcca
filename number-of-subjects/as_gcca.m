%% TEST VARIOUS NUMBERS OF TRAINING DATA FOR MAXVAR-GCCA

% Author: Simon Geirnaert, KU Leuven, ESAT & Dept. of Neurosciences
% Correspondence: simon.geirnaert@esat.kuleuven.be

clear; close all; clc;

%% Setup: parameters
params.subjects = 1:19; % subjects to include in experiments
params.datapath = 'C:\Users\sgeirnae\Documents\data\broderick'; % path where your data is
params.saving.save = false; params.saving.name = 'test'; % saving parameters
params.windowLength = 60; % correlation window length to use (in seconds)
params.nbFolds = 5; % number of folds to evaluate decoders in CV loop

params.numberOfSubjects = 2:19; % number of subjects for GEVD computation
params.nbReps = 25; % amount of repetitions of sampling combinations of subjects

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
pairwiseCorrs = cell(length(params.numberOfSubjects),1); % pairwise Pearson correlation coefficients

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
    end
    eeg(:,:,:,s) = eegS;
end

%% Loop over number of subjects
nbTrials = size(eeg,3);

for nbS = 1:length(params.numberOfSubjects)

    % list all possible combinations of subjects
    pairCoding = nchoosek(1:params.numberOfSubjects(nbS),2); % to index corrs and indicate the pairing of subjects
    possibleCombos = nchoosek(1:length(params.subjects),params.numberOfSubjects(nbS));
    possibleCombos = possibleCombos(randperm(size(possibleCombos,1)),:);
    maxNbReps = min(size(possibleCombos,1),params.nbReps);
    pairwiseCorrs{nbS} = zeros(size(pairCoding,1),size(eeg,3),maxNbReps,params.decoder.Q);

    corrTrs = pairwiseCorrs{nbS};
    for rep = 1:maxNbReps
        fprintf('Subject size %i/%i, rep %i/%i\n',nbS,length(params.numberOfSubjects),rep,maxNbReps);

        cv = cvpartition(nbTrials,'KFold',params.nbFolds);
        for fold = 1:params.nbFolds

            % determine training and test data
            idxTrain = cv.training(fold);
            idxTest = cv.test(fold);

            Xtest = eeg(:,:,idxTest,possibleCombos(rep,:));
            Xtrain = eeg(:,:,idxTrain,possibleCombos(rep,:));

            %% train GCCA decoders
            W = trainGCCAdecoders(reshape(permute(Xtrain,[1,3,2,4]),[size(Xtrain,1)*size(Xtrain,3),size(Xtrain,2),size(Xtrain,4)]),params.decoder.Q,params.regularization);

            %% apply decoders
            yTest = zeros(size(Xtest,1),size(Xtest,3),size(Xtest,4),params.decoder.Q);
            for s = 1:params.numberOfSubjects(nbS)
                yTest(:,:,s,:) = permute(squeeze(tmprod(Xtest(:,:,:,s),squeeze(W(:,s,:))',2)),[1,3,2]);
            end

            %% cut in windows and compute ISC and p-value

            % compute pairwise correlations
            for cmb = 1:size(pairCoding,1)
                for q = 1:params.decoder.Q
                    pairwiseCorrs{nbS}(cmb,idxTest,rep,q) = diag(corr(yTest(:,:,pairCoding(cmb,1),q),yTest(:,:,pairCoding(cmb,2),q),'Type','Pearson'));
                end
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
    save(['results-',params.saving.saveName],'pairwiseCorrs','ISC','params');
end