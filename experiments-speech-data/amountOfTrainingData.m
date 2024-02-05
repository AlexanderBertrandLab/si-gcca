%% TEST A SPECIFIC AMOUNT OF TRAINING DATA

% Author: Simon Geirnaert, KU Leuven, ESAT & Dept. of Neurosciences
% Correspondence: simon.geirnaert@esat.kuleuven.be

clear; close all; clc;

%% Setup: parameters
params.amountOfTrainingData = 30; % amount of training data for GEVD computation
params.methods = {'corrCA','corrCA-l2','SI-corrCA','GCCA','GCCA-l2','SI-GCCA'}; % options: 'GCCA', 'SI-GCCA', 'corrCA', 'SI-corrCA', 'GCCA-l2', 'corrCA-l2'

params.datapath = 'C:\Users\sgeirnae\Documents\data\broderick'; % path where your data is
params.windowLength = 60; % decision window length to test (in seconds)
params.nbReps = 5; % amount of repetitions of sampling the data
params.subjects = 1:6; % subjects to include in experiments

% preprocessing
% params.preprocessing.eegChanSel = [];
% params.preprocessing.eegChanSel = [1,3,5,7,10,15,17,19,21,23,25,28,30,32,34,36,39,42,43,45,46,48,50,52,54,56,57,59,61,63,68,69,71,72,75,76,79,80,81,83,85,87,88,89,92,93,94,100,101,103,104,106,108,110,112,115,117,119,120,122,124,125,127,128]; % 64-channel
params.preprocessing.eegChanSel = [1,3,7,15,19,21,28,36,43,54,58,59,68,71,80,83,85,93,100,103,115,119,127]; % smarting mbraintrain for biosemi 128-channel
params.preprocessing.normalization = true; % 1: with normalization of regression matrices (column-wise), 0: without normalization
params.preprocessing.filtering.low = 1; % lower band for filtering
params.preprocessing.filtering.high = 4; % upper band for filtering

% decoder and result estimation
params.decoder.Leeg = [-0.25,0.25]; % integration window ([-250,250]ms)
params.decoder.Laudio = [-1.25,0]; % integration window audio
params.decoder.gamma.validationSize = 0.25; % percentage of left-out data as validation data for gamma
% params.decoder.gamma.values = [0,10.^(-2:0.5:8)];
params.decoder.gamma.values = [0,10.^-0.5];
params.decoder.mu.values = [0,10.^(1)]; % weight for L2-regularization in GCCA/corrCA estimation -> one value in case of 'fixed', range of values with 'validated'
params.decoder.nbComponents = 1; % number of GCCA components to use

% backward decoding
params.bwDecoder.L = [0,0.25];
params.bwDecoder.regularization.method = 'lwcov';

% saving parameters
params.saving.save = false; params.saving.naming = 'test'; % saving parameters

%% Initialization
pairCoding = nchoosek(1:length(params.subjects),2); % to index corrs and indicate the pairing of subjects

%% Load all data and construct data matrices
fprintf('Loading all data\n');
for s = 1:length(params.subjects)
    fprintf('   Subject %i/%i\n',s,length(params.subjects))

    %% Load all data (EEG and speech envelopes, already preprocessed) and build data matrices
    sb = params.subjects(s);
    [eegS,~,fs,~] = loadDataBroderick(sb,params.preprocessing,params.windowLength,params.datapath);

    % time-delay embedding
    eegS = timeDelayEmbedding(eegS,params.decoder.Leeg,fs);

    % initialize eeg variable
    if s == 1
        eeg = zeros(size(eegS,1),size(eegS,2),size(eegS,3),length(params.subjects));
        validationSize = max(0,round(params.decoder.gamma.validationSize*(size(eeg,3)-params.amountOfTrainingData)));
        testSize = size(eeg,3)-params.amountOfTrainingData-validationSize;
        pairwiseCorrs = zeros(size(pairCoding,1),testSize,params.nbReps,params.decoder.nbComponents,length(params.methods)); % pair x window x repetition x component x method
        bwCorr = zeros(testSize,params.decoder.nbComponents,length(params.subjects)+1,2,params.nbReps,length(params.methods)); % window x component x subject + avg subspace x repetitions x method
    end
    eeg(:,:,:,s) = eegS;
end
clear eegS;

% load envelope
[~,envelope,fs,trialLength] = loadDataBroderick(sb,params.preprocessing,params.windowLength,params.datapath);

% time-delay embedding envelope
envelope = timeDelayEmbedding(reshape(envelope,[size(envelope,1),1,size(envelope,2)]),params.decoder.Laudio,fs);

%% Loop over amounts of training data
fprintf('Training and testing...\n');
nbTrials = size(eeg,3);
nbMethods = length(params.methods);

%% Loop over repetitions
for rep = 1:params.nbReps
    fprintf('%.2f%% completed\n',(rep-1)/params.nbReps*100);

    % determine training and test data
    ri = randperm(nbTrials,params.amountOfTrainingData+validationSize); % randomly pick training and validation segments
    idxTrain = false(nbTrials,1);
    idxTrain(ri(1:params.amountOfTrainingData)) = 1;
    idxValidation = false(nbTrials,1);
    idxValidation(ri(params.amountOfTrainingData+1:end)) = 1;
    idxTest = logical(1-idxTrain-idxValidation);

    X = struct;
    X.test = eeg(:,:,idxTest,:);
    X.val = eeg(:,:,idxValidation,:);
    X.train = eeg(:,:,idxTrain,:);

    env = struct;
    env.test = envelope(:,:,idxTest);
    env.val = envelope(:,:,idxValidation);
    env.train = envelope(:,:,idxTrain);

    %% train group decoders
    for meI = 1:nbMethods
        method = params.methods{meI};
        %% validate optimal hyperparameter
        if strcmp('SI',method(1:2)) || contains(method,'l2')
            if strcmp('SI',method(1:2))
                hyperparams = params.decoder.gamma.values;
            elseif contains(method,'l2')
                hyperparams = params.decoder.mu.values;
            end
            ISCtemp = zeros(length(hyperparams),1);
            for hypCnt = 1:length(hyperparams)
                yVal = zeros(size(X.val,1),size(X.val,3),size(X.val,4),params.decoder.nbComponents);
                % train neural decoders
                switch method
                    case 'SI-GCCA'
                        reg = struct; reg.method = 'lwcov';
                        [W,~] = trainSIGCCAdecoders(reshape(permute(X.train,[1,3,2,4]),[size(X.train,1)*size(X.train,3),size(X.train,2),size(X.train,4)]),reshape(permute(env.train,[1,3,2]),[size(env.train,1)*size(env.train,3),size(env.train,2)]),params.decoder.nbComponents,hyperparams(hypCnt),reg);
                    case 'SI-corrCA'
                        reg = struct; reg.method = 'lwcov';
                        [W,~] = trainSIcorrCAdecoders(reshape(permute(X.train,[1,3,2,4]),[size(X.train,1)*size(X.train,3),size(X.train,2),size(X.train,4)]),reshape(permute(env.train,[1,3,2]),[size(env.train,1)*size(env.train,3),size(env.train,2)]),params.decoder.nbComponents,hyperparams(hypCnt),reg);
                        W = permute(repmat(W,[1,1,length(params.subjects)]),[1,3,2]);
                    case 'GCCA-l2'
                        reg = struct; reg.method = 'l2'; reg.mu = hyperparams(hypCnt);
                        W = trainGCCAdecoders(reshape(permute(X.train,[1,3,2,4]),[size(X.train,1)*size(X.train,3),size(X.train,2),size(X.train,4)]),params.decoder.nbComponents,reg);
                    case 'corrCA-l2'
                        reg = struct; reg.method = 'l2'; reg.mu = hyperparams(hypCnt);
                        W = trainCorrCAdecoders(reshape(permute(X.train,[1,3,2,4]),[size(X.train,1)*size(X.train,3),size(X.train,2),size(X.train,4)]),params.decoder.nbComponents,reg);
                        W = permute(repmat(W,[1,1,length(params.subjects)]),[1,3,2]);
                end

                % output validation signals
                for s = 1:length(params.subjects)
                    yVal(:,:,s,:) = permute(squeeze(tmprod(X.val(:,:,:,s),squeeze(W(:,s,:))',2)),[1,3,2]);
                end

                % compute pairwise correlations on validation set
                corrTemp = zeros(size(pairCoding,1),size(yVal,2),params.decoder.nbComponents);
                for cmb = 1:size(pairCoding,1)
                    for c = 1:params.decoder.nbComponents
                        corrTemp(cmb,:,c) = diag(corr(yVal(:,:,pairCoding(cmb,1),c),yVal(:,:,pairCoding(cmb,2),c),'Type','Pearson'));
                    end
                end
                ISCtemp(hypCnt) = mean(mean(corrTemp(:,:,1)));
            end
            [~,hypInd] = max(ISCtemp);
            if strcmp('SI',method(1:2))
                gamma = params.decoder.gamma.values(hypInd);
            elseif contains(method,'l2')
                mu = params.decoder.mu.values(hypInd);
            end
        end

        %% train neural decoders
        switch method
            case 'GCCA'
                reg = struct; reg.method = 'none';
                W = trainGCCAdecoders(reshape(permute(X.train,[1,3,2,4]),[size(X.train,1)*size(X.train,3),size(X.train,2),size(X.train,4)]),params.decoder.nbComponents,reg);
            case 'GCCA-l2'
                reg = struct; reg.method = 'l2'; reg.mu = mu;
                W = trainGCCAdecoders(reshape(permute(X.train,[1,3,2,4]),[size(X.train,1)*size(X.train,3),size(X.train,2),size(X.train,4)]),params.decoder.nbComponents,reg);
            case 'SI-GCCA'
                reg = struct; reg.method = 'lwcov';
                [W,~] = trainSIGCCAdecoders(reshape(permute(X.train,[1,3,2,4]),[size(X.train,1)*size(X.train,3),size(X.train,2),size(X.train,4)]),reshape(permute(env.train,[1,3,2]),[size(env.train,1)*size(env.train,3),size(env.train,2)]),params.decoder.nbComponents,gamma,reg);
            case 'corrCA'
                reg = struct; reg.method = 'none';
                W = trainCorrCAdecoders(reshape(permute(X.train,[1,3,2,4]),[size(X.train,1)*size(X.train,3),size(X.train,2),size(X.train,4)]),params.decoder.nbComponents,reg);
            case 'corrCA-l2'
                reg = struct; reg.method = 'l2'; reg.mu = mu;
                W = trainCorrCAdecoders(reshape(permute(X.train,[1,3,2,4]),[size(X.train,1)*size(X.train,3),size(X.train,2),size(X.train,4)]),params.decoder.nbComponents,reg);
            case 'SI-corrCA'
                reg = struct; reg.method = 'lwcov';
                [W,~] = trainSIcorrCAdecoders(reshape(permute(X.train,[1,3,2,4]),[size(X.train,1)*size(X.train,3),size(X.train,2),size(X.train,4)]),reshape(permute(env.train,[1,3,2]),[size(env.train,1)*size(env.train,3),size(env.train,2)]),params.decoder.nbComponents,gamma,reg);
        end

        %% apply decoders
        yTrain = zeros(size(X.train,1),size(X.train,3),size(X.train,4),params.decoder.nbComponents);
        yTest = zeros(size(X.test,1),size(X.test,3),size(X.test,4),params.decoder.nbComponents);
        for s = 1:length(params.subjects)
            if contains(method,'corrCA')
                yTrain(:,:,s,:) = permute(squeeze(tmprod(X.train(:,:,:,s),W',2)),[1,3,2]);
                yTest(:,:,s,:) = permute(squeeze(tmprod(X.test(:,:,:,s),W',2)),[1,3,2]);
            elseif contains(method,'GCCA')
                yTrain(:,:,s,:) = permute(squeeze(tmprod(X.train(:,:,:,s),squeeze(W(:,s,:))',2)),[1,3,2]);
                yTest(:,:,s,:) = permute(squeeze(tmprod(X.test(:,:,:,s),squeeze(W(:,s,:))',2)),[1,3,2]);
            end
        end

        %% perform backward decoding and apply stimulus decoders
        envTrain = env.train(:,1,:); envTrain = envTrain(:);

        % individual decoding
        yBw = zeros(size(yTest,1),size(yTest,2),params.decoder.nbComponents,length(params.subjects)+1,2); % 4th dimension: every subject + hidden subspace, last dimension: individual components + incremental subspace
        for s = 1:length(params.subjects)
            % time-delay embedding
            yTrainEmbed = timeDelayEmbedding(permute(yTrain(:,:,s,:),[1,4,2,3]),params.bwDecoder.L,fs);
            yTrainEmbed = reshape(permute(yTrainEmbed,[1,3,2]),[size(yTrainEmbed,1)*size(yTrainEmbed,3),size(yTrainEmbed,2)]);
            yTestEmbed = timeDelayEmbedding(permute(yTest(:,:,s,:),[1,4,2,3]),params.bwDecoder.L,fs);

            % compute decoder and output
            nbLagsBw = size(yTrainEmbed,2)/params.decoder.nbComponents;
            for c = 1:params.decoder.nbComponents
                d = trainStimulusDecoder(yTrainEmbed(:,(c-1)*nbLagsBw+1:nbLagsBw*c),envTrain,params.bwDecoder.regularization);
                yBw(:,:,c,s,1) = squeeze(tmprod(yTestEmbed(:,(c-1)*nbLagsBw+1:nbLagsBw*c,:),d',2));

                d = trainStimulusDecoder(yTrainEmbed(:,1:nbLagsBw*c),envTrain,params.bwDecoder.regularization);
                yBw(:,:,c,s,2) = squeeze(tmprod(yTestEmbed(:,1:nbLagsBw*c,:),d',2));
            end
        end

        % hidden subspace decoding
        sTrainEmbed = timeDelayEmbedding(permute(mean(yTrain,3),[1,4,2,3]),params.bwDecoder.L,fs); % time-delay embedding of hidden subspace
        sTestEmbed = timeDelayEmbedding(permute(mean(yTest,3),[1,4,2,3]),params.bwDecoder.L,fs);
        sTrainEmbed = reshape(permute(sTrainEmbed,[1,3,2]),[size(sTrainEmbed,1)*size(sTrainEmbed,3),size(sTrainEmbed,2)]);
        nbLagsBw = size(sTrainEmbed,2)/params.decoder.nbComponents;

        for c = 1:params.decoder.nbComponents
            d = trainStimulusDecoder(sTrainEmbed(:,(c-1)*nbLagsBw+1:nbLagsBw*c),envTrain,params.bwDecoder.regularization);
            yBw(:,:,c,length(params.subjects)+1,1) = squeeze(tmprod(sTestEmbed(:,(c-1)*nbLagsBw+1:nbLagsBw*c,:),d',2));
            d = trainStimulusDecoder(sTrainEmbed(:,1:nbLagsBw*c),envTrain,params.bwDecoder.regularization);
            yBw(:,:,c,length(params.subjects)+1,2) = squeeze(tmprod(sTestEmbed(:,1:nbLagsBw*c,:),d',2));
        end

        %% cut in windows and compute ISC
        % segmentation
        yWindowedTest = segmentize(yTest,'Segsize',params.windowLength*fs);
        yWindowedTest = reshape(yWindowedTest,[size(yWindowedTest,1),size(yWindowedTest,2)*size(yWindowedTest,3),size(yWindowedTest,4),size(yWindowedTest,5)]);

        yBw = segmentize(yBw,'Segsize',params.windowLength*fs);
        yBw = reshape(yBw,[size(yBw,1),size(yBw,2)*size(yBw,3),size(yBw,4),size(yBw,5),size(yBw,6)]);
        envTest = squeeze(env.test(:,1,:));
        envTest = segmentize(envTest,'Segsize',params.windowLength*fs);
        envTest = reshape(envTest,[size(envTest,1),size(envTest,2)*size(envTest,3)]);

        % compute stimulus correlations
        for s = 1:length(params.subjects)+1
            for c = 1:params.decoder.nbComponents
                for setting = 1:2
                    bwCorr(:,c,s,setting,rep,meI) = diag(corr(yBw(:,:,c,s,setting),envTest,'Type','Pearson'));
                end
            end
        end

        % compute pairwise correlations
        for cmb = 1:size(pairCoding,1)
            for c = 1:params.decoder.nbComponents
                pairwiseCorrs(cmb,:,rep,c,meI) = diag(corr(yWindowedTest(:,:,pairCoding(cmb,1),c),yWindowedTest(:,:,pairCoding(cmb,2),c),'Type','Pearson'));
            end
        end
    end
end

%% collect and display results
ISC = squeeze(mean(pairwiseCorrs,1));

% display
fprintf('\n Average ISC (component x method):\n');
disp(squeeze(mean(mean(ISC,2),1)));

fprintf('\n Average individual comp stimulus correlation (component x subject + subspace x method):\n');
disp(squeeze(mean(mean(bwCorr(:,:,:,1,:,:),1),5)));

fprintf('\n Average cumulative comp stimulus correlation (component x subject + subspace x method):\n');
disp(squeeze(mean(mean(bwCorr(:,:,:,2,:,:),1),5)));

%% save results
if params.saving.save
    save(['results-',params.saving.saveName],'pairwiseCorrs','ISC','bwCorr','params');
end

