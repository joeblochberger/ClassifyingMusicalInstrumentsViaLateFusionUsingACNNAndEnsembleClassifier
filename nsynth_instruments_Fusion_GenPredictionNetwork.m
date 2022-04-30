% Joe Blochberger, March 2022
% uses MATLAB R2020b
parpool(1);delete(gcp);
clear all; close all; clc;
datasetFolder_train = fullfile('..\nsynth-train.jsonwav.tar\nsynth-train.jsonwav\nsynth-train');
datasetFolder_test = fullfile('..\nsynth-test.jsonwav.tar\nsynth-test.jsonwav\nsynth-test');
datasetFolder_valid = fullfile('..\nsynth-valid.jsonwav.tar\nsynth-valid.jsonwav\nsynth-valid');

%% Simulation forloop
start=1;
finish=10;
for SimulationNum = start:finish
    tic
    %% set up training meta data from .txt file
    metadata_train = readtable(fullfile(datasetFolder_train,'meta_train_JAB_no_synth_lead.txt'), ...
        'Delimiter',{'\t'}, ...
        'ReadVariableNames',false);
    metadata_train.Properties.VariableNames = {'FileName','InstrumentFamily','SpecificInstrument'};
    head(metadata_train)
    
    %% set up testing metadata from .txt file
    metadata_test = readtable(fullfile(datasetFolder_test,'meta_test.txt'), ...
        'Delimiter',{'\t'}, ...
        'ReadVariableNames',false);
    metadata_test.Properties.VariableNames = {'FileName','InstrumentFamily','SpecificInstrument'};
    head(metadata_test)
    
    %% set up validation metadata from .txt file
    metadata_valid = readtable(fullfile(datasetFolder_valid,'meta_valid.txt'), ...
        'Delimiter',{'\t'}, ...
        'ReadVariableNames',false);
    metadata_valid.Properties.VariableNames = {'FileName','InstrumentFamily','SpecificInstrument'};
    head(metadata_valid)
    
    %% check if recordings are contaminating training and testing data
    sharedRecordingLocations = intersect(metadata_test.SpecificInstrument,metadata_train.SpecificInstrument);
    fprintf('Number of specific recording locations in both train and test sets = %d\n',numel(sharedRecordingLocations))
    
    %% define filepaths for training and testing data
    train_filePaths = fullfile(datasetFolder_train,'audio',metadata_train.FileName);
    test_filePaths = fullfile(datasetFolder_test,'audio',metadata_test.FileName);
    valid_filePaths = fullfile(datasetFolder_valid,'audio',metadata_valid.FileName);
    
    %% Create an audio datastore with wav and mp3 files from the samples folder
    adsTrain = audioDatastore(train_filePaths, ...
        'Labels',categorical(metadata_train.InstrumentFamily), ...
        'IncludeSubfolders',true);
    display(countEachLabel(adsTrain))
    
    adsTest = audioDatastore(test_filePaths, ...
        'Labels',categorical(metadata_test.InstrumentFamily), ...
        'IncludeSubfolders',true);
    display(countEachLabel(adsTest))
    
    adsValid = audioDatastore(valid_filePaths, ...
        'Labels',categorical(metadata_valid.InstrumentFamily), ...
        'IncludeSubfolders',true);
    display(countEachLabel(adsValid))
    
    %% Reduce dataset if needed for testing
    % 80% data is training
    % 20% data is validation
    % test data is new data
    
    reduceDataset = false;
%     reduceDataset = true;
%     if reduceDataset
%         % reduce the datasets
%         reduction = 141*10
%         adsTrain = splitEachLabel(adsTrain,reduction,'randomized'); % 33.8 GB (283704 files)
%         
%         % make the labels all equal, 141 labels in Test and Validation Data
%         adsTest = splitEachLabel(adsTest,141,'randomized'); % 500 MB (4096 files)
%         adsValid = splitEachLabel(adsValid,141,'randomized'); % 1.54 GB (12678 files)
%     end
    display(countEachLabel(adsTrain))
    display(countEachLabel(adsTest))
    display(countEachLabel(adsValid))
    
    %% read in audioDatastore information to see if it works
    [data,adsInfo] = read(adsTrain);
    data = data./max(data,[],'all');
    fs = adsInfo.SampleRate;
    sound(data,fs)
    fprintf('Instrument = %s\n',adsTrain.Labels(1))
    
    %% Reset audioDatastore once you're good
    reset(adsTrain)
    
    %% Feature extraction for CNN
    dataMono = mean(data,2); % the NSynth dataset is monophonic: https://magenta.tensorflow.org/datasets/nsynth
    
    %% Buffer the data for feature extraction - No need to
%     segmentLength = 4; % Four-second .wav file clips
%     segmentOverlap = 0; % arbitrary overlap default is 0.5
    
    %% FFT information
    windowLength = 2048*2;
    samplesPerHop = 1024*2;
    samplesOverlap = round(0.5*windowLength);
    fftLength = 2*windowLength; %--> windowLength
    numBands = 128*2; % default was 128
    
    % Spectrograms with mel scale on y-axis
    spec = melSpectrogram(dataMono,fs, ...
        'Window',hann(windowLength,'periodic'), ...
        'OverlapLength',samplesOverlap, ...
        'FFTLength',fftLength, ...
        'NumBands',numBands);
    
    % Do not want any -Inf to show up
    spec = log10(spec+eps);
    
    % Reshape spectrogram output for feature extraction
    X = reshape(spec,size(spec,1),size(spec,2),size(data,2),[]);
    
    % Plot some figures of the melSpectrogram
    %     plotflag = false;
    plotflag = true;
    if plotflag
        figure
        melSpectrogram(dataMono,fs, ...
            'Window',hamming(windowLength,'periodic'), ...
            'OverlapLength',samplesOverlap, ...
            'FFTLength',fftLength, ...
            'NumBands',numBands);
        title(['Instrument = ',adsTrain.Labels(1)])
        colormap turbo
    end
    
    %% Use parallel processing for training
    pp = parpool('IdleTimeout',inf);
    
    train_set_tall = tall(adsTrain);
    xTrain = cellfun(@(x)HelperSegmentedMelSpectrograms_JABmono(x,fs, ...
        'WindowLength',windowLength, ...
        'HopLength',samplesPerHop, ...
        'NumBands',numBands, ...
        'FFTLength',fftLength), ...
        train_set_tall, ...
        'UniformOutput',false);
    xTrain = gather(xTrain);
    xTrain = cat(4,xTrain{:});
    
    test_set_tall = tall(adsTest);
    xTest = cellfun(@(x)HelperSegmentedMelSpectrograms_JABmono(x,fs, ...
        'WindowLength',windowLength, ...
        'HopLength',samplesPerHop, ...
        'NumBands',numBands, ...
        'FFTLength',fftLength), ...
        test_set_tall, ...
        'UniformOutput',false);
    xTest = gather(xTest);
    xTest = cat(4,xTest{:});
    
    valid_set_tall = tall(adsValid);
    xValid = cellfun(@(x)HelperSegmentedMelSpectrograms_JABmono(x,fs, ...
        'WindowLength',windowLength, ...
        'HopLength',samplesPerHop, ...
        'NumBands',numBands, ...
        'FFTLength',fftLength), ...
        valid_set_tall, ...
        'UniformOutput',false);
    xValid = gather(xValid);
    xValid = cat(4,xValid{:});
    
    %% Replicate the labels of the training, test, and validation sets so that they are in one-to-one correspondence with the segments.
    size(xTest)
    
    % numSegmentsPer10seconds = size(dataBuffered,2)/2;
    numSegmentsPer4seconds = 1;
    yTrain = repmat(adsTrain.Labels,1,numSegmentsPer4seconds)';
    yTrain = yTrain(:);
    yTest = repmat(adsTest.Labels,1,numSegmentsPer4seconds)';
    yTest = yTest(:);
    yValid = repmat(adsValid.Labels,1,numSegmentsPer4seconds)';
    yValid = yValid(:);
    
    size(yTest)
    
    %% Data Augmentation for CNN - Training data only
    xTrainExtra = xTrain;
    yTrainExtra = yTrain;
    lambda = 0.5;
    for i = 1:size(xTrain,4)
        
        % Find all available spectrograms with different labels.
        availableSpectrograms = find(yTrain~=yTrain(i));
        
        % Randomly choose one of the available spectrograms with a different label.
        numAvailableSpectrograms = numel(availableSpectrograms);
        idx = randi([1,numAvailableSpectrograms]);
        
        % Mix.
        xTrainExtra(:,:,:,i) = lambda*xTrain(:,:,:,i) + (1-lambda)*xTrain(:,:,:,availableSpectrograms(idx));
        
        % Specify the label as randomly set by lambda.
        if rand > lambda
            yTrainExtra(i) = yTrain(availableSpectrograms(idx));
        end
    end
    xTrain = cat(4,xTrain,xTrainExtra);
    yTrain = [yTrain;yTrainExtra];
    % note validation is not augmented
    
    %% Summary
    size(xTrain)
    size(yTrain)
    summary(yTrain)
    
    %% Define and Train CNN
    
    imgSize = [size(xTrain,1),size(xTrain,2),size(xTrain,3)];
    numF = 2^7; % number of filter (neurons); default was 32
    layers = [ ...
        imageInputLayer(imgSize)
        ...
        batchNormalizationLayer
        ...
        convolution2dLayer(3,numF,'Padding','same')
        batchNormalizationLayer
        reluLayer
        convolution2dLayer(3,numF,'Padding','same')
        batchNormalizationLayer
        reluLayer
        ...
        maxPooling2dLayer(3,'Stride',2,'Padding','same')
        ...
        convolution2dLayer(3,numF*2,'Padding','same')
        batchNormalizationLayer
        reluLayer
        convolution2dLayer(3,numF*2,'Padding','same')
        batchNormalizationLayer
        reluLayer
        ...
        maxPooling2dLayer(3,'Stride',2,'Padding','same')
        ...
        convolution2dLayer(3,numF*4,'Padding','same')
        batchNormalizationLayer
        reluLayer
        convolution2dLayer(3,numF*4,'Padding','same')
        batchNormalizationLayer
        reluLayer
        ...
        maxPooling2dLayer(3,'Stride',2,'Padding','same')
        ...
        convolution2dLayer(3,numF*8,'Padding','same')
        batchNormalizationLayer
        reluLayer
        convolution2dLayer(3,numF*8,'Padding','same')
        batchNormalizationLayer
        reluLayer
        ...
        globalAveragePooling2dLayer
        ...
        dropoutLayer(0.5)
        ...
        fullyConnectedLayer(size(countEachLabel(adsTest),1)) % make sure output size equals ads table Label size
        softmaxLayer
        classificationLayer];
    
    %% Training Options
    miniBatchSize = 128; %128 was used in acoustic scene classification
    tuneme = 128; %128 was used in acoustic scene classification
    lr = (0.01)*miniBatchSize/tuneme;
    options = trainingOptions(...
        'sgdm', ...
        'Momentum',0.99, ...
        'L2Regularization',0.005, ...
        ...
        'MiniBatchSize',miniBatchSize, ...
        'MaxEpochs',8*5, ...
        'Shuffle','every-epoch', ...
        ...
        'Plots','training-progress', ...
        'Verbose',false, ...
        ...
        'InitialLearnRate',lr, ...
        'LearnRateSchedule','piecewise', ...
        'LearnRateDropPeriod',1000, ...
        'LearnRateDropFactor',0.02, ...
        ...
        'ValidationData',{xValid,yValid}, ...
        'ValidationFrequency',floor(size(xValid,4)/miniBatchSize));
    
    %% Train the Network
    trainedNet = trainNetwork(xTrain,yTrain,layers,options);
    
    %% Evaluate CNN
    cnnResponsesPerSegment = predict(trainedNet,xTest);
    
    %% Average the responses over each 4-second audio clip.
    classes = trainedNet.Layers(end).Classes;
    numFiles = numel(adsTest.Files);
    
    cnnResponses = zeros(numFiles,numel(classes));
    cnnResponses = cnnResponsesPerSegment;
    
    %% For each 4-second audio clip, choose the maximum of the predictions, then map it to the corresponding predicted location.
    [cnnMaxScore,classIdx] = max(cnnResponses,[],2);
    cnnPredictedLabels = classes(classIdx);
    
    %% Call confusionchart (Deep Learning Toolbox) to visualize the accuracy on the test set.
    figure('Units','normalized','Position',[0.2 0.2 0.5 0.5])
    cm_CNN = confusionchart(adsTest.Labels,cnnPredictedLabels,'title','Test Accuracy - CNN');
    cm_CNN.ColumnSummary = 'column-normalized';
    cm_CNN.RowSummary = 'row-normalized';
    cm_CNN.Normalization = 'row-normalized';
    sortClasses(cm_CNN,'descending-diagonal')
    cm_CNN.Normalization = 'absolute';
    
    fprintf('Average accuracy of CNN = %0.2f\n',mean(adsTest.Labels==cnnPredictedLabels)*100)
    
    %% Feature Extraction for Ensemble Classifier
    sf = waveletScattering('SignalLength',size(data,1), ...
        'SamplingFrequency',fs, ...
        'InvarianceScale',0.75, ...
        'QualityFactors',[8 1]);
    
    scatteringCoeffients = featureMatrix(sf,dataMono,'Transform','log');
    
    featureVector = mean(scatteringCoeffients,2);
    fprintf('Number of wavelet features per 4-second clip = %d\n',numel(featureVector))
    
    scatteringTrain = cellfun(@(x)HelperWaveletFeatureVector(x,sf),train_set_tall,'UniformOutput',false);
    xTrain = gather(scatteringTrain);
    xTrain = cell2mat(xTrain')';
    
    scatteringTest = cellfun(@(x)HelperWaveletFeatureVector(x,sf),test_set_tall,'UniformOutput',false);
    xTest = gather(scatteringTest);
    xTest = cell2mat(xTest')';
    
    subspaceDimension = min(150,size(xTrain,2) - 1);
    numLearningCycles = 30;
    classificationEnsemble = fitcensemble(xTrain,adsTrain.Labels, ...
        'Method','Subspace', ...
        'NumLearningCycles',numLearningCycles, ...
        'Learners','discriminant', ...
        'NPredToSample',subspaceDimension, ...
        'ClassNames',removecats(unique(adsTrain.Labels)));
    
    [waveletPredictedLabels,waveletResponses] = predict(classificationEnsemble,xTest);
    [maxWaveletScores,classIdx] = max(waveletResponses,[],2);
    
    %% Call confusionchart (Deep Learning Toolbox) to visualize the accuracy on the test set.
    figure('Units','normalized','Position',[0.2 0.2 0.5 0.5])
    cm_WE = confusionchart(adsTest.Labels,waveletPredictedLabels,'title','Test Accuracy - Wavelet Scattering');
    cm_WE.ColumnSummary = 'column-normalized';
    cm_WE.RowSummary = 'row-normalized';
    sortClasses(cm_WE,'descending-diagonal')
    cm_WE.Normalization = 'absolute';
    
    fprintf('Average accuracy of classifier = %0.2f\n',mean(adsTest.Labels==waveletPredictedLabels)*100)
    
    %% Apply late fusion
    fused = waveletResponses .* cnnResponses;
    [maxFusedScores,classIdx] = max(fused,[],2);
    fusedpredictedLabels = classes(classIdx);
    
    %% Evaluate late fusion
    figure('Units','normalized','Position',[0.2 0.2 0.5 0.5])
    cm_Fusion = confusionchart(adsTest.Labels,fusedpredictedLabels,'title','Test Accuracy - Fusion');
    cm_Fusion.ColumnSummary = 'column-normalized';
    cm_Fusion.RowSummary = 'row-normalized';
    sortClasses(cm_Fusion,'descending-diagonal')
    cm_Fusion.Normalization = 'absolute';
    fprintf('Average accuracy of fused models = %0.2f\n',mean(adsTest.Labels==fusedpredictedLabels)*100)
    
    %% shut off parallel processing pool
    delete(pp)
    toc
    
    %% save off mat files and clear command window
    save(['Fused_Run_',num2str(SimulationNum),'tosave.mat'],'trainedNet','fusedpredictedLabels','fused','waveletPredictedLabels','waveletResponses','adsTest','adsTrain','adsInfo','cm_CNN','cm_Fusion','cm_WE','cnnResponsesPerSegment','cnnResponses','cnnPredictedLabels','cnnMaxScore','maxWaveletScores','maxFusedScores')
    diary(['CommandHistory',num2str(SimulationNum),'.txt'])
    clc;
    
end