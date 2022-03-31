%% Simulation forloop
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
    
    %% check if recordings are contaminating training and testing data
    sharedRecordingLocations = intersect(metadata_test.SpecificInstrument,metadata_train.SpecificInstrument);
    fprintf('Number of specific recording locations in both train and test sets = %d\n',numel(sharedRecordingLocations))
    
    %% define filepaths for training and testing data
    train_filePaths = fullfile(datasetFolder_train,'audio',metadata_train.FileName);
    test_filePaths = fullfile(datasetFolder_test,'audio',metadata_test.FileName);
    
    %% Create an audio datastore with wav and mp3 files from the samples folder
    adsTrain = audioDatastore(train_filePaths, ...
        'Labels',categorical(metadata_train.InstrumentFamily), ...
        'IncludeSubfolders',true);
    display(countEachLabel(adsTrain))
    
    adsTest = audioDatastore(test_filePaths, ...
        'Labels',categorical(metadata_test.InstrumentFamily), ...
        'IncludeSubfolders',true);
    display(countEachLabel(adsTest))
    
    %% Reduce dataset if needed for testing
    % reduceDataset = false;
    reduceDataset = true;
    if reduceDataset
        adsTrain = splitEachLabel(adsTrain,30,'randomized'); % 33.8 GB (283704 files)
        adsTest = splitEachLabel(adsTest,141,'randomized'); % 500 MB (4096 files)
    end
    
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
    
    %% Buffer the data for feature extraction
    segmentLength = 1; % Four-second .wav file clips divided into one-second segments
    segmentOverlap = 0.5; % arbitrary overlap default is 0.5
    
    [dataBuffered] = buffer(dataMono(:,1),round(segmentLength*fs),round(segmentOverlap*fs),'nodelay');
    
    %% FFT information
    windowLength = 2^11;
    samplesPerHop = 2^10;
    samplesOverlap = round(0.95*windowLength);
    fftLength = 2*windowLength;
    numBands = 2^10; % default was 128
    
    %% Spectrograms with mel scale on y-axis
    spec = melSpectrogram(dataMono,fs, ...
        'Window',hann(windowLength,'periodic'), ...
        'OverlapLength',samplesOverlap, ...
        'FFTLength',fftLength, ...
        'NumBands',numBands);
    
    %% Do not want any -Inf to show up
    spec = log10(spec+eps);
    
    %% Reshape spectragram output for feature extraction
    X = reshape(spec,size(spec,1),size(spec,2),size(data,2),[]);
    
    %% Plot some figures of the melSpectrograms: 6 segments
    plotflag = false;
    % plotflag = true;
    if plotflag
        figure
        melSpectrogram(dataMono,fs, ...
            'Window',hamming(windowLength,'periodic'), ...
            'OverlapLength',samplesOverlap, ...
            'FFTLength',fftLength, ...
            'NumBands',numBands);
        %     title(sprintf('Segment %d',ceil(channel/2)))
        colormap turbo
        
        for channel = 1:size(dataBuffered,2)
            figure
            melSpectrogram(dataBuffered(:,channel),fs, ...
                'Window',hamming(windowLength,'periodic'), ...
                'OverlapLength',samplesOverlap, ...
                'FFTLength',fftLength, ...
                'NumBands',numBands);
            title(sprintf('Segment %d',ceil(channel/2)))
            colormap turbo
        end
    end
    %% Use parallel processing for training
    pp = parpool('IdleTimeout',inf);
    
    train_set_tall = tall(adsTrain);
    xTrain = cellfun(@(x)HelperSegmentedMelSpectrograms_JABmono(x,fs, ...
        'SegmentLength',segmentLength, ...
        'SegmentOverlap',segmentOverlap, ...
        'WindowLength',windowLength, ...
        'HopLength',samplesPerHop, ...
        'NumBands',numBands, ...
        'FFTLength',fftLength), ...
        train_set_tall, ...
        'UniformOutput',false);
    xTrain = gather(xTrain); %@readDataStoreAudio is struggling
    xTrain = cat(4,xTrain{:});
    
    test_set_tall = tall(adsTest);
    xTest = cellfun(@(x)HelperSegmentedMelSpectrograms_JABmono(x,fs, ...
        'SegmentLength',segmentLength, ...
        'SegmentOverlap',segmentOverlap, ...
        'WindowLength',windowLength, ...
        'HopLength',samplesPerHop, ...
        'NumBands',numBands, ...
        'FFTLength',fftLength), ...
        test_set_tall, ...
        'UniformOutput',false);
    xTest = gather(xTest);
    xTest = cat(4,xTest{:});
    
    %% Replicate the labels of the training and test sets so that they are in one-to-one correspondence with the segments.
    % numSegmentsPer10seconds = size(dataBuffered,2)/2;
    numSegmentsPer4seconds = 7;
    yTrain = repmat(adsTrain.Labels,1,numSegmentsPer4seconds)';
    yTrain = yTrain(:);
    
    %% Data Augmentation for CNN
    
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
    
    %% Summary
    size(xTrain)
    size(yTrain)
    summary(yTrain)
    
    %% Define and Train CNN
    
    imgSize = [size(xTrain,1),size(xTrain,2),size(xTrain,3)];
    numF = 32; % number of filter (neurons); default was 32
    layers = [ ...
        % An image input layer inputs 2-D images to a network and applies data normalization.
        imageInputLayer(imgSize)
        
        % A batch normalization layer normalizes a mini-batch of data across all observations for each channel independently. To speed up training of the convolutional neural network and reduce the sensitivity to network initialization, use batch normalization layers between convolutional layers and nonlinearities, such as ReLU layers.
        batchNormalizationLayer
        
        % A 2-D convolutional layer applies sliding convolutional filters to 2-D input.
        convolution2dLayer(3,numF,'Padding','same')
        batchNormalizationLayer
        % A ReLU layer performs a threshold operation to each element of the input, where any value less than zero is set to zero.
        reluLayer
        convolution2dLayer(3,numF,'Padding','same')
        batchNormalizationLayer
        reluLayer
        % A 2-D max pooling layer performs downsampling by dividing the input into rectangular pooling regions, then computing the maximum of each region.
        maxPooling2dLayer(3,'Stride',2,'Padding','same')
        
        convolution2dLayer(3,2*numF,'Padding','same')
        batchNormalizationLayer
        reluLayer
        convolution2dLayer(3,2*numF,'Padding','same')
        batchNormalizationLayer
        reluLayer
        maxPooling2dLayer(3,'Stride',2,'Padding','same')
        
        convolution2dLayer(3,3*numF,'Padding','same')
        batchNormalizationLayer
        reluLayer
        convolution2dLayer(3,3*numF,'Padding','same')
        batchNormalizationLayer
        reluLayer
        maxPooling2dLayer(3,'Stride',2,'Padding','same')
        
        convolution2dLayer(3,6*numF,'Padding','same')
        batchNormalizationLayer
        reluLayer
        convolution2dLayer(3,6*numF,'Padding','same')
        batchNormalizationLayer
        reluLayer
        maxPooling2dLayer(3,'Stride',2,'Padding','same')
        
        convolution2dLayer(3,8*numF,'Padding','same')
        batchNormalizationLayer
        reluLayer
        convolution2dLayer(3,8*numF,'Padding','same')
        batchNormalizationLayer
        reluLayer
        
        % A 2-D global average pooling layer performs downsampling by computing the mean of the height and width dimensions of the input.
        globalAveragePooling2dLayer
        
        % A dropout layer randomly sets input elements to zero with a given probability.
        dropoutLayer(0.5)
        
        % A fully connected layer multiplies the input by a weight matrix and then adds a bias vector.
        fullyConnectedLayer(size(countEachLabel(adsTest),1)) % make sure output size equals ads table Label size
        
        % A softmax layer applies a softmax function to the input.
        softmaxLayer
        
        % A classification layer computes the cross-entropy loss for classification and weighted classification tasks with mutually exclusive classes.
        classificationLayer];
    
    %% Training Options
    miniBatchSize = 2^7; %128 was used in acoustic scene classification
    tuneme = 2^7;
    lr = (1e-2)*miniBatchSize/tuneme;
    options = trainingOptions('sgdm', ...
        'InitialLearnRate',lr, ...
        'MiniBatchSize',miniBatchSize, ...
        'Momentum',0.99, ...
        'L2Regularization',0.005, ...
        'MaxEpochs',100, ...
        'Shuffle','every-epoch', ...
        'Plots','training-progress', ...
        'Verbose',false, ...
        'LearnRateSchedule','piecewise', ...
        'LearnRateDropPeriod',1000, ...
        'LearnRateDropFactor',(1e-3));
    
    %% Train the Network
    trainedNet = trainNetwork(xTrain,yTrain,layers,options);
    
    %% Evaluate CNN
    cnnResponsesPerSegment = predict(trainedNet,xTest);
    
    %% Average the responses over each 4-second audio clip.
    classes = trainedNet.Layers(end).Classes;
    numFiles = numel(adsTest.Files);
    
    counter = 1;
    cnnResponses = zeros(numFiles,numel(classes));
    for channel = 1:numFiles
        cnnResponses(channel,:) = sum(cnnResponsesPerSegment(counter:counter+numSegmentsPer4seconds-1,:),1)/numSegmentsPer4seconds;
        counter = counter + numSegmentsPer4seconds;
    end
    
    %% For each 4-second audio clip, choose the maximum of the predictions, then map it to the corresponding predicted location.
    [~,classIdx] = max(cnnResponses,[],2);
    cnnPredictedLabels = classes(classIdx);
    
    %% Call confusionchart (Deep Learning Toolbox) to visualize the accuracy on the test set.
    figure('Units','normalized','Position',[0.2 0.2 0.5 0.5])
    cm = confusionchart(adsTest.Labels,cnnPredictedLabels,'title','Test Accuracy - CNN');
    cm.ColumnSummary = 'column-normalized';
    cm.RowSummary = 'row-normalized';
    cm.Normalization = 'row-normalized';
    sortClasses(cm,'descending-diagonal')
    cm.Normalization = 'absolute';
    
    fprintf('Average accuracy of CNN = %0.2f\n',mean(adsTest.Labels==cnnPredictedLabels)*100)
    
    % %% Feature Extraction for Ensemble Classifier
    % sf = waveletScattering('SignalLength',size(data,1), ...
    %                        'SamplingFrequency',fs, ...
    %                        'InvarianceScale',0.75, ...
    %                        'QualityFactors',[8 1]);
    %
    % %%
    % scatteringCoeffients = featureMatrix(sf,dataMono,'Transform','log');
    %
    % %%
    % featureVector = mean(scatteringCoeffients,2);
    % fprintf('Number of wavelet features per 4-second clip = %d\n',numel(featureVector))
    %
    % %%
    % scatteringTrain = cellfun(@(x)HelperWaveletFeatureVector(x,sf),train_set_tall,'UniformOutput',false);
    % xTrain = gather(scatteringTrain);
    % xTrain = cell2mat(xTrain')';
    %
    % scatteringTest = cellfun(@(x)HelperWaveletFeatureVector(x,sf),test_set_tall,'UniformOutput',false);
    % xTest = gather(scatteringTest);
    % xTest = cell2mat(xTest')';
    %
    % %%
    % subspaceDimension = min(150,size(xTrain,2) - 1);
    % numLearningCycles = 30;
    % classificationEnsemble = fitcensemble(xTrain,adsTrain.Labels, ...
    %     'Method','Subspace', ...
    %     'NumLearningCycles',numLearningCycles, ...
    %     'Learners','discriminant', ...
    %     'NPredToSample',subspaceDimension, ...
    %     'ClassNames',removecats(unique(adsTrain.Labels)));
    %
    % %%
    % [waveletPredictedLabels,waveletResponses] = predict(classificationEnsemble,xTest);
    %
    % figure('Units','normalized','Position',[0.2 0.2 0.5 0.5])
    % cm = confusionchart(adsTest.Labels,waveletPredictedLabels,'title','Test Accuracy - Wavelet Scattering');
    % cm.ColumnSummary = 'column-normalized';
    % cm.RowSummary = 'row-normalized';
    % % cm.Normalization = 'row-normalized';
    % sortClasses(cm,'descending-diagonal')
    % cm.Normalization = 'absolute';
    %
    % fprintf('Average accuracy of classifier = %0.2f\n',mean(adsTest.Labels==waveletPredictedLabels)*100)
    %
    % %% Apply late fusion
    % fused = waveletResponses .* cnnResponses;
    % [~,classIdx] = max(fused,[],2);
    %
    % predictedLabels = classes(classIdx);
    %
    % %% Evaluate late fusion
    % figure('Units','normalized','Position',[0.2 0.2 0.5 0.5])
    % cm = confusionchart(adsTest.Labels,predictedLabels,'title','Test Accuracy - Fusion');
    % cm.ColumnSummary = 'column-normalized';
    % cm.RowSummary = 'row-normalized';
    % % cm.Normalization = 'row-normalized';
    % sortClasses(cm,'descending-diagonal')
    % cm.Normalization = 'absolute';
    % fprintf('Average accuracy of fused models = %0.2f\n',mean(adsTest.Labels==predictedLabels)*100)
    
    %% shut off parallel processing pool
    delete(pp)
    toc
    
    %% save off mat files and clear command window
    
    save(['CNN_Run_',num2str(SimulationNum),'tosave.mat'],'adsTest','adsTrain','adsInfo','cm','cnnResponses','cnnPredictedLabels')
    diary(['CommandHistory',num2str(SimulationNum),'.txt'])
    
    clc; close all;
end
