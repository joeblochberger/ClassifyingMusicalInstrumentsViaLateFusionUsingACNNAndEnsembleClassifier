clear all; close all; clc;
load('Fused_Run_2tosave.mat')

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
% sortClasses(cm_CNN,'descending-diagonal')
cm_CNN.Normalization = 'absolute';
fprintf('Average accuracy of CNN = %0.2f\n',mean(adsTest.Labels==cnnPredictedLabels)*100)
quickpretty
%
c_matrix = cm_CNN.NormalizedValues
n_class=numel(classes);
TP_all=zeros(1,n_class);
FN_all=zeros(1,n_class);
FP_all=zeros(1,n_class);
TN_all=zeros(1,n_class);
for i=1:n_class
    TP_all(i)=c_matrix(i,i);
    FN_all(i)=sum(c_matrix(i,:))-c_matrix(i,i);
    FP_all(i)=sum(c_matrix(:,i))-c_matrix(i,i);
    TN_all(i)=sum(c_matrix(:))-TP_all(i)-FP_all(i)-FN_all(i);
end

TP = sum(TP_all);
FP = sum(FP_all);
FN = sum(FN_all);
TN = sum(TN_all);

% All metrics in percentages; formulae from Amidi and Amidi cheatsheet
Accuracy = (TP+TN)./(TP+TN+FP+FN)*100
Precision = (TP)./(TP+FP)*100
Recall = (TP)./(TP+FN)*100
Specificity = (TN)./(TN+FP)*100
F1_score = (2*TP)./(2*TP+FP+FN)*100
TPR = (TP)./(TP+FN)*100
FPR = (FP)./(TN+FP)*100

% 2x2 matrix
figure
bic_matrix = [TP FN; FP TN]
labels = {'+','-'};
confusionchart(bic_matrix,labels)
quickpretty
%% Call confusionchart (Deep Learning Toolbox) to visualize the accuracy on the test set.
figure('Units','normalized','Position',[0.2 0.2 0.5 0.5])
cm_WE = confusionchart(adsTest.Labels,waveletPredictedLabels,'title','Test Accuracy - Wavelet Scattering');
cm_WE.ColumnSummary = 'column-normalized';
cm_WE.RowSummary = 'row-normalized';
% sortClasses(cm_WE,'descending-diagonal')
cm_WE.Normalization = 'absolute';
cm_WE.NormalizedValues
fprintf('Average accuracy of classifier = %0.2f\n',mean(adsTest.Labels==waveletPredictedLabels)*100)
quickpretty
c_matrix = cm_WE.NormalizedValues
n_class=numel(classes);
TP=zeros(1,n_class);
FN=zeros(1,n_class);
FP=zeros(1,n_class);
TN=zeros(1,n_class);
for i=1:n_class
    TP(i)=c_matrix(i,i);
    FN(i)=sum(c_matrix(i,:))-c_matrix(i,i);
    FP(i)=sum(c_matrix(:,i))-c_matrix(i,i);
    TN(i)=sum(c_matrix(:))-TP(i)-FP(i)-FN(i);
end

% All metrics in percentages; formulae from Amidi and Amidi cheatsheet
Accuracy = (TP+TN)/(TP+TN+FP+FN)*100
Precision = (TP)/(TP+FP)*100
Recall = (TP)/(TP+FN)*100
Specificity = (TN)/(TN+FP)*100
F1_score = (2*TP)/(2*TP+FP+FN)*100
TPR = (TP)/(TP+FN)*100
FPR = (FP)/(TN+FP)*100

% 2x2 matrix
figure
bic_matrix = [sum(TP) sum(FN); sum(FP) sum(TN)]
labels = {'+','-'};
confusionchart(bic_matrix,labels)
quickpretty
%% Apply late fusion
fused = waveletResponses .* cnnResponses;
[maxFusedScores,classIdx] = max(fused,[],2);
fusedpredictedLabels = classes(classIdx);

%% Evaluate late fusion
figure('Units','normalized','Position',[0.2 0.2 0.5 0.5])
cm_Fusion = confusionchart(adsTest.Labels,fusedpredictedLabels,'title','Test Accuracy - Fusion');
cm_Fusion.ColumnSummary = 'column-normalized';
cm_Fusion.RowSummary = 'row-normalized';
% sortClasses(cm_Fusion,'descending-diagonal')
cm_Fusion.Normalization = 'absolute';
cm_Fusion.NormalizedValues
fprintf('Average accuracy of fused models = %0.2f\n',mean(adsTest.Labels==fusedpredictedLabels)*100)
quickpretty
c_matrix = cm_Fusion.NormalizedValues
n_class=numel(classes);
TP=zeros(1,n_class);
FN=zeros(1,n_class);
FP=zeros(1,n_class);
TN=zeros(1,n_class);
for i=1:n_class
    TP(i)=c_matrix(i,i);
    FN(i)=sum(c_matrix(i,:))-c_matrix(i,i);
    FP(i)=sum(c_matrix(:,i))-c_matrix(i,i);
    TN(i)=sum(c_matrix(:))-TP(i)-FP(i)-FN(i);
end

% All metrics in percentages; formulae from Amidi and Amidi cheatsheet
Accuracy = (TP+TN)/(TP+TN+FP+FN)*100
Precision = (TP)/(TP+FP)*100
Recall = (TP)/(TP+FN)*100
Specificity = (TN)/(TN+FP)*100
F1_score = (2*TP)/(2*TP+FP+FN)*100
TPR = (TP)/(TP+FN)*100
FPR = (FP)/(TN+FP)*100

% 2x2 matrix
figure
bic_matrix = [sum(TP) sum(FN); sum(FP) sum(TN)]
labels = {'+','-'};
confusionchart(bic_matrix,labels)
quickpretty