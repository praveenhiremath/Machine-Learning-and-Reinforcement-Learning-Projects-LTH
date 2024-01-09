% Exercise 6: Classification of handwritten digits
%%
clear;
close all;
addpath('./'); 
load('./models/network_trained_with_momentum.mat')

weights_conv_layer1 = net.layers{1,2}.params.weights;
biases_conv_layer1 = net.layers{1,2}.params.biases;
kernels = 16;

% Plotting the kernels learned by first convolution layer
figure; 
hold on;
%tcl = tiledlayout(4,4);
pos = 0;
for i = 1:kernels
    pos = pos+1;
    subplot(4,4,pos)
    imshow(weights_conv_layer1(:,:,:,i));
end     

sgtitle('Kernels learned by first convolution layer','FontSize',30)
set(gcf,'Position',[10 1000 1500 1000])
saveas(gcf,'Ass3_mnist_conv_layer1_kernels.png');        
%hold off

%% Plotting missclassified images

clear;
close all;
addpath(genpath('./'))
load('./models/network_trained_with_momentum.mat');

testset_imgs = loadMNISTImages('./data/mnist/t10k-images.idx3-ubyte');
testset_labels = loadMNISTLabels('./data/mnist/t10k-labels.idx1-ubyte');


num_testdata = length(testset_labels);

testset_labels(testset_labels == 0) = 10;
testset_imgs = reshape(testset_imgs, [net.layers{1,1}.params.size(1,1), net.layers{1,1}.params.size(1,2), net.layers{1,1}.params.size(1,3), num_testdata]);
%net

test_batch = 16;
test_pred = zeros(num_testdata,1);

batch_start_ids = 1:test_batch:num_testdata;
for i = 1:length(batch_start_ids)
    batch_ids = batch_start_ids(i):(batch_start_ids(i)+test_batch-1);
    [outs,~] = evaluate(net,testset_imgs(:,:,:,batch_ids),testset_labels(batch_ids));    % Evaluate the 'net' for predicting test dataset
    out_probs = outs{end-1};  %pre-final layer outputs
    [max_vals,maxprob_ids] = max(out_probs,[],1);
    test_pred(batch_ids) = maxprob_ids;
end
% Converting the labels '10' back to '0' 
sprintf('Done loop\n')
test_pred(test_pred == 10) = 0;
testset_labels(testset_labels == 10) = 0;
size(test_pred)
size(testset_labels)

wrong_preds = test_pred(test_pred ~= testset_labels);

figure;
hold on;
pos = 0;
wrong_pred_ids = find(test_pred ~= testset_labels);

for i = 1:20
    pos = pos+1;
    subplot(5,4,pos)
    imshow(testset_imgs(:,:,1,wrong_pred_ids(i)))
    title(['Pred class: ',num2str(test_pred(wrong_pred_ids(i))), ', True class: ',num2str(testset_labels(wrong_pred_ids(i)))]);
end

sgtitle('Missclassified images','FontSize',30);
set(gcf,'Position',[10 1000 1500 1000]);
saveas(gcf,sprintf('Ass3_mnist_E6_Missclassified_images.png'))
hold off
    
%% Confusion matrix and other metrics
confusion_matrix = confusionmat(testset_labels',test_pred);
figure(2);
confusionchart(testset_labels',test_pred,'FontSize',30,'title',{"Confusion matrix chart using Matlab's", "confusionmat() and confusionchart()"})
%cm.set(gca,'FontSize',30)
%cm.title("Confusion matrix chart using Matlab's \n confusionmat() and confusionchart()")  
set(gcf,'Position',[10 1000 1500 1000]);
saveas(gcf,sprintf('Ass3_mnist_E6_confusionchart.png'))

%% Metrics: Precision and recall
ind_class_precision = zeros(10,1);
ind_class_recall = zeros(10,1);
CorrectPreds = diag(confusion_matrix);
for C = 1:10
    tot_num_class_preds = sum(confusion_matrix(:,C));
    ind_class_precision(C,1) = (CorrectPreds(C)/tot_num_class_preds)    % Precision for individual digits
    tot_num_preds_notpreds_class = sum(confusion_matrix(C,:));
    ind_class_recall(C,1) = (CorrectPreds(C)/tot_num_preds_notpreds_class)  %recall for individual digits
    avg_precision = mean(ind_class_precision)
    avg_recall = mean(ind_class_recall)
end       

%% Number of parameters in all layers
net_length = numel(net.layers);
Ws = zeros(net_length,1);   % weights
bs = zeros(net_length,1);   % biases
i = 1;
count = 0;
for l = 1:net_length
    if (or(strcmp(net.layers{1,l}.type,'convolution'),strcmp(net.layers{1,l}.type,'fully_connected')))
        Ws(l,1) = numel(net.layers{1,l}.params.weights);
        bs(l,1) = numel(net.layers{1,l}.params.biases);
        sprintf("%2d-th layer has %5d trainable W-parameters",l,Ws(l,1))
        sprintf("%2d-th layer has %5d trainable b-parameters",l,bs(l,1))
        sprintf("%2d-th layer has %5d trainable parameters",l,Ws(l,1)+bs(l,1))
   
    end
end

tot_num_Ws = sum(Ws(:,1))
tot_num_bs = sum(bs(:,1))
tot_params = tot_num_Ws+tot_num_bs


