% Exercise 7: CIFAR Dataset (baseline)
%%
clear;
close all;
addpath('./'); 
load('./models/cifar10_baseline.mat')

weights_conv_layer1 = rescale(net.layers{1,2}.params.weights); % rescaling weights to fall in the range [0,1]
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
saveas(gcf,'Ass3_cifar_conv_layer1_kernels.png');        
%hold off

%% Plotting missclassified images

clear;
close all;
addpath(genpath('./'))
load('./models/cifar10_baseline.mat');

[trainset_imgs, trainset_labels, testset_imgs, testset_labels, classes] = load_cifar10(2);  
mean_of_test_imgs = mean(mean(mean(testset_imgs,1),2),4);   % taking mean of RGB channels
testset_imgs_mean = bsxfun(@minus, testset_imgs, mean_of_test_imgs);   % testset_imgs-mean_of_test_imgs


num_testdata = length(testset_labels);

%testset_labels(testset_labels == 0) = 10;
%testset_imgs = reshape(testset_imgs, [net.layers{1,1}.params.size(1,1), net.layers{1,1}.params.size(1,2), net.layers{1,1}.params.size(1,3), num_testdata]);
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
sprintf('Done evaluation loop\n')
%test_pred(test_pred == 10) = 0;
%testset_labels(testset_labels == 10) = 0;
size(test_pred)
size(testset_labels)

wrong_preds = test_pred(test_pred ~= testset_labels);
%%
figure;
hold on;
pos = 0;
wrong_pred_ids = find(test_pred ~= testset_labels);

for i = 1:20
    pos = pos+1;
    subplot(5,4,pos)
    imshow(testset_imgs(:,:,:,wrong_pred_ids(i))/255)
    %title(['Pred class: ',num2str(test_pred(wrong_pred_ids(i))), ', True class: ',num2str(testset_labels(wrong_pred_ids(i)))]);
    xlabel(strcat('Pred class: ',classes(test_pred(wrong_pred_ids(i))), ', True class: ', classes(testset_labels(wrong_pred_ids(i)))));
end

sgtitle('Missclassified images','FontSize',30);
set(gcf,'Position',[10 1000 1500 1000]);
saveas(gcf,sprintf('Ass3_cifar_E7_Missclassified_images.png'))
hold off
    
%% Correct predictions

%correct_preds = setdiff(test_pred, test_pred(test_pred ~=testset_labels));
correct_pred_ids = find(test_pred == testset_labels);
classes(test_pred(correct_pred_ids(i)))

pos2= 0;
for i = 1:20
    pos2 = pos2+1;
    subplot(5,4,pos2)
    imshow(testset_imgs(:,:,:,correct_pred_ids(i))/255)
    %title(['Pred class: ',num2str(test_pred(wrong_pred_ids(i))), ', True class: ',num2str(testset_labels(wrong_pred_ids(i)))]);
%    title(['PC: ',classes(test_pred(correct_pred_ids(i))), ', TC: ', classes(testset_labels(correct_pred_ids(i)))]);
    xlabel(strcat('PC: ',classes(test_pred(correct_pred_ids(i))), ', TC: ', classes(testset_labels(correct_pred_ids(i)))));
end

sgtitle('Correctly classified images','FontSize',30);
set(gcf,'Position',[10 1000 1500 1000]);
saveas(gcf,sprintf('Ass3_cifar_E7_correctly_classified_images.png'))
hold off

%% Confusion matrix and other metrics

%nums_testset_labels = ascii2str(testset_labels);
confusion_matrix = confusionmat(double(testset_labels'),test_pred);
figure(2);
confusionchart(classes(testset_labels'),classes(test_pred),'FontSize',30,'title',{"Confusion matrix chart using Matlab's", "confusionmat() and confusionchart()"})
%cm.set(gca,'FontSize',30)
%cm.title("Confusion matrix chart using Matlab's \n confusionmat() and confusionchart()")  
set(gcf,'Position',[10 1000 1500 1000]);
saveas(gcf,sprintf('Ass3_cifar_E7_confusionchart.png'))

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


