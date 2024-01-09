% Exercise E7: Classification of MNIST digits using SVM with gaussian kernelfunction and varying beta.

clear;
close all;
load('A2_data.mat');

temp_val = 0.2:0.2:4;
beta_val = temp_val';

% Training set:
train_X = train_data_01;
labels_train_X = train_labels_01;
train_T = labels_train_X;

% Test set
test_X = test_data_01;
labels_test_X = test_labels_01;
test_T = labels_test_X;
missclassification_rate_test = zeros(length(beta_val),1);

% Running svm for different beta values of the gaussian kernel function
for iter = 1:length(beta_val)
    sprintf('Trying \x03b2 = %.2f',beta_val(iter))
    % Training 
    train_model = fitcsvm(train_X', train_T, 'KernelFunction', 'gaussian', 'KernelScale', beta_val(iter));
    [pred_class_labels_train, pred_class_train_scores] = predict(train_model,train_X');

    % Computation of the matrix (also called confusion matrix) elements' in the assignment description (Training)

    true_zeros = sum(train_T(pred_class_labels_train == 0) == 0);
    false_zeros = sum(train_T(pred_class_labels_train == 0) == 1);
    true_ones = sum(train_T(pred_class_labels_train == 1) == 1);
    false_ones = sum(train_T(pred_class_labels_train == 1) == 0);

    matrix = [true_zeros false_zeros ; false_ones true_ones];

    %Test
    test_model = fitcsvm(test_X', test_T, 'KernelFunction', 'gaussian', 'KernelScale', beta_val(iter));
    [pred_class_labels_test, pred_class_test_scores] = predict(test_model,test_X');
    
    
    % Computation of the matrix (also called confusion matrix) elements' in the assignment description (Test)
    true_zeros_test = sum(test_T(pred_class_labels_test == 0) == 0);
    false_zeros_test = sum(test_T(pred_class_labels_test == 0) == 1);
    true_ones_test = sum(test_T(pred_class_labels_test == 1) == 1);
    false_ones_test = sum(test_T(pred_class_labels_test == 1) == 0);
    
    matrix_test = [true_zeros_test false_zeros_test ; false_ones_test true_ones_test];
    missclassification_rate_test(iter) = (matrix_test(1,2)+matrix_test(2,1))*100/length(test_X);

end

figure
hold on

plot(beta_val,missclassification_rate_test(:,1),'-x','color','m','MarkerSize',10)
set(gca,'FontSize',24)
xlabel('\beta-values')
ylabel('Missclassification rate of test data')
legend('Test missclassification rate')
title('Task E7: Test missclassification rate vs \beta-values')
set(gcf,'Position',[10 1000 1500 1000])
saveas(gcf,'taskE7_missclassification_rate_vs_beta_values.png')

