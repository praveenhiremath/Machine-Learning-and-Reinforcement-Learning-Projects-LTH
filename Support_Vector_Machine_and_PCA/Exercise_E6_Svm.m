% Exercise E6: Classification of MNIST digits using SVM

clear;
close all;
load('A2_data.mat');

% Training set:
train_X = train_data_01;
labels_train_X = train_labels_01;
train_T = labels_train_X;

%train_model = fitcsvm(train_X', train_T, 'KernelFunction', 'gaussian', 'KernelScale', 1);
train_model = fitcsvm(train_X', train_T, 'KernelFunction', 'linear','KernelScale', 1);
[pred_class_labels_train, pred_class_train_scores] = predict(train_model,train_X');


% Computation of the matrix (also called confusion matrix) elements' in the assignment description 

%num_classes = length(unique(train_T));   % number of classes in train_T
%matrix = zeros(num_classes,num_classes);

true_zeros = sum(train_T(pred_class_labels_train == 0) == 0);
false_zeros = sum(train_T(pred_class_labels_train == 0) == 1);
true_ones = sum(train_T(pred_class_labels_train == 1) == 1);
false_ones = sum(train_T(pred_class_labels_train == 1) == 0);

matrix = [true_zeros false_zeros ; false_ones true_ones]

% Test set
test_X = test_data_01;
labels_test_X = test_labels_01;
test_T = labels_test_X;

test_model = fitcsvm(test_X', test_T, 'KernelFunction', 'linear','KernelScale', 1);
[pred_class_labels_test, pred_class_test_scores] = predict(test_model,test_X');


% Computation of the matrix (also called confusion matrix) elements' in the assignment description 
true_zeros_test = sum(test_T(pred_class_labels_test == 0) == 0);
false_zeros_test = sum(test_T(pred_class_labels_test == 0) == 1);
true_ones_test = sum(test_T(pred_class_labels_test == 1) == 1);
false_ones_test = sum(test_T(pred_class_labels_test == 1) == 0);

matrix_test = [true_zeros_test false_zeros_test ; false_ones_test true_ones_test]
