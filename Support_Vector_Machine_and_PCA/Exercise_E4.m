% Exercise E4: K-means clustering for classification

clear;
close all;
load('A2_data.mat');

K = 2;
% Classification of training data

train_set_X = train_data_01;
labels_train_set_X = train_labels_01; % (True labels in train_data_01)

[train_set_ys,~] = K_means_clustering(train_set_X, K); 
train_set_Kmeans_clust_labels = K_means_classifier(labels_train_set_X,K,train_set_ys)  

% Extracting data for table in assignment description
K_clust_0_labels = zeros(K,1);
K_clust_1_labels = zeros(K,1);
miss_class_1_in_K = zeros(K,1);
miss_class_0_in_K = zeros(K,1);

for k = 1:K
        K_clust_0_labels(k) = sum(labels_train_set_X(train_set_ys == k) == 0); 
        K_clust_1_labels(k) = sum(labels_train_set_X(train_set_ys == k) == 1); 
        if (train_set_Kmeans_clust_labels(k)==0)
            miss_class_1_in_K(k) = K_clust_1_labels(k);
            sprintf("Train set: Number of 0 labels in cluster %d is %5d", k, K_clust_0_labels(k))
            sprintf("Train set: Number of label 1 missclassified into cluster %d is %5d", k, K_clust_1_labels(k))
        else
            miss_class_0_in_K(k) = K_clust_0_labels(k);       
            sprintf("Train set: Number of 1 labels in cluster %d is %5d", k, K_clust_1_labels(k))
            sprintf("Train set: Number of label 0 missclassified into cluster %d is %5d", k, K_clust_0_labels(k))
        end
end

total_miss_class_0_label = sum(miss_class_0_in_K);            
total_miss_class_1_label = sum(miss_class_1_in_K);            

total_true_0s = sum(labels_train_set_X==0);
total_true_1s = sum(labels_train_set_X==1);

total_miss_class_label = (total_miss_class_0_label + total_miss_class_1_label);
missclassification_rate = (100*total_miss_class_label)/(total_true_0s + total_true_1s);
sprintf("Train set: Total misclassification = %.5f",total_miss_class_label)
sprintf("Train set: misclassification rate = %.5f",missclassification_rate)


% Classification of test data

test_set_X = test_data_01;
labels_test_set_X = test_labels_01;  % (True labels in test_data_01)

[test_set_ys,~] = K_means_clustering(test_set_X, K); 
test_set_Kmeans_clust_labels = K_means_classifier(labels_test_set_X,K,test_set_ys)


% Extracting data for table in assignment description
test_K_clust_0_labels = zeros(K,1);
test_K_clust_1_labels = zeros(K,1);
test_miss_class_1_in_K = zeros(K,1);
test_miss_class_0_in_K = zeros(K,1);

for tk = 1:K
        test_K_clust_0_labels(tk) = sum(labels_test_set_X(test_set_ys == tk) == 0); 
        test_K_clust_1_labels(tk) = sum(labels_test_set_X(test_set_ys == tk) == 1); 
        if (test_set_Kmeans_clust_labels(tk)==0)
            test_miss_class_1_in_K(tk) = test_K_clust_1_labels(tk);
            sprintf("Test set: Number of 0 labels in cluster %d is %5d", tk, test_K_clust_0_labels(tk))
            sprintf("Test set: Number of label 1 missclassified into cluster %d is %5d", tk, test_K_clust_1_labels(tk))
        else
            test_miss_class_0_in_K(tk) = test_K_clust_0_labels(tk);            
            sprintf("Test set: Number of 1 labels in cluster %d is %5d", tk, test_K_clust_1_labels(tk))
            sprintf("Test set: Number of label 0 missclassified into cluster %d is %5d", tk, test_K_clust_0_labels(tk))
        end
end

test_total_miss_class_0_label = sum(test_miss_class_0_in_K);            
test_total_miss_class_1_label = sum(test_miss_class_1_in_K);            

test_total_true_0s = sum(labels_test_set_X==0);
test_total_true_1s = sum(labels_test_set_X==1);

test_total_miss_class_label = (test_total_miss_class_0_label + test_total_miss_class_1_label);
test_missclassification_rate = (100*test_total_miss_class_label)/(test_total_true_0s + test_total_true_1s);
sprintf("Test set: Total misclassification = %.5f",test_total_miss_class_label)
sprintf("Test set: misclassification rate = %.5f",test_missclassification_rate)

