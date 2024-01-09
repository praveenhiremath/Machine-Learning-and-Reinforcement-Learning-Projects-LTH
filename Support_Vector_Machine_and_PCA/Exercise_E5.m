% Exercise E5: K-means clustering for classification: Evaluate missclassification rate for different K-values

clear;
close all;
load('A2_data.mat');

K = [2 3 4 5 6 7 8 9 10];

[~,len] = size(K);

train_missclass_rates = zeros(len,1);
test_missclass_rates = zeros(len,1);

size(train_missclass_rates)
size(test_missclass_rates)

for iter = 1:len

% Classification of training data
    train_set_X = train_data_01;
    labels_train_set_X = train_labels_01; % (True labels in train_data_01)

    [train_set_ys,~] = K_means_clustering(train_set_X, K(iter)); 
    train_set_Kmeans_clust_labels = K_means_classifier(labels_train_set_X,K(iter),train_set_ys);  

% Extracting data for table in assignment description
    K_clust_0_labels = zeros(K(iter),1);
    K_clust_1_labels = zeros(K(iter),1);
    miss_class_1_in_K = zeros(K(iter),1);
    miss_class_0_in_K = zeros(K(iter),1);

    for k = 1:K(iter)
            K_clust_0_labels(k) = sum(labels_train_set_X(train_set_ys == k) == 0); 
            K_clust_1_labels(k) = sum(labels_train_set_X(train_set_ys == k) == 1); 
            if (train_set_Kmeans_clust_labels(k)==0)
                miss_class_1_in_K(k) = K_clust_1_labels(k);
            else
                miss_class_0_in_K(k) = K_clust_0_labels(k);       
            end
    end

    total_miss_class_0_label = sum(miss_class_0_in_K);            
    total_miss_class_1_label = sum(miss_class_1_in_K);            

    total_true_0s = sum(labels_train_set_X==0);
    total_true_1s = sum(labels_train_set_X==1);

    total_miss_class_label = (total_miss_class_0_label + total_miss_class_1_label);
    train_missclass_rates(iter) = (100*total_miss_class_label)/(total_true_0s + total_true_1s);
    sprintf("Train set: For K = %d, misclassification rate = %.5f", K(iter), train_missclass_rates(iter))
    

% Classification of test data
    test_set_X = test_data_01;
    labels_test_set_X = test_labels_01;  % (True labels in test_data_01)

    [test_set_ys,~] = K_means_clustering(test_set_X, K(iter)); 
    test_set_Kmeans_clust_labels = K_means_classifier(labels_test_set_X,K(iter),test_set_ys);


% Extracting data for table in assignment description
    test_K_clust_0_labels = zeros(K(iter),1);
    test_K_clust_1_labels = zeros(K(iter),1);
    test_miss_class_1_in_K = zeros(K(iter),1);
    test_miss_class_0_in_K = zeros(K(iter),1);

    for tk = 1:K(iter)
            test_K_clust_0_labels(tk) = sum(labels_test_set_X(test_set_ys == tk) == 0); % total number of "0" labels in cluster k
            test_K_clust_1_labels(tk) = sum(labels_test_set_X(test_set_ys == tk) == 1); % total number of "1" labels in cluster k
            if (test_set_Kmeans_clust_labels(tk)==0)
                test_miss_class_1_in_K(tk) = test_K_clust_1_labels(tk);
            else
                test_miss_class_0_in_K(tk) = test_K_clust_0_labels(tk);            
            end
    end

    test_total_miss_class_0_label = sum(test_miss_class_0_in_K);            
    test_total_miss_class_1_label = sum(test_miss_class_1_in_K);            

    test_total_true_0s = sum(labels_test_set_X==0);
    test_total_true_1s = sum(labels_test_set_X==1);

    test_total_miss_class_label = (test_total_miss_class_0_label + test_total_miss_class_1_label);
    test_missclass_rates(iter) = (100*test_total_miss_class_label)/(test_total_true_0s + test_total_true_1s);
    sprintf("Test set: For K = %d, misclassification rate = %.5f", K(iter), test_missclass_rates(iter))
end

figure
hold on

plot(K,train_missclass_rates(:,1),'-x','color','m','MarkerSize',10)
plot(K,test_missclass_rates(:,1),'-o','color','b','MarkerSize',5)
set(gca,'FontSize',24)
xlabel('K-values')
ylabel('Missclassification rate')
legend('Training missclassification rate','Test missclassification rate')
title('\fontsize{24} Task E5: Missclassification rate vs K-values')
set(gcf,'Position',[10 1000 1500 1000])
saveas(gcf,'taskE5_missclassification_rate_vs_K_values.png')




