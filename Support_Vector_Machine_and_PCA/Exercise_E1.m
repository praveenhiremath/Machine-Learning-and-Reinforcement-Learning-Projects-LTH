% Exercise E1: Visualize data in d = 2
close all
clear all
load('A2_data.mat')  % MNIST images data

X_data = train_data_01;
X_data_project = linear_pca(train_data_01);

% Plotting d=2 dim reduced training data
figure;
hold on;

plot(X_data_project(1,train_labels_01==0), X_data_project(2,train_labels_01==0), 's', 'MarkerSize', 10,'color', 'r') % train data of category/label = 0
plot(X_data_project(1,train_labels_01==1), X_data_project(2,train_labels_01==1), 'o', 'MarkerSize', 10,'color', 'b') % train data of category/label = 1

set(gca, 'FontSize', 24)
set(gcf,'Position',[10 1000 1500 1000])
xlabel('Principal component #1')
ylabel('Principal Component #2')
legend('Category #1', 'Category #2')
title('\fontsize{24} Principal component analysis of the data (d=2)')
saveas(gcf,'PCA_taskE1.png')
