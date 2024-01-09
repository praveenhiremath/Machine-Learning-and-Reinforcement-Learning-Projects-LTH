% Exercise E3: Display K=2 and K=5 cluster centroids as images
close all;
clear all;
load('A2_data.mat');  % MNIST images data
load('Clusters_from_E2_K2.mat');  % results of task E2: K-means clustering with K = 2

% Centroids of clusters for K=2
K2_ys = ys;
K2_Cs = Cs;
reshaped_K2_C1 = reshape(K2_Cs(:,1), [28 28]);
reshaped_K2_C2 = reshape(K2_Cs(:,2), [28 28]);


load('Clusters_from_E2_K5.mat');  % results of task E2: K-means clustering with K = 5
% Centroids of clusters for K=5
K5_ys = ys;
K5_Cs = Cs;
reshaped_K5_C1 = reshape(K5_Cs(:,1), [28 28]);
reshaped_K5_C2 = reshape(K5_Cs(:,2), [28 28]);
reshaped_K5_C3 = reshape(K5_Cs(:,3), [28 28]);
reshaped_K5_C4 = reshape(K5_Cs(:,4), [28 28]);
reshaped_K5_C5 = reshape(K5_Cs(:,5), [28 28]);

% Plotting centroids as images
% K = 2 images
figure
hold on

subplot(2,5,[1,2]);
imshow(reshaped_K2_C1);

title("Cluster #1",'FontSize',30);
subplot(2,5,[4,5]);
imshow(reshaped_K2_C2);
text(-30.0,-3.5,{'K = 2 Clusters'},'FontSize',40);
title("Cluster #2",'FontSize',30);

% K = 5 images
subplot(2,5,[6,6]);
imshow(reshaped_K5_C1);
title("Cluster #1",'FontSize',30);
subplot(2,5,[7,7]);
imshow(reshaped_K5_C2);
title("Cluster #2",'FontSize',30);
subplot(2,5,[8,8]);
imshow(reshaped_K5_C3);
text(-8.0,-11.0,{'K = 5 Clusters'},'FontSize',40);
title("Cluster #3",'FontSize',30);
subplot(2,5,[9,9]);
imshow(reshaped_K5_C4);
title("Cluster #4",'FontSize',30);
subplot(2,5,[10,10]);
imshow(reshaped_K5_C5);
title("Cluster #5",'FontSize',30);
set(gcf,'Position',[10 1000 1500 1000])
saveas(gcf,sprintf('Ass2_E3_Images_of_clusters.png'))
close()


