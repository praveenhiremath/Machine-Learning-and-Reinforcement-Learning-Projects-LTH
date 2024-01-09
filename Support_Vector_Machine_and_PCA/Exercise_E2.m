% Exercise E2: K-means clustering (For K=2 and K=5 clusters)
close all;
clear all;
load('A2_data.mat')  % MNIST images data

X_data = train_data_01;
K_vals = [2 5]; % 2 clusters and 5 clusters

for K = 1:length(K_vals)
    ys = [];
    Cs = [];
    [ys,Cs] = K_means_clustering(X_data,K_vals(K));    
    dim_red_X_data = linear_pca(X_data);
    save(['Clusters_from_E2_K',num2str(K_vals(K)),'.mat'],'ys','Cs');    

    figure('Name','K_means_clusters1')
    gscatter(dim_red_X_data(1,:), dim_red_X_data(2,:), ys, 'rgbcm', 'sosox')
    set(gca, 'FontSize', 24)
    title(['\fontsize{24} K-means clustering with K=',num2str(K_vals(K)),' clusters'])  
    legend('Cluster #1','Cluster #2','Cluster #3','Cluster #4','Cluster #5','location','northeast');
    set(gcf,'Position',[10 1000 1500 1000])
    saveas(gcf,sprintf('Ass2_E2_K%02d.png',K_vals(K)))
    close()
    
end
