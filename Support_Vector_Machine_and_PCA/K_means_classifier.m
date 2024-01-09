% Assigning labels of the closest cluster centroids to data points. Using the labels provided in the data i.e. (i) load('A2_data.mat') (ii)"X = train_data_01;" 
% then (iii) "labels = train_labels_01" 
% there are 2 labels in "A2_data.mat" because "unique(labels) = [0 1]"


function centroid_labels = K_means_classifier(labels_from_X, K, y)
    centroid_labels = zeros(K,1);
    for k = 1:K 
% The function "K_means_clustering" assigns N-datapoints to K clusters and is stored in y of dimension (N,1). 
% So now the labels in "y" are compared with actual labels from data "train_data_01". Then the majority of labels in y that match with those in "train_data_01" becomes 
% the label of the centroid of that cluster K.
        if ((sum(labels_from_X(y == k) == 1)) > (sum(labels_from_X(y == k) == 0)))   % if this condition fails, centroid label remains "0".
            centroid_labels(k) = 1;
        else
            centroid_labels(k) = 0;
        end
    end
end
                 
