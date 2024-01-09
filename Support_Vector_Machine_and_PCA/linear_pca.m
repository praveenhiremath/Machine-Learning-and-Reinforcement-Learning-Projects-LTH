function X_projected = linear_pca(X)
% Create zero mean dataset
X_zeromean = X - mean(X,2);
% Calculating [U,S,V] using svd() in matlab
[U,S,V] = svd(X_zeromean);
% Obtaining 1st and 2nd principal components
princ_comps = U(:,1:2);
% Projecting on 1st and 2nd principal components
X_projected = princ_comps'*X_zeromean;
end
