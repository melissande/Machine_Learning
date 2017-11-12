function [] = knn_density_perform( X,K, dataset_Name )
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here


% Find the k nearest neighbors
[idx, D] = knnsearch(X, X, 'K', K+1);

% Compute the density
density = 1./(sum(D(:,2:end),2)/K);

% Sort the densities
[y,i] = sort(density);

% Plot outlier scores
mfig([dataset_Name,' KNN density: outlier score']); clf;
bar(y(1:20));


end

