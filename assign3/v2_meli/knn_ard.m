function [ ] = knn_ard( X,K, dataset_Name)
%UNTITLED7 Summary of this function goes here
%   Detailed explanation goes here
% Compute the average relative density
[idx, D] = knnsearch(X, X, 'K', K+1);
density = 1./(sum(D(:,2:end),2)/K);

avg_rel_density=density./(sum(density(idx(:,2:end)),2)/K);

% Sort the densities
[y,i] = sort(avg_rel_density);

% Plot outlier scores
mfig([dataset_Name,' KNN average relative density: outlier score']); clf;
bar(y(1:20));



end

