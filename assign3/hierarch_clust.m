function [Z,i]  = hierarch_clust( Kmax,X ,dataSet_name)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here

%% Hierarchical clustering

% Compute hierarchical clustering
Z = linkage(X, 'single', 'euclidean');

% Compute clustering by thresholding the dendrogram
i = cluster(Z, 'Maxclust', Kmax);

%% Plot results

% Plot dendrogram
mfig([dataSet_name,' Dendrogram ',Kmax]); clf;
dendrogram(Z);



end

