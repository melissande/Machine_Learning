function [ i,X_c,Sigma_c  ] = GMM_perform( K,X,dataset_Name,y,classNames )

% Fit model
G = gmdistribution.fit(X, K, 'Replicates', 10,'Regularize',10^(-6));

% Compute clustering
i = cluster(G, X);

%% Extract cluster centers
X_c = G.mu;
Sigma_c=G.Sigma;

%% Plot results

% Plot clustering
mfig([dataset_Name,' GMM: Clustering']); clf; 
clusterplot(X, y, i, X_c, Sigma_c);
legend(classNames)
end

