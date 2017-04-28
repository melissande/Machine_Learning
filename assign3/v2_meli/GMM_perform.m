function [ i,X_c,Sigma_c  ] = GMM_perform( K,X,dataset_Name,y)

% Fit model
OPTIONS = statset('MaxIter',300,'TolFun',1e-6);

G = gmdistribution.fit(X, K, 'Replicates', 10,'Regularize',10^(-6),'options',OPTIONS);

% Compute clustering
i = cluster(G, X);

%% Extract cluster centers
X_c = G.mu;
Sigma_c=G.Sigma;

%% Plot results

% Plot clustering
mfig([dataset_Name,' GMM: Clustering']); clf; 
size(X)
clusterplot(X, y, i, X_c, Sigma_c);

end

