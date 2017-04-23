function [ ] = GMM_k_sel(X,dataset_Name,Kmax)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
%i,X_c,Sigma_c 
N=size(X,1); 

% Range of K's to try
KRange = 1:Kmax;
T = length(KRange);

% Allocate variables
BIC = nan(T,1);
AIC = nan(T,1);
CVE = zeros(T,1);

% Create crossvalidation partition for evaluation
CV = cvpartition(N, 'Kfold', 10);

% For each model order
for t = 1:T    
    % Get the current K
    K = KRange(t);
    
    % Display information
    fprintf('Fitting model for K=%d\n', K);
    
    % Fit model
    G = gmdistribution.fit(X, K, 'Replicates', 10,'Regularize',10^(-6));
    
    % Get BIC and AIC
    BIC(t) = G.BIC;
    AIC(t) = G.AIC;
    
    % For each crossvalidation fold
    for k = 1:CV.NumTestSets
        % Extract the training and test set
        X_train = X(CV.training(k), :);
        X_test = X(CV.test(k), :);
        
        % Fit model to training set
        G = gmdistribution.fit(X_train, K, 'Replicates', 10,'Regularize',10^(-6));
        
        % Evaluation crossvalidation error
        [~, NLOGL] = posterior(G, X_test);
        CVE(t) = CVE(t)+NLOGL;
    end
end


%% Plot results

mfig([dataset_Name,' GMM: Number of clusters']); clf; hold all
plot(KRange, BIC);
plot(KRange, AIC);
plot(KRange, 2*CVE);
legend('BIC', 'AIC', 'Crossvalidation');
xlabel('K');

%

end

