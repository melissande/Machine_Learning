%% Neural Network
M=zscore(MET);
K=15;
CV = cvpartition(size(M,1), 'Kfold', K);

NHiddenUnits = 1;  % Number of hidden units optimal
NTrain =1; % Number of re-trains of neural network
% Variable for regression error
Error_train_nn = nan(K,1);
Error_test_nn = nan(K,1);
Error_train_nofeatures = nan(K,1);
Error_test_nofeatures = nan(K,1);
bestnet=cell(K,1);
% For regression
w=nan(size(M,2)+1,K);
Error_train_reg=nan(K,1);
Error_test_reg=nan(K,1);
w_temp=nan(size(M,2)+1);

for k = 1:K % For each crossvalidation fold
    fprintf('Crossvalidation fold %d/%d\n', k, CV.NumTestSets);

    % Extract training and test set
    X_train = M(CV.training(k), :);
    y_train = area2(CV.training(k));
    X_test = M(CV.test(k), :);
    y_test = area2(CV.test(k));
 
    % Fit neural network to training set
    MSEBest = inf;
    for t = 1:NTrain
        netwrk = nr_main(X_train, y_train, X_test, y_test, NHiddenUnits);
        if netwrk.mse_train(end)<MSEBest, bestnet{k} = netwrk; MSEBest=netwrk.mse_train(end); MSEBest=netwrk.mse_train(end); end
    end
    %Fit linear reg
    w(:,k)=glmfit(X_train, y_train) ;
    w_temp=w(:,k);
    y_train_est_reg= glmval(w_temp,X_train,'identity'); %Regression based on the features selected matrix
    y_test_est_reg=glmval(w_temp,X_test,'identity');
    
%Reg lin error
Error_train_reg(k) = sum((y_train-y_train_est_reg).^2);
    Error_test_reg(k) = sum((y_test-y_test_est_reg).^2); 
 %Neural network error
  % Predict model on test and training data    
    y_train_est_nn = bestnet{k}.t_pred_train;    
    y_test_est_nn = bestnet{k}.t_pred_test;        
    
    % Compute least squares error
    Error_train_nn(k) = sum((y_train-y_train_est_nn).^2);
    Error_test_nn(k) = sum((y_test-y_test_est_nn).^2);
            
    % Compute least squares error when predicting output to be mean of
    % training data
    Error_train_nofeatures(k) = sum((y_train-mean(y_train)).^2);
    Error_test_nofeatures(k) = sum((y_test-mean(y_train)).^2);  
end
Error_gene_nn= sum(Error_test_nn)/sum(CV.TestSize)
Error_gene_reg=sum(Error_test_reg)/sum(CV.TestSize)
z=(Error_test_reg-Error_test_nn)
zb=mean(z)
sig = sqrt( mean( (z-zb).^2) / (K-1));
nu=K-1;
alpha = 0.05; 
[zLH] = zb + sig * tinv([alpha/2, 1-alpha/2], nu)

%% Same with K=10

M=zscore(MET);
K=10;
CV = cvpartition(size(M,1), 'Kfold', K);

NHiddenUnits = 1;  % Number of hidden units optimal
NTrain =1; % Number of re-trains of neural network
% Variable for regression error
Error_train_nn = nan(K,1);
Error_test_nn = nan(K,1);
Error_train_nofeatures = nan(K,1);
Error_test_nofeatures = nan(K,1);
bestnet=cell(K,1);
% For regression
w=nan(size(M,2)+1,K);
Error_train_reg=nan(K,1);
Error_test_reg=nan(K,1);
w_temp=nan(size(M,2)+1);

for k = 1:K % For each crossvalidation fold
    fprintf('Crossvalidation fold %d/%d\n', k, CV.NumTestSets);

    % Extract training and test set
    X_train = M(CV.training(k), :);
    y_train = area2(CV.training(k));
    X_test = M(CV.test(k), :);
    y_test = area2(CV.test(k));
 
    % Fit neural network to training set
    MSEBest = inf;
    for t = 1:NTrain
        netwrk = nr_main(X_train, y_train, X_test, y_test, NHiddenUnits);
        if netwrk.mse_train(end)<MSEBest, bestnet{k} = netwrk; MSEBest=netwrk.mse_train(end); MSEBest=netwrk.mse_train(end); end
    end
    %Fit linear reg
    w(:,k)=glmfit(X_train, y_train) ;
    w_temp=w(:,k);
    y_train_est_reg= glmval(w_temp,X_train,'identity'); %Regression based on the features selected matrix
    y_test_est_reg=glmval(w_temp,X_test,'identity');
    
%Reg lin error
Error_train_reg(k) = sum((y_train-y_train_est_reg).^2);
    Error_test_reg(k) = sum((y_test-y_test_est_reg).^2); 
 %Neural network error
  % Predict model on test and training data    
    y_train_est_nn = bestnet{k}.t_pred_train;    
    y_test_est_nn = bestnet{k}.t_pred_test;        
    
    % Compute least squares error
    Error_train_nn(k) = sum((y_train-y_train_est_nn).^2);
    Error_test_nn(k) = sum((y_test-y_test_est_nn).^2);
            
    % Compute least squares error when predicting output to be mean of
    % training data
    Error_train_nofeatures(k) = sum((y_train-mean(y_train)).^2);
    Error_test_nofeatures(k) = sum((y_test-mean(y_train)).^2);  
end
Error_gene_nn= sum(Error_test_nn)/sum(CV.TestSize)
Error_gene_reg=sum(Error_test_reg)/sum(CV.TestSize)
z=(Error_test_reg-Error_test_nn)
zb=mean(z)
sig = sqrt( mean( (z-zb).^2) / (K-1));
nu=K-1;
alpha = 0.05; 
[zLH] = zb + sig * tinv([alpha/2, 1-alpha/2], nu)

