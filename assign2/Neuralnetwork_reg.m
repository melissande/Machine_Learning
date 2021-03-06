remove_outliers_pro2
p = path();
cdir = fileparts(mfilename('fullpath')); 
rmpath(fullfile(cdir,'../Tools/nc_multiclass'));
addpath(fullfile(cdir,'../Tools/nc_binclass'));
%% First part : M=FWI
%% Important initial data
%Matrix chosen
M=FWI;
attributes = size(M,2);
attributeName=attributeNames_met;
K = 5; 
CV = cvpartition(size(M,1), 'Kfold', K);

%% Loop (cross-validation) to choose the appropriate number of Hidden Nodes
% Parameters for neural network classifier
I=9;
Err_gene = nan(I,1);
Err_train = nan(I,1);
for i=1:I
NHiddenUnits = i;  % Number of hidden units
NTrain =2; % Number of re-trains of neural network

% Variable for regression error
Error_train = nan(K,1);
Error_test = nan(K,1);
Error_train_nofeatures = nan(K,1);
Error_test_nofeatures = nan(K,1);
bestnet=cell(K,1);


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
    
  % Predict model on test and training data    
    y_train_est = bestnet{k}.t_pred_train;    
    y_test_est = bestnet{k}.t_pred_test;        
    
    % Compute least squares error
    Error_train(k) = sum((y_train-y_train_est).^2);
    Error_test(k) = sum((y_test-y_test_est).^2); 
        
    % Compute least squares error when predicting output to be mean of
    % training data
    Error_train_nofeatures(k) = sum((y_train-mean(y_train)).^2);
    Error_test_nofeatures(k) = sum((y_test-mean(y_train)).^2);  
end
Err_gene(i)= sum(Error_test)/sum(CV.TestSize) ;
Err_train(i)=  sum(Error_train)/sum(CV.TrainSize);

% end
%% Display results
fprintf('\n');
fprintf('Neural network regression without feature selection:\n');
fprintf('- Training error: %8.2f\n', sum(Error_train)/sum(CV.TrainSize));
fprintf('- Test error:     %8.2f\n', sum(Error_test)/sum(CV.TestSize)); %error généralisée
fprintf('- R^2 train:     %8.2f\n', (sum(Error_train_nofeatures)-sum(Error_train))/sum(Error_train_nofeatures));
fprintf('- R^2 test:     %8.2f\n', (sum(Error_test_nofeatures)-sum(Error_test))/sum(Error_test_nofeatures));
% fprintf('%.1f%%\n', Er_rate)
end


%% Display the error
Err_gene
mfig('Generalized error per hidden nodes'); clf;
plot(1:I,Err_gene,1:I,Err_train);
legend('Generalized error', 'av train error')
xlabel('Hidden nodes');
ylabel('Error');
%Selected : HiddenNodes  = 4
[val,hi_opt]= min(Err_gene);
%% Neural Network chosen displayed 
K=5;
NHiddenUnits = hi_opt;  % Number of hidden units optimal
NTrain =1; % Number of re-trains of neural network

% Variable for regression error
Error_train = nan(K,1);
Error_test = nan(K,1);
Error_train_nofeatures = nan(K,1);
Error_test_nofeatures = nan(K,1);
bestnet=cell(K,1);


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
    
  % Predict model on test and training data    
    y_train_est = bestnet{k}.t_pred_train;    
    y_test_est = bestnet{k}.t_pred_test;        
    
    % Compute least squares error
    Error_train_op(k) = sum((y_train-y_train_est).^2);
    Error_test_op(k) = sum((y_test-y_test_est).^2); 
        
    % Compute least squares error when predicting output to be mean of
    % training data
    Error_train_nofeatures(k) = sum((y_train-mean(y_train)).^2);
    Error_test_nofeatures(k) = sum((y_test-mean(y_train)).^2);  
end
Error_opt= sum(Error_test_op)/sum(CV.TestSize)
% Display the trained network 
mfig('Trained Network');
k=1; % cross-validation fold...  the minimum error chosen : k=1
displayNetworkRegression(bestnet{k});

%Display the area estimated thanks to NN against the real one
%area_NN = [y_train

%% Weight analysis
W_input= bestnet{k}.Wi;
W_out=bestnet{k}.Wo;

mfig('Weights input');
plot(W_input)
title('Weigths input')
xlabel('attributes')
xticks([1:(size(M,2)+1)]);
attrib_intercpt=[attributeName,{'Intercept'}];
xticklabels(attrib_intercpt);
ylabel('Hidden nodes')
title('Weights input')
mfig('Weights output');

W_out
title('Weigths output')
ylabel('Weight value')
xticks([1:(hi_opt+1)]);
%% Important initial data
%Matrix chosen
M=MET;
attributes = size(M,2);
attributeName=attributeNames_met;
K = 5; 
CV = cvpartition(size(M,1), 'Kfold', K);

%% Loop (cross-validation) to choose the appropriate number of Hidden Nodes
% Parameters for neural network classifier
I=9;
Err_gene = nan(I,1);
Err_train = nan(I,1);
for i=1:I
NHiddenUnits = i;  % Number of hidden units
NTrain =2; % Number of re-trains of neural network

% Variable for regression error
Error_train = nan(K,1);
Error_test = nan(K,1);
Error_train_nofeatures = nan(K,1);
Error_test_nofeatures = nan(K,1);
bestnet=cell(K,1);


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
    
  % Predict model on test and training data    
    y_train_est = bestnet{k}.t_pred_train;    
    y_test_est = bestnet{k}.t_pred_test;        
    
    % Compute least squares error
    Error_train(k) = sum((y_train-y_train_est).^2);
    Error_test(k) = sum((y_test-y_test_est).^2); 
        
    % Compute least squares error when predicting output to be mean of
    % training data
    Error_train_nofeatures(k) = sum((y_train-mean(y_train)).^2);
    Error_test_nofeatures(k) = sum((y_test-mean(y_train)).^2);  
end
Err_gene(i)= sum(Error_test)/sum(CV.TestSize) ;
Err_train(i)=  sum(Error_train)/sum(CV.TrainSize);

% end
%% Display results
fprintf('\n');
fprintf('Neural network regression without feature selection:\n');
fprintf('- Training error: %8.2f\n', sum(Error_train)/sum(CV.TrainSize));
fprintf('- Test error:     %8.2f\n', sum(Error_test)/sum(CV.TestSize)); %error généralisée
fprintf('- R^2 train:     %8.2f\n', (sum(Error_train_nofeatures)-sum(Error_train))/sum(Error_train_nofeatures));
fprintf('- R^2 test:     %8.2f\n', (sum(Error_test_nofeatures)-sum(Error_test))/sum(Error_test_nofeatures));
% fprintf('%.1f%%\n', Er_rate)
end


%% Display the error
Err_gene
mfig('Generalized error per hidden nodes'); clf;
plot(1:I,Err_gene,1:I,Err_train);
legend('Generalized error', 'av train error')
xlabel('Hidden nodes');
ylabel('Error');
%Selected : HiddenNodes  = 4
[val,hi_opt]= min(Err_gene);
%% Neural Network chosen displayed 
K=5;
NHiddenUnits = hi_opt;  % Number of hidden units optimal
NTrain =1; % Number of re-trains of neural network

% Variable for regression error
Error_train = nan(K,1);
Error_test = nan(K,1);
Error_train_nofeatures = nan(K,1);
Error_test_nofeatures = nan(K,1);
bestnet=cell(K,1);


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
    
  % Predict model on test and training data    
    y_train_est = bestnet{k}.t_pred_train;    
    y_test_est = bestnet{k}.t_pred_test;        
    
    % Compute least squares error
    Error_train_op(k) = sum((y_train-y_train_est).^2);
    Error_test_op(k) = sum((y_test-y_test_est).^2); 
        
    % Compute least squares error when predicting output to be mean of
    % training data
    Error_train_nofeatures(k) = sum((y_train-mean(y_train)).^2);
    Error_test_nofeatures(k) = sum((y_test-mean(y_train)).^2);  
end
Error_opt= sum(Error_test_op)/sum(CV.TestSize)
% Display the trained network 
mfig('Trained Network');
k=1; % cross-validation fold...  the minimum error chosen : k=1
displayNetworkRegression(bestnet{k});

%Display the area estimated thanks to NN against the real one
%area_NN = [y_train

%% Weight analysis
W_input= bestnet{k}.Wi;
W_out=bestnet{k}.Wo;

mfig('Weights input');
plot(W_input)
title('Weigths input')
xlabel('attributes')
xticks([1:(size(M,2)+1)]);
attrib_intercpt=[attributeName,{'Intercept'}];
xticklabels(attrib_intercpt);
ylabel('Hidden nodes')
title('Weights input')
mfig('Weights output');

W_out
title('Weigths output')
ylabel('Weight value')
xticks([1:(hi_opt+1)]);
xticklabels([1:hi_opt,{'Intercept'}]);
xlabel('Hidden nodes')
title('Weights output')
xticklabels([1:hi_opt,{'Intercept'}]);
xlabel('Hidden nodes')
title('Weights output')

%% Same thing with M=MET
%Important initial data
%Matrix chosen
M=zscore(MET);
attributes = size(M,2);
attributeName=attributeNames_met;
K = 5; 
CV = cvpartition(size(M,1), 'Kfold', K);

% Loop (cross-validation) to choose the appropriate number of Hidden Nodes
% Parameters for neural network classifier
I=4;
Err_gene = nan(I,1);
Err_train = nan(I,1);
for i=1:I
NHiddenUnits = i;  % Number of hidden units
NTrain =2; % Number of re-trains of neural network

% Variable for regression error
Error_train = nan(K,1);
Error_test = nan(K,1);
Error_train_nofeatures = nan(K,1);
Error_test_nofeatures = nan(K,1);
bestnet=cell(K,1);


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
    
  % Predict model on test and training data    
    y_train_est = bestnet{k}.t_pred_train;    
    y_test_est = bestnet{k}.t_pred_test;        
    
    % Compute least squares error
    Error_train(k) = sum((y_train-y_train_est).^2);
    Error_test(k) = sum((y_test-y_test_est).^2); 
        
    % Compute least squares error when predicting output to be mean of
    % training data
    Error_train_nofeatures(k) = sum((y_train-mean(y_train)).^2);
    Error_test_nofeatures(k) = sum((y_test-mean(y_train)).^2);  
end
Err_gene(i)= sum(Error_test)/sum(CV.TestSize) ;
Err_train(i)=  sum(Error_train)/sum(CV.TrainSize);

% end
% Display results
fprintf('\n');
fprintf('Neural network regression without feature selection:\n');
fprintf('- Training error: %8.2f\n', sum(Error_train)/sum(CV.TrainSize));
fprintf('- Test error:     %8.2f\n', sum(Error_test)/sum(CV.TestSize)); %error généralisée
fprintf('- R^2 train:     %8.2f\n', (sum(Error_train_nofeatures)-sum(Error_train))/sum(Error_train_nofeatures));
fprintf('- R^2 test:     %8.2f\n', (sum(Error_test_nofeatures)-sum(Error_test))/sum(Error_test_nofeatures));
% fprintf('%.1f%%\n', Er_rate)
end


% Display the error
Err_gene
mfig('Generalized error per hidden nodes'); clf;
plot(1:I,Err_gene,1:I,Err_train);
legend('Generalized error', 'av train error')
xlabel('Hidden nodes');
ylabel('Error');
%Selected : HiddenNodes  = 4
[val,hi_opt]= min(Err_gene);
%% Neural Network chosen displayed 
K=5;
NHiddenUnits = 1;  % Number of hidden units optimal
NTrain =1; % Number of re-trains of neural network

% Variable for regression error
Error_train = nan(K,1);
Error_test = nan(K,1);
Error_train_nofeatures = nan(K,1);
Error_test_nofeatures = nan(K,1);
bestnet=cell(K,1);


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
    
  % Predict model on test and training data    
    y_train_est = bestnet{k}.t_pred_train;    
    y_test_est = bestnet{k}.t_pred_test;        
    
    % Compute least squares error
    Error_train_op(k) = sum((y_train-y_train_est).^2);
    Error_test_op(k) = sum((y_test-y_test_est).^2); 
        
    % Compute least squares error when predicting output to be mean of
    % training data
    Error_train_nofeatures(k) = sum((y_train-mean(y_train)).^2);
    Error_test_nofeatures(k) = sum((y_test-mean(y_train)).^2);  
end
Error_opt= sum(Error_test_op)/sum(CV.TestSize)
% Display the trained network 
mfig('Trained Network');
k=1; % cross-validation fold...  the minimum error chosen : k=1
displayNetworkRegression(bestnet{k});

%Display the area estimated thanks to NN against the real one
%area_NN = [y_train

%% Weight analysis
W_input= bestnet{k}.Wi;
W_out=bestnet{k}.Wo;

mfig('Weights input');
plot(W_input)
title('Weigths input')
xlabel('attributes')
xticks([1:(size(M,2)+1)]);
attrib_intercpt=[attributeName,{'Intercept'}];
xticklabels(attrib_intercpt);
ylabel('Hidden nodes')
title('Weights input')
mfig('Weights output');
W_input
W_out
title('Weigths output')
ylabel('Weight value')
xticks([1:(hi_opt+1)]);
xticklabels([1:hi_opt,{'Intercept'}]);
xlabel('Hidden nodes')
title('Weights output')
path(p); %reset path.