%% K neighbors - intro
close all;
clear all;
clc;
run main_project1.m
run remove_outliers_pro2.m
%% definition of class names and preparation
area3= log(M_data(:,end)+1);
classNames={'small','medium','large'}';

range_S_area = [min(area3), median(area3)] 
range_M_area = [prctile(area3,50), prctile(area3,75)]
range_L_area = [prctile(area3,75) prctile(area3,100)]

y=area3;
y((min(range_S_area)<=area3) & (area3<=max(range_S_area)))=0;
y((min(range_M_area)<=area3) & (area3<=max(range_M_area)))=1;
y((min(range_L_area)<=area3) & (area3<=max(range_L_area)))=2;

%% Full dataset - Naive Bayes classification

X= M2_data;


N=length(X);



% K-fold crossvalidation
K = 10;
CV = cvpartition(y, 'Kfold', K);

% Parameters for naive Bayes classifier
Distribution = 'mvmn';
Prior = 'empirical';

% Variable for classification error
Error = nan(K,1);


for k = 1:K % For each crossvalidation fold
    %fprintf('Crossvalidation fold %d/%d\n', k, CV.NumTestSets);

    % Extract training and test set
    X_train = X(CV.training(k), :);
    y_train = y(CV.training(k));
    X_test = X(CV.test(k), :);
    y_test = y(CV.test(k));

    % Fit naive Bayes classifier to training set
    NB = NaiveBayes.fit(X_train, y_train, 'Distribution', Distribution, 'Prior', Prior);
    
    % Predict model on test data    
    y_test_est = predict(NB, X_test);
    
    % Compute error rate
    Error(k) = sum(y_test~=y_test_est); % Count the number of errors
end

% Print the error rate
fprintf('Full dataset Error rate: %.1f%%\n', sum(Error)./sum(CV.TestSize)*100);

% Full dataset Error rate: 70.6%
%% FWI - Naive Bayes classification

X= FWI;


N=length(X);



% K-fold crossvalidation
K = 10;
CV = cvpartition(y, 'Kfold', K);

% Parameters for naive Bayes classifier
Distribution = 'mvmn';
Prior = 'empirical';

% Variable for classification error
Error = nan(K,1);


for k = 1:K % For each crossvalidation fold
    %fprintf('Crossvalidation fold %d/%d\n', k, CV.NumTestSets);

    % Extract training and test set
    X_train = X(CV.training(k), :);
    y_train = y(CV.training(k));
    X_test = X(CV.test(k), :);
    y_test = y(CV.test(k));

    % Fit naive Bayes classifier to training set
    NB = NaiveBayes.fit(X_train, y_train, 'Distribution', Distribution, 'Prior', Prior);
    
    % Predict model on test data    
    y_test_est = predict(NB, X_test);
    
    % Compute error rate
    Error(k) = sum(y_test~=y_test_est); % Count the number of errors
end

% Print the error rate
fprintf('FWI Error rate: %.1f%%\n', sum(Error)./sum(CV.TestSize)*100);
%FWI Error rate: 63.2%
%% STM - Naive Bayes classification

X= STM;


N=length(X);



% K-fold crossvalidation
K = 10;
CV = cvpartition(y, 'Kfold', K);

% Parameters for naive Bayes classifier
Distribution = 'mvmn';
Prior = 'empirical';

% Variable for classification error
Error = nan(K,1);


for k = 1:K % For each crossvalidation fold
    %fprintf('Crossvalidation fold %d/%d\n', k, CV.NumTestSets);

    % Extract training and test set
    X_train = X(CV.training(k), :);
    y_train = y(CV.training(k));
    X_test = X(CV.test(k), :);
    y_test = y(CV.test(k));

    % Fit naive Bayes classifier to training set
    NB = NaiveBayes.fit(X_train, y_train, 'Distribution', Distribution, 'Prior', Prior);
    
    % Predict model on test data    
    y_test_est = predict(NB, X_test);
    
    % Compute error rate
    Error(k) = sum(y_test~=y_test_est); % Count the number of errors
end

% Print the error rate
fprintf('STM Error rate: %.1f%%\n', sum(Error)./sum(CV.TestSize)*100);
%STM Error rate: 64.9%

%% STFWI - Naive Bayes classification

X= STFWI;


N=length(X);



% K-fold crossvalidation
K = 10;
CV = cvpartition(y, 'Kfold', K);

% Parameters for naive Bayes classifier
Distribution = 'mvmn';
Prior = 'empirical';

% Variable for classification error
Error = nan(K,1);


for k = 1:K % For each crossvalidation fold
    %fprintf('Crossvalidation fold %d/%d\n', k, CV.NumTestSets);

    % Extract training and test set
    X_train = X(CV.training(k), :);
    y_train = y(CV.training(k));
    X_test = X(CV.test(k), :);
    y_test = y(CV.test(k));

    % Fit naive Bayes classifier to training set
    NB = NaiveBayes.fit(X_train, y_train, 'Distribution', Distribution, 'Prior', Prior);
    
    % Predict model on test data    
    y_test_est = predict(NB, X_test);
    
    % Compute error rate
    Error(k) = sum(y_test~=y_test_est); % Count the number of errors
end

% Print the error rate
fprintf('STFWI Error rate: %.1f%%\n', sum(Error)./sum(CV.TestSize)*100);
%STFWI Error rate: 62.8%
%% MET - Naive Bayes classification

X= MET;


N=length(X);



% K-fold crossvalidation
K = 10;
CV = cvpartition(y, 'Kfold', K);

% Parameters for naive Bayes classifier
Distribution = 'mvmn';
Prior = 'empirical';

% Variable for classification error
Error = nan(K,1);


for k = 1:K % For each crossvalidation fold
    %fprintf('Crossvalidation fold %d/%d\n', k, CV.NumTestSets);

    % Extract training and test set
    X_train = X(CV.training(k), :);
    y_train = y(CV.training(k));
    X_test = X(CV.test(k), :);
    y_test = y(CV.test(k));

    % Fit naive Bayes classifier to training set
    NB = NaiveBayes.fit(X_train, y_train, 'Distribution', Distribution, 'Prior', Prior);
    
    % Predict model on test data    
    y_test_est = predict(NB, X_test);
    
    % Compute error rate
    Error(k) = sum(y_test~=y_test_est); % Count the number of errors
end

% Print the error rate
fprintf('MET Error rate: %.1f%%\n', sum(Error)./sum(CV.TestSize)*100);

%MET Error rate: 61.7%