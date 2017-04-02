%% Methods comparison
close all;
clear all;
clc;
run main_project1.m
run remove_outliers_pro2.m

%% definition of class names and preparation
area3= M_data(:,end);
classNames={'small','medium','large'}';

range_S_area = [min(area3), median(area3)] 
range_M_area = [prctile(area3,50), prctile(area3,75)]
range_L_area = [prctile(area3,75) prctile(area3,100)]

y=area3;
y((min(range_S_area)<=area3) & (area3<=max(range_S_area)))=0;
y((min(range_M_area)<=area3) & (area3<=max(range_M_area)))=1;
y((min(range_L_area)<=area3) & (area3<=max(range_L_area)))=2;

%% Full dataset - Method comparison


X= M2_data;


% K-fold crossvalidation out loop
K1 = 10;
CV1 = cvpartition(y, 'Kfold', K1);
% K-fold crossvalidation inner loop
    K2 = 10;

% Parameters for naive Bayes classifier
Distribution = 'mvmn';
Prior = 'empirical';

% Parameters for Decision Tree classifier
prune =10;

% Parameters for K neighbor classifier
L =12;
Distance = 'euclidean'; % Distance measure



% Variable for classification error

Error_in_gen=nan(3,1);
Error_out_gen=nan(K1,2);

for k1 = 1:K1 % For each crossvalidation fold
   % fprintf('Crossvalidation fold %d/%d\n', k1, CV.NumTestSets);

    % Extract training and test set
    X_train = X(CV1.training(k1), :);
    y_train = y(CV1.training(k1));
    X_test = X(CV1.test(k1), :);
    y_test = y(CV1.test(k1));
    
    Error_Bayes = nan(K2,1);
    Error_dt=nan(K2,1);
    Error_kclass=nan(K2,1);
    
    for k2 = 1:K2
        
     
    CV2 = cvpartition(y_train, 'Kfold', K2);

        
    X_train2 = X(CV2.training(k2), :);
    y_train2 = y(CV2.training(k2));
    X_test2 = X(CV2.test(k2), :);
    y_test2 = y(CV2.test(k2));
    %%NAIVE BAYES
    % Fit naive Bayes classifier to training set
    NB = NaiveBayes.fit(X_train2, y_train2, 'Distribution', Distribution, 'Prior', Prior);
    
    % Predict model on test data    
    y_test_est2 = predict(NB, X_test2);
    
    % Compute error rate
    Error_Bayes(k2) = sum(y_test2~=y_test_est2); % Count the number of errors
    
    
    
     %%DECISION TREE
     % Fit classification tree to training set
    T = classregtree(X_train2, classNames(y_train2+1), ...
        'method', 'classification', ...
        'splitcriterion', 'gdi', ...
        'categorical', [], ...
        'names', attributeNames_M2, ...
        'prune', 'on', ...
        'minparent', 10);

    Error_dt(k2) = sum(~strcmp(classNames(y_test2+1), eval(T, X_test2, prune)));
    
    
    %%k neighbor
    % Use knnclassify to find the L nearest neighbors
   y_test_est2 = knnclassify(X_test2, X_train2, y_train2, L, Distance);
    
     Error_kclass(k2) = sum(y_test2~=y_test_est2); % Count the number of errors 
     
    end
    
    Error_in_gen(1)=sum(CV2.TestSize)*sum(Error_Bayes)/sum(CV1.TrainSize);
    Error_in_gen(2)=sum(CV2.TestSize)*sum(Error_dt)/sum(CV1.TrainSize);
    Error_in_gen(3)=sum(CV2.TestSize)*sum(Error_kclass)/sum(CV1.TrainSize);
    
    [val,ind]=min(Error_in_gen);
    
    Error_out_gen(k1,1)=ind;
    switch ind
        case 1
            NB = NaiveBayes.fit(X_train, y_train, 'Distribution', Distribution, 'Prior', Prior);
    
            % Predict model on test data    
             y_test_est = predict(NB, X_test);
    
             % Compute error rate
              Error_out_gen(k1,2) = sum(y_test~=y_test_est); % Count the number of errors
        case 2
             %%DECISION TREE
            % Fit classification tree to training set
            T = classregtree(X_train, classNames(y_train+1), ...
        'method', 'classification', ...
        'splitcriterion', 'gdi', ...
        'categorical', [], ...
        'names', attributeNames_M2, ...
        'prune', 'on', ...
        'minparent', 10);

            Error_out_gen(k1,2) = sum(~strcmp(classNames(y_test+1), eval(T, X_test, prune)));
        case 3
            
             
        %%k neighbor
        % Use knnclassify to find the L nearest neighbors
        y_test_est = knnclassify(X_test, X_train, y_train, L, Distance);
            
           Error_out_gen(k1,2) = sum(y_test~=y_test_est); % Count the number of errors 
    end
    
            
      
    
end
% Print the error rate
fprintf('Full Dataset Gene Error: %.1f\n', sum(CV1.TestSize)*sum(Error_out_gen(:,2))/N);
%%
%Plot
vec=[1:K1]';
figure(1)
plot(vec(Error_out_gen(:,1)==1,1),Error_out_gen(Error_out_gen(:,1)==1,2),'r.','MarkerSize',12)
hold on;
plot(vec(Error_out_gen(:,1)==2,1),Error_out_gen(Error_out_gen(:,1)==2,2),'b.','MarkerSize',12)
hold on;
plot(vec(Error_out_gen(:,1)==3,1),Error_out_gen(Error_out_gen(:,1)==3,2),'g.','MarkerSize',12)
hold off;
legend('Naive Bayes','Classification Tree','KNN')
xlabel('K fold')
ylabel('Test Error')
title('Full Dataset Comparison Models')

%% FWI- Method comparison


X= FWI;


% K-fold crossvalidation out loop
K1 = 10;
CV1 = cvpartition(y, 'Kfold', K1);
% K-fold crossvalidation inner loop
    K2 = 10;

% Parameters for naive Bayes classifier
Distribution = 'mvmn';
Prior = 'empirical';

% Parameters for Decision Tree classifier
prune =10;

% Parameters for K neighbor classifier
L =20;
Distance = 'euclidean'; % Distance measure



% Variable for classification error

Error_in_gen=nan(3,1);
Error_out_gen=nan(K1,2);

for k1 = 1:K1 % For each crossvalidation fold
   % fprintf('Crossvalidation fold %d/%d\n', k1, CV.NumTestSets);

    % Extract training and test set
    X_train = X(CV1.training(k1), :);
    y_train = y(CV1.training(k1));
    X_test = X(CV1.test(k1), :);
    y_test = y(CV1.test(k1));
    
    Error_Bayes = nan(K2,1);
    Error_dt=nan(K2,1);
    Error_kclass=nan(K2,1);
    
    for k2 = 1:K2
        
     
    CV2 = cvpartition(y_train, 'Kfold', K2);

        
    X_train2 = X(CV2.training(k2), :);
    y_train2 = y(CV2.training(k2));
    X_test2 = X(CV2.test(k2), :);
    y_test2 = y(CV2.test(k2));
    %%NAIVE BAYES
    % Fit naive Bayes classifier to training set
    NB = NaiveBayes.fit(X_train2, y_train2, 'Distribution', Distribution, 'Prior', Prior);
    
    % Predict model on test data    
    y_test_est2 = predict(NB, X_test2);
    
    % Compute error rate
    Error_Bayes(k2) = sum(y_test2~=y_test_est2); % Count the number of errors
    
    
    
     %%DECISION TREE
     % Fit classification tree to training set
    T = classregtree(X_train2, classNames(y_train2+1), ...
        'method', 'classification', ...
        'splitcriterion', 'gdi', ...
        'categorical', [], ...
        'names', attributeNames_fwi, ...
        'prune', 'on', ...
        'minparent', 10);

    Error_dt(k2) = sum(~strcmp(classNames(y_test2+1), eval(T, X_test2, prune)));
    
    
    %%k neighbor
    % Use knnclassify to find the L nearest neighbors
   y_test_est2 = knnclassify(X_test2, X_train2, y_train2, L, Distance);
    
     Error_kclass(k2) = sum(y_test2~=y_test_est2); % Count the number of errors 
     
    end
    
    Error_in_gen(1)=sum(CV2.TestSize)*sum(Error_Bayes)/sum(CV1.TrainSize);
    Error_in_gen(2)=sum(CV2.TestSize)*sum(Error_dt)/sum(CV1.TrainSize);
    Error_in_gen(3)=sum(CV2.TestSize)*sum(Error_kclass)/sum(CV1.TrainSize);
    
    [val,ind]=min(Error_in_gen);
    
    Error_out_gen(k1,1)=ind;
    switch ind
        case 1
            NB = NaiveBayes.fit(X_train, y_train, 'Distribution', Distribution, 'Prior', Prior);
    
            % Predict model on test data    
             y_test_est = predict(NB, X_test);
    
             % Compute error rate
              Error_out_gen(k1,2) = sum(y_test~=y_test_est); % Count the number of errors
        case 2
             %%DECISION TREE
            % Fit classification tree to training set
            T = classregtree(X_train, classNames(y_train+1), ...
        'method', 'classification', ...
        'splitcriterion', 'gdi', ...
        'categorical', [], ...
        'names', attributeNames_fwi, ...
        'prune', 'on', ...
        'minparent', 10);

            Error_out_gen(k1,2) = sum(~strcmp(classNames(y_test+1), eval(T, X_test, prune)));
        case 3
            
             
        %%k neighbor
        % Use knnclassify to find the L nearest neighbors
        y_test_est = knnclassify(X_test, X_train, y_train, L, Distance);
            
           Error_out_gen(k1,2) = sum(y_test~=y_test_est); % Count the number of errors 
    end
    
            
      
    
end

% Print the error rate
fprintf('FWI Gene Error: %.1f\n', sum(CV1.TestSize)*sum(Error_out_gen(:,2))/N);

%Plot
vec=[1:K1]';
figure(2)
plot(vec(Error_out_gen(:,1)==1,1),Error_out_gen(Error_out_gen(:,1)==1,2),'r.','MarkerSize',12)
hold on;
plot(vec(Error_out_gen(:,1)==2,1),Error_out_gen(Error_out_gen(:,1)==2,2),'b.','MarkerSize',12)
hold on;
plot(vec(Error_out_gen(:,1)==3,1),Error_out_gen(Error_out_gen(:,1)==3,2),'g.','MarkerSize',12)
hold off;
legend('Naive Bayes','Classification Tree','KNN')
xlabel('K fold')
ylabel('Test Error')
title('FWI Comparison Models')