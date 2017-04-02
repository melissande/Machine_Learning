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



%% Full dataset - K neighbors classification

X= M2_data;
% K fold crossvalidation

N=length(X);

CV = cvpartition(N, 'Leaveout');
K = CV.NumTestSets;
% K-nearest neighbors parameters
Distance = 'euclidean'; % Distance measure
L = 40; % Maximum number of neighbors

% Variable for classification error
Error = nan(K,L);sum(CV.TestSize)




for k = 1:K % For each crossvalidation fold
    fprintf('Crossvalidation fold %d/%d\n', k, CV.NumTestSets);

    % Extract training and test set
    X_train = X(CV.training(k), :);
    y_train = y(CV.training(k));
    X_test = X(CV.test(k), :);
    y_test = y(CV.test(k));

    for l = 1:L % For each number of neighbors
        
        % Use knnclassify to find the l nearest neighbors
        y_test_est = knnclassify(X_test, X_train, y_train, l, Distance);
        
        % Compute number of classification errors
        Error(k,l) = sum(y_test~=y_test_est); % Count the number of errors
    end
end


mfig('Full dataset - Error rate');
plot(sum(Error)/sum(CV.TestSize)*100);
xlabel('Number of neighbors');
ylabel('Classification error rate (%)');

[min_val,L_fin]=min(sum(Error)/sum(CV.TestSize)*100);

% Create holdout partition to display a confusion matrix
CV = cvpartition(classNames(y+1), 'holdout',0.5);


X_train = X(CV.training, :);
y_train = y(CV.training);
X_test = X(CV.test, :);
y_test = y(CV.test);

y_test_est = knnclassify(X_test, X_train, y_train, L_fin, Distance);


% Plot confusion matrix
mfig('Full dataset - Confusion matrix');
confmatplot(classNames(y_test+1), classNames(y_test_est+1));

 fprintf('Full Datset - nb Neighbors %d \n',L_fin );

%% FWI - K neighbors classification

X= FWI;
% K fold crossvalidation

N=length(X);

CV = cvpartition(N, 'Leaveout');
K = CV.NumTestSets;
% K-nearest neighbors parameters
Distance = 'euclidean'; % Distance measure
L = 80; % Maximum number of neighbors

% Variable for classification error
Error = nan(K,L);

for k = 1:K % For each crossvalidation fold
    fprintf('Crossvalidation fold %d/%d\n', k, CV.NumTestSets);

    % Extract training and test set
    X_train = X(CV.training(k), :);
    y_train = y(CV.training(k));
    X_test = X(CV.test(k), :);
    y_test = y(CV.test(k));

    for l = 1:L % For each number of neighbors
        
        % Use knnclassify to find the l nearest neighbors
        y_test_est = knnclassify(X_test, X_train, y_train, l, Distance);
        
        % Compute number of classification errors
        Error(k,l) = sum(y_test~=y_test_est); % Count the number of errors
    end
end




mfig('FWI -Error rate');
plot(sum(Error)./sum(CV.TestSize)*100);
xlabel('Number of neighbors');
ylabel('Classification error rate (%)');

[min_val,L_fin]=min(sum(Error)./sum(CV.TestSize)*100);
% Create holdout partition to display a confusion matrix
CV = cvpartition(classNames(y+1), 'holdout',0.5);


X_train = X(CV.training, :);
y_train = y(CV.training);
X_test = X(CV.test, :);
y_test = y(CV.test);

y_test_est = knnclassify(X_test, X_train, y_train, L_fin, Distance);

% Plot confusion matrix
mfig('Confusion matrix');
confmatplot(classNames(y_test+1), classNames(y_test_est+1));
fprintf('FWI - nb Neighbors %d \n',L_fin );

%% STM - K neighbors classification

X= STM;
% K fold crossvalidation

N=length(X);

CV = cvpartition(N, 'Leaveout');
K = CV.NumTestSets;
% K-nearest neighbors parameters
Distance = 'euclidean'; % Distance measure
L = 80; % Maximum number of neighbors

% Variable for classification error
Error = nan(K,L);

for k = 1:K % For each crossvalidation fold
    fprintf('Crossvalidation fold %d/%d\n', k, CV.NumTestSets);

    % Extract training and test set
    X_train = X(CV.training(k), :);
    y_train = y(CV.training(k));
    X_test = X(CV.test(k), :);
    y_test = y(CV.test(k));

    for l = 1:L % For each number of neighbors
        
        % Use knnclassify to find the l nearest neighbors
        y_test_est = knnclassify(X_test, X_train, y_train, l, Distance);
        
        % Compute number of classification errors
        Error(k,l) = sum(y_test~=y_test_est); % Count the number of errors
    end
end


mfig('STM -Error rate');
plot(sum(Error)./sum(CV.TestSize)*100);
xlabel('Number of neighbors');
ylabel('Classification error rate (%)');

[min_val,L_fin]=min(sum(Error)./sum(CV.TestSize)*100);
% Create holdout partition to display a confusion matrix
CV = cvpartition(classNames(y+1), 'holdout',0.5);


X_train = X(CV.training, :);
y_train = y(CV.training);
X_test = X(CV.test, :);
y_test = y(CV.test);

y_test_est = knnclassify(X_test, X_train, y_train, L_fin, Distance);
% Plot confusion matrix
mfig('STM Confusion matrix');
confmatplot(classNames(y_test+1), classNames(y_test_est+1));

fprintf('STM - nb Neighbors %d \n',L_fin );
%% STFWI - K neighbors classification

X= STFWI;
% K fold crossvalidation

N=length(X);

CV = cvpartition(N, 'Leaveout');
K = CV.NumTestSets;
% K-nearest neighbors parameters
Distance = 'euclidean'; % Distance measure
L = 40; % Maximum number of neighbors

% Variable for classification error
Error = nan(K,L);

for k = 1:K % For each crossvalidation fold
    fprintf('Crossvalidation fold %d/%d\n', k, CV.NumTestSets);

    % Extract training and test set
    X_train = X(CV.training(k), :);
    y_train = y(CV.training(k));
    X_test = X(CV.test(k), :);
    y_test = y(CV.test(k));

    for l = 1:L % For each number of neighbors
        
        % Use knnclassify to find the l nearest neighbors
        y_test_est = knnclassify(X_test, X_train, y_train, l, Distance);
        
        % Compute number of classification errors
        Error(k,l) = sum(y_test~=y_test_est); % Count the number of errors
    end
end


mfig('STFWI -Error rate');
plot(sum(Error)./sum(CV.TestSize)*100);
xlabel('Number of neighbors');
ylabel('Classification error rate (%)');

[min_val,L_fin]=min(sum(Error)./sum(CV.TestSize)*100);
% Create holdout partition to display a confusion matrix
CV = cvpartition(classNames(y+1), 'holdout',0.5);


X_train = X(CV.training, :);
y_train = y(CV.training);
X_test = X(CV.test, :);
y_test = y(CV.test);

y_test_est = knnclassify(X_test, X_train, y_train, L_fin, Distance);
% Plot confusion matrix
mfig('STFWI Confusion matrix');
confmatplot(classNames(y_test+1), classNames(y_test_est+1));

fprintf('STFWI - nb Neighbors %d \n',L_fin );
%% MET - K neighbors classification

X= MET;
% K fold crossvalidation

N=length(X);

CV = cvpartition(N, 'Leaveout');
K = CV.NumTestSets;
% K-nearest neighbors parameters
Distance = 'euclidean'; % Distance measure
L = 80; % Maximum number of neighbors

% Variable for classification error
Error = nan(K,L);

for k = 1:K % For each crossvalidation fold
    fprintf('Crossvalidation fold %d/%d\n', k, CV.NumTestSets);

    % Extract training and test set
    X_train = X(CV.training(k), :);
    y_train = y(CV.training(k));
    X_test = X(CV.test(k), :);
    y_test = y(CV.test(k));

    for l = 1:L % For each number of neighbors
        
        % Use knnclassify to find the l nearest neighbors
        y_test_est = knnclassify(X_test, X_train, y_train, l, Distance);
        
        % Compute number of classification errors
        Error(k,l) = sum(y_test~=y_test_est); % Count the number of errors
    end
end


mfig('MET -Error rate');
plot(sum(Error)./sum(CV.TestSize)*100);
xlabel('Number of neighbors');
ylabel('Classification error rate (%)');

[min_val,L_fin]=min(sum(Error)./sum(CV.TestSize)*100);


% Create holdout partition to display a confusion matrix
CV = cvpartition(classNames(y+1), 'holdout',0.5);


X_train = X(CV.training, :);
y_train = y(CV.training);
X_test = X(CV.test, :);
y_test = y(CV.test);

y_test_est = knnclassify(X_test, X_train, y_train, L_fin, Distance);
% Plot confusion matrix
mfig('MET Confusion matrix');
confmatplot(classNames(y_test+1), classNames(y_test_est+1));



fprintf('MET - nb Neighbors %d \n',L_fin );


