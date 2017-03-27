%% Decisions trees - intro
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

%% Decisions Tree - Full dataset

X=M2_data;
% Number of folds for crossvalidation
K = 250;


% Create holdout crossvalidation partition
CV = cvpartition(classNames(y+1), 'Kfold', K);

% Pruning levels
prune = 0:15;

% Variable for classification error
Error_train = nan(K,length(prune));
Error_test = nan(K,length(prune));

for k = 1:K
    fprintf('Crossvalidation fold %d/%d\n', k, K);

    % Extract training and test set
    X_train = X(CV.training(k), :);
    y_train = y(CV.training(k));
    X_test = X(CV.test(k), :);
    y_test = y(CV.test(k));

    % Fit classification tree to training set
    T = classregtree(X_train, classNames(y_train+1), ...
        'method', 'classification', ...
        'splitcriterion', 'gdi', ...
        'categorical', [], ...
        'names', attributeNames_M2, ...
        'prune', 'on', ...
        'minparent', 10);

    % Compute classification error
    for n = 1:length(prune) % For each pruning level
        Error_train(k,n) = sum(~strcmp(classNames(y_train+1), eval(T, X_train, prune(n))));
        Error_test(k,n) = sum(~strcmp(classNames(y_test+1), eval(T, X_test, prune(n))));
    end    
end
    
% Plot classification error
mfig('Full Dataset - Fire decision tree: K-fold crossvalidation'); clf; hold all;
plot(prune, sum(Error_train)/sum(CV.TrainSize));
plot(prune, sum(Error_test)/sum(CV.TestSize));
xlabel('Pruning level');
ylabel('Classification error');
legend('Training error', 'Test error');



%% Decisions Tree -  STFWI
X=FWI;
% Number of folds for crossvalidation
K = 100;


% Create holdout crossvalidation partition
CV = cvpartition(classNames(y+1), 'Kfold', K);

% Pruning levels
prune = 0:10;

% Variable for classification error
Error_train = nan(K,length(prune));
Error_test = nan(K,length(prune));

for k = 1:K
    fprintf('Crossvalidation fold %d/%d\n', k, K);

    % Extract training and test set
    X_train = X(CV.training(k), :);
    y_train = y(CV.training(k));
    X_test = X(CV.test(k), :);
    y_test = y(CV.test(k));

    % Fit classification tree to training set
    T = classregtree(X_train, classNames(y_train+1), ...
        'method', 'classification', ...
        'splitcriterion', 'gdi', ...
        'categorical', [], ...
        'names', attributeNames_fwi, ...
        'prune', 'on', ...
        'minparent', 10);

    % Compute classification error
    for n = 1:length(prune) % For each pruning level
        Error_train(k,n) = sum(~strcmp(classNames(y_train+1), eval(T, X_train, prune(n))));
        Error_test(k,n) = sum(~strcmp(classNames(y_test+1), eval(T, X_test, prune(n))));
    end    
end
    
% Plot classification error
mfig('STFWI - Fire decision tree: K-fold crossvalidatoin'); clf; hold all;
plot(prune, sum(Error_train)/sum(CV.TrainSize));
plot(prune, sum(Error_test)/sum(CV.TestSize));
xlabel('Pruning level');
ylabel('Classification error');
legend('Training error', 'Test error');


%% Decisions Tree -  STM
X=STM;
% Number of folds for crossvalidation
K = 200;


% Create holdout crossvalidation partition
CV = cvpartition(classNames(y+1), 'Kfold', K);

% Pruning levels
prune = 0:12;

% Variable for classification error
Error_train = nan(K,length(prune));
Error_test = nan(K,length(prune));

for k = 1:K
    fprintf('Crossvalidation fold %d/%d\n', k, K);

    % Extract training and test set
    X_train = X(CV.training(k), :);
    y_train = y(CV.training(k));
    X_test = X(CV.test(k), :);
    y_test = y(CV.test(k));

    % Fit classification tree to training set
    T = classregtree(X_train, classNames(y_train+1), ...
        'method', 'classification', ...
        'splitcriterion', 'gdi', ...
        'categorical', [], ...
        'names', attributeNames_stm, ...
        'prune', 'on', ...
        'minparent', 10);

    % Compute classification error
    for n = 1:length(prune) % For each pruning level
        Error_train(k,n) = sum(~strcmp(classNames(y_train+1), eval(T, X_train, prune(n))));
        Error_test(k,n) = sum(~strcmp(classNames(y_test+1), eval(T, X_test, prune(n))));
    end    
end
    
% Plot classification error
mfig('STM - Fire decision tree: K-fold crossvalidatoin'); clf; hold all;
plot(prune, sum(Error_train)/sum(CV.TrainSize));
plot(prune, sum(Error_test)/sum(CV.TestSize));
xlabel('Pruning level');
ylabel('Classification error');
legend('Training error', 'Test error');

%% Decisions Tree -  MET
X=MET;
% Number of folds for crossvalidation
K = 100;


% Create holdout crossvalidation partition
CV = cvpartition(classNames(y+1), 'Kfold', K);

% Pruning levels
prune = 0:10;

% Variable for classification error
Error_train = nan(K,length(prune));
Error_test = nan(K,length(prune));

for k = 1:K
    fprintf('Crossvalidation fold %d/%d\n', k, K);

    % Extract training and test set
    X_train = X(CV.training(k), :);
    y_train = y(CV.training(k));
    X_test = X(CV.test(k), :);
    y_test = y(CV.test(k));

    % Fit classification tree to training set
    T = classregtree(X_train, classNames(y_train+1), ...
        'method', 'classification', ...
        'splitcriterion', 'gdi', ...
        'categorical', [], ...
        'names', attributeNames_met, ...
        'prune', 'on', ...
        'minparent', 10);

    % Compute classification error
    for n = 1:length(prune) % For each pruning level
        Error_train(k,n) = sum(~strcmp(classNames(y_train+1), eval(T, X_train, prune(n))));
        Error_test(k,n) = sum(~strcmp(classNames(y_test+1), eval(T, X_test, prune(n))));
    end    
end
    
% Plot classification error
mfig('MET - Fire decision tree: K-fold crossvalidatoin'); clf; hold all;
plot(prune, sum(Error_train)/sum(CV.TrainSize));
plot(prune, sum(Error_test)/sum(CV.TestSize));
xlabel('Pruning level');
ylabel('Classification error');
legend('Training error', 'Test error');