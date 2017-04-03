%% Ttest
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

%% Full Dataset Crossvalidation

X=M2_data;
% Create 10-fold crossvalidation partition for evaluation
K = 10;
CV = cvpartition(y, 'Kfold', K);

Distance='euclidean';
% Initialize variables
Error_knn13 = nan(1,K);
Error_knn9 = nan(1,K);
Error_lg_class = nan(1,K);

% For each crossvalidation fold
for k = 1:K
    fprintf('Crossvalidation fold %d/%d\n', k, K);
    
    % Extract the training and test set
    X_train = X(CV.training(k), :); 
    y_train = y(CV.training(k));
    X_test = X(CV.test(k), :); 
    y_test = y(CV.test(k));

     %KNN with neighb=13
     
     y_test_est = knnclassify(X_test, X_train, y_train, 13, Distance);
        
     
        Error_knn13(k) = sum(y_test~=y_test_est); % Count the number of errors
        

    
     %KNN with neighb=9
     
     y_test_est = knnclassify(X_test, X_train, y_train, 9, Distance);
        
     
        Error_knn9(k) = sum(y_test~=y_test_est); % Count the number of errors
        
     %Largest class
      [larg_class,nb]=mode(y_train);
    y_test_est=larg_class*ones(CV.TestSize(1,k),1);
    Error_lg_class(k) =sum(y_test_est~=y_test);
end



% Determine if classifiers are significantly different
mfig('Error rates Knn13/Knn9 ');
boxplot([Error_knn13./CV.TestSize; Error_knn9./CV.TestSize]'*100, ...
    'labels', {'DT pruning=13', 'KNN  N=9'});
ylabel('Error rate (%)');

disp('Knn13/Knn9')
[H,P,CI] =ttest(Error_knn13, Error_knn9);
P
CI
if H 
    disp('Classifiers are significantly different');
else
    disp('Classifiers are NOT significantly different');
end


% Determine if classifiers are significantly different
mfig('Error rates Error knn13/LC');
boxplot([Error_knn13./CV.TestSize; Error_lg_class./CV.TestSize]'*100, ...
    'labels', {'KNN N=13', 'Larger Class'});
ylabel('Error rate (%)');

disp('KNN13/LC')
[H,P,CI] =ttest(Error_knn13,Error_lg_class);
P
CI
if H 
    disp('Classifiers are significantly different');
else
    disp('Classifiers are NOT significantly different');
end

% Determine if classifiers are significantly different
mfig('Error rates KNN9/LC');
boxplot([Error_lg_class./CV.TestSize; Error_knn9./CV.TestSize]'*100, ...
    'labels', {'Larger Class', 'KNN  N=9'});
ylabel('Error rate (%)');

disp('Knn9 Vs Larger class')
[H,P,CI] =ttest(Error_knn9,Error_lg_class );
P
CI
if H    
    disp('Classifiers are significantly different');
else
    disp('Classifiers are NOT significantly different');
end
% Knn13/Knn9
% 
% P =
% 
%     0.0665
% 
% 
% CI =
% 
%    -0.1260    3.1260
% 
% Classifiers are NOT significantly different
% KNN13/LC
% 
% P =
% 
%     0.0343
% 
% 
% CI =
% 
%     0.1840    3.8160
% 
% Classifiers are significantly different
% Knn9 Vs Larger class
% 
% P =
% 
%     0.6008
% 
% 
% CI =
% 
%    -1.5856    2.5856
% 
% Classifiers are NOT significantly different

%% FWI Dataset Crossvalidation


X=FWI;
% Create 10-fold crossvalidation partition for evaluation
K = 10;
CV = cvpartition(y, 'Kfold', K);

Distribution = 'mvmn';
Prior = 'empirical';
Distance = 'euclidean'; 

% Initialize variables
Error_dt6 = nan(1,K);
Error_knn9 = nan(1,K);
Error_lg_class = nan(1,K);

% For each crossvalidation fold
for k = 1:K
    fprintf('Crossvalidation fold %d/%d\n', k, K);
    
    % Extract the training and test set
    X_train = X(CV.training(k), :); 
    y_train = y(CV.training(k));
    X_test = X(CV.test(k), :); 
    y_test = y(CV.test(k));


    % Decision tree with pruning 6
        T = classregtree(X_train, classNames(y_train+1), ...
        'method', 'classification', ...
        'splitcriterion', 'gdi', ...
        'categorical', [], ...
        'names', attributeNames_fwi, ...
        'prune', 'on', ...
        'minparent', 10);
     Error_dt6(k) =sum(~strcmp(classNames(y_test+1), eval(T, X_test, 6)));
     %KNN with neighb=9
     
     y_test_est = knnclassify(X_test, X_train, y_train, 9, Distance);
        
     
        Error_knn9(k) = sum(y_test~=y_test_est); % Count the number of errors
        
     %Largest class
      [larg_class,nb]=mode(y_train);
    y_test_est=larg_class*ones(CV.TestSize(1,k),1);
    Error_lg_class(k) =sum(y_test_est~=y_test);
end



% Determine if classifiers are significantly different
mfig('Error rates DT6/Knn9 ');
boxplot([Error_dt6./CV.TestSize; Error_knn9./CV.TestSize]'*100, ...
    'labels', {'DT pruning=6', 'KNN  N=9'});
ylabel('Error rate (%)');

disp('DT6/Knn9')
[H,P,CI] =ttest(Error_dt6, Error_knn9);
fprintf('P=%f and CI=[%f:%f]\n',P,CI(1),CI(2))
if H    
    disp('Classifiers are significantly different');
else
    disp('Classifiers are NOT significantly different');
end


% Determine if classifiers are significantly different
mfig('Error rates Error dt6/LC');
boxplot([Error_dt6./CV.TestSize; Error_lg_class./CV.TestSize]'*100, ...
    'labels', {'DT pruning=6', 'Larger Class'});
ylabel('Error rate (%)');

disp('dt6/LC')
[H,P,CI] =ttest(Error_dt6, Error_lg_class);
fprintf('P=%f and CI=[%f:%f]\n',P,CI(1),CI(2))
if H
    disp('Classifiers are significantly different');
else
    disp('Classifiers are NOT significantly different');
end

% Determine if classifiers are significantly different
mfig('Error rates KNN9/LC');
boxplot([Error_lg_class./CV.TestSize; Error_knn9./CV.TestSize]'*100, ...
    'labels', {'Larger Class', 'KNN  N=9'});
ylabel('Error rate (%)');

disp('Knn9 Vs Larger class')
[H,P,CI] =ttest(Error_knn9, Error_lg_class);
fprintf('P=%f and CI=[%f:%f]\n',P,CI(1),CI(2))
if H
    disp('Classifiers are significantly different');
    
else
    disp('Classifiers are NOT significantly different');
end
% DT6/Knn9
% P=0.112506 and CI=[-0.486752:3.886752]
% Classifiers are NOT significantly different
% dt6/LC
% P=0.088219 and CI=[-0.348290:4.148290]
% Classifiers are NOT significantly different
% Knn9 Vs Larger class
% P=0.817275 and CI=[-1.701648:2.101648]
% Classifiers are NOT significantly different
%% STM Crossvalidation

X=STM;
% Create 10-fold crossvalidation partition for evaluation
K = 10;
CV = cvpartition(y, 'Kfold', K);
Distribution = 'mvmn';
Prior = 'empirical';
Distance = 'euclidean'; 

% Initialize variables
Error_bayes = nan(1,K);
Error_knn17 = nan(1,K);
Error_lg_class = nan(1,K);

% For each crossvalidation fold
for k = 1:K
    fprintf('Crossvalidation fold %d/%d\n', k, K);
    
    % Extract the training and test set
    X_train = X(CV.training(k), :); 
    y_train = y(CV.training(k));
    X_test = X(CV.test(k), :); 
    y_test = y(CV.test(k));


    % Fit naive Bayes classifier to training set
    NB = NaiveBayes.fit(X_train, y_train, 'Distribution', Distribution, 'Prior', Prior);
    
    % Predict model on test data    
    y_test_est = predict(NB, X_test);
    
    % Compute error rate
    Error_bayes(k) = sum(y_test~=y_test_est); % Count the number of errors
     %KNN with neighb=17
     
     y_test_est = knnclassify(X_test, X_train, y_train, 17, Distance);
        
     
        Error_knn17(k) = sum(y_test~=y_test_est); % Count the number of errors
        
     %Largest class
      [larg_class,nb]=mode(y_train);
    y_test_est=larg_class*ones(CV.TestSize(1,k),1);
    Error_lg_class(k) =sum(y_test_est~=y_test);
end



% Determine if classifiers are significantly different
mfig('Error rates bayes/Knn17 ');
boxplot([Error_bayes./CV.TestSize; Error_knn17./CV.TestSize]'*100, ...
    'labels', {'Naive Bayes', 'KNN  N=17'});
ylabel('Error rate (%)');

disp('bayes/Knn17')
[H,P,CI] =ttest(Error_bayes, Error_knn17);
fprintf('P=%f and CI=[%f:%f]\n',P,CI(1),CI(2))
if H    
    disp('Classifiers are significantly different');
else
    disp('Classifiers are NOT significantly different');
end


% Determine if classifiers are significantly different
mfig('Error rates bayes/LC');
boxplot([Error_bayes./CV.TestSize; Error_lg_class./CV.TestSize]'*100, ...
    'labels', {'Naive Bayes', 'Larger Class'});
ylabel('Error rate (%)');

disp('bayes/LC')
[H,P,CI] =ttest(Error_bayes, Error_lg_class);
fprintf('P=%f and CI=[%f:%f]\n',P,CI(1),CI(2))
if H
    disp('Classifiers are significantly different');
else
    disp('Classifiers are NOT significantly different');
end

% Determine if classifiers are significantly different
mfig('Error rates KNN17/LC');
boxplot([Error_lg_class./CV.TestSize; Error_knn17./CV.TestSize]'*100, ...
    'labels', {'Larger Class', 'KNN  N=17'});
ylabel('Error rate (%)');

disp('Knn17 Vs Larger class')
[H,P,CI] =ttest(Error_knn17, Error_lg_class);
fprintf('P=%f and CI=[%f:%f]\n',P,CI(1),CI(2))
if H
    disp('Classifiers are significantly different');
    
else
    disp('Classifiers are NOT significantly different');
end

% bayes/Knn17
% P=0.001510 and CI=[3.076366:9.323634]
% Classifiers are significantly different
% bayes/LC
% P=0.000271 and CI=[4.617916:10.582084]
% Classifiers are significantly different
% Knn17 Vs Larger class
% P=0.088590 and CI=[-0.258915:3.058915]
% Classifiers are NOT significantly different

%% STFWI Crossvalidation

X=STFWI;
% Create 10-fold crossvalidation partition for evaluation
K = 10;
CV = cvpartition(y, 'Kfold', K);
Distribution = 'mvmn';
Prior = 'empirical';
Distance = 'euclidean'; 


% Initialize variables
Error_dt7 = nan(1,K);
Error_knn11 = nan(1,K);
Error_lg_class = nan(1,K);

% For each crossvalidation fold
for k = 1:K
    fprintf('Crossvalidation fold %d/%d\n', k, K);
    
    % Extract the training and test set
    X_train = X(CV.training(k), :); 
    y_train = y(CV.training(k));
    X_test = X(CV.test(k), :); 
    y_test = y(CV.test(k));


    % Decision tree with pruning 7
        T = classregtree(X_train, classNames(y_train+1), ...
        'method', 'classification', ...
        'splitcriterion', 'gdi', ...
        'categorical', [], ...
        'names', attributeNames_stfwi, ...
        'prune', 'on', ...
        'minparent', 10);
     Error_dt7(k) =sum(~strcmp(classNames(y_test+1), eval(T, X_test, 7)));
     %KNN with neighb=11
     
     y_test_est = knnclassify(X_test, X_train, y_train, 11, Distance);
        
     
        Error_knn11(k) = sum(y_test~=y_test_est); % Count the number of errors
        
     %Largest class
      [larg_class,nb]=mode(y_train);
    y_test_est=larg_class*ones(CV.TestSize(1,k),1);
    Error_lg_class(k) =sum(y_test_est~=y_test);
end



% Determine if classifiers are significantly different
mfig('Error rates DT7/Knn11 ');
boxplot([Error_dt7./CV.TestSize; Error_knn11./CV.TestSize]'*100, ...
    'labels', {'DT pruning=7', 'KNN  N=11'});
ylabel('Error rate (%)');

disp('DT7/Knn11')
[H,P,CI] =ttest(Error_dt7, Error_knn11);
fprintf('P=%f and CI=[%f:%f]\n',P,CI(1),CI(2))
if H    
    disp('Classifiers are significantly different');
else
    disp('Classifiers are NOT significantly different');
end


% Determine if classifiers are significantly different
mfig('Error rates Error dt7/LC');
boxplot([Error_dt7./CV.TestSize; Error_lg_class./CV.TestSize]'*100, ...
    'labels', {'DT pruning=7', 'Larger Class'});
ylabel('Error rate (%)');

disp('dt7/LC')
[H,P,CI] =ttest(Error_dt7, Error_lg_class);
fprintf('P=%f and CI=[%f:%f]\n',P,CI(1),CI(2))
if H
    disp('Classifiers are significantly different');
else
    disp('Classifiers are NOT significantly different');
end

% Determine if classifiers are significantly different
mfig('Error rates KNN11/LC');
boxplot([Error_lg_class./CV.TestSize; Error_knn11./CV.TestSize]'*100, ...
    'labels', {'Larger Class', 'KNN  N=11'});
ylabel('Error rate (%)');

disp('Knn11 Vs Larger class')
[H,P,CI] =ttest(Error_knn11, Error_lg_class);
fprintf('P=%f and CI=[%f:%f]\n',P,CI(1),CI(2))
if H
    disp('Classifiers are significantly different');
    
else
    disp('Classifiers are NOT significantly different');
end
% 
% DT7/Knn11
% P=0.014587 and CI=[0.699453:4.900547]
% Classifiers are significantly different
% dt7/LC
% P=0.069829 and CI=[-0.289383:6.089383]
% Classifiers are NOT significantly different
% Knn11 Vs Larger class
% P=0.925336 and CI=[-2.247272:2.447272]
% Classifiers are NOT significantly different

%% MET Crossvalidation

X=MET;
% Create 10-fold crossvalidation partition for evaluation
K = 10;
CV = cvpartition(y, 'Kfold', K);

Distribution = 'mvmn';
Prior = 'empirical';
Distance = 'euclidean'; 

% Initialize variables
Error_dt9 = nan(1,K);
Error_knn20 = nan(1,K);
Error_lg_class = nan(1,K);

% For each crossvalidation fold
for k = 1:K
    fprintf('Crossvalidation fold %d/%d\n', k, K);
    
    % Extract the training and test set
    X_train = X(CV.training(k), :); 
    y_train = y(CV.training(k));
    X_test = X(CV.test(k), :); 
    y_test = y(CV.test(k));


    % Decision tree with pruning 9
        T = classregtree(X_train, classNames(y_train+1), ...
        'method', 'classification', ...
        'splitcriterion', 'gdi', ...
        'categorical', [], ...
        'names', attributeNames_met, ...
        'prune', 'on', ...
        'minparent', 10);
     Error_dt9(k) =sum(~strcmp(classNames(y_test+1), eval(T, X_test, 9)));
     %KNN with neighb=20
     
     y_test_est = knnclassify(X_test, X_train, y_train, 20, Distance);
        
     
        Error_knn20(k) = sum(y_test~=y_test_est); % Count the number of errors
        
     %Largest class
      [larg_class,nb]=mode(y_train);
    y_test_est=larg_class*ones(CV.TestSize(1,k),1);
    Error_lg_class(k) =sum(y_test_est~=y_test);
end



% Determine if classifiers are significantly different
mfig('Error rates DT9/Knn20 ');
boxplot([Error_dt9./CV.TestSize; Error_knn20./CV.TestSize]'*100, ...
    'labels', {'DT pruning=9', 'KNN  N=20'});
ylabel('Error rate (%)');

disp('DT9/Knn20')
[H,P,CI] =ttest(Error_dt9, Error_knn20);
fprintf('P=%f and CI=[%f:%f]\n',P,CI(1),CI(2))
if H    
    disp('Classifiers are significantly different');
else
    disp('Classifiers are NOT significantly different');
end


% Determine if classifiers are significantly different
mfig('Error rates Error dt9/LC');
boxplot([Error_dt9./CV.TestSize; Error_lg_class./CV.TestSize]'*100, ...
    'labels', {'DT pruning=9', 'Larger Class'});
ylabel('Error rate (%)');

disp('dt9/LC')
[H,P,CI] =ttest(Error_dt9, Error_lg_class);
fprintf('P=%f and CI=[%f:%f]\n',P,CI(1),CI(2))
if H
    disp('Classifiers are significantly different');
else
    disp('Classifiers are NOT significantly different');
end

% Determine if classifiers are significantly different
mfig('Error rates KNN20/LC');
boxplot([Error_lg_class./CV.TestSize; Error_knn20./CV.TestSize]'*100, ...
    'labels', {'Larger Class', 'KNN  N=20'});
ylabel('Error rate (%)');

disp('Knn20 Vs Larger class')
[H,P,CI] =ttest(Error_knn20, Error_lg_class);
fprintf('P=%f and CI=[%f:%f]\n',P,CI(1),CI(2))
if H
    disp('Classifiers are significantly different');
    
else
    disp('Classifiers are NOT significantly different');
end

% DT9/Knn20
% P=0.403568 and CI=[-3.222922:1.422922]
% Classifiers are NOT significantly different
% dt9/LC
% P=0.890531 and CI=[-1.497808:1.697808]
% Classifiers are NOT significantly different
% Knn20 Vs Larger class
% P=0.195389 and CI=[-0.617262:2.617262]
% Classifiers are NOT significantly different